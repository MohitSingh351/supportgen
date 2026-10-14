"""Response-generation providers.

``GenerationProvider`` is the common interface; the web app and the model-comparison
endpoint only talk to this interface, so a local Transformer and the optional Gemini
benchmark are interchangeable.
"""
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .postprocess import clean_generated_text

DEFAULT_PROMPT = (
    "Generate a professional customer support response.\n"
    "Subject: {subject}\n"
    "Customer ticket: {body}"
)
DEFAULT_GENERATION = {"num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True}


class ProviderUnavailable(RuntimeError):
    """The provider cannot be used (missing model files, missing API key, ...)."""


class GenerationError(RuntimeError):
    """The provider was available but generation failed."""


@dataclass
class GenerationResult:
    text: str
    provider: str
    model: str
    latency_ms: int


def build_prompt(template: str, subject: str, body: str) -> str:
    return template.format(subject=subject.strip() or "(no subject)", body=body.strip())


class GenerationProvider(ABC):
    name: str = "provider"

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def generate(self, subject: str, body: str) -> GenerationResult:
        ...


class LocalTransformerProvider(GenerationProvider):
    """A seq2seq Transformer (FLAN-T5) loaded once and kept in memory.

    ``source`` is either a local directory (our fine-tuned model) or a Hugging Face
    model id (the original pretrained model). If the directory contains
    ``training_config.json`` the prompt, max lengths and decoding settings used
    during training are reused, so training and inference stay consistent.
    """

    def __init__(self, name: str, source: str | Path, device: str = "cpu",
                 prompt_template: str = DEFAULT_PROMPT, max_input_len: int = 160,
                 max_new_tokens: int = 128, generation_kwargs: dict | None = None,
                 clean_output: bool = True):
        self.name = name
        self.source = str(source)
        self.device = device
        self.prompt_template = prompt_template
        self.max_input_len = max_input_len
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = dict(generation_kwargs or DEFAULT_GENERATION)
        self.clean_output = clean_output
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._is_local = Path(self.source).is_absolute() or Path(self.source).exists()
        self._load_training_config()

    def _load_training_config(self):
        cfg_path = Path(self.source) / "training_config.json"
        if self._is_local and cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            self.prompt_template = cfg.get("prompt_template", self.prompt_template)
            self.max_input_len = cfg.get("max_input_len", self.max_input_len)
            self.max_new_tokens = cfg.get("max_target_len", self.max_new_tokens)
            self.generation_kwargs = cfg.get("generation", self.generation_kwargs)

    @property
    def model_id(self) -> str:
        return Path(self.source).name if self._is_local else self.source

    def is_available(self) -> bool:
        if self._is_local:
            return (Path(self.source) / "config.json").exists()
        return True  # Hugging Face id: downloaded on first use

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.is_available():
                raise ProviderUnavailable(
                    f"Model not found at {self.source}. Unzip the Colab export "
                    "(supportgen_models.zip) into the project folder or set SUPPORTGEN_MODEL_DIR.")
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(self.source)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.source).to(self.device).eval()
            except Exception as exc:  # corrupt files, no network for a hub id, ...
                raise ProviderUnavailable(f"Could not load model '{self.source}': {exc}") from exc
            self._tokenizer, self._model = tokenizer, model

    def generate_from_prompt(self, prompt: str, max_new_tokens: int | None = None) -> str:
        import torch

        self._ensure_loaded()
        try:
            enc = self._tokenizer(prompt, max_length=self.max_input_len, truncation=True,
                                  return_tensors="pt").to(self.device)
            with self._lock, torch.no_grad():
                ids = self._model.generate(**enc, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                           **self.generation_kwargs)
            text = self._tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        except Exception as exc:
            raise GenerationError(f"{self.name} generation failed: {exc}") from exc
        return clean_generated_text(text) if self.clean_output else text

    def generate(self, subject: str, body: str) -> GenerationResult:
        t0 = time.perf_counter()
        text = self.generate_from_prompt(build_prompt(self.prompt_template, subject, body))
        return GenerationResult(text, self.name, self.model_id, int((time.perf_counter() - t0) * 1000))


class GeminiProvider(GenerationProvider):
    """Optional benchmark. Only available when GEMINI_API_KEY is set and google-genai is installed."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, prompt_template: str = DEFAULT_PROMPT):
        self.api_key = api_key
        self.model = model
        self.prompt_template = prompt_template
        self._client = None

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import google.genai  # noqa: F401
        except ImportError:
            return False
        return True

    def generate(self, subject: str, body: str) -> GenerationResult:
        if not self.is_available():
            raise ProviderUnavailable("Gemini is not configured (set GEMINI_API_KEY and install google-genai).")
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        t0 = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=build_prompt(self.prompt_template, subject, body),
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=1024))
        except Exception as exc:
            raise GenerationError(f"Gemini request failed: {exc}") from exc
        return GenerationResult((response.text or "").strip(), self.name, self.model,
                                int((time.perf_counter() - t0) * 1000))
