"""Configuration for the AI layer, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass
from pathlib import Path

# backend/ai/config.py → project root is two levels above backend/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env_name: str, default: str) -> Path:
    path = Path(os.getenv(env_name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class AIConfig:
    model_dir: Path            # fine-tuned FLAN-T5 exported from Colab
    classifier_dir: Path       # TF-IDF classifiers exported from Colab
    results_dir: Path          # evaluation results exported from Colab
    base_model: str            # pretrained FLAN-T5 (Model A) for the live comparison
    aux_model: str             # zero-shot model for summary / recommended action
    sentiment_model: str       # pretrained sentiment model (inferred, not evaluated)
    device: str                # "cpu", "mps", "cuda" or "auto"
    max_ticket_chars: int
    gemini_api_key: str
    gemini_model: str


def load_config() -> AIConfig:
    return AIConfig(
        model_dir=_path("SUPPORTGEN_MODEL_DIR", "models/flan-t5-supportgen"),
        classifier_dir=_path("SUPPORTGEN_CLASSIFIER_DIR", "models/classifiers"),
        results_dir=_path("SUPPORTGEN_RESULTS_DIR", "results"),
        base_model=os.getenv("SUPPORTGEN_BASE_MODEL", "google/flan-t5-small"),
        aux_model=os.getenv("SUPPORTGEN_AUX_MODEL", "google/flan-t5-base"),
        sentiment_model=os.getenv("SUPPORTGEN_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
        # Measured on an M4: FLAN-T5-small generates faster on CPU (~0.9 s) than on MPS (~1.7 s).
        device=os.getenv("SUPPORTGEN_DEVICE", "cpu"),
        max_ticket_chars=int(os.getenv("SUPPORTGEN_MAX_TICKET_CHARS", "5000")),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    )


def resolve_device(preference: str) -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
