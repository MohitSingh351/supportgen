"""Process-wide singletons: every model is created once and loaded lazily on first use,
never per request."""
from functools import lru_cache

from .classify import TicketClassifier
from .config import load_config, resolve_device
from .providers import GeminiProvider, LocalTransformerProvider
from .sentiment import SentimentAnalyzer
from .services import TicketAnalyzer


@lru_cache(maxsize=1)
def get_config():
    return load_config()


@lru_cache(maxsize=1)
def get_finetuned_provider() -> LocalTransformerProvider:
    cfg = get_config()
    return LocalTransformerProvider("finetuned", cfg.model_dir, device=resolve_device(cfg.device))


@lru_cache(maxsize=1)
def get_base_provider() -> LocalTransformerProvider:
    """Original pretrained FLAN-T5 with the same prompt/decoding as the fine-tuned model."""
    cfg = get_config()
    ft = get_finetuned_provider()
    return LocalTransformerProvider("base", cfg.base_model, device=resolve_device(cfg.device),
                                    prompt_template=ft.prompt_template, max_input_len=ft.max_input_len,
                                    max_new_tokens=ft.max_new_tokens, generation_kwargs=ft.generation_kwargs)


@lru_cache(maxsize=1)
def get_gemini_provider() -> GeminiProvider:
    cfg = get_config()
    return GeminiProvider(cfg.gemini_api_key, cfg.gemini_model, prompt_template=get_finetuned_provider().prompt_template)


@lru_cache(maxsize=1)
def get_analyzer() -> TicketAnalyzer:
    cfg = get_config()
    device = resolve_device(cfg.device)
    aux = LocalTransformerProvider("aux", cfg.aux_model, device=device, max_input_len=256,
                                   clean_output=True) if cfg.aux_model else None
    sentiment = SentimentAnalyzer(cfg.sentiment_model, device=device) if cfg.sentiment_model else None
    return TicketAnalyzer(get_finetuned_provider(), TicketClassifier(cfg.classifier_dir), sentiment, aux)
