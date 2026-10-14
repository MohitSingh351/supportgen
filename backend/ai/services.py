"""Orchestrates all AI components for one ticket."""
import os
import time
from dataclasses import asdict, dataclass, field

from . import assist
from .classify import TicketClassifier
from .providers import GenerationError, LocalTransformerProvider, ProviderUnavailable, build_prompt
from .sentiment import SentimentAnalyzer


@dataclass
class AnalysisResult:
    summary: str
    summary_source: str            # "generated" | "extractive_fallback"
    category: str | None
    category_confidence: float | None
    priority: str | None
    priority_confidence: float | None
    ticket_type: str | None
    type_confidence: float | None
    sentiment: str | None          # inferred by a pretrained model, not a dataset label
    sentiment_score: float | None
    recommended_action: str
    action_source: str             # "rule_based" | "generated"
    generated_response: str
    model_used: str
    timings_ms: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class TicketAnalyzer:
    def __init__(self, generator: LocalTransformerProvider, classifier: TicketClassifier,
                 sentiment: SentimentAnalyzer | None, aux: LocalTransformerProvider | None,
                 action_mode: str | None = None):
        self.generator = generator
        self.classifier = classifier
        self.sentiment = sentiment
        self.aux = aux
        self.action_mode = action_mode or os.getenv("SUPPORTGEN_ACTION_MODE", "rules")

    def status(self) -> dict:
        return {
            "response_model": {"name": self.generator.model_id, "available": self.generator.is_available(),
                               "path": self.generator.source},
            "classifiers": {"available": self.classifier.is_available(), "path": str(self.classifier.classifier_dir)},
            "summary_model": self.aux.model_id if self.aux else None,
            "sentiment_model": self.sentiment.model_name if self.sentiment else None,
            "action_mode": self.action_mode,
        }

    def analyze(self, subject: str, body: str) -> AnalysisResult:
        timings, warnings = {}, []

        # 1. Core GenAI component: fine-tuned FLAN-T5 response. Required — errors propagate (→ HTTP 503/500).
        t0 = time.perf_counter()
        response = self.generator.generate(subject, body)
        timings["response"] = int((time.perf_counter() - t0) * 1000)

        # 2. Classifiers (optional: the response is still useful without them)
        preds = {}
        if self.classifier.is_available():
            t0 = time.perf_counter()
            preds = self.classifier.predict(subject, body)
            timings["classification"] = int((time.perf_counter() - t0) * 1000)
        else:
            warnings.append(f"Classifiers not found at {self.classifier.classifier_dir}.")
        queue, priority, ttype = preds.get("queue_group"), preds.get("priority"), preds.get("type")

        # 3. Inferred sentiment (optional)
        sentiment = None
        if self.sentiment is not None:
            t0 = time.perf_counter()
            sentiment = self.sentiment.predict(f"{subject}. {body}")
            timings["sentiment"] = int((time.perf_counter() - t0) * 1000)
            if sentiment is None:
                warnings.append("Sentiment model unavailable.")

        # 4. Summary: zero-shot generation with extractive fallback
        summary, summary_source = None, "extractive_fallback"
        if self.aux is not None:
            t0 = time.perf_counter()
            try:
                generated = self.aux.generate_from_prompt(
                    build_prompt(assist.SUMMARY_TEMPLATE, subject, body), max_new_tokens=64)
                if not assist.is_degenerate(generated, subject):
                    summary, summary_source = generated, "generated"
            except (ProviderUnavailable, GenerationError) as exc:
                warnings.append(f"Summary model unavailable: {exc}")
            timings["summary"] = int((time.perf_counter() - t0) * 1000)
        if summary is None:
            summary = assist.extractive_summary(body)

        # 5. Recommended action: rules by default, optional zero-shot generation
        action, action_source = None, "rule_based"
        if self.action_mode == "generate" and self.aux is not None:
            try:
                generated = self.aux.generate_from_prompt(
                    build_prompt(assist.ACTION_TEMPLATE, subject, body), max_new_tokens=48)
                if not assist.is_degenerate(generated, subject, min_words=4):
                    action, action_source = generated, "generated"
            except (ProviderUnavailable, GenerationError) as exc:
                warnings.append(f"Action generation unavailable: {exc}")
        if action is None:
            action = assist.rule_based_action(queue.label if queue else None, priority.label if priority else None)

        return AnalysisResult(
            summary=summary, summary_source=summary_source,
            category=queue.label if queue else None, category_confidence=queue.confidence if queue else None,
            priority=priority.label if priority else None, priority_confidence=priority.confidence if priority else None,
            ticket_type=ttype.label if ttype else None, type_confidence=ttype.confidence if ttype else None,
            sentiment=sentiment[0] if sentiment else None, sentiment_score=sentiment[1] if sentiment else None,
            recommended_action=action, action_source=action_source,
            generated_response=response.text, model_used=response.model,
            timings_ms=timings, warnings=warnings,
        )
