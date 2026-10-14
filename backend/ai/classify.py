"""TF-IDF + linear-model classifiers trained in Colab (notebook 05)."""
import json
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Prediction:
    label: str
    confidence: float


class TicketClassifier:
    """Loads ``queue_group``, ``priority`` and ``type`` pipelines once, on first use."""

    def __init__(self, classifier_dir: Path):
        self.classifier_dir = Path(classifier_dir)
        self._pipelines = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return (self.classifier_dir / "meta.json").exists()

    def _ensure_loaded(self):
        if self._pipelines is not None:
            return
        with self._lock:
            if self._pipelines is not None:
                return
            import joblib

            meta = json.loads((self.classifier_dir / "meta.json").read_text())
            self._pipelines = {t: joblib.load(self.classifier_dir / f"{t}.joblib") for t in meta["targets"]}

    @staticmethod
    def to_text(subject: str, body: str) -> str:
        # must match the training input in notebook 05: subject + " . " + body
        return f"{subject.strip()} . {body.strip()}"

    def predict(self, subject: str, body: str) -> dict[str, Prediction]:
        self._ensure_loaded()
        text = self.to_text(subject, body)
        out = {}
        for target, pipe in self._pipelines.items():
            proba = pipe.predict_proba([text])[0]
            best = int(proba.argmax())
            out[target] = Prediction(str(pipe.classes_[best]), round(float(proba[best]), 4))
        return out
