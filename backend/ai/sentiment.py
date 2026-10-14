"""Inferred sentiment from a pretrained model.

The dataset has no sentiment labels, so this is an auxiliary signal that was never
evaluated against ground truth. The UI labels it as "inferred".
"""
import threading


class SentimentAnalyzer:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._pipe = None
        self._failed = False
        self._lock = threading.Lock()

    def predict(self, text: str) -> tuple[str, float] | None:
        """Returns (label, score) or None when the model cannot be loaded."""
        if self._failed:
            return None
        if self._pipe is None:
            with self._lock:
                if self._pipe is None and not self._failed:
                    try:
                        from transformers import pipeline

                        self._pipe = pipeline("sentiment-analysis", model=self.model_name, device=self.device)
                    except Exception:
                        self._failed = True
                        return None
        result = self._pipe(text, truncation=True, max_length=512)[0]
        return result["label"].lower(), round(float(result["score"]), 4)
