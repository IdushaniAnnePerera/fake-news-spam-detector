from __future__ import annotations

from typing import Any, Dict

from transformers import pipeline


class TransformerDetector:
    def __init__(self) -> None:
        self._classifier = None

    def _load(self):
        if self._classifier is None:
            self._classifier = pipeline(
                "zero-shot-classification", model="facebook/bart-large-mnli"
            )
        return self._classifier

    def predict(self, task: str, text: str) -> Dict[str, Any]:
        classifier = self._load()

        if task == "spam":
            labels = ["spam", "ham"]
        elif task == "fake_news":
            labels = ["fake", "real"]
        else:
            raise ValueError("Unsupported task for transformer prediction")

        result: Dict[str, Any] = classifier(text, labels, multi_label=False)
        top_label = str(result["labels"][0])
        top_score = float(result["scores"][0])
        scores = {
            str(label): float(score)
            for label, score in zip(result["labels"], result["scores"])
        }

        return {
            "task": task,
            "label": top_label,
            "confidence": round(top_score, 4),
            "scores": scores,
            "model": "transformer",
        }
