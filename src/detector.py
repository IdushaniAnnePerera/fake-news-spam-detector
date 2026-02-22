from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np


class NLPDetector:
    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = Path(models_dir)
        self._models: Dict[str, Any] = {}

    def load(self, task: str) -> Any:
        if task in self._models:
            return self._models[task]

        model_file = self.models_dir / f"{task}_detector.joblib"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model for task '{task}' not found at {model_file}. Run train.py first."
            )

        model = joblib.load(model_file)
        self._models[task] = model
        return model

    def predict(self, task: str, text: str) -> Dict[str, Any]:
        model = self.load(task)
        probabilities = model.predict_proba([text])[0]
        best_idx = int(np.argmax(probabilities))
        label = model.classes_[best_idx]
        confidence = float(probabilities[best_idx])

        all_scores = {
            str(cls): float(probabilities[i]) for i, cls in enumerate(model.classes_)
        }

        return {
            "task": task,
            "label": str(label),
            "confidence": round(confidence, 4),
            "scores": all_scores,
            "model": "sklearn",
        }
