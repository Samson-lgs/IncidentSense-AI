from __future__ import annotations

import joblib

from .config import ANOMALY_PATH, MODEL_PATH
from .features import prepare_model_frame
from .models import AnomalyModel


class InferenceEngine:
    def __init__(self, model_path=MODEL_PATH, anomaly_path=ANOMALY_PATH):
        self.pipeline = joblib.load(model_path)
        self.anomaly = AnomalyModel.load(anomaly_path)

    def predict(self, df):
        features = prepare_model_frame(df)
        probabilities = self.pipeline.predict_proba(features)[0]
        classes = self.pipeline.classes_
        idx = int(probabilities.argmax())
        label = str(classes[idx])
        confidence = float(probabilities[idx])
        top = sorted(
            ((str(c), float(p)) for c, p in zip(classes, probabilities)),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        return {"root_cause": label, "confidence": confidence, "top_probabilities": top}

    def explain(self, df, limit=6):
        features = self.pipeline.named_steps["features"]
        model = self.pipeline.named_steps["model"]
        class_names = list(self.pipeline.classes_)
        pred_idx = class_names.index(self.predict(df)["root_cause"])
        names = features.get_feature_names_out()
        weights = model.coef_[pred_idx]
        order = weights.argsort()[::-1]
        return [{"feature": names[i], "weight": round(float(weights[i]), 4)} for i in order[:limit]]
