from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import MODEL_TEXT_COLUMN, prepare_model_frame, numeric_matrix


TEXT_COL = MODEL_TEXT_COLUMN
CAT_COLS = ["service", "environment", "severity", "error_code"]
NUM_COLS = ["latency_ms", "http_status", "severity_level", "hour"]


@dataclass
class RootCauseModel:
    pipeline: Pipeline

    def predict(self, df: pd.DataFrame) -> tuple[str, float, dict[str, float]]:
        features = prepare_model_frame(df)
        probs = self.pipeline.predict_proba(features)[0]
        classes = self.pipeline.classes_
        idx = int(np.argmax(probs))
        return str(classes[idx]), float(probs[idx]), {str(c): float(p) for c, p in zip(classes, probs)}


def make_root_cause_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "text",
                Pipeline(
                    [
                        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000)),
                    ]
                ),
                TEXT_COL,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CAT_COLS,
            ),
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUM_COLS,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", preprocessor),
            ("model", LogisticRegression(max_iter=1800, class_weight="balanced", C=1.2)),
        ]
    )


class AnomalyModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=250, contamination=0.04, random_state=42)

    def fit(self, df: pd.DataFrame) -> "AnomalyModel":
        matrix = numeric_matrix(df)
        scaled = self.scaler.fit_transform(matrix)
        self.model.fit(scaled)
        return self

    def score(self, df: pd.DataFrame) -> dict:
        matrix = numeric_matrix(df)
        scaled = self.scaler.transform(matrix)
        raw = float(self.model.decision_function(scaled)[0])
        label = int(self.model.predict(scaled)[0])
        return {"is_anomaly": label == -1, "anomaly_score": raw}

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "AnomalyModel":
        return joblib.load(path)
