from pathlib import Path

import pandas as pd

from incidentsense.config import ANOMALY_PATH, MODEL_PATH
from incidentsense.inference import InferenceEngine


def test_artifacts_exist():
    assert Path(MODEL_PATH).exists()
    assert Path(ANOMALY_PATH).exists()


def test_prediction_shape():
    engine = InferenceEngine()
    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-09-23T10:00:00Z",
                "service": "orders",
                "environment": "production",
                "severity": "ERROR",
                "error_code": "DB_QUERY",
                "http_status": 500,
                "latency_ms": 1400,
                "log_message": "request failed after query exceeded timeout",
            }
        ]
    )
    result = engine.predict(df)
    assert result["root_cause"] in engine.pipeline.classes_
    assert 0 <= result["confidence"] <= 1
