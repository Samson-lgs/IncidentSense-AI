import pandas as pd

from incidentsense.features import build_text, prepare_model_frame


def test_prepare_model_frame_adds_model_features():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-09-23T10:00:00Z",
                "service": "payments",
                "environment": "production",
                "severity": "ERROR",
                "error_code": "TIMEOUT",
                "http_status": 500,
                "latency_ms": 1000,
                "log_message": "operation timed out",
            }
        ]
    )
    out = prepare_model_frame(df)
    assert "model_text" in out.columns
    assert out.loc[0, "severity_level"] == 2
    assert "TIMEOUT" in build_text(df).iloc[0]
