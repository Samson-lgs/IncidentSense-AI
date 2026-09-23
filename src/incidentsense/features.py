from __future__ import annotations

import pandas as pd

MODEL_TEXT_COLUMN = "model_text"


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return out


def build_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["log_message"].astype(str)
        + " error_code="
        + df["error_code"].astype(str)
        + " service="
        + df["service"].astype(str)
        + " severity="
        + df["severity"].astype(str)
    )


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_temporal_features(df)
    out[MODEL_TEXT_COLUMN] = build_text(out)
    severity_map = {"INFO": 0, "WARN": 1, "ERROR": 2, "CRITICAL": 3}
    out["severity_level"] = out["severity"].map(severity_map).fillna(0).astype(float)
    return out


def numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_model_frame(df)
    return out[["latency_ms", "http_status", "severity_level", "hour"]].astype(float)
