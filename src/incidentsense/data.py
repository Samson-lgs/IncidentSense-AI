from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "incident_id",
    "timestamp",
    "service",
    "environment",
    "severity",
    "error_code",
    "http_status",
    "latency_ms",
    "log_message",
    "root_cause",
}


def load_incidents(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Incident dataset is empty")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["log_message"] = df["log_message"].fillna("").astype(str)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce").fillna(0)
    df["http_status"] = pd.to_numeric(df["http_status"], errors="coerce").fillna(0)
    return df
