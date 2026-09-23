from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "demo_incidents.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = ARTIFACT_DIR / "root_cause_model.joblib"
ANOMALY_PATH = ARTIFACT_DIR / "anomaly_model.joblib"
REFERENCE_STATS_PATH = ARTIFACT_DIR / "reference_stats.json"
METRICS_PATH = REPORT_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "service",
    "environment",
    "severity",
    "error_code",
    "http_status",
    "latency_ms",
    "log_message",
]
TARGET_COLUMN = "root_cause"
