from __future__ import annotations

import json
import os

import joblib

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from .config import ANOMALY_PATH, ARTIFACT_DIR, DATA_PATH, METRICS_PATH, MODEL_PATH, REPORT_DIR
from .data import load_incidents
from .features import prepare_model_frame
from .models import AnomalyModel, make_root_cause_pipeline


def train() -> dict:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = load_incidents(DATA_PATH)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["root_cause"], random_state=42
    )

    X_train = prepare_model_frame(train_df.drop(columns=["root_cause"]))
    y_train = train_df["root_cause"]
    X_test = prepare_model_frame(test_df.drop(columns=["root_cause"]))
    y_test = test_df["root_cause"]

    pipeline = make_root_cause_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "weighted_f1": float(f1_score(y_test, preds, average="weighted")),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }

    joblib.dump(pipeline, MODEL_PATH)
    anomaly = AnomalyModel().fit(train_df)
    anomaly.save(ANOMALY_PATH)

    if mlflow is not None:
        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
            mlflow.set_experiment("IncidentSense-AI")
            with mlflow.start_run(run_name="baseline-logistic-regression"):
                mlflow.log_params({"model": "TF-IDF + LogisticRegression", "seed": 42})
                mlflow.log_metrics({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
                mlflow.sklearn.log_model(pipeline, "root_cause_model")
        except Exception as exc:
            metrics["mlflow_warning"] = str(exc)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps({k: v for k, v in result.items() if k != "classification_report"}, indent=2))
