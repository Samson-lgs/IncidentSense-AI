from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import DATA_PATH, METRICS_PATH
from .data import load_incidents
from .inference import InferenceEngine
from .monitoring import drift_status, numeric_drift
from .similarity import SimilarIncidentIndex

engine = None
similarity = None
reference_df = None
startup_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, similarity, reference_df, startup_error
    try:
        engine = InferenceEngine()
        reference_df = load_incidents(DATA_PATH)
        similarity = SimilarIncidentIndex().fit(reference_df)
        startup_error = None
    except Exception as exc:
        engine = None
        similarity = None
        reference_df = None
        startup_error = str(exc)
    yield


app = FastAPI(
    title="IncidentSense AI API",
    version="0.1.0",
    description="ML-powered incident classification, anomaly detection, similarity search and monitoring.",
    lifespan=lifespan,
)


class IncidentRequest(BaseModel):
    timestamp: str = Field(default="2026-09-23T10:00:00Z")
    service: str
    environment: str = "production"
    severity: str = "ERROR"
    error_code: str = "UNKNOWN"
    http_status: int = 500
    latency_ms: float = 500.0
    log_message: str


def _frame(req: IncidentRequest) -> pd.DataFrame:
    return pd.DataFrame([req.model_dump()])


@app.get("/health")
def health():
    return {
        "status": "ok" if engine is not None else "degraded",
        "model_loaded": engine is not None,
        "similarity_index_loaded": similarity is not None,
        "startup_error": startup_error,
    }


@app.get("/metrics")
def metrics():
    if not Path(METRICS_PATH).exists():
        raise HTTPException(status_code=404, detail="Training metrics not found. Run scripts/train_model.py first.")
    return json.loads(Path(METRICS_PATH).read_text())


@app.post("/predict")
def predict(req: IncidentRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
    df = _frame(req)
    return {**engine.predict(df), "explanation": engine.explain(df)}


@app.post("/analyze")
def analyze(req: IncidentRequest):
    if engine is None or similarity is None:
        raise HTTPException(status_code=503, detail="ML services are not loaded")
    df = _frame(req)
    pred = engine.predict(df)
    anomaly = engine.anomaly.score(df)
    similar = similarity.query(df, top_k=5)
    return {"prediction": pred, "anomaly": anomaly, "similar_incidents": similar}


@app.post("/drift")
def drift(sample: list[IncidentRequest]):
    if reference_df is None:
        raise HTTPException(status_code=503, detail="Reference dataset is not loaded")
    if not sample:
        raise HTTPException(status_code=400, detail="Sample batch cannot be empty")
    current = pd.DataFrame([x.model_dump() for x in sample])
    scores = numeric_drift(reference_df, current)
    return {"metrics": {k: {"psi": v, "status": drift_status(v)} for k, v in scores.items()}}
