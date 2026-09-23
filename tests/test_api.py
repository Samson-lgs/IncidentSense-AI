from fastapi.testclient import TestClient

from incidentsense.api import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in {"ok", "degraded"}


def test_predict():
    payload = {
        "service": "orders",
        "environment": "production",
        "severity": "ERROR",
        "error_code": "DB_QUERY",
        "http_status": 500,
        "latency_ms": 1300,
        "log_message": "database query exceeded timeout",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code in {200, 503}
        if response.status_code == 200:
            assert "root_cause" in response.json()
