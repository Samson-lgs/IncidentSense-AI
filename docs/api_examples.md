# API Examples

## Health

`GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "similarity_index_loaded": true,
  "startup_error": null
}
```

## Predict

`POST /predict`

```json
{
  "service": "orders",
  "environment": "production",
  "severity": "ERROR",
  "error_code": "DB_QUERY",
  "http_status": 500,
  "latency_ms": 1300,
  "log_message": "database query exceeded timeout"
}
```

The response contains the predicted root cause, confidence, top class probabilities, and model-weighted explanation features.
