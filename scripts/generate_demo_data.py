"""Generate a deliberately noisy, deterministic incident dataset for demos."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "demo_incidents.csv"
RNG = random.Random(2026)

ROOT_CAUSES = ["database", "dependency", "auth", "network", "resource", "application"]
SERVICES = ["checkout", "payments", "identity", "orders", "search", "gateway"]
ENVIRONMENTS = ["production", "staging", "qa"]
SEVERITIES = ["INFO", "WARN", "ERROR", "CRITICAL"]
COMMON_MESSAGES = [
    "request failed while processing operation",
    "worker reported an unexpected operational error",
    "request exceeded configured service threshold",
    "operation returned an error and was retried",
    "background job failed during execution",
    "service response could not be completed",
    "transaction did not complete successfully",
    "request aborted after repeated retries",
]
CAUSE_MESSAGES = {
    "database": ["connection pool pressure", "transaction lock contention", "query exceeded timeout", "storage session rejected"],
    "dependency": ["upstream provider unavailable", "third-party response delayed", "external service contract mismatch", "dependency circuit opened"],
    "auth": ["credential validation rejected", "token authorization failed", "identity lookup rejected", "role mapping denied"],
    "network": ["peer reset connection", "name resolution failed", "secure handshake expired", "route to host unavailable"],
    "resource": ["memory pressure detected", "compute saturation detected", "disk capacity pressure", "queue backlog increased"],
    "application": ["invalid application state", "payload validation failed", "unexpected runtime state", "request handler error"],
}
COMMON_CODES = ["TIMEOUT", "SERVICE_ERR", "REQUEST_FAIL", "RETRY_EXHAUSTED", "UPSTREAM_ERR", "VALIDATION_ERR", "CONNECTION_ERR"]
SPECIFIC_CODES = {
    "database": ["DB_POOL", "DB_LOCK", "DB_QUERY"],
    "dependency": ["UPSTREAM_503", "EXT_API"],
    "auth": ["AUTH_401", "TOKEN_ERR"],
    "network": ["DNS_FAIL", "TLS_ERR"],
    "resource": ["MEM_HIGH", "CPU_HIGH", "DISK_HIGH"],
    "application": ["STATE_ERR", "NPE"],
}


def make_row(idx: int) -> dict:
    label = RNG.choice(ROOT_CAUSES)
    service = RNG.choice(SERVICES)
    env = RNG.choices(ENVIRONMENTS, weights=[60, 25, 15])[0]
    severity = RNG.choices(SEVERITIES, weights=[10, 24, 52, 14])[0]

    status = RNG.choices([200, 400, 401, 404, 429, 500, 502, 503], weights=[12, 5, 5, 3, 3, 28, 22, 22])[0]
    base_latency = {
        "database": 1000,
        "dependency": 1200,
        "auth": 350,
        "network": 900,
        "resource": 1400,
        "application": 550,
    }[label]
    latency = max(15, int(RNG.gauss(base_latency, 420)))

    if RNG.random() < 0.18:
        status = RNG.choice([200, 400, 401, 429, 500, 502, 503])
    if RNG.random() < 0.15:
        latency = max(20, int(RNG.gauss(700, 500)))

    code = RNG.choice(SPECIFIC_CODES[label]) if RNG.random() < 0.62 else RNG.choice(COMMON_CODES)
    if RNG.random() < 0.12:
        code = RNG.choice(COMMON_CODES)

    message_parts = [RNG.choice(COMMON_MESSAGES)]
    if RNG.random() < 0.58:
        message_parts.append(RNG.choice(CAUSE_MESSAGES[label]))
    if RNG.random() < 0.3:
        message_parts.append(f"retry={RNG.randint(0, 4)}")
    if RNG.random() < 0.25:
        message_parts.append(f"queue={RNG.randint(0, 800)}")

    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=idx * 11)
    return {
        "incident_id": f"INC-{idx:06d}",
        "timestamp": timestamp.isoformat(),
        "service": service,
        "environment": env,
        "severity": severity,
        "error_code": code,
        "http_status": status,
        "latency_ms": latency,
        "log_message": " | ".join(message_parts),
        "root_cause": label,
        "resolved_minutes": max(2, int(RNG.gauss({"database": 18, "dependency": 22, "auth": 9, "network": 26, "resource": 30, "application": 14}[label], 7))),
    }


def main() -> None:
    rows = [make_row(i) for i in range(1, 3001)]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} incidents to {OUT}")


if __name__ == "__main__":
    main()
