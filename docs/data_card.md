# Data Card — Demo Incident Dataset

## Source

Generated locally with a deterministic seed. No private production logs are included.

## Records

3,000 incident records across six root-cause classes.

## Features

- timestamp
- service
- environment
- severity
- error_code
- HTTP status
- latency in milliseconds
- log message
- root cause label
- resolution time

## Why synthetic data is included

The repository can be cloned and tested without licensing, privacy, or credentials. The generator intentionally introduces overlap between classes so a perfect score is not the default outcome.

## Production migration

Replace `scripts/generate_demo_data.py` with a connector that reads approved, anonymized CI/test logs or incident records. Preserve the same schema contract where possible so the downstream training and API layers remain reusable.
