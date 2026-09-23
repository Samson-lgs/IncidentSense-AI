# Model Card — IncidentSense AI Baseline

## Model

**TF-IDF + Logistic Regression** with categorical one-hot features and standardized numeric telemetry.

## Intended use

Rank likely root causes for application, infrastructure, and CI/test incidents to assist human triage.

## Classes

- application
- auth
- database
- dependency
- network
- resource

## Evaluation

| Metric | Demo result |
|---|---:|
| Accuracy | 88.33% |
| Macro F1 | 88.33% |
| Weighted F1 | 88.34% |
| Test rows | 600 |

The evaluation uses a fixed random seed and a stratified 80/20 hold-out split of the included synthetic dataset.

## Limitations

- The dataset is synthetic and does not establish production accuracy.
- Log vocabulary and class distribution may differ substantially in real systems.
- A prediction is decision support, not an automated root-cause verdict.
- Similarity retrieval is lexical TF-IDF retrieval in the baseline and can miss semantic equivalence.
- Drift monitoring covers selected numeric features in this version.

## Upgrade path

1. Train on anonymized real CI and application incidents.
2. Add transformer embeddings for semantic retrieval.
3. Calibrate probabilities and establish human-review thresholds.
4. Add precision/recall monitoring using newly reviewed incidents.
5. Introduce champion/challenger model promotion with MLflow.
