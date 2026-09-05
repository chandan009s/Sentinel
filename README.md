# Sentinel

> An AI-powered, stateful transaction risk detection and incident management platform.

Sentinel is a production-oriented risk detection system that combines **machine-learning anomaly detection, behavioral analysis, stateful incident management, business policy, persistence, observability, simulation, and reproducible CI/CD**.

Instead of treating every anomalous transaction as an isolated classification problem, Sentinel evaluates **behavior over time** and turns sustained suspicious activity into actionable incidents.

---

## Why Sentinel?

Traditional anomaly detection can answer:

> "Does this transaction look unusual?"

Sentinel goes further:

> "Is this unusual behavior sustained enough to represent a real incident, what evidence supports it, and what should the business do?"

The system therefore separates:

```
Anomaly Detection
       ↓
Behavioral Evidence
       ↓
Incident State
       ↓
Business Decision
       ↓
Audit + Observability
```

---

## Core Features

### AI-powered anomaly detection

Sentinel uses an Isolation Forest model to detect unusual transaction behavior.

The production feature vector contains:

- `velocity_ratio`
- `amount_ratio`
- `velocity_acceleration_1m`
- `amount_acceleration_1m`

The model is versioned and accompanied by:

- model metadata
- feature version
- feature contract version
- threshold version
- training configuration
- reference statistics
- risk bands

### Stateful incident detection

A single anomalous prediction does not automatically become a blocking action.

Sentinel maintains per-merchant state:

```
NORMAL
   ↓
SUSPECTED
   ↓
ACTIVE
   ↓
RECOVERY
   ↓
RESOLVED
```

The transition into `ACTIVE` requires sustained behavioral evidence.

Recovery uses hysteresis to avoid rapidly oscillating between normal and risky states.

### Risk evidence

When an incident becomes actionable, Sentinel records supporting evidence such as:

- anomaly score
- threshold
- first alert score
- score increase
- velocity acceleration
- amount acceleration
- alert count
- detection count
- time to activation
- decision/action

This makes the system auditable, rather than producing only an opaque model prediction.

### Business risk policy

Detection and business action are intentionally separated.

Current policy:

| Event State | Risk Level  | Decision |
|-------------|-------------|----------|
| NORMAL      | Any         | ALLOW    |
| SUSPECTED   | LOW / MEDIUM| ALLOW    |
| SUSPECTED   | HIGH        | REVIEW   |
| ACTIVE      | Any         | BLOCK    |
| RECOVERY    | Any         | REVIEW   |
| RESOLVED    | Any         | ALLOW    |

This allows business policy to evolve independently from the ML model.

### Idempotency and auditability

Prediction requests support event IDs.

Duplicate events are protected so repeated submissions do not create duplicate risk-event records.

PostgreSQL stores:

- risk events
- risk incidents
- incident evidence
- model metadata
- state transitions

---

## Architecture

```
Transaction / Simulation Input
            │
            ▼
       Go API Gateway
            │
            ▼
      FastAPI ML Service
            │
            ▼
     Isolation Forest Model
            │
            ▼
      Anomaly Score
            │
            ▼
   Stateful Incident Engine
            │
     ┌──────┼─────────┐
     ▼      ▼         ▼
  NORMAL SUSPECTED   ACTIVE
                      │
                      ▼
                  RECOVERY
                      │
                      ▼
                  RESOLVED
            │
            ├──────────────► Business Policy
            │                 ALLOW / REVIEW / BLOCK
            │
            ├──────────────► PostgreSQL
            │                 Events / Incidents / Evidence
            │
            └──────────────► Prometheus
                              │
                              ▼
                           Grafana
```

Detailed architecture: [`architecture.md`](./architecture.md)

---

## System Components

| Component    | Technology                     | Responsibility                    |
|--------------|---------------------------------|------------------------------------|
| API Gateway  | Go                              | Request handling and orchestration |
| ML Service   | Python / FastAPI                | Model inference                    |
| Model        | scikit-learn Isolation Forest   | Anomaly detection                  |
| Database     | PostgreSQL                      | Events, incidents, evidence        |
| Metrics      | Prometheus                      | Runtime monitoring                 |
| Dashboard    | Grafana                         | Operational visibility             |
| Containers   | Docker Compose                  | Local deployment                   |
| CI/CD        | GitHub Actions                  | Automated validation               |
| Simulation   | Python                          | Scenario replay                    |
| Testing      | Go + pytest                     | Functional and concurrency testing |

---

## Machine Learning Pipeline

```
Synthetic Transaction Generation
              ↓
        Feature Engineering
              ↓
        Feature Contract
              ↓
      Train Isolation Forest
              ↓
        Threshold Selection
              ↓
        Model Evaluation
              ↓
       Reference Statistics
              ↓
       Runtime Inference
```

The canonical V2 pipeline uses deterministic generation and a fixed random seed so the evaluation can be reproduced.

### Model and Feature Versioning

Sentinel exposes model lifecycle metadata at runtime.

Current production metadata:

```
model_name              = isolation_forest
model_version           = v2
feature_version         = v2
feature_contract_version = v1
threshold_version       = v1
```

The locked production anomaly threshold is:

```
0.637129
```

The runtime ML service also exposes model metadata through:

```
GET /metadata
```

---

## Monitoring and Drift Detection

The Go API exposes Prometheus metrics covering:

- prediction volume
- risk-level counts
- prediction latency
- runtime feature aggregates
- anomaly score aggregates
- model/version information
- reference feature statistics

Prometheus recording rules calculate rolling five-minute runtime means for:

- `velocity_ratio`
- `amount_ratio`
- `velocity_acceleration_1m`
- `amount_acceleration_1m`
- `anomaly_score`

These runtime statistics are compared against reference training statistics to detect potential feature drift.

Grafana provides an operational dashboard for:

- runtime traffic
- anomaly scores
- feature behavior
- model version
- monitoring volume
- drift signals

---

## Simulation Platform

Sentinel includes scenario-driven replay through the real production API.

Supported scenarios include:

- `NORMAL`
- `GRADUAL`
- `VELOCITY`
- `AMOUNT`
- `COMBINED`
- `RECOVERY`

Each scenario generates unique merchant and event IDs and records:

- anomaly scores
- risk levels
- state transitions
- decisions
- raw alerts
- activations

Run a recovery demonstration:

```bash
python3 scripts/replay_demo.py --scenario recovery --delay 0.2
```

Run all scenarios:

```bash
python3 scripts/replay_demo.py --scenario all --delay 0.2
```

Example state progression:

```
SUSPECTED → ACTIVE → RECOVERY
```

with corresponding business decisions:

```
ALLOW → BLOCK → REVIEW
```

---

## Evaluation

Sentinel uses a chronological train / validation / held-out test split.

**Canonical incident-aware benchmark**

Threshold:

```
0.637129
```

### Transaction-level performance

| Metric               | Result |
|-----------------------|--------|
| Precision              | 0.6199 |
| Recall                 | 0.8662 |
| F1                     | 0.7226 |
| False Positive Rate    | 0.1667 |

### Incident-level performance

| Metric                      | Result   |
|-------------------------------|----------|
| Total Incidents                | 2        |
| Detected                       | 2        |
| Missed                         | 0        |
| Incident Recall                | 1.0000   |
| Median Time to Detection       | 11.790 s |
| P95 Time to Detection          | 14.206 s |

Held-out incidents:

- **M002** — GRADUAL_SPIKE — Time to detection: 14.474 s
- **M003** — VELOCITY_SPIKE + SUSTAINED_SPIKE — Time to detection: 9.105 s

Incident-level recall is emphasized because the system is designed to identify sustained risk incidents, not merely maximize row-level classification accuracy.

---

## Load Testing

Sentinel includes a concurrent load-testing harness.

Run:

```bash
python3 scripts/load_test.py \
  --requests 1000 \
  --concurrency 50
```

Measured development baseline:

| Requests | Concurrency | Success | Throughput  | P95    | P99    |
|----------|-------------|---------|-------------|--------|--------|
| 100      | 10          | 100%    | 27.34 req/s | 0.77 s | 0.78 s |
| 500      | 25          | 100%    | 33.02 req/s | 1.19 s | 2.15 s |
| 1000     | 50          | 100%    | 35.92 req/s | 1.97 s | 2.56 s |

At 1000 requests / 50 concurrent workers:

```
Successful: 1000
Failed:        0
Throughput: 35.92 req/s
P95:          1.97 s
P99:          2.56 s
```

These figures are development baselines, not production capacity guarantees.

---

## Reproducible CI/CD

CI is designed to work from a fresh Git checkout without depending on locally generated datasets or model artifacts.

The Go test pipeline reproduces its required artifacts:

```
generate_transactions_v2.py
            ↓
synthetic_transactions_v2.csv
            ↓
train_model.py
            ↓
isolation_forest.pkl
            ↓
generate_prediction_stream.py
            ↓
v2_prediction_stream.csv
            ↓
Go tests
```

GitHub Actions validates:

- Go tests
- Go race detector
- Python tests
- Python syntax
- Deterministic ML artifact generation
- Prometheus rules
- Prometheus configuration
- Docker builds

Generated datasets and reports are intentionally excluded from version control.

---

## Running Sentinel

### Prerequisites

- Docker
- Docker Compose
- Python 3.12+
- Go 1.26+

### Start the stack

```bash
docker compose up -d --build
```

Check services:

```bash
docker compose ps
```

### Backend health

```bash
curl http://localhost:8080/health
```

### Backend readiness

```bash
curl http://localhost:8080/readyz
```

### ML service health

```bash
curl http://localhost:8000/health
```

### ML metadata

```bash
curl http://localhost:8000/metadata
```

### Run the simulator

```bash
python3 scripts/replay_demo.py --scenario recovery --delay 0.2
```

---

## Testing

### Go tests

```bash
cd backend
go test ./...
```

### Go race detector

```bash
go test -race ./...
```

### Python tests

```bash
cd ..
python3 -m pytest -q
```

### Prometheus rules

```bash
docker run --rm \
  -v "$PWD/monitoring:/etc/prometheus" \
  --entrypoint promtool \
  prom/prometheus:latest \
  check rules /etc/prometheus/alerts.yml
```

### Prometheus configuration

```bash
docker run --rm \
  -v "$PWD/monitoring:/etc/prometheus" \
  --entrypoint promtool \
  prom/prometheus:latest \
  check config /etc/prometheus/prometheus.yml
```

---

## Repository Structure

```
sentinel/
├── backend/
│   ├── database/
│   ├── risk/
│   ├── incident_coordinator.go
│   └── main.go
│
├── database/
│   └── schema.sql
│
├── ml/
│   ├── features/
│   ├── inference/
│   ├── models/
│   └── training/
│
├── model_service/
│   └── app.py
│
├── monitoring/
│   ├── alerts.yml
│   ├── prometheus.yml
│   └── grafana/
│
├── scripts/
│   ├── load_test.py
│   └── replay_demo.py
│
├── models/
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── architecture.md
└── README.md
```

> The repository directory may retain the existing `ai-risk-manager` name even though the project itself is branded as Sentinel.

---

## Design Principles

**Detection is not the decision**

```
ML prediction
     ≠
incident state
     ≠
business action
```

Each layer has a separate responsibility.

**Behavior matters**

Sustained and accelerating behavior is more informative than a single isolated anomaly.

**Evidence matters**

Risk decisions should be explainable and auditable.

**Reproducibility matters**

Training, test fixtures, evaluation, and CI should be deterministic and reproducible.

**Observability matters**

A production ML system must expose not only predictions, but also runtime behavior, model metadata, latency, and drift signals.

---

## Project Status

- ✅ Machine-learning anomaly detection
- ✅ Behavioral feature engineering
- ✅ Feature contract
- ✅ Stateful incident detection
- ✅ Confirmation logic
- ✅ Recovery / hysteresis
- ✅ Risk evidence
- ✅ Business policy
- ✅ Idempotency
- ✅ Model versioning
- ✅ Reference statistics
- ✅ Drift monitoring
- ✅ Grafana dashboard
- ✅ Scenario simulation
- ✅ Load testing
- ✅ Held-out incident evaluation
- ✅ Reproducible CI/CD

---

## License

See the repository license file.
