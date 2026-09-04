# AI Risk Manager

An end-to-end anomaly and risk detection system for identifying abnormal transaction activity using machine learning, a Go backend, PostgreSQL, and production-style observability.

## Overview

AI Risk Manager detects unusual transaction behavior using two model features:

- `velocity_ratio` — transaction activity relative to a historical baseline
- `amount_ratio` — transaction amount relative to a historical baseline

An Isolation Forest produces an anomaly score. The ML service applies the configured anomaly threshold to produce the model prediction, while the Go backend applies the operational risk policy and classifies the result as:

- **LOW**
- **MEDIUM**
- **HIGH**

The system exposes a prediction API, persists risk events, and exports Prometheus metrics for monitoring through Grafana.

## Architecture

```text
                    Transaction / API Request
                              |
                              v
                   +----------------------+
                   |     Go Backend       |
                   |       :8080          |
                   +----------+-----------+
                              |
                              | HTTP /predict
                              v
                   +----------------------+
                   |   FastAPI ML Service |
                   |       :8000          |
                   +----------+-----------+
                              |
                              v
                   +----------------------+
                   |   Isolation Forest   |
                   |    anomaly score     |
                   +----------+-----------+
                              |
                              | score + threshold
                              v
                   +----------------------+
                   |   Go Risk Engine     |
                   | LOW / MEDIUM / HIGH  |
                   +----------+-----------+
                              |
                 +------------+-------------+
                 |                          |
                 v                          v
          +-------------+             +-------------+
          | PostgreSQL  |             | Prometheus  |
          | risk_events |             |   metrics   |
          +-------------+             +------+------+
                                              |
                                              v
                                        +-----------+
                                        |  Grafana  |
                                        +-----------+
```

## Technology Stack

| Component      | Technology                       |
|----------------|----------------------------------|
| Backend API    | Go                               |
| ML Service     | Python / FastAPI                 |
| ML Model       | Scikit-learn Isolation Forest    |
| Database       | PostgreSQL                       |
| Metrics        | Prometheus                       |
| Visualization  | Grafana                          |
| Containers     | Docker / Docker Compose          |
| CI             | GitHub Actions                   |

## ML Pipeline

The feature engineering pipeline builds time-based transaction features using merchant-level rolling windows and historical baselines.

The model uses two features:

- `velocity_ratio`
- `amount_ratio`

Training uses a chronological split:

**70% Training | 15% Validation | 15% Test**

The Isolation Forest is fitted only on normal transactions from the training portion.

The anomaly threshold is selected using the validation set. Candidate thresholds are evaluated using a cost function that penalizes false negatives more heavily than false positives.

The test set remains separate for final evaluation.

### Feature Engineering

The system calculates transaction activity over multiple time windows and derives historical baselines.

The baseline calculations use lagged rolling windows:

```text
Current transaction
       |
       v
Historical transaction activity
       |
       v
Lagged rolling baseline
       |
       v
Velocity / Amount ratio
       |
       v
Isolation Forest
```

This prevents the current activity from immediately becoming part of its own baseline.

## Risk Decision Engine

The ML service is responsible for generating the anomaly score and model prediction.

The Go backend owns the operational risk classification.

The current decision boundaries are:

```text
score < threshold
    -> LOW

threshold <= score < threshold + 0.05
    -> MEDIUM

score >= threshold + 0.05
    -> HIGH
```

The active threshold is stored in `models/threshold.json`.

Separating model scoring from risk classification keeps operational risk policy independent from the ML implementation.

## Data Leakage Prevention

- The training and evaluation pipeline uses **chronological splits** instead of random splits.
- The model is trained only on the training portion and only on transactions labelled as normal.
- Threshold candidates are derived from training anomaly scores and evaluated using validation data.
- The test set is not used to fit the model or select the threshold percentile.
- Feature baselines use lagged historical windows so the current spike does not directly define its own baseline.

The implementation was reviewed for train/test and threshold-selection leakage. No obvious leakage was found based on the current code.

## Backend API

### `GET /health`

Returns the backend health status.

### `POST /predict`

Runs an anomaly and risk prediction.

**Example request:**

```json
{
  "velocity_ratio": 4.0,
  "amount_ratio": 4.0
}
```

**Example response:**

```json
{
  "anomaly_score": 0.8023,
  "threshold": 0.6577,
  "prediction": 1,
  "risk_level": "HIGH"
}
```

### `GET /risk-events`

Returns recently persisted risk events from PostgreSQL.

### `GET /metrics`

Exposes Prometheus metrics.

## Observability

The backend exposes the following metrics:

- `risk_predictions_total`
- `risk_low_predictions_total`
- `risk_medium_predictions_total`
- `risk_high_predictions_total`
- `risk_prediction_latency_seconds`

The Grafana dashboard provides visibility into:

- Total predictions
- LOW predictions
- MEDIUM predictions
- HIGH predictions
- Prediction rate
- Prediction latency
- HIGH-risk rate

The dashboard configuration is version-controlled at `monitoring/grafana/ai-risk-manager-dashboard.json`.

## Live Replay Demo

A deterministic replay script sends controlled feature values through the real Go `/predict` API:

```bash
python scripts/replay_demo.py
```

The replay moves through three traffic conditions:

```text
Normal traffic
      |
      v
LOW risk
      |
      v
Elevated traffic
      |
      v
MEDIUM risk
      |
      v
Traffic spike
      |
      v
HIGH risk
```

**Example observed output:**

```text
Normal traffic       velocity=1.0  amount=1.0  score=0.3822 risk=LOW
Normal traffic       velocity=1.1  amount=1.0  score=0.4222 risk=LOW
Elevated traffic     velocity=2.0  amount=2.0  score=0.6839 risk=MEDIUM
Elevated traffic     velocity=2.2  amount=2.1  score=0.6841 risk=MEDIUM
Spike                velocity=4.0  amount=4.0  score=0.8023 risk=HIGH
Spike                velocity=5.0  amount=5.0  score=0.8149 risk=HIGH
Spike                velocity=6.0  amount=6.0  score=0.8149 risk=HIGH
```

Because the replay uses the real backend, the requests flow through:

```text
Replay Script
     |
     v
Go API
     |
     v
FastAPI ML Service
     |
     v
Isolation Forest
     |
     v
Go Risk Engine
     |
     +------> PostgreSQL
     |
     +------> Prometheus
                    |
                    v
                 Grafana
```

This provides a reproducible end-to-end demonstration of the system.

## Running the System

### Prerequisites

- Docker
- Docker Compose
- Python 3.12+
- Go 1.26.4+

### Start the complete stack

```bash
docker compose up -d --build
```

Check the services:

```bash
docker compose ps
```

Expected services:

- `backend`
- `ml-service`
- `postgres`
- `prometheus`
- `grafana`

### Service Endpoints

| Service     | URL                     |
|-------------|-------------------------|
| Backend     | http://localhost:8080   |
| ML Service  | http://localhost:8000   |
| Prometheus  | http://localhost:9090   |
| Grafana     | http://localhost:3000   |
| PostgreSQL  | localhost:5432          |

### Example Prediction

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"velocity_ratio":4.0,"amount_ratio":4.0}'
```

### View Persisted Risk Events

```bash
curl http://localhost:8080/risk-events
```

### View Prometheus Metrics

```bash
curl http://localhost:8080/metrics
```

## Testing

### Go Tests

```bash
cd backend
go test ./...
```

### Python Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the ML unit tests:

```bash
python -m pytest ml/models/test_isolation_forest_unit.py -v
```

The automated ML tests verify:

- Model fitting
- Anomaly score generation
- Prediction output shape
- Valid Isolation Forest prediction values

## CI

GitHub Actions runs on pushes and pull requests targeting `main`.

The CI pipeline verifies:

```text
Push / Pull Request
        |
        +------> Go Tests
        |
        +------> Python ML Tests
        |
        +------> Docker Build
```

Workflow definition: `.github/workflows/ci.yml`

## Key Engineering Findings

### Baseline Contamination

A transaction spike can influence a rolling baseline if current activity enters the baseline too quickly.

The feature-engineering pipeline therefore uses lagged rolling baselines rather than allowing the current bucket to immediately define its own reference point.

### Threshold Sensitivity

Isolation Forest produces continuous anomaly scores.

The operational decision depends on the selected threshold.

The project therefore separates model training from threshold selection and evaluates candidate thresholds against validation data.

### ML vs Operational Responsibility

The ML service produces:

- `anomaly_score`
- `prediction`

The Go backend determines:

- `LOW`
- `MEDIUM`
- `HIGH`

This separation allows operational risk policy to be changed independently of the ML implementation. It also makes the risk boundaries independently testable.

### Reproducibility

The system is designed to be reproducible through:

- Docker Compose
- Environment-based configuration
- Version-controlled Grafana dashboard
- Automated Go tests
- Automated Python ML tests
- GitHub Actions CI
- Deterministic replay demo

## Project Structure

```text
ai-risk-manager/
├── backend/
│   ├── main.go
│   ├── main_test.go
│   ├── database/
│   ├── Dockerfile
│   ├── go.mod
│   └── go.sum
│
├── ml/
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── ...
│   ├── models/
│   │   ├── isolation_forest.py
│   │   ├── isolation_forest.pkl
│   │   └── test_isolation_forest_unit.py
│   ├── inference/
│   └── training/
│
├── model_service/
│   └── app.py
│
├── database/
│   └── schema.sql
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── ai-risk-manager-dashboard.json
│
├── models/
│   └── threshold.json
│
├── scripts/
│   └── replay_demo.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Engineering Focus

This project focuses on the complete ML-serving lifecycle rather than model training alone:

```text
Feature Engineering
        |
        v
  Model Training
        |
        v
 Threshold Selection
        |
        v
    ML Serving
        |
        v
    Go API
        |
        v
    Risk Decision
        |
        v
    PostgreSQL
        |
        v
    Prometheus
        |
        v
    Grafana

```