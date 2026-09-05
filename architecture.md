# Sentinel Architecture

Sentinel is a production-oriented transaction risk detection platform that combines machine-learning anomaly detection with deterministic behavioral state management, business policy, persistence, observability, simulation, and reproducible CI/CD.

The architecture intentionally separates:

```
ML Detection
     ↓
Behavioral Evidence
     ↓
Incident State
     ↓
Business Decision
     ↓
Persistence
     ↓
Observability
```

---

## 1. High-Level Architecture

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

---

## 2. Component Responsibilities

### Go API

The Go service is the runtime orchestration layer.

Responsibilities:

- request validation
- event-id idempotency
- ML-service communication
- risk-level assignment
- incident coordination
- business policy evaluation
- PostgreSQL persistence
- Prometheus metrics
- API responses

Primary endpoints:

```
GET  /health
GET  /readyz
POST /predict
GET  /risk-events
GET  /risk-incidents
GET  /risk-incidents/{id}/evidence
GET  /metrics
```

### FastAPI ML Service

The ML service is intentionally separated from the Go application.

Responsibilities:

- load model artifact
- load metadata
- validate features
- generate anomaly scores
- apply threshold
- expose model metadata
- expose health information

Endpoints:

```
GET  /health
GET  /metadata
POST /predict
```

### Isolation Forest

Sentinel uses an Isolation Forest anomaly detector.

Production features:

- `velocity_ratio`
- `amount_ratio`
- `velocity_acceleration_1m`
- `amount_acceleration_1m`

The detector operates independently of the business decision policy.

This allows the detection algorithm to evolve without coupling the ML implementation directly to business actions.

---

## 3. Runtime Prediction Flow

A prediction request follows:

```
Client
  │
  ▼
POST /predict
  │
  ▼
Input validation
  │
  ▼
Event ID / idempotency check
  │
  ▼
FastAPI inference
  │
  ▼
Anomaly score + threshold
  │
  ▼
Risk level
  │
  ▼
Incident state evaluation
  │
  ▼
Evidence creation
  │
  ▼
Business decision
  │
  ▼
Monitoring metrics
  │
  ▼
PostgreSQL persistence
  │
  ▼
Response
```

The important design property is that the model prediction does not directly determine the final business action.

---

## 4. Feature Contract

The canonical production feature vector is:

- `velocity_ratio`
- `amount_ratio`
- `velocity_acceleration_1m`
- `amount_acceleration_1m`

The project maintains a shared feature contract so that offline training and online serving use compatible feature definitions.

Version information:

```
feature_version = v2
feature_contract_version = v1
```

This reduces the risk of training-serving skew.

---

## 5. Stateful Incident Engine

### Why a state machine?

A point anomaly does not necessarily represent a true incident.

Sentinel therefore evaluates:

- anomaly score
- threshold crossing
- score increase
- velocity acceleration
- amount acceleration
- alert count
- detection count
- elapsed time

before activating an incident.

This reduces dependence on a single noisy prediction.

---

## 6. Confirmation and Hysteresis

The incident engine uses temporal confirmation and recovery windows.

Conceptually:

```
Anomaly
   ↓
Suspicion
   ↓
Repeated evidence
   ↓
Activation
```

Recovery is intentionally slower than a single return below threshold:

```
ACTIVE
  ↓
behavior normalizes
  ↓
RECOVERY
  ↓
recovery window
  ↓
RESOLVED
```

This prevents rapid state oscillation.

Reappearance of suspicious behavior during recovery can reactivate the incident.

---

## 7. Business Policy

Business policy consumes incident state and risk level.

Current policy:

| State     | Risk         | Decision |
|-----------|--------------|----------|
| NORMAL    | Any          | ALLOW    |
| SUSPECTED | LOW / MEDIUM | ALLOW    |
| SUSPECTED | HIGH         | REVIEW   |
| ACTIVE    | Any          | BLOCK    |
| RECOVERY  | Any          | REVIEW   |
| RESOLVED  | Any          | ALLOW    |

This separation gives the architecture a clean boundary:

- **Model:** "How unusual is this?"
- **Incident engine:** "Is this behavior becoming a sustained incident?"
- **Policy:** "What should the business do?"

---

## 8. Evidence Model

Activated incidents preserve supporting evidence.

An evidence record can contain:

- `incident_id`
- `merchant_id`
- `observed_at`
- `anomaly_score`
- `threshold`
- `first_alert_score`
- `score_increase`
- `velocity_acceleration_1m`
- `amount_acceleration_1m`
- `alert_count`
- `time_to_activation`
- `decision`
- `action`

Example:

```
first score
     ↓
current score
     ↓
score increase
     ↓
behavioral acceleration
     ↓
activation evidence
     ↓
business action
```

This provides an auditable explanation for incident activation.

---

## 9. Persistence Architecture

PostgreSQL stores three primary objects.

### `risk_events`

Individual prediction-level records containing:

- event ID
- merchant ID
- timestamp
- feature values
- anomaly score
- threshold
- prediction
- risk level
- decision
- state
- counters
- time to activation
- model metadata
- feature metadata
- threshold metadata

A unique partial index protects event IDs from duplicate persistence.

### `risk_incidents`

Incident lifecycle information:

- merchant
- start time
- activation time
- resolution time
- state
- alert count
- detection count
- time to activation

### `incident_evidence`

Supporting evidence associated with an incident.

Foreign-key protection ensures evidence follows the incident lifecycle.

---

## 10. Idempotency

The prediction endpoint supports event identifiers.

The runtime flow is:

```
POST /predict
     ↓
event_id exists?
    /   \
  yes    no
  ↓       ↓
return   process
stored   prediction
result      ↓
         persist
```

There is also a database uniqueness constraint as the final safety boundary.

This prevents duplicate event records even when requests are retried.

---

## 11. Model Lifecycle

Model metadata is stored alongside the model artifact.

Current metadata:

```
model_name               = isolation_forest
model_version            = v2
feature_version          = v2
feature_contract_version = v1
threshold_version        = v1
```

The production threshold is:

```
0.637129
```

The runtime ML service exposes this information through `/metadata`.

This makes model identity visible to operators and monitoring systems.

---

## 12. Monitoring

The Go API exposes Prometheus metrics for:

- prediction count
- risk-level count
- prediction latency
- runtime feature aggregates
- runtime anomaly score
- model metadata
- reference feature statistics

Runtime aggregate metrics are maintained as cumulative counters.

Prometheus converts these into rolling averages using recording rules.

Conceptually:

```
runtime feature sum
        ÷
runtime sample count
        ↓
5-minute runtime mean
        ↓
compare with reference mean
        ↓
drift signal
```

---

## 13. Drift Detection

Reference feature statistics are derived from the training population.

Tracked features include:

- `velocity_ratio`
- `amount_ratio`
- `velocity_acceleration_1m`
- `amount_acceleration_1m`

Prometheus maintains rolling five-minute means.

Recent drift rules compare these means against reference statistics.

This creates an operational path:

```
Runtime behavior
       ↓
Rolling statistics
       ↓
Reference comparison
       ↓
Drift condition
       ↓
Prometheus alert
       ↓
Grafana visualization
```

---

## 14. Grafana

Grafana provides operational visibility into:

- prediction rate
- anomaly score
- runtime feature means
- model/version information
- monitoring sample volume
- drift indicators
- operational state

The dashboard is version-controlled as:

```
monitoring/grafana/ai-risk-manager-dashboard.json
```

---

## 15. Simulation Platform

Sentinel includes scenario-driven replay against the actual `/predict` endpoint.

Supported scenarios:

- `NORMAL`
- `GRADUAL`
- `VELOCITY`
- `AMOUNT`
- `COMBINED`
- `RECOVERY`

Each scenario:

- creates a unique merchant
- creates unique event IDs
- generates behavioral observations
- calls the Go API
- captures anomaly scores
- captures risk levels
- captures incident states
- captures decisions
- summarizes alerts and activations

Example:

```bash
python3 scripts/replay_demo.py --scenario recovery --delay 0.2
```

---

## 16. ML Reproducibility

The V2 dataset is generated deterministically with a fixed random seed.

The reproducible pipeline is:

```
generate_transactions_v2.py
             │
             ▼
synthetic_transactions_v2.csv
             │
             ▼
train_model.py
             │
             ▼
isolation_forest.pkl
             │
             ▼
generate_prediction_stream.py
             │
             ▼
v2_prediction_stream.csv
             │
             ▼
stateful replay tests
```

The generated artifacts are not required to be stored in Git.

This is especially important for CI because a fresh runner can recreate everything from source.

---

## 17. CI/CD

GitHub Actions validates the system through independent jobs.

The CI system therefore checks more than compilation:

- correctness
- concurrency safety
- reproducibility
- monitoring configuration
- container buildability

---

## 18. Resilience and Load Testing

The project includes a concurrent load-testing harness.

Measured development baseline:

| Requests | Concurrency | Success | Throughput  | P95    | P99    |
|----------|-------------|---------|-------------|--------|--------|
| 100      | 10          | 100%    | 27.34 req/s | 0.77 s | 0.78 s |
| 500      | 25          | 100%    | 33.02 req/s | 1.19 s | 2.15 s |
| 1000     | 50          | 100%    | 35.92 req/s | 1.97 s | 2.56 s |

At the highest tested load:

```
1000 requests
50 concurrent workers
0 failures
35.92 req/s
```

The benchmark establishes a repeatable local baseline.

---

## 19. Validation Strategy

Sentinel uses chronological evaluation:

```
TRAIN
  ↓
VALIDATION
  ↓
HELD-OUT TEST
```

The canonical incident-aware test evaluates:

```
Transaction Precision  = 0.6199
Transaction Recall     = 0.8662
Transaction F1         = 0.7226
Transaction FPR        = 0.1667

Incident Recall        = 1.0000
Median TTD             = 11.790 s
P95 TTD                = 14.206 s
```

Held-out incidents:

- **M002** — GRADUAL_SPIKE — TTD = 14.474 s
- **M003** — VELOCITY_SPIKE + SUSTAINED_SPIKE — TTD = 9.105 s

The distinction between transaction-level and incident-level evaluation is deliberate.

A system can miss individual anomalous rows while still successfully detecting the sustained incident.

---

## 20. Deployment Topology

The local stack is containerized with Docker Compose.

Services:

- backend
- ml-service
- postgres
- prometheus
- grafana

---

## 21. Main Design Trade-offs

**Isolation Forest**

Chosen as a practical unsupervised anomaly-detection approach for a synthetic risk-monitoring environment.

**Stateful engine**

Chosen because sustained risk is temporal and cannot be represented well by independent row-level predictions alone.

**Separate policy**

Chosen so business actions can change without modifying the ML detector.

**Evidence persistence**

Chosen to make incidents auditable and explainable.

**Deterministic simulation**

Chosen so tests and demonstrations are reproducible.

**Prometheus + Grafana**

Chosen to provide an operational monitoring layer without introducing unnecessary infrastructure complexity.

---

## 22. Architecture Principle

The central Sentinel design principle is:

```
Detect unusual behavior
          ↓
Accumulate evidence
          ↓
Determine incident state
          ↓
Apply business policy
          ↓
Persist the outcome
          ↓
Observe the system
          ↓
Evaluate the system
```

Sentinel is therefore designed as a risk-management platform, not simply an anomaly-classification API.
