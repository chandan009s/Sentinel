CREATE TABLE IF NOT EXISTS risk_events (
    id BIGSERIAL PRIMARY KEY,
    event_id VARCHAR(100),
    merchant_id VARCHAR(100),
    timestamp DOUBLE PRECISION,
    velocity_ratio DOUBLE PRECISION NOT NULL,
    amount_ratio DOUBLE PRECISION NOT NULL,
    velocity_acceleration_1m DOUBLE PRECISION NOT NULL DEFAULT 0,
    amount_acceleration_1m DOUBLE PRECISION NOT NULL DEFAULT 0,
    anomaly_score DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    prediction INTEGER NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    decision VARCHAR(20) NOT NULL DEFAULT 'ALLOW',
    event_state VARCHAR(20) NOT NULL DEFAULT 'NORMAL',
    alert_count INTEGER NOT NULL DEFAULT 0,
    detection_count INTEGER NOT NULL DEFAULT 0,
    time_to_activation DOUBLE PRECISION NOT NULL DEFAULT 0,
    model_name VARCHAR(100) NOT NULL DEFAULT 'isolation_forest',
    model_version VARCHAR(50) NOT NULL DEFAULT 'v2',
    feature_version VARCHAR(50) NOT NULL DEFAULT 'v2',
    threshold_version VARCHAR(50) NOT NULL DEFAULT 'v1',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_events_created_at
ON risk_events (created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_risk_events_event_id
ON risk_events (event_id)
WHERE event_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS risk_incidents (
    id BIGSERIAL PRIMARY KEY,
    merchant_id VARCHAR(100) NOT NULL,
    started_at DOUBLE PRECISION NOT NULL,
    activated_at DOUBLE PRECISION,
    resolved_at DOUBLE PRECISION,
    state VARCHAR(20) NOT NULL,
    alert_count INTEGER NOT NULL DEFAULT 0,
    detection_count INTEGER NOT NULL DEFAULT 0,
    time_to_activation DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_incidents_merchant_started
ON risk_incidents (merchant_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_risk_incidents_state
ON risk_incidents (state);

CREATE TABLE IF NOT EXISTS incident_evidence (
    id BIGSERIAL PRIMARY KEY,
    incident_id BIGINT NOT NULL REFERENCES risk_incidents(id) ON DELETE CASCADE,
    merchant_id VARCHAR(100) NOT NULL,
    observed_at DOUBLE PRECISION NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    first_alert_score DOUBLE PRECISION NOT NULL,
    score_increase DOUBLE PRECISION NOT NULL,
    velocity_acceleration_1m DOUBLE PRECISION NOT NULL,
    amount_acceleration_1m DOUBLE PRECISION NOT NULL,
    alert_count INTEGER NOT NULL,
    time_to_activation DOUBLE PRECISION NOT NULL,
    decision VARCHAR(50) NOT NULL,
    action VARCHAR(20) NOT NULL DEFAULT 'ALLOW',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_incident_evidence_incident
ON incident_evidence (incident_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_incident_evidence_merchant
ON incident_evidence (merchant_id, observed_at DESC);
