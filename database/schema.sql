CREATE TABLE IF NOT EXISTS risk_events (
    id BIGSERIAL PRIMARY KEY,
    merchant_id VARCHAR(100),
    timestamp DOUBLE PRECISION,
    velocity_ratio DOUBLE PRECISION NOT NULL,
    amount_ratio DOUBLE PRECISION NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    prediction INTEGER NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_events_created_at
ON risk_events (created_at DESC);
