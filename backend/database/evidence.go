package database

import (
	"database/sql"
	"fmt"

	"ai-risk-manager/backend/risk"
)

type EvidenceStore struct {
	db *sql.DB
}

func NewEvidenceStore(db *sql.DB) *EvidenceStore {
	return &EvidenceStore{db: db}
}

func (s *EvidenceStore) Create(
	incidentID int64,
	observation risk.Observation,
	event risk.Event,
	action risk.Decision,
) error {
	_, err := s.db.Exec(`
		INSERT INTO incident_evidence (
			incident_id,
			merchant_id,
			observed_at,
			anomaly_score,
			threshold,
			first_alert_score,
			score_increase,
			velocity_acceleration_1m,
			amount_acceleration_1m,
			alert_count,
			time_to_activation,
			decision,
			action
		)
		VALUES (
			$1, $2, $3, $4, $5, $6, $7,
			$8, $9, $10, $11, $12, $13
		)
	`,
		incidentID,
		event.MerchantID,
		observation.Timestamp,
		observation.AnomalyScore,
		observation.Threshold,
		event.FirstAlertScore,
		event.ScoreIncrease,
		observation.VelocityAcceleration,
		observation.AmountAcceleration,
		event.AlertCount,
		event.TimeToActivation,
		"ACTIVATED",
		action,
	)

	if err != nil {
		return fmt.Errorf("insert incident evidence: %w", err)
	}

	return nil
}

func (s *EvidenceStore) ListByIncident(
	incidentID int64,
) ([]map[string]interface{}, error) {
	rows, err := s.db.Query(`
		SELECT
			id,
			incident_id,
			merchant_id,
			observed_at,
			anomaly_score,
			threshold,
			first_alert_score,
			score_increase,
			velocity_acceleration_1m,
			amount_acceleration_1m,
			alert_count,
			time_to_activation,
			decision,
			action,
			created_at
		FROM incident_evidence
		WHERE incident_id = $1
		ORDER BY observed_at DESC
	`, incidentID)
	if err != nil {
		return nil, fmt.Errorf("query incident evidence: %w", err)
	}
	defer rows.Close()

	var evidence []map[string]interface{}

	for rows.Next() {
		var (
			id                   int64
			incidentIDValue      int64
			merchantID           string
			observedAt           float64
			anomalyScore         float64
			threshold            float64
			firstAlertScore      float64
			scoreIncrease        float64
			velocityAcceleration float64
			amountAcceleration   float64
			alertCount           int
			timeToActivation     float64
			decision             string
			action               string
			createdAt            string
		)

		if err := rows.Scan(
			&id,
			&incidentIDValue,
			&merchantID,
			&observedAt,
			&anomalyScore,
			&threshold,
			&firstAlertScore,
			&scoreIncrease,
			&velocityAcceleration,
			&amountAcceleration,
			&alertCount,
			&timeToActivation,
			&decision,
			&action,
			&createdAt,
		); err != nil {
			return nil, fmt.Errorf("read incident evidence: %w", err)
		}

		evidence = append(evidence, map[string]interface{}{
			"id":                       id,
			"incident_id":              incidentIDValue,
			"merchant_id":              merchantID,
			"observed_at":              observedAt,
			"anomaly_score":            anomalyScore,
			"threshold":                threshold,
			"first_alert_score":        firstAlertScore,
			"score_increase":           scoreIncrease,
			"velocity_acceleration_1m": velocityAcceleration,
			"amount_acceleration_1m":   amountAcceleration,
			"alert_count":              alertCount,
			"time_to_activation":       timeToActivation,
			"decision":                 decision,
			"action":                   action,
			"created_at":               createdAt,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate incident evidence: %w", err)
	}

	if evidence == nil {
		evidence = []map[string]interface{}{}
	}

	return evidence, nil
}
