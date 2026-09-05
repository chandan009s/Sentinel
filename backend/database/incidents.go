package database

import (
	"database/sql"
	"fmt"

	"ai-risk-manager/backend/risk"
)

type IncidentStore struct {
	db *sql.DB
}

func NewIncidentStore(db *sql.DB) *IncidentStore {
	return &IncidentStore{
		db: db,
	}
}

func (s *IncidentStore) Create(
	event risk.Event,
) (int64, error) {
	var id int64

	err := s.db.QueryRow(`
		INSERT INTO risk_incidents (
			merchant_id,
			started_at,
			activated_at,
			resolved_at,
			state,
			alert_count,
			detection_count,
			time_to_activation
		)
		VALUES ($1, $2, $3, NULL, $4, $5, $6, $7)
		RETURNING id
	`,
		event.MerchantID,
		event.StartedAt,
		nullableActivatedAt(event),
		string(event.State),
		event.AlertCount,
		event.DetectionCount,
		nullableTimeToActivation(event),
	).Scan(&id)

	if err != nil {
		return 0, fmt.Errorf(
			"create risk incident: %w",
			err,
		)
	}

	return id, nil
}

func (s *IncidentStore) Update(
	id int64,
	event risk.Event,
	timestamp float64,
) error {
	_, err := s.db.Exec(`
		UPDATE risk_incidents
		SET
			activated_at = $1,
			resolved_at = $2,
			state = $3,
			alert_count = $4,
			detection_count = $5,
			time_to_activation = $6,
			updated_at = CURRENT_TIMESTAMP
		WHERE id = $7
	`,
		nullableActivatedAt(event),
		nullableResolvedAt(event, timestamp),
		string(event.State),
		event.AlertCount,
		event.DetectionCount,
		nullableTimeToActivation(event),
		id,
	)
	if err != nil {
		return fmt.Errorf(
			"update risk incident %d: %w",
			id,
			err,
		)
	}

	return nil
}

func nullableActivatedAt(event risk.Event) any {
	if event.TimeToActivation <= 0 {
		return nil
	}

	return event.StartedAt + event.TimeToActivation
}

func nullableResolvedAt(
	event risk.Event,
	timestamp float64,
) any {
	if event.State != risk.StateResolved {
		return nil
	}

	return timestamp
}

func nullableTimeToActivation(event risk.Event) any {
	if event.TimeToActivation <= 0 {
		return nil
	}

	return event.TimeToActivation
}
