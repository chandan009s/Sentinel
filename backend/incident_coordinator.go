package main

import (
	"sync"

	"ai-risk-manager/backend/database"
	"ai-risk-manager/backend/risk"
)

type IncidentRepository interface {
	Create(event risk.Event) (int64, error)
	Update(id int64, event risk.Event, timestamp float64) error
}

type EvidenceRepository interface {
	Create(
		incidentID int64,
		observation risk.Observation,
		event risk.Event,
		action risk.Decision,
	) error
}

type IncidentCoordinator struct {
	mu           sync.Mutex
	eventManager *risk.EventManager
	repository   IncidentRepository
	evidence     EvidenceRepository
	incidentIDs  map[string]int64
}

func NewIncidentCoordinator(
	repository IncidentRepository,
	evidence EvidenceRepository,
) *IncidentCoordinator {
	return &IncidentCoordinator{
		eventManager: risk.NewEventManager(),
		repository:   repository,
		evidence:     evidence,
		incidentIDs:  make(map[string]int64),
	}
}

func (c *IncidentCoordinator) Process(
	observation risk.Observation,
) (risk.Event, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	event := c.eventManager.Process(observation)

	switch {
	case event.PreviousState == risk.StateSuspected &&
		event.State == risk.StateActive:

		id, err := c.repository.Create(event)
		if err != nil {
			return event, err
		}

		if err := c.evidence.Create(
			id,
			observation,
			event,
			risk.DetermineDecision(
				event.State,
				"MEDIUM",
			),
		); err != nil {
			return event, err
		}

		c.incidentIDs[event.MerchantID] = id

	case event.PreviousState == risk.StateRecovery &&
		event.State == risk.StateActive:

		id, ok := c.incidentIDs[event.MerchantID]
		if !ok {
			id, err := c.repository.Create(event)
			if err != nil {
				return event, err
			}

			c.incidentIDs[event.MerchantID] = id
		} else if err := c.repository.Update(
			id,
			event,
			observation.Timestamp,
		); err != nil {
			return event, err
		}

	case event.PreviousState == risk.StateActive &&
		event.State == risk.StateRecovery:

		if id, ok := c.incidentIDs[event.MerchantID]; ok {
			if err := c.repository.Update(
				id,
				event,
				observation.Timestamp,
			); err != nil {
				return event, err
			}
		}

	case event.PreviousState == risk.StateRecovery &&
		event.State == risk.StateResolved:

		if id, ok := c.incidentIDs[event.MerchantID]; ok {
			if err := c.repository.Update(
				id,
				event,
				observation.Timestamp,
			); err != nil {
				return event, err
			}

			delete(c.incidentIDs, event.MerchantID)
		}
	}

	return event, nil
}

var _ IncidentRepository = (*database.IncidentStore)(nil)
var _ EvidenceRepository = (*database.EvidenceStore)(nil)
