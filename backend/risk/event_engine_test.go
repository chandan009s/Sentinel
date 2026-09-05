package risk

import "testing"

func observationAt(
	timestamp float64,
	prediction int,
	score float64,
) Observation {
	return Observation{
		MerchantID:           "M001",
		Timestamp:            timestamp,
		AnomalyScore:         score,
		Threshold:            0.637,
		Prediction:           prediction,
		VelocityAcceleration: 0.10,
		AmountAcceleration:   0.10,
	}
}

func observation(
	timestamp float64,
	prediction int,
) Observation {
	return observationAt(
		timestamp,
		prediction,
		0.70,
	)
}

func TestEventEngineActivatesAfterThreeIncreasingAlerts(t *testing.T) {
	engine := NewEventEngine("M001")

	engine.Process(
		observationAt(100, 1, 0.64),
	)

	engine.Process(
		observationAt(105, 1, 0.66),
	)

	event := engine.Process(
		observationAt(110, 1, 0.68),
	)

	if event.State != StateActive {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateActive,
		)
	}

	if event.TimeToActivation != 10 {
		t.Fatalf(
			"got TimeToActivation %v, want 10",
			event.TimeToActivation,
		)
	}
}

func TestEventEnginePlateauDoesNotActivate(t *testing.T) {
	engine := NewEventEngine("M001")

	engine.Process(
		observationAt(100, 1, 0.70),
	)

	engine.Process(
		observationAt(105, 1, 0.70),
	)

	event := engine.Process(
		observationAt(110, 1, 0.70),
	)

	if event.State != StateSuspected {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateSuspected,
		)
	}

	if event.AlertCount != 3 {
		t.Fatalf(
			"got alert count %d, want 3",
			event.AlertCount,
		)
	}

	if event.TimeToActivation != 0 {
		t.Fatalf(
			"got TimeToActivation %v, want 0",
			event.TimeToActivation,
		)
	}
}

func TestEventEngineSingleAlertDoesNotActivate(t *testing.T) {
	engine := NewEventEngine("M001")

	event := engine.Process(
		observation(100, 1),
	)

	if event.State != StateSuspected {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateSuspected,
		)
	}

	if event.AlertCount != 1 {
		t.Fatalf(
			"got alert count %d, want 1",
			event.AlertCount,
		)
	}
}

func TestEventEngineAlertsOutsideWindowDoNotActivate(t *testing.T) {
	engine := NewEventEngine("M001")

	engine.Process(
		observation(100, 1),
	)

	engine.Process(
		observation(120, 1),
	)

	event := engine.Process(
		observation(140, 1),
	)

	if event.State != StateSuspected {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateSuspected,
		)
	}

	if event.AlertCount != 1 {
		t.Fatalf(
			"got alert count %d, want 1",
			event.AlertCount,
		)
	}

	if event.TimeToActivation != 0 {
		t.Fatalf(
			"got TimeToActivation %v, want 0",
			event.TimeToActivation,
		)
	}
}

func TestEventEngineResetsAfterLongGap(t *testing.T) {
	engine := NewEventEngine("M001")

	engine.Process(
		observation(100, 1),
	)

	event := engine.Process(
		observation(116, 0),
	)

	if event.State != StateNormal {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateNormal,
		)
	}
}

func TestEventEngineRecoversAndResolves(t *testing.T) {
	engine := NewEventEngine("M001")

	engine.Process(
		observationAt(100, 1, 0.64),
	)

	engine.Process(
		observationAt(105, 1, 0.66),
	)

	event := engine.Process(
		observationAt(110, 1, 0.68),
	)

	if event.State != StateActive {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateActive,
		)
	}

	event = engine.Process(
		observation(126, 0),
	)

	if event.State != StateRecovery {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateRecovery,
		)
	}

	event = engine.Process(
		observation(142, 0),
	)

	if event.State != StateResolved {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateResolved,
		)
	}
}

func TestEventEngineNewIncidentAfterResolution(t *testing.T) {
	engine := NewEventEngine("M001")

	engine.Process(
		observationAt(100, 1, 0.64),
	)

	engine.Process(
		observationAt(105, 1, 0.66),
	)

	engine.Process(
		observationAt(110, 1, 0.68),
	)

	engine.Process(
		observation(126, 0),
	)

	engine.Process(
		observation(142, 0),
	)

	event := engine.Process(
		observationAt(200, 1, 0.65),
	)

	if event.State != StateSuspected {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateSuspected,
		)
	}
}

func TestEventEngineNegativeBehavioralAccelerationDoesNotActivate(t *testing.T) {
	engine := NewEventEngine("M001")

	engine.Process(
		Observation{
			MerchantID:           "M001",
			Timestamp:            100,
			AnomalyScore:         0.64,
			Threshold:            0.637,
			Prediction:           1,
			VelocityAcceleration: -0.20,
			AmountAcceleration:   -0.30,
		},
	)

	engine.Process(
		Observation{
			MerchantID:           "M001",
			Timestamp:            105,
			AnomalyScore:         0.66,
			Threshold:            0.637,
			Prediction:           1,
			VelocityAcceleration: -0.10,
			AmountAcceleration:   -0.20,
		},
	)

	event := engine.Process(
		Observation{
			MerchantID:           "M001",
			Timestamp:            110,
			AnomalyScore:         0.68,
			Threshold:            0.637,
			Prediction:           1,
			VelocityAcceleration: -0.05,
			AmountAcceleration:   -0.10,
		},
	)

	if event.State != StateSuspected {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateSuspected,
		)
	}
}

func TestEventEnginePositiveBehavioralAccelerationActivates(t *testing.T) {
	engine := NewEventEngine("M001")

	engine.Process(
		observationAt(100, 1, 0.64),
	)

	engine.Process(
		Observation{
			MerchantID:           "M001",
			Timestamp:            105,
			AnomalyScore:         0.66,
			Threshold:            0.637,
			Prediction:           1,
			VelocityAcceleration: 0.10,
			AmountAcceleration:   0.20,
		},
	)

	event := engine.Process(
		Observation{
			MerchantID:           "M001",
			Timestamp:            110,
			AnomalyScore:         0.68,
			Threshold:            0.637,
			Prediction:           1,
			VelocityAcceleration: 0.20,
			AmountAcceleration:   0.30,
		},
	)

	if event.State != StateActive {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateActive,
		)
	}
}

func TestEventEngineRecoveryDoesNotReactivateWithoutEscalation(t *testing.T) {
	engine := NewEventEngine("M001")

	engine.Process(
		observationAt(100, 1, 0.64),
	)

	engine.Process(
		observationAt(105, 1, 0.66),
	)

	engine.Process(
		observationAt(110, 1, 0.68),
	)

	event := engine.Process(
		observation(126, 0),
	)

	if event.State != StateRecovery {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateRecovery,
		)
	}

	event = engine.Process(
		observationAt(127, 1, 0.69),
	)

	if event.State != StateRecovery {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateRecovery,
		)
	}

	event = engine.Process(
		observationAt(132, 1, 0.70),
	)

	if event.State != StateRecovery {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateRecovery,
		)
	}
}

func TestEventManagerMaintainsSeparateMerchantState(t *testing.T) {
	manager := NewEventManager()

	manager.Process(Observation{
		MerchantID:           "M001",
		Timestamp:            100,
		AnomalyScore:         0.64,
		Threshold:            0.637,
		Prediction:           1,
		VelocityAcceleration: 0.10,
		AmountAcceleration:   0.10,
	})

	manager.Process(Observation{
		MerchantID:   "M002",
		Timestamp:    100,
		AnomalyScore: 0.50,
		Threshold:    0.637,
		Prediction:   0,
	})

	if state := manager.State("M001"); state != StateSuspected {
		t.Fatalf("got M001 state %q, want %q", state, StateSuspected)
	}

	if state := manager.State("M002"); state != StateNormal {
		t.Fatalf("got M002 state %q, want %q", state, StateNormal)
	}
}

func TestEventManagerReturnsTransitionMetadata(t *testing.T) {
	manager := NewEventManager()

	event := manager.Process(Observation{
		MerchantID:           "M001",
		Timestamp:            100,
		AnomalyScore:         0.64,
		Threshold:            0.637,
		Prediction:           1,
		VelocityAcceleration: 0.10,
		AmountAcceleration:   0.10,
	})

	if !event.Transitioned {
		t.Fatal("expected first alert to transition NORMAL -> SUSPECTED")
	}

	if event.PreviousState != StateNormal {
		t.Fatalf(
			"got previous state %q, want %q",
			event.PreviousState,
			StateNormal,
		)
	}

	if event.State != StateSuspected {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateSuspected,
		)
	}

	event = manager.Process(Observation{
		MerchantID:           "M001",
		Timestamp:            101,
		AnomalyScore:         0.50,
		Threshold:            0.637,
		Prediction:           0,
		VelocityAcceleration: -0.10,
		AmountAcceleration:   -0.10,
	})

	if event.Transitioned {
		t.Fatal("did not expect a transition for an in-window non-alert")
	}

	if event.PreviousState != StateSuspected {
		t.Fatalf(
			"got previous state %q, want %q",
			event.PreviousState,
			StateSuspected,
		)
	}

	if event.State != StateSuspected {
		t.Fatalf(
			"got state %q, want %q",
			event.State,
			StateSuspected,
		)
	}
}
