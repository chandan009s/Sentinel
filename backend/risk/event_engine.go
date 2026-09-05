package risk

const (
	confirmationWindowSeconds = 15.0
	recoveryWindowSeconds     = 15.0
	requiredConfirmations     = 3
	minimumScoreIncrease      = 0.01
)

type EventState string

const (
	StateNormal    EventState = "NORMAL"
	StateSuspected EventState = "SUSPECTED"
	StateActive    EventState = "ACTIVE"
	StateRecovery  EventState = "RECOVERY"
	StateResolved  EventState = "RESOLVED"
)

type Observation struct {
	MerchantID           string
	Timestamp            float64
	AnomalyScore         float64
	Threshold            float64
	Prediction           int
	RiskLevel            string
	VelocityAcceleration float64
	AmountAcceleration   float64
}

type Event struct {
	MerchantID       string
	PreviousState    EventState
	State            EventState
	Transitioned     bool
	StartedAt        float64
	LastAlertAt      float64
	FirstAlertScore  float64
	ScoreIncrease    float64
	AlertCount       int
	DetectionCount   int
	TimeToActivation float64
}

type EventEngine struct {
	state                   EventState
	merchantID              string
	startedAt               float64
	lastAlertAt             float64
	alertCount              int
	detectionCount          int
	firstAlertAt            float64
	firstAlertScore         float64
	activationAt            float64
	recoveryStartedAt       float64
	recoveryAlertCount      int
	recoveryFirstAlertAt    float64
	recoveryFirstAlertScore float64
}

func NewEventEngine(merchantID string) *EventEngine {
	return &EventEngine{
		state:      StateNormal,
		merchantID: merchantID,
	}
}

func (e *EventEngine) State() EventState {
	return e.state
}

func (e *EventEngine) Process(observation Observation) Event {
	isAlert := observation.Prediction == 1
	previousState := e.state

	switch e.state {
	case StateNormal:
		if isAlert {
			e.startSuspected(
				observation.Timestamp,
				observation.AnomalyScore,
			)
		}

	case StateSuspected:
		if !isAlert {
			if observation.Timestamp-e.lastAlertAt > confirmationWindowSeconds {
				e.reset()
			}
			break
		}

		if observation.Timestamp-e.lastAlertAt <= confirmationWindowSeconds {
			e.alertCount++
			e.detectionCount++
			e.lastAlertAt = observation.Timestamp

			behavioralEscalation :=
				observation.VelocityAcceleration > 0 ||
					observation.AmountAcceleration > 0

			if e.alertCount >= requiredConfirmations &&
				observation.AnomalyScore >=
					e.firstAlertScore+minimumScoreIncrease &&
				behavioralEscalation {
				e.state = StateActive
				e.activationAt = observation.Timestamp
			}
		} else {
			e.startSuspected(
				observation.Timestamp,
				observation.AnomalyScore,
			)
		}

	case StateActive:
		if isAlert {
			e.lastAlertAt = observation.Timestamp
			e.detectionCount++
			e.recoveryStartedAt = 0
			break
		}

		if observation.Timestamp-e.lastAlertAt >= recoveryWindowSeconds {
			e.state = StateRecovery
			e.recoveryStartedAt = observation.Timestamp

			e.recoveryAlertCount = 0
			e.recoveryFirstAlertAt = 0
			e.recoveryFirstAlertScore = 0
		}

	case StateRecovery:
		if isAlert {
			if e.recoveryAlertCount == 0 ||
				observation.Timestamp-e.lastAlertAt > confirmationWindowSeconds {
				e.recoveryAlertCount = 1
				e.recoveryFirstAlertAt = observation.Timestamp
				e.recoveryFirstAlertScore = observation.AnomalyScore
				e.lastAlertAt = observation.Timestamp
				e.detectionCount++
				break
			}

			e.recoveryAlertCount++
			e.lastAlertAt = observation.Timestamp
			e.detectionCount++

			behavioralEscalation :=
				observation.VelocityAcceleration > 0 ||
					observation.AmountAcceleration > 0

			scoreProgression :=
				observation.AnomalyScore >=
					e.recoveryFirstAlertScore+minimumScoreIncrease

			if e.recoveryAlertCount >= requiredConfirmations &&
				scoreProgression &&
				behavioralEscalation {
				e.state = StateActive
				e.activationAt = observation.Timestamp
				e.recoveryStartedAt = 0
				e.recoveryAlertCount = 0
				e.recoveryFirstAlertAt = 0
				e.recoveryFirstAlertScore = 0
			}

			break
		}

		if observation.Timestamp-e.recoveryStartedAt >= recoveryWindowSeconds {
			e.state = StateResolved
		}

	case StateResolved:
		if isAlert {
			e.startSuspected(
				observation.Timestamp,
				observation.AnomalyScore,
			)
		}
	}

	event := e.snapshot(previousState)
	if e.firstAlertScore > 0 {
		event.ScoreIncrease = observation.AnomalyScore - e.firstAlertScore
	}
	return event
}

func (e *EventEngine) startSuspected(timestamp float64, score float64) {
	e.state = StateSuspected
	e.startedAt = timestamp
	e.firstAlertAt = timestamp
	e.firstAlertScore = score
	e.lastAlertAt = timestamp
	e.alertCount = 1
	e.detectionCount = 1
	e.activationAt = 0
	e.recoveryStartedAt = 0
	e.recoveryAlertCount = 0
	e.recoveryFirstAlertAt = 0
	e.recoveryFirstAlertScore = 0
}

func (e *EventEngine) snapshot(previousState EventState) Event {
	timeToActivation := 0.0

	if e.activationAt > 0 {
		timeToActivation = e.activationAt - e.firstAlertAt
	}

	return Event{
		MerchantID:       e.merchantID,
		PreviousState:    previousState,
		State:            e.state,
		Transitioned:     previousState != e.state,
		StartedAt:        e.startedAt,
		LastAlertAt:      e.lastAlertAt,
		FirstAlertScore:  e.firstAlertScore,
		ScoreIncrease:    0,
		AlertCount:       e.alertCount,
		DetectionCount:   e.detectionCount,
		TimeToActivation: timeToActivation,
	}
}

func (e *EventEngine) reset() {
	e.state = StateNormal
	e.startedAt = 0
	e.lastAlertAt = 0
	e.alertCount = 0
	e.detectionCount = 0
	e.firstAlertAt = 0
	e.firstAlertScore = 0
	e.activationAt = 0
	e.recoveryStartedAt = 0
	e.recoveryAlertCount = 0
	e.recoveryFirstAlertAt = 0
	e.recoveryFirstAlertScore = 0
}
