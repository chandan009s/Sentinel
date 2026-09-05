package risk

import "sync"

type EventManager struct {
	mu      sync.Mutex
	engines map[string]*EventEngine
}

func NewEventManager() *EventManager {
	return &EventManager{
		engines: make(map[string]*EventEngine),
	}
}

func (m *EventManager) Process(observation Observation) Event {
	m.mu.Lock()
	defer m.mu.Unlock()

	engine, ok := m.engines[observation.MerchantID]
	if !ok {
		engine = NewEventEngine(observation.MerchantID)
		m.engines[observation.MerchantID] = engine
	}

	return engine.Process(observation)
}

func (m *EventManager) State(merchantID string) EventState {
	m.mu.Lock()
	defer m.mu.Unlock()

	engine, ok := m.engines[merchantID]
	if !ok {
		return StateNormal
	}

	return engine.State()
}
