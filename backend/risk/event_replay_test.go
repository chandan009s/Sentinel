package risk

import "testing"

func TestReplayPredictionsUsesSeparateMerchantState(t *testing.T) {
	summary, err := ReplayPredictions(
		"../../data/reports/v2_prediction_stream.csv",
	)

	if err != nil {
		t.Fatalf(
			"ReplayPredictions() error = %v",
			err,
		)
	}

	if summary.TotalRows != 7755 {
		t.Fatalf(
			"got %d rows, want 7755",
			summary.TotalRows,
		)
	}

	if summary.RawAlerts != 2016 {
		t.Fatalf(
			"got %d raw alerts, want 2016",
			summary.RawAlerts,
		)
	}

	if len(summary.Results) != 4 {
		t.Fatalf(
			"got %d merchants, want 4",
			len(summary.Results),
		)
	}

	for _, result := range summary.Results {
		if result.MerchantID == "" {
			t.Fatal("merchant ID is empty")
		}
	}
}

func TestReplayPredictionsStatefullyDetectsExpectedIncidents(t *testing.T) {
	summary, err := ReplayPredictionsStatefully(
		"../../data/reports/v2_prediction_stream.csv",
		1210.30,
	)

	if err != nil {
		t.Fatalf(
			"ReplayPredictionsStatefully() error = %v",
			err,
		)
	}

	if summary.TotalRows != 7755 {
		t.Fatalf(
			"got %d total rows, want 7755",
			summary.TotalRows,
		)
	}

	if summary.WarmupRows != 3783 {
		t.Fatalf(
			"got %d warmup rows, want 3783",
			summary.WarmupRows,
		)
	}

	if summary.EvaluatedRows != 3972 {
		t.Fatalf(
			"got %d evaluated rows, want 3972",
			summary.EvaluatedRows,
		)
	}

	if summary.RawAlerts != 2016 {
		t.Fatalf(
			"got %d raw alerts, want 2016",
			summary.RawAlerts,
		)
	}

	if summary.EventsActivated != 2 {
		t.Fatalf(
			"got %d activated events, want 2",
			summary.EventsActivated,
		)
	}

	var activations []ReplayEvent

	for _, event := range summary.Events {
		if event.State == StateActive {
			activations = append(activations, event)
		}
	}

	if len(activations) != 2 {
		t.Fatalf(
			"got %d activation transitions, want 2",
			len(activations),
		)
	}

	expected := map[string]float64{
		"M002": 1321.220,
		"M003": 1500.811,
	}

	for _, activation := range activations {
		expectedTimestamp, ok := expected[activation.MerchantID]
		if !ok {
			t.Fatalf(
				"unexpected activation for merchant %s",
				activation.MerchantID,
			)
		}

		if activation.PreviousState != StateSuspected {
			t.Fatalf(
				"merchant %s activated from %q, want %q",
				activation.MerchantID,
				activation.PreviousState,
				StateSuspected,
			)
		}

		if diff := activation.Timestamp - expectedTimestamp; diff < -0.01 || diff > 0.01 {
			t.Fatalf(
				"merchant %s activated at %.3f, want %.3f",
				activation.MerchantID,
				activation.Timestamp,
				expectedTimestamp,
			)
		}
	}
}
