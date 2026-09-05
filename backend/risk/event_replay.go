package risk

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
)

type ReplayResult struct {
	MerchantID      string
	EventsActivated int
	EventsResolved  int
	FinalState      EventState
}

type ReplaySummary struct {
	TotalRows       int
	RawAlerts       int
	EventsActivated int
	EventsResolved  int
	Results         []ReplayResult
}

type ReplayEvent struct {
	MerchantID    string
	Timestamp     float64
	State         EventState
	PreviousState EventState
}

type StatefulReplaySummary struct {
	TotalRows       int
	RawAlerts       int
	WarmupRows      int
	EvaluatedRows   int
	EventsActivated int
	EventsResolved  int
	Events          []ReplayEvent
}

func readPredictionHeader(
	reader *csv.Reader,
) (map[string]int, error) {
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	columns := make(map[string]int, len(header))

	for index, name := range header {
		columns[name] = index
	}

	required := []string{
		"timestamp",
		"merchant_id",
		"anomaly_score",
		"threshold",
		"prediction",
		"velocity_acceleration_1m",
		"amount_acceleration_1m",
	}

	for _, column := range required {
		if _, ok := columns[column]; !ok {
			return nil, fmt.Errorf(
				"missing required column %q",
				column,
			)
		}
	}

	return columns, nil
}

type predictionRow struct {
	Timestamp            float64
	MerchantID           string
	AnomalyScore         float64
	Threshold            float64
	Prediction           int
	VelocityAcceleration float64
	AmountAcceleration   float64
}

func parsePredictionRow(
	record []string,
	columns map[string]int,
) (predictionRow, error) {
	timestamp, err := strconv.ParseFloat(
		record[columns["timestamp"]],
		64,
	)
	if err != nil {
		return predictionRow{}, fmt.Errorf(
			"parse timestamp: %w",
			err,
		)
	}

	anomalyScore, err := strconv.ParseFloat(
		record[columns["anomaly_score"]],
		64,
	)
	if err != nil {
		return predictionRow{}, fmt.Errorf(
			"parse anomaly_score: %w",
			err,
		)
	}

	threshold, err := strconv.ParseFloat(
		record[columns["threshold"]],
		64,
	)
	if err != nil {
		return predictionRow{}, fmt.Errorf(
			"parse threshold: %w",
			err,
		)
	}

	prediction, err := strconv.Atoi(
		record[columns["prediction"]],
	)
	if err != nil {
		return predictionRow{}, fmt.Errorf(
			"parse prediction: %w",
			err,
		)
	}

	velocityAcceleration, err := strconv.ParseFloat(
		record[columns["velocity_acceleration_1m"]],
		64,
	)
	if err != nil {
		return predictionRow{}, fmt.Errorf(
			"parse velocity_acceleration_1m: %w",
			err,
		)
	}

	amountAcceleration, err := strconv.ParseFloat(
		record[columns["amount_acceleration_1m"]],
		64,
	)
	if err != nil {
		return predictionRow{}, fmt.Errorf(
			"parse amount_acceleration_1m: %w",
			err,
		)
	}

	return predictionRow{
		Timestamp:            timestamp,
		MerchantID:           record[columns["merchant_id"]],
		AnomalyScore:         anomalyScore,
		Threshold:            threshold,
		Prediction:           prediction,
		VelocityAcceleration: velocityAcceleration,
		AmountAcceleration:   amountAcceleration,
	}, nil
}

func ReplayPredictions(path string) (ReplaySummary, error) {
	file, err := os.Open(path)
	if err != nil {
		return ReplaySummary{}, fmt.Errorf(
			"open prediction file: %w",
			err,
		)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	columns, err := readPredictionHeader(reader)
	if err != nil {
		return ReplaySummary{}, err
	}

	engines := make(map[string]*EventEngine)
	activated := make(map[string]int)
	resolved := make(map[string]int)

	summary := ReplaySummary{}

	for {
		record, err := reader.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			return ReplaySummary{}, fmt.Errorf(
				"read prediction row: %w",
				err,
			)
		}

		summary.TotalRows++

		row, err := parsePredictionRow(
			record,
			columns,
		)
		if err != nil {
			return ReplaySummary{}, err
		}

		if row.Prediction == 1 {
			summary.RawAlerts++
		}

		engine, ok := engines[row.MerchantID]
		if !ok {
			engine = NewEventEngine(row.MerchantID)
			engines[row.MerchantID] = engine
		}

		previousState := engine.State()

		event := engine.Process(
			Observation{
				MerchantID:           row.MerchantID,
				Timestamp:            row.Timestamp,
				AnomalyScore:         row.AnomalyScore,
				Threshold:            row.Threshold,
				Prediction:           row.Prediction,
				VelocityAcceleration: row.VelocityAcceleration,
				AmountAcceleration:   row.AmountAcceleration,
			},
		)

		if previousState != StateActive &&
			event.State == StateActive {
			activated[row.MerchantID]++
		}

		if previousState != StateResolved &&
			event.State == StateResolved {
			resolved[row.MerchantID]++
		}
	}

	results := make([]ReplayResult, 0, len(engines))

	for merchantID, engine := range engines {
		results = append(
			results,
			ReplayResult{
				MerchantID:      merchantID,
				EventsActivated: activated[merchantID],
				EventsResolved:  resolved[merchantID],
				FinalState:      engine.State(),
			},
		)
	}

	sort.Slice(
		results,
		func(i, j int) bool {
			return results[i].MerchantID < results[j].MerchantID
		},
	)

	for _, result := range results {
		summary.EventsActivated += result.EventsActivated
		summary.EventsResolved += result.EventsResolved
	}

	summary.Results = results

	return summary, nil
}

func ReplayPredictionsStatefully(
	path string,
	evaluationStart float64,
) (StatefulReplaySummary, error) {
	file, err := os.Open(path)
	if err != nil {
		return StatefulReplaySummary{}, fmt.Errorf(
			"open prediction file: %w",
			err,
		)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	columns, err := readPredictionHeader(reader)
	if err != nil {
		return StatefulReplaySummary{}, err
	}

	engines := make(map[string]*EventEngine)

	summary := StatefulReplaySummary{
		Events: make([]ReplayEvent, 0),
	}

	for {
		record, err := reader.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			return StatefulReplaySummary{}, fmt.Errorf(
				"read prediction row: %w",
				err,
			)
		}

		summary.TotalRows++

		row, err := parsePredictionRow(
			record,
			columns,
		)
		if err != nil {
			return StatefulReplaySummary{}, err
		}

		if row.Prediction == 1 {
			summary.RawAlerts++
		}

		engine, ok := engines[row.MerchantID]
		if !ok {
			engine = NewEventEngine(row.MerchantID)
			engines[row.MerchantID] = engine
		}

		previousState := engine.State()

		event := engine.Process(
			Observation{
				MerchantID:           row.MerchantID,
				Timestamp:            row.Timestamp,
				AnomalyScore:         row.AnomalyScore,
				Threshold:            row.Threshold,
				Prediction:           row.Prediction,
				VelocityAcceleration: row.VelocityAcceleration,
				AmountAcceleration:   row.AmountAcceleration,
			},
		)

		if row.Timestamp < evaluationStart {
			summary.WarmupRows++
			continue
		}

		summary.EvaluatedRows++

		if previousState != StateActive &&
			event.State == StateActive {
			summary.EventsActivated++

			summary.Events = append(
				summary.Events,
				ReplayEvent{
					MerchantID:    row.MerchantID,
					Timestamp:     row.Timestamp,
					PreviousState: previousState,
					State:         event.State,
				},
			)
		}

		if previousState != StateResolved &&
			event.State == StateResolved {
			summary.EventsResolved++

			summary.Events = append(
				summary.Events,
				ReplayEvent{
					MerchantID:    row.MerchantID,
					Timestamp:     row.Timestamp,
					PreviousState: previousState,
					State:         event.State,
				},
			)
		}
	}

	return summary, nil
}
