package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"

	"ai-risk-manager/backend/database"
	"ai-risk-manager/backend/risk"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"time"
)

var mlServiceURL string

var db *sql.DB

var incidentCoordinator *IncidentCoordinator

type modelMetadata struct {
	ModelName              string  `json:"model_name"`
	ModelVersion           string  `json:"model_version"`
	FeatureVersion         string  `json:"feature_version"`
	FeatureContractVersion string  `json:"feature_contract_version"`
	ThresholdVersion       string  `json:"threshold_version"`
	Threshold              float64 `json:"threshold"`

	ReferenceStatistics map[string]struct {
		Mean float64 `json:"mean"`
		Std  float64 `json:"std"`
	} `json:"reference_statistics"`
}

func loadModelMetadata() (modelMetadata, error) {
	var metadata modelMetadata

	data, err := os.ReadFile(
		"ml/models/isolation_forest_v1_meta.json",
	)
	if err != nil {
		return metadata, err
	}

	if err := json.Unmarshal(data, &metadata); err != nil {
		return metadata, err
	}

	return metadata, nil
}

var monitoringMu sync.Mutex

var monitoringSamples uint64
var runtimePredictionSum float64

var runtimeFeatureSums = map[string]float64{
	"velocity_ratio":           0,
	"amount_ratio":             0,
	"velocity_acceleration_1m": 0,
	"amount_acceleration_1m":   0,
}

var runtimeScoreSum float64

var (
	predictionsTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "risk_predictions_total",
			Help: "Total number of risk predictions.",
		},
	)

	highRiskTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "risk_high_predictions_total",
			Help: "Total number of HIGH risk predictions.",
		},
	)

	mediumRiskTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "risk_medium_predictions_total",
			Help: "Total number of MEDIUM risk predictions.",
		},
	)

	lowRiskTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "risk_low_predictions_total",
			Help: "Total number of LOW risk predictions.",
		},
	)

	predictionLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name: "risk_prediction_latency_seconds",
			Help: "Time taken to process a prediction.",
		},
	)
	runtimeVelocityRatioSum = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "risk_runtime_velocity_ratio_sum_total",
			Help: "Cumulative sum of runtime velocity ratio values.",
		},
	)

	runtimeAmountRatioSum = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "risk_runtime_amount_ratio_sum_total",
			Help: "Cumulative sum of runtime amount ratio values.",
		},
	)

	runtimeVelocityAccelerationSum = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "risk_runtime_velocity_acceleration_1m_sum_total",
			Help: "Cumulative sum of runtime velocity acceleration values.",
		},
	)

	runtimeAmountAccelerationSum = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "risk_runtime_amount_acceleration_1m_sum_total",
			Help: "Cumulative sum of runtime amount acceleration values.",
		},
	)

	runtimeAnomalyScoreSum = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "risk_runtime_anomaly_score_sum_total",
			Help: "Cumulative sum of runtime anomaly score values.",
		},
	)
	referenceFeatureMean = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "risk_reference_feature_mean",
			Help: "Training reference mean for a production feature.",
		},
		[]string{"feature"},
	)

	referenceAnomalyScoreMean = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "risk_reference_anomaly_score_mean",
			Help: "Reference anomaly score mean.",
		},
	)
	featureVelocityMean = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "risk_feature_velocity_ratio_mean",
			Help: "Observed mean velocity ratio.",
		},
	)

	featureAmountMean = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "risk_feature_amount_ratio_mean",
			Help: "Observed mean amount ratio.",
		},
	)

	featureVelocityAccelerationMean = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "risk_feature_velocity_acceleration_1m_mean",
			Help: "Observed mean velocity acceleration.",
		},
	)

	featureAmountAccelerationMean = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "risk_feature_amount_acceleration_1m_mean",
			Help: "Observed mean amount acceleration.",
		},
	)

	anomalyScoreMean = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "risk_anomaly_score_mean",
			Help: "Observed mean anomaly score.",
		},
	)

	predictionRate = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "risk_prediction_rate",
			Help: "Fraction of observed predictions classified as anomalous.",
		},
	)

	monitoringSamplesTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "risk_monitoring_samples_total",
			Help: "Total predictions included in runtime monitoring.",
		},
	)

	modelInfo = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "risk_model_info",
			Help: "Metadata for the currently serving model.",
		},
		[]string{
			"model_name",
			"model_version",
			"feature_version",
			"feature_contract_version",
			"threshold_version",
		},
	)
)

func init() {
	prometheus.MustRegister(predictionsTotal)
	prometheus.MustRegister(highRiskTotal)
	prometheus.MustRegister(mediumRiskTotal)
	prometheus.MustRegister(lowRiskTotal)
	prometheus.MustRegister(predictionLatency)
	prometheus.MustRegister(runtimeVelocityRatioSum)
	prometheus.MustRegister(runtimeAmountRatioSum)
	prometheus.MustRegister(runtimeVelocityAccelerationSum)
	prometheus.MustRegister(runtimeAmountAccelerationSum)
	prometheus.MustRegister(runtimeAnomalyScoreSum)
	prometheus.MustRegister(referenceFeatureMean)
	prometheus.MustRegister(referenceAnomalyScoreMean)
	prometheus.MustRegister(featureVelocityMean)
	prometheus.MustRegister(featureAmountMean)
	prometheus.MustRegister(featureVelocityAccelerationMean)
	prometheus.MustRegister(featureAmountAccelerationMean)
	prometheus.MustRegister(anomalyScoreMean)
	prometheus.MustRegister(predictionRate)
	prometheus.MustRegister(monitoringSamplesTotal)
	prometheus.MustRegister(modelInfo)
}

type PredictRequest struct {
	EventID              string  `json:"event_id"`
	MerchantID           string  `json:"merchant_id,omitempty"`
	Timestamp            float64 `json:"timestamp,omitempty"`
	VelocityRatio        float64 `json:"velocity_ratio"`
	AmountRatio          float64 `json:"amount_ratio"`
	VelocityAcceleration float64 `json:"velocity_acceleration_1m"`
	AmountAcceleration   float64 `json:"amount_acceleration_1m"`
}

type PredictResponse struct {
	MerchantID             string          `json:"merchant_id,omitempty"`
	Timestamp              float64         `json:"timestamp,omitempty"`
	AnomalyScore           float64         `json:"anomaly_score"`
	Threshold              float64         `json:"threshold"`
	Prediction             int             `json:"prediction"`
	RiskLevel              string          `json:"risk_level"`
	Decision               risk.Decision   `json:"decision"`
	EventState             risk.EventState `json:"event_state"`
	AlertCount             int             `json:"alert_count"`
	DetectionCount         int             `json:"detection_count"`
	TimeToActivation       float64         `json:"time_to_activation"`
	ModelName              string          `json:"model_name"`
	ModelVersion           string          `json:"model_version"`
	FeatureVersion         string          `json:"feature_version"`
	FeatureContractVersion string          `json:"feature_contract_version"`
	ThresholdVersion       string          `json:"threshold_version"`
}

const mediumRiskMargin = 0.05

func determineRiskLevel(score, threshold float64) string {
	if score < threshold {
		return "LOW"
	}

	if score < threshold+mediumRiskMargin {
		return "MEDIUM"
	}
	return "HIGH"
}

func getExistingRiskEvent(eventID string) (*PredictResponse, error) {
	var response PredictResponse

	err := db.QueryRow(`
		SELECT
			merchant_id,
			timestamp,
			anomaly_score,
			threshold,
			prediction,
			risk_level,
			decision,
			event_state,
			alert_count,
			detection_count,
			time_to_activation,
			model_name,
			model_version,
			feature_version,
			feature_contract_version,
			threshold_version
		FROM risk_events
		WHERE event_id = $1
	`,
		eventID,
	).Scan(
		&response.MerchantID,
		&response.Timestamp,
		&response.AnomalyScore,
		&response.Threshold,
		&response.Prediction,
		&response.RiskLevel,
		&response.Decision,
		&response.EventState,
		&response.AlertCount,
		&response.DetectionCount,
		&response.TimeToActivation,
		&response.ModelName,
		&response.ModelVersion,
		&response.FeatureVersion,
		&response.FeatureContractVersion,
		&response.ThresholdVersion,
	)

	if err != nil {
		return nil, err
	}

	return &response, nil
}

func eventExists(eventID string) (bool, error) {
	var exists bool

	err := db.QueryRow(`
		SELECT EXISTS (
			SELECT 1
			FROM risk_events
			WHERE event_id = $1
		)
	`, eventID).Scan(&exists)

	if err != nil {
		return false, err
	}

	return exists, nil
}

func recordMonitoringSample(
	request PredictRequest,
	prediction PredictResponse,
) {
	monitoringMu.Lock()
	defer monitoringMu.Unlock()

	monitoringSamples++
	runtimeVelocityRatioSum.Add(request.VelocityRatio)
	runtimeAmountRatioSum.Add(request.AmountRatio)
	runtimeVelocityAccelerationSum.Add(request.VelocityAcceleration)
	runtimeAmountAccelerationSum.Add(request.AmountAcceleration)
	runtimeAnomalyScoreSum.Add(prediction.AnomalyScore)
	runtimePredictionSum += float64(prediction.Prediction)

	runtimeFeatureSums["velocity_ratio"] += request.VelocityRatio
	runtimeFeatureSums["amount_ratio"] += request.AmountRatio
	runtimeFeatureSums["velocity_acceleration_1m"] += request.VelocityAcceleration
	runtimeFeatureSums["amount_acceleration_1m"] += request.AmountAcceleration

	runtimeScoreSum += prediction.AnomalyScore

	count := float64(monitoringSamples)

	featureVelocityMean.Set(
		runtimeFeatureSums["velocity_ratio"] / count,
	)

	featureAmountMean.Set(
		runtimeFeatureSums["amount_ratio"] / count,
	)

	featureVelocityAccelerationMean.Set(
		runtimeFeatureSums["velocity_acceleration_1m"] / count,
	)

	featureAmountAccelerationMean.Set(
		runtimeFeatureSums["amount_acceleration_1m"] / count,
	)

	anomalyScoreMean.Set(
		runtimeScoreSum / count,
	)

	predictionRate.Set(
		runtimePredictionSum / count,
	)

	monitoringSamplesTotal.Inc()

	modelInfo.WithLabelValues(
		prediction.ModelName,
		prediction.ModelVersion,
		prediction.FeatureVersion,
		prediction.FeatureContractVersion,
		prediction.ThresholdVersion,
	).Set(1)
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()

	defer func() {
		predictionLatency.Observe(
			time.Since(start).Seconds(),
		)
	}()

	var request PredictRequest

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if math.IsNaN(request.VelocityRatio) ||
		math.IsInf(request.VelocityRatio, 0) ||
		math.IsNaN(request.AmountRatio) ||
		math.IsInf(request.AmountRatio, 0) ||
		math.IsNaN(request.VelocityAcceleration) ||
		math.IsInf(request.VelocityAcceleration, 0) ||
		math.IsNaN(request.AmountAcceleration) ||
		math.IsInf(request.AmountAcceleration, 0) ||
		request.VelocityRatio < 0 ||
		request.AmountRatio < 0 {
		http.Error(w, "feature values must be finite and ratios must be non-negative", http.StatusBadRequest)
		return
	}

	if request.EventID != "" {
		exists, err := eventExists(request.EventID)
		if err != nil {
			log.Printf("failed to check event_id: %v", err)

			http.Error(
				w,
				"failed to check event_id",
				http.StatusInternalServerError,
			)
			return
		}

		if exists {
			existing, err := getExistingRiskEvent(request.EventID)
			if err != nil {
				log.Printf("failed to retrieve existing risk event: %v", err)

				http.Error(
					w,
					"failed to retrieve existing risk event",
					http.StatusInternalServerError,
				)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(existing)
			return
		}
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		http.Error(w, "failed to encode request", http.StatusInternalServerError)
		return
	}

	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	response, err := client.Post(
		mlServiceURL,
		"application/json",
		bytes.NewBuffer(requestBody),
	)
	if err != nil {
		http.Error(w, "ML service unavailable", http.StatusBadGateway)
		return
	}

	defer response.Body.Close()

	body, err := io.ReadAll(response.Body)
	if err != nil {
		http.Error(w, "failed to read ML response", http.StatusBadGateway)
		return
	}

	if response.StatusCode != http.StatusOK {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)

		json.NewEncoder(w).Encode(map[string]interface{}{
			"error":       "ML service returned an error",
			"ml_status":   response.Status,
			"ml_response": string(body),
		})
		return
	}

	var prediction PredictResponse

	if err := json.Unmarshal(body, &prediction); err != nil {
		http.Error(w, "failed to decode ML response", http.StatusBadGateway)
		return
	}

	prediction.RiskLevel = determineRiskLevel(
		prediction.AnomalyScore,
		prediction.Threshold,
	)

	event, err := incidentCoordinator.Process(
		risk.Observation{
			MerchantID:           request.MerchantID,
			Timestamp:            request.Timestamp,
			AnomalyScore:         prediction.AnomalyScore,
			Threshold:            prediction.Threshold,
			Prediction:           prediction.Prediction,
			RiskLevel:            prediction.RiskLevel,
			VelocityAcceleration: request.VelocityAcceleration,
			AmountAcceleration:   request.AmountAcceleration,
		},
	)

	if err != nil {
		log.Printf("failed to process risk incident: %v", err)

		http.Error(
			w,
			"failed to process risk incident",
			http.StatusInternalServerError,
		)
		return
	}

	prediction.EventState = event.State
	prediction.AlertCount = event.AlertCount
	prediction.DetectionCount = event.DetectionCount
	prediction.TimeToActivation = event.TimeToActivation
	prediction.Decision = risk.DetermineDecision(
		event.State,
		prediction.RiskLevel,
	)
	recordMonitoringSample(
		request,
		prediction,
	)

	_, err = db.Exec(`
                INSERT INTO risk_events (
						event_id,
                        merchant_id,
                        timestamp,
                        velocity_ratio,
                        amount_ratio,
                        velocity_acceleration_1m,
                        amount_acceleration_1m,
                        anomaly_score,
                        threshold,
                        prediction,
                        risk_level,
						decision,
						event_state,
						alert_count,
						detection_count,
						time_to_activation,
						model_name,
						model_version,
						feature_version,
						feature_contract_version,
						threshold_version
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
        `,
		request.EventID,
		request.MerchantID,
		request.Timestamp,
		request.VelocityRatio,
		request.AmountRatio,
		request.VelocityAcceleration,
		request.AmountAcceleration,
		prediction.AnomalyScore,
		prediction.Threshold,
		prediction.Prediction,
		prediction.RiskLevel,
		prediction.Decision,
		prediction.EventState,
		prediction.AlertCount,
		prediction.DetectionCount,
		prediction.TimeToActivation,
		prediction.ModelName,
		prediction.ModelVersion,
		prediction.FeatureVersion,
		prediction.FeatureContractVersion,
		prediction.ThresholdVersion,
	)

	if err != nil {
		if strings.Contains(err.Error(), "idx_risk_events_event_id") {
			existing, lookupErr := getExistingRiskEvent(request.EventID)
			if lookupErr != nil {
				log.Printf("failed to retrieve existing risk event: %v", lookupErr)

				http.Error(
					w,
					"failed to retrieve existing risk event",
					http.StatusInternalServerError,
				)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(existing)
			return
		}
		log.Printf("failed to save risk event: %v", err)

		http.Error(
			w,
			"failed to save prediction",
			http.StatusInternalServerError,
		)
		return
	}

	predictionsTotal.Inc()

	if prediction.RiskLevel == "HIGH" {
		highRiskTotal.Inc()
	} else if prediction.RiskLevel == "MEDIUM" {
		mediumRiskTotal.Inc()
	} else if prediction.RiskLevel == "LOW" {
		lowRiskTotal.Inc()
	}

	w.Header().Set("Content-Type", "application/json")

	json.NewEncoder(w).Encode(prediction)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	json.NewEncoder(w).Encode(map[string]string{
		"status":  "ok",
		"service": "go-api",
	})
}

func riskEventsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	rows, err := db.Query(`
                SELECT
                        id,
                        merchant_id,
                        timestamp,
                        velocity_ratio,
                        amount_ratio,
                        velocity_acceleration_1m,
                        amount_acceleration_1m,
                        anomaly_score,
                        threshold,
                        prediction,
                        risk_level,
						decision,
						model_name,
						model_version,
						feature_version,
						feature_contract_version,
						threshold_version,
                        created_at
		FROM risk_events
		ORDER BY created_at DESC
		LIMIT 50
	`)

	if err != nil {
		http.Error(
			w,
			"failed to query risk events",
			http.StatusInternalServerError,
		)
		return
	}

	defer rows.Close()

	var events []map[string]interface{}

	for rows.Next() {
		var (
			id                     int64
			merchantID             sql.NullString
			timestamp              float64
			velocityRatio          float64
			amountRatio            float64
			velocityAcceleration   float64
			amountAcceleration     float64
			anomalyScore           float64
			threshold              float64
			prediction             int
			riskLevel              string
			decision               string
			modelName              string
			modelVersion           string
			featureVersion         string
			featureContractVersion string
			thresholdVersion       string
			createdAt              string
		)

		err := rows.Scan(
			&id,
			&merchantID,
			&timestamp,
			&velocityRatio,
			&amountRatio,
			&velocityAcceleration,
			&amountAcceleration,
			&anomalyScore,
			&threshold,
			&prediction,
			&riskLevel,
			&decision,
			&modelName,
			&modelVersion,
			&featureVersion,
			&featureContractVersion,
			&thresholdVersion,
			&createdAt,
		)

		if err != nil {
			http.Error(
				w,
				"failed to read risk event",
				http.StatusInternalServerError,
			)
			return
		}

		events = append(events, map[string]interface{}{
			"id":                       id,
			"merchant_id":              merchantID.String,
			"timestamp":                timestamp,
			"velocity_ratio":           velocityRatio,
			"amount_ratio":             amountRatio,
			"velocity_acceleration_1m": velocityAcceleration,
			"amount_acceleration_1m":   amountAcceleration,
			"anomaly_score":            anomalyScore,
			"threshold":                threshold,
			"prediction":               prediction,
			"risk_level":               riskLevel,
			"decision":                 decision,
			"model_name":               modelName,
			"model_version":            modelVersion,
			"feature_version":          featureVersion,
			"feature_contract_version": featureContractVersion,
			"threshold_version":        thresholdVersion,
			"created_at":               createdAt,
		})
	}

	if err := rows.Err(); err != nil {
		http.Error(
			w,
			"failed to read risk events",
			http.StatusInternalServerError,
		)
		return
	}

	if events == nil {
		events = []map[string]interface{}{}
	}

	w.Header().Set("Content-Type", "application/json")

	json.NewEncoder(w).Encode(events)
}

func riskIncidentsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	rows, err := db.Query(`
		SELECT
			id,
			merchant_id,
			started_at,
			activated_at,
			resolved_at,
			state,
			alert_count,
			detection_count,
			time_to_activation,
			created_at,
			updated_at
		FROM risk_incidents
		ORDER BY created_at DESC
		LIMIT 50
	`)

	if err != nil {
		http.Error(
			w,
			"failed to query risk incidents",
			http.StatusInternalServerError,
		)
		return
	}

	defer rows.Close()

	var incidents []map[string]interface{}

	for rows.Next() {
		var (
			id               int64
			merchantID       string
			startedAt        float64
			activatedAt      sql.NullFloat64
			resolvedAt       sql.NullFloat64
			state            string
			alertCount       int
			detectionCount   int
			timeToActivation sql.NullFloat64
			createdAt        string
			updatedAt        string
		)

		if err := rows.Scan(
			&id,
			&merchantID,
			&startedAt,
			&activatedAt,
			&resolvedAt,
			&state,
			&alertCount,
			&detectionCount,
			&timeToActivation,
			&createdAt,
			&updatedAt,
		); err != nil {
			http.Error(
				w,
				"failed to read risk incident",
				http.StatusInternalServerError,
			)
			return
		}

		incident := map[string]interface{}{
			"id":              id,
			"merchant_id":     merchantID,
			"started_at":      startedAt,
			"state":           state,
			"alert_count":     alertCount,
			"detection_count": detectionCount,
			"created_at":      createdAt,
			"updated_at":      updatedAt,
		}

		if activatedAt.Valid {
			incident["activated_at"] = activatedAt.Float64
		} else {
			incident["activated_at"] = nil
		}

		if resolvedAt.Valid {
			incident["resolved_at"] = resolvedAt.Float64
		} else {
			incident["resolved_at"] = nil
		}

		if timeToActivation.Valid {
			incident["time_to_activation"] = timeToActivation.Float64
		} else {
			incident["time_to_activation"] = nil
		}

		incidents = append(incidents, incident)
	}

	if err := rows.Err(); err != nil {
		http.Error(
			w,
			"failed to read risk incidents",
			http.StatusInternalServerError,
		)
		return
	}

	if incidents == nil {
		incidents = []map[string]interface{}{}
	}

	w.Header().Set("Content-Type", "application/json")

	json.NewEncoder(w).Encode(incidents)
}

func riskIncidentEvidenceHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	const prefix = "/risk-incidents/"
	path := r.URL.Path

	if len(path) <= len(prefix) || path[:len(prefix)] != prefix {
		http.Error(w, "invalid incident path", http.StatusBadRequest)
		return
	}

	parts := strings.Split(strings.Trim(path[len(prefix):], "/"), "/")

	if len(parts) != 2 || parts[1] != "evidence" {
		http.Error(w, "invalid incident path", http.StatusBadRequest)
		return
	}

	incidentID, err := strconv.ParseInt(parts[0], 10, 64)
	if err != nil || incidentID <= 0 {
		http.Error(w, "invalid incident id", http.StatusBadRequest)
		return
	}

	evidenceStore := database.NewEvidenceStore(db)

	evidence, err := evidenceStore.ListByIncident(incidentID)
	if err != nil {
		log.Printf("failed to query incident evidence: %v", err)

		http.Error(
			w,
			"failed to query incident evidence",
			http.StatusInternalServerError,
		)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	if err := json.NewEncoder(w).Encode(evidence); err != nil {
		log.Printf("failed to encode incident evidence: %v", err)
	}
}

func main() {
	var err error

	mlServiceURL = os.Getenv("ML_SERVICE_URL")

	if mlServiceURL == "" {
		log.Fatal("ML_SERVICE_URL is not set")
	}

	metadata, err := loadModelMetadata()
	if err != nil {
		log.Fatal("failed to load model metadata: ", err)
	}

	modelInfo.WithLabelValues(
		metadata.ModelName,
		metadata.ModelVersion,
		metadata.FeatureVersion,
		metadata.FeatureContractVersion,
		metadata.ThresholdVersion,
	).Set(1)

	for feature, stats := range metadata.ReferenceStatistics {
		referenceFeatureMean.
			WithLabelValues(feature).
			Set(stats.Mean)
	}

	db, err = database.Connect()
	if err != nil {
		log.Fatal("PostgreSQL connection failed: ", err)
	}

	defer db.Close()

	incidentCoordinator = NewIncidentCoordinator(
		database.NewIncidentStore(db),
		database.NewEvidenceStore(db),
	)

	log.Println("PostgreSQL connected")

	http.HandleFunc("/predict", predictHandler)
	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/readyz", healthHandler)
	http.HandleFunc("/risk-events", riskEventsHandler)
	http.HandleFunc("/risk-incidents", riskIncidentsHandler)
	http.HandleFunc("/risk-incidents/", riskIncidentEvidenceHandler)
	http.Handle("/metrics", promhttp.Handler())

	log.Println("Go API running on :8080")

	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
