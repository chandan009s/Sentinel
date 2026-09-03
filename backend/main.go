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

	"ai-risk-manager/backend/database"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"time"
)

var mlServiceURL string

var db *sql.DB

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
)

func init() {
	prometheus.MustRegister(predictionsTotal)
	prometheus.MustRegister(highRiskTotal)
	prometheus.MustRegister(lowRiskTotal)
	prometheus.MustRegister(predictionLatency)
}

type PredictRequest struct {
	MerchantID    string  `json:"merchant_id,omitempty"`
	Timestamp     float64 `json:"timestamp,omitempty"`
	VelocityRatio float64 `json:"velocity_ratio"`
	AmountRatio   float64 `json:"amount_ratio"`
}

type PredictResponse struct {
	MerchantID   string  `json:"merchant_id,omitempty"`
	Timestamp    float64 `json:"timestamp,omitempty"`
	AnomalyScore float64 `json:"anomaly_score"`
	Threshold    float64 `json:"threshold"`
	Prediction   int     `json:"prediction"`
	RiskLevel    string  `json:"risk_level"`
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
		request.VelocityRatio < 0 ||
		request.AmountRatio < 0 {
		http.Error(w, "ratios must be non-negative finite numbers", http.StatusBadRequest)
		return
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

	_, err = db.Exec(`
		INSERT INTO risk_events (
			merchant_id,
			timestamp,
			velocity_ratio,
			amount_ratio,
			anomaly_score,
			threshold,
			prediction,
			risk_level
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`,
		request.MerchantID,
		request.Timestamp,
		request.VelocityRatio,
		request.AmountRatio,
		prediction.AnomalyScore,
		prediction.Threshold,
		prediction.Prediction,
		prediction.RiskLevel,
	)

	if err != nil {
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
			anomaly_score,
			threshold,
			prediction,
			risk_level,
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
			id            int64
			merchantID    sql.NullString
			timestamp     float64
			velocityRatio float64
			amountRatio   float64
			anomalyScore  float64
			threshold     float64
			prediction    int
			riskLevel     string
			createdAt     string
		)

		err := rows.Scan(
			&id,
			&merchantID,
			&timestamp,
			&velocityRatio,
			&amountRatio,
			&anomalyScore,
			&threshold,
			&prediction,
			&riskLevel,
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
			"id":             id,
			"merchant_id":    merchantID.String,
			"timestamp":      timestamp,
			"velocity_ratio": velocityRatio,
			"amount_ratio":   amountRatio,
			"anomaly_score":  anomalyScore,
			"threshold":      threshold,
			"prediction":     prediction,
			"risk_level":     riskLevel,
			"created_at":     createdAt,
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

func main() {
	var err error

	mlServiceURL = os.Getenv("ML_SERVICE_URL")

	if mlServiceURL == "" {
		log.Fatal("ML_SERVICE_URL is not set")
	}

	db, err = database.Connect()
	if err != nil {
		log.Fatal("PostgreSQL connection failed: ", err)
	}

	defer db.Close()

	log.Println("PostgreSQL connected")

	http.HandleFunc("/predict", predictHandler)
	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/risk-events", riskEventsHandler)
	http.Handle("/metrics", promhttp.Handler())

	log.Println("Go API running on :8080")

	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
