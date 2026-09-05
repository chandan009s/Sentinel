import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from ml.models.isolation_forest import AnomalyDetector


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "isolation_forest.pkl"
)

MODEL_METADATA_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "isolation_forest_v1_meta.json"
)


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

with MODEL_METADATA_PATH.open() as f:
    MODEL_METADATA = json.load(f)

ANOMALY_THRESHOLD = float(
    MODEL_METADATA["threshold"]
)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Risk Manager - ML Service",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

detector = AnomalyDetector()
detector.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):

    velocity_ratio: float
    amount_ratio: float
    velocity_acceleration_1m: float
    amount_acceleration_1m: float

    merchant_id: str | None = None
    timestamp: float | None = None

    @field_validator(
        "velocity_ratio",
        "amount_ratio",
    )
    @classmethod
    def ratios_must_be_non_negative(cls, v, info):

        if v < 0 or not float("-inf") < v < float("inf"):
            raise ValueError(
                f"{info.field_name} must be a "
                f"non-negative, finite number"
            )

        return v

    @field_validator(
        "velocity_acceleration_1m",
        "amount_acceleration_1m",
    )
    @classmethod
    def accelerations_must_be_finite(cls, v, info):

        if not float("-inf") < v < float("inf"):
            raise ValueError(
                f"{info.field_name} must be a "
                f"finite number"
            )

        return v


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": MODEL_METADATA["model_name"],
        "model_version": MODEL_METADATA["model_version"],
        "feature_version": MODEL_METADATA["feature_version"],
        "feature_contract_version": MODEL_METADATA["feature_contract_version"],
        "threshold_version": MODEL_METADATA["threshold_version"],
        "threshold": ANOMALY_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

@app.get("/metadata")
def metadata():

    return {
        "model_name": MODEL_METADATA["model_name"],
        "model_version": MODEL_METADATA["model_version"],
        "feature_version": MODEL_METADATA["feature_version"],
        "feature_contract_version": MODEL_METADATA["feature_contract_version"],
        "threshold_version": MODEL_METADATA["threshold_version"],
        "threshold": ANOMALY_THRESHOLD,
        "features": MODEL_METADATA["features"],
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    features = pd.DataFrame(
        [
            {
                "velocity_ratio": request.velocity_ratio,
                "amount_ratio": request.amount_ratio,
                "velocity_acceleration_1m": (
                    request.velocity_acceleration_1m
                ),
                "amount_acceleration_1m": (
                    request.amount_acceleration_1m
                ),
            }
        ]
    )

    score = float(
        detector.anomaly_score(features)[0]
    )

    prediction = int(
        score >= ANOMALY_THRESHOLD
    )

    return {
        "merchant_id": request.merchant_id,
        "timestamp": request.timestamp,
        "anomaly_score": score,
        "threshold": ANOMALY_THRESHOLD,
        "prediction": prediction,
        "model_name": MODEL_METADATA["model_name"],
        "model_version": MODEL_METADATA["model_version"],
        "feature_version": MODEL_METADATA["feature_version"],
        "feature_contract_version": MODEL_METADATA["feature_contract_version"],
        "threshold_version": MODEL_METADATA["threshold_version"],
    }