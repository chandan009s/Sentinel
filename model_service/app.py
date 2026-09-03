import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from ml.models.isolation_forest import AnomalyDetector

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR/"ml"/"models"/"isolation_forest.pkl"
)

THRESHOLD_PATH = (
    BASE_DIR/"models"/"threshold.json"
)


app = FastAPI(
    title="AI Risk Manager - ML Service",
    version="1.0.0",
)

detector = AnomalyDetector()
detector.load(MODEL_PATH)

with open(THRESHOLD_PATH) as f:
    threshold_config = json.load(f)

ANOMALY_THRESHOLD = threshold_config["threshold"]

class PredictionRequest(BaseModel):

    velocity_ratio: float
    amount_ratio: float

    merchant_id: str | None = None
    timestamp: float | None = None

    @field_validator(
        "velocity_ratio",
        "amount_ratio",
    )
    @classmethod
    def must_be_non_negative(cls, v, info):

        if v < 0 or not float("-inf") < v < float("inf"):
            raise ValueError(
                f"{info.field_name} must be a "
                f"non-negative, finite number"
            )

        return v

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": "isolation_forest",
        "threshold": ANOMALY_THRESHOLD,
    }

@app.post("/predict")
def predict(request: PredictionRequest):

    features = pd.DataFrame(
        [
            {
                "velocity_ratio": request.velocity_ratio,
                "amount_ratio": request.amount_ratio,
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
    }
