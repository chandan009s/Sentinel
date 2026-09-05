import pandas as pd
import json

from ml.features.feature_contract import (
    FEATURE_COLUMNS,
    validate_feature_frame,
)
from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector

DATA_PATH = "data/raw/synthetic_transactions_v2.csv"
THRESHOLD_PATH = "models/threshold.json"
MODEL_PATH = "ml/models/isolation_forest.pkl"

def main():
    with open(THRESHOLD_PATH) as f:
        threshold_config = json.load(f)

    anomaly_threshold = threshold_config["threshold"]

    df = pd.read_csv(DATA_PATH)

    features = create_time_features(df)

    features = features.dropna(
        subset=[
            "transaction_count_baseline",
            "amount_baseline",
            "velocity_ratio",
            "amount_ratio",
            "velocity_acceleration_1m",
            "amount_acceleration_1m",
        ]
    ).copy()

    features = features.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    validate_feature_frame(features)

    detector = AnomalyDetector()
    detector.load(MODEL_PATH)

    scores = detector.anomaly_score(
        features[FEATURE_COLUMNS]
    )

    features["anomaly_score"] = scores

    features["prediction"] = (
        features["anomaly_score"] >= anomaly_threshold
    ).astype(int)

    print("\n================================")
    print("ANOMALY DETECTION")
    print("================================")

    print(
        f"\nThreshold: {anomaly_threshold:.6f}"
    )

    print("\nPredictions:")
    print(
        features["prediction"].value_counts()
    )

    print("\nHighest anomaly scores:")

    print(
        features[
            [
                "merchant_id",
                "timestamp",
                "velocity_ratio",
                "amount_ratio",
                "anomaly_score",
                "prediction",
            ]
        ]
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .head(20)
    )

if __name__ == "__main__":
    main()