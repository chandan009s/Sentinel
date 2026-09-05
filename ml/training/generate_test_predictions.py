from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector


DATA_PATH = Path(
    "data/raw/synthetic_transactions_v2.csv"
)

MODEL_PATH = Path(
    "ml/models/isolation_forest.pkl"
)

THRESHOLD_PATH = Path(
    "models/threshold.json"
)

OUTPUT_PATH = Path(
    "data/reports/v2_test_predictions.csv"
)

TRAIN_END = 707.06
VALIDATION_END = 1210.30

FEATURE_COLUMNS = [
    "velocity_ratio",
    "amount_ratio",
    "velocity_acceleration_1m",
    "amount_acceleration_1m",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df = df.sort_values(
        ["timestamp", "merchant_id"]
    ).reset_index(drop=True)

    featured = create_time_features(df)

    featured = featured.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    train = featured[
        featured["timestamp"] < TRAIN_END
    ].copy()

    test = featured[
        featured["timestamp"] >= VALIDATION_END
    ].copy()

    train_normal = train[
        train["fraud_spike"] == 0
    ].copy()

    if train_normal.empty:
        raise RuntimeError(
            "No normal training samples found."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold not found: {THRESHOLD_PATH}"
        )

    with THRESHOLD_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        threshold_config = json.load(f)

    threshold = float(
        threshold_config["threshold"]
    )

    detector = AnomalyDetector(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )

    # Load exactly the production model artifact.
    detector.load(MODEL_PATH)

    scores = detector.anomaly_score(
        test[FEATURE_COLUMNS].to_numpy()
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    output = test[
        [
            "timestamp",
            "merchant_id",
            "scenario",
            "fraud_spike",
        ]
    ].copy()

    output["anomaly_score"] = scores
    output["threshold"] = threshold
    output["prediction"] = predictions

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n================================")
    print("CANONICAL V2 TEST PREDICTIONS")
    print("================================")

    print(
        f"Test rows:       {len(output):,}"
    )

    print(
        f"Threshold:       {threshold:.6f}"
    )

    print(
        f"Raw alerts:      "
        f"{int(predictions.sum()):,}"
    )

    print(
        f"True alerts:     "
        f"{int(((predictions == 1) & (output['fraud_spike'] == 1)).sum()):,}"
    )

    print(
        f"False alerts:    "
        f"{int(((predictions == 1) & (output['fraud_spike'] == 0)).sum()):,}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
