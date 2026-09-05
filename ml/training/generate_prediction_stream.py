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
    "data/reports/v2_prediction_stream.csv"
)

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

    features = create_time_features(df)

    features = features.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

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
        threshold = float(
            json.load(f)["threshold"]
        )

    detector = AnomalyDetector()
    detector.load(MODEL_PATH)

    scores = detector.anomaly_score(
        features[FEATURE_COLUMNS]
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    output = features[
        [
            "timestamp",
            "merchant_id",
            "scenario",
            "fraud_spike",
            "velocity_ratio",
            "amount_ratio",
            "velocity_acceleration_1m",
            "amount_acceleration_1m",
        ]
    ].copy()

    output["anomaly_score"] = scores
    output["threshold"] = threshold
    output["prediction"] = predictions

    output = output.sort_values(
        ["timestamp", "merchant_id"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n================================")
    print("FULL V2 PREDICTION STREAM")
    print("================================")

    print(
        f"Rows:          {len(output):,}"
    )

    print(
        f"Start:         "
        f"{output['timestamp'].min():.3f}s"
    )

    print(
        f"End:           "
        f"{output['timestamp'].max():.3f}s"
    )

    print(
        f"Threshold:     "
        f"{threshold:.6f}"
    )

    print(
        f"Raw alerts:    "
        f"{int(output['prediction'].sum()):,}"
    )

    print(
        f"True alerts:   "
        f"{int(((output['prediction'] == 1) & (output['fraud_spike'] == 1)).sum()):,}"
    )

    print(
        f"False alerts:  "
        f"{int(((output['prediction'] == 1) & (output['fraud_spike'] == 0)).sum()):,}"
    )

    print("\nColumns:")
    for column in output.columns:
        print(f"  - {column}")

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
