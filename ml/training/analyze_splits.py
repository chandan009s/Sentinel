import json

import numpy as np
import pandas as pd

from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector


DATA_PATH = "data/raw/synthetic_transactions.csv"
THRESHOLD_PATH = "models/threshold.json"

FEATURE_COLUMNS = [
    "velocity_ratio",
    "amount_ratio",
]


def summarize_split(name, df, scores, threshold):
    normal = df["fraud_spike"] == 0
    spike = df["fraud_spike"] == 1

    print(f"\n===== {name.upper()} =====")
    print(f"Rows: {len(df)}")
    print(f"Normal: {normal.sum()}")
    print(f"Spikes: {spike.sum()}")

    print(
        f"Time range: "
        f"{df['timestamp'].min():.3f} -> "
        f"{df['timestamp'].max():.3f}"
    )

    for label, mask in [
        ("NORMAL", normal),
        ("SPIKE", spike),
    ]:
        if mask.sum() == 0:
            continue

        split_scores = scores[mask.to_numpy()]
        velocity = df.loc[mask, "velocity_ratio"]
        amount = df.loc[mask, "amount_ratio"]

        print(f"\n{label}")
        print(
            f"  velocity median: {velocity.median():.4f}"
        )
        print(
            f"  velocity max:    {velocity.max():.4f}"
        )
        print(
            f"  amount median:   {amount.median():.4f}"
        )
        print(
            f"  amount max:      {amount.max():.4f}"
        )
        print(
            f"  score median:    {np.median(split_scores):.6f}"
        )
        print(
            f"  score max:       {np.max(split_scores):.6f}"
        )
        print(
            f"  above threshold: "
            f"{int((split_scores >= threshold).sum())}"
        )


def main():
    df = pd.read_csv(DATA_PATH)

    with open(THRESHOLD_PATH) as f:
        threshold_config = json.load(f)

    threshold = float(threshold_config["threshold"])
    percentile = float(threshold_config["percentile"])

    features = create_time_features(df)

    features = features.dropna(
        subset=[
            "transaction_count_baseline",
            "amount_baseline",
            "velocity_ratio",
            "amount_ratio",
        ]
    ).copy()

    features = (
        features
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    n = len(features)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = features.iloc[:train_end].copy()
    validation = features.iloc[
        train_end:validation_end
    ].copy()
    test = features.iloc[validation_end:].copy()

    normal_train = train[
        train["fraud_spike"] == 0
    ].copy()

    detector = AnomalyDetector()
    detector.fit(normal_train[FEATURE_COLUMNS])

    print("\n================================")
    print("TEMPORAL SPLIT ANALYSIS")
    print("================================")

    print(f"\nLocked percentile: {percentile:.1f}")
    print(f"Locked threshold:  {threshold:.6f}")

    train_scores = detector.anomaly_score(
        train[FEATURE_COLUMNS]
    )

    validation_scores = detector.anomaly_score(
        validation[FEATURE_COLUMNS]
    )

    test_scores = detector.anomaly_score(
        test[FEATURE_COLUMNS]
    )

    summarize_split(
        "TRAIN",
        train,
        train_scores,
        threshold,
    )

    summarize_split(
        "VALIDATION",
        validation,
        validation_scores,
        threshold,
    )

    summarize_split(
        "TEST",
        test,
        test_scores,
        threshold,
    )


if __name__ == "__main__":
    main()
