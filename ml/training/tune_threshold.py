import json
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector

DATA_PATH = "data/raw/synthetic_transactions.csv"

FEATURE_COLUMNS = [
    "velocity_ratio",
    "amount_ratio",
]

def main():

    df = pd.read_csv(DATA_PATH)

    features = create_time_features(df)
    features = features.dropna(
        subset=[
            "transaction_count_baseline",
            "amount_baseline",
            "velocity_ratio",
            "amount_ratio",
        ]
    ).copy()

    features = features.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    n = len(features)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = features.iloc[:train_end].copy()
    validation = features.iloc[
        train_end:validation_end
    ].copy()

    normal_train = train[
        train["fraud_spike"] == 0
    ].copy()

    X_train = normal_train[FEATURE_COLUMNS]
    X_validation = validation[FEATURE_COLUMNS]

    y_validation = validation["fraud_spike"]

    detector = AnomalyDetector()
    detector.fit(X_train)

    train_scores = detector.anomaly_score(
        X_train
    )

    scores = detector.anomaly_score(
        X_validation
    )

    validation["anomaly_score"] = scores

    percentiles = range(80, 100)

    results = []

    for percentile in percentiles:

        threshold = np.percentile(
            train_scores,
            percentile
        )

        predictions = (
            scores >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        false_negatives = (
            (y_validation == 1) & (predictions == 0)
        ).sum()

        false_positives = (
            (y_validation == 0) & (predictions == 1)
        ).sum()

        total_cost = (
            false_negatives * 20
            + false_positives * 1
        )

        results.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "total_cost": total_cost,
            }
        )

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df["total_cost"].idxmin()
    ]

    Path("models").mkdir(exist_ok=True)

    with open("models/threshold.json", "w") as f:
        json.dump(
            {
                "percentile": float(best["percentile"]),
                "threshold": float(best["threshold"]),
            },
            f,
            indent=4,
        )

    print(
        f"\nSaved threshold: "
        f"{best['threshold']:.6f}"
        f" -> models/threshold.json"
    )

    print("\nThreshold evaluation:")
    print(
        results_df.sort_values(
            "total_cost",
            ascending=True,
        ).head(10)
    )

    print("\nBest validation threshold:")

    print(
        f"Percentile: {best['percentile']:.1f}"
    )

    print(
        f"Threshold: {best['threshold']:.6f}"
    )

    print(
        f"False positives:  {int(best['false_positives'])}"
    )

    print(
        f"False negatives:  {int(best['false_negatives'])}"
    )

    print(
        f"Total cost:       {int(best['total_cost'])}"
    )

    print(
        f"Precision: {best['precision']:.4f}"
    )

    print(
        f"Recall:    {best['recall']:.4f}"
    )

    print(
        f"F1:        {best['f1']:.4f}"
    )

if __name__ == "__main__":
    main()