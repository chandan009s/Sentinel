import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector


DATA_PATH = "data/raw/synthetic_transactions.csv"
THRESHOLD_PATH = "models/threshold.json"

FEATURE_COLUMNS = [
    "velocity_ratio",
    "amount_ratio",
]

PERCENTILES = [
    85,
    87,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
]

FN_COST = 20
FP_COST = 1


def main():
    df = pd.read_csv(DATA_PATH)

    with open(THRESHOLD_PATH) as f:
        threshold_config = json.load(f)

    locked_percentile = threshold_config["percentile"]

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
    test = features.iloc[validation_end:].copy()

    normal_train = train[
        train["fraud_spike"] == 0
    ].copy()

    X_train = normal_train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["fraud_spike"]

    detector = AnomalyDetector()
    detector.fit(X_train)

    train_scores = detector.anomaly_score(X_train)
    test_scores = detector.anomaly_score(X_test)

    print("\n================================")
    print("THRESHOLD OPERATING-POINT ANALYSIS")
    print("================================")

    print(f"\nLocked percentile: {locked_percentile:.1f}")
    print(f"FN cost: {FN_COST}")
    print(f"FP cost: {FP_COST}")

    rows = []

    for percentile in PERCENTILES:
        threshold = np.percentile(
            train_scores,
            percentile,
        )

        predictions = (
            test_scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
        ).ravel()

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        total_cost = (
            fn * FN_COST
            + fp * FP_COST
        )

        rows.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": fp,
                "false_negatives": fn,
                "false_positive_rate": false_positive_rate,
                "cost": total_cost,
            }
        )

    results = pd.DataFrame(rows)

    print(
        results.to_string(
            index=False,
            formatters={
                "threshold": "{:.6f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "false_positive_rate": "{:.4f}".format,
            },
        )
    )

    best = results.loc[
        results["cost"].idxmin()
    ]

    print("\n================================")
    print("LOWEST-COST OPERATING POINT")
    print("================================")

    print(
        f"Percentile: {best['percentile']:.1f}"
    )

    print(
        f"Threshold:  {best['threshold']:.6f}"
    )

    print(
        f"Precision:  {best['precision']:.4f}"
    )

    print(
        f"Recall:     {best['recall']:.4f}"
    )

    print(
        f"F1:         {best['f1']:.4f}"
    )

    print(
        f"FP:         {int(best['false_positives'])}"
    )

    print(
        f"FN:         {int(best['false_negatives'])}"
    )

    print(
        f"Total cost: {int(best['cost'])}"
    )


if __name__ == "__main__":
    main()
