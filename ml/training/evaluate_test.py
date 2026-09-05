import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
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


def main():
    df = pd.read_csv(DATA_PATH)

    with open(THRESHOLD_PATH) as f:
        threshold_config = json.load(f)

    anomaly_percentile = threshold_config["percentile"]

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

    anomaly_threshold = np.percentile(
        train_scores,
        anomaly_percentile,
    )

    test["anomaly_score"] = detector.anomaly_score(
        X_test
    )

    test["prediction"] = (
        test["anomaly_score"] >= anomaly_threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        test["prediction"],
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test["prediction"],
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test["prediction"],
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        test["prediction"],
    )

    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    print("\n================================")
    print("FINAL TEST EVALUATION")
    print("================================")

    print(
        f"\nLocked percentile: "
        f"{anomaly_percentile:.1f}"
    )

    print(
        f"Calculated threshold: "
        f"{anomaly_threshold:.6f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test["prediction"],
            zero_division=0,
        )
    )

    print("\nPredictions:")
    print(
        test["prediction"].value_counts()
    )

    print("\nHighest anomaly scores:")

    print(
        test[
            [
                "merchant_id",
                "timestamp",
                "fraud_spike",
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
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # False-negative analysis
    # ---------------------------------------------------------

    false_negatives = test[
        (test["fraud_spike"] == 1)
        & (test["prediction"] == 0)
    ].copy()

    false_negatives["distance_below_threshold"] = (
        anomaly_threshold
        - false_negatives["anomaly_score"]
    )

    print("\n================================")
    print("FALSE NEGATIVE ANALYSIS")
    print("================================")

    print(
        f"Missed spikes: {len(false_negatives)}"
    )

    print(
        f"Miss rate: "
        f"{len(false_negatives) / max(tp + fn, 1):.4f}"
    )

    if false_negatives.empty:
        print("\nNo false negatives found.")
        return

    print("\nMissed spikes ordered by closest to threshold:")

    print(
        false_negatives[
            [
                "merchant_id",
                "timestamp",
                "velocity_ratio",
                "amount_ratio",
                "anomaly_score",
                "distance_below_threshold",
            ]
        ]
        .sort_values(
            "distance_below_threshold",
            ascending=True,
        )
        .to_string(index=False)
    )

    print("\nMissed spikes ordered by lowest anomaly score:")

    print(
        false_negatives[
            [
                "merchant_id",
                "timestamp",
                "velocity_ratio",
                "amount_ratio",
                "anomaly_score",
                "distance_below_threshold",
            ]
        ]
        .sort_values(
            "anomaly_score",
            ascending=True,
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
