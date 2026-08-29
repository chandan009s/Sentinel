import pandas as pd
import json

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector

DATA_PATH = "data/raw/synthetic_transactions.csv"

MODEL_META_PATH = "ml/models/isolation_forest_v1_meta.json"

FEATURE_COLUMNS = [
    "velocity_ratio",
    "amount_ratio",
]

def main():

    df = pd.read_csv(DATA_PATH)

    with open(MODEL_META_PATH) as f:
        metadata = json.load(f)

    anomaly_threshold = metadata["threshold"]

    features = create_time_features(df)

    features = features.dropna(
        subset=[
            "transaction_count_baseline",
            "amount_baseline",
            "velocity_ratio",
            "amount_ratio",
        ]
    ).copy()

    # Preserve chronological order
    features = features.sort_values(
        "timestamp"
    ).reset_index(drop=True)
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

    print("\n================================")
    print("FINAL TEST EVALUATION")
    print("================================")

    print(
        f"\nLocked threshold: "
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

    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

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
    )

if __name__ == "__main__":
    main()