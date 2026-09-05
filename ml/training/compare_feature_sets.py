from __future__ import annotations

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


DATA_PATH = "data/raw/synthetic_transactions_v2.csv"

TRAIN_END = 707.06
VALIDATION_END = 1210.30

FN_COST = 20.0
FP_COST = 1.0

V1_FEATURES = [
    "velocity_ratio",
    "amount_ratio",
]

V2_FEATURES = [
    "velocity_ratio",
    "amount_ratio",
    "velocity_acceleration_1m",
    "amount_acceleration_1m",
]


def choose_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    candidates = np.quantile(
        scores,
        np.linspace(0.80, 0.99, 100),
    )

    best_threshold = None
    best_cost = float("inf")

    for threshold in candidates:
        predictions = (
            scores >= threshold
        ).astype(int)

        fp = int(
            ((predictions == 1) & (labels == 0)).sum()
        )

        fn = int(
            ((predictions == 0) & (labels == 1)).sum()
        )

        cost = FN_COST * fn + FP_COST * fp

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    if best_threshold is None:
        raise RuntimeError(
            "Unable to select threshold."
        )

    return best_threshold


def evaluate_feature_set(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> dict:

    train = data[
        data["timestamp"] < TRAIN_END
    ].copy()

    validation = data[
        (data["timestamp"] >= TRAIN_END)
        & (data["timestamp"] < VALIDATION_END)
    ].copy()

    test = data[
        data["timestamp"] >= VALIDATION_END
    ].copy()

    train = train.dropna(
        subset=feature_columns
    )

    validation = validation.dropna(
        subset=feature_columns
    )

    test = test.dropna(
        subset=feature_columns
    )

    train_normal = train[
        train["fraud_spike"] == 0
    ]

    detector = AnomalyDetector(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )

    detector.fit(
        train_normal[feature_columns].to_numpy()
    )

    validation_scores = detector.anomaly_score(
        validation[feature_columns].to_numpy()
    )

    test_scores = detector.anomaly_score(
        test[feature_columns].to_numpy()
    )

    validation_labels = (
        validation["fraud_spike"]
        .astype(int)
        .to_numpy()
    )

    test_labels = (
        test["fraud_spike"]
        .astype(int)
        .to_numpy()
    )

    threshold = choose_threshold(
        validation_scores,
        validation_labels,
    )

    validation_predictions = (
        validation_scores >= threshold
    ).astype(int)

    test_predictions = (
        test_scores >= threshold
    ).astype(int)

    precision = precision_score(
        test_labels,
        test_predictions,
        zero_division=0,
    )

    recall = recall_score(
        test_labels,
        test_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        test_labels,
        test_predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        test_labels,
        test_predictions,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    return {
        "features": feature_columns,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "validation_positive_predictions": int(
            validation_predictions.sum()
        ),
        "test_positive_predictions": int(
            test_predictions.sum()
        ),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df = df.sort_values(
        ["timestamp", "merchant_id"]
    ).reset_index(drop=True)

    # Feature engineering is intentionally performed
    # over the complete timeline to preserve streaming
    # history across evaluation boundaries.
    featured = create_time_features(df)

    featured = featured.dropna(
        subset=[
            "velocity_ratio",
            "amount_ratio",
        ]
    ).copy()

    print("\n================================")
    print("FEATURE-SET COMPARISON")
    print("================================")

    results = []

    for name, features in [
        ("V1", V1_FEATURES),
        ("V2", V2_FEATURES),
    ]:
        result = evaluate_feature_set(
            featured,
            features,
        )

        result["name"] = name
        results.append(result)

        print(f"\n{name}")
        print("-" * 30)

        print(
            "Features:   "
            + ", ".join(features)
        )

        print(
            f"Train rows: {result['train_rows']:,}"
        )

        print(
            f"Val rows:   {result['validation_rows']:,}"
        )

        print(
            f"Test rows:  {result['test_rows']:,}"
        )

        print(
            f"Threshold:  {result['threshold']:.6f}"
        )

        print(
            f"Precision:  {result['precision']:.4f}"
        )

        print(
            f"Recall:     {result['recall']:.4f}"
        )

        print(
            f"F1:         {result['f1']:.4f}"
        )

        print(
            f"FPR:        {result['fpr']:.4f}"
        )

        print(
            "Confusion:  "
            f"TN={result['tn']} "
            f"FP={result['fp']} "
            f"FN={result['fn']} "
            f"TP={result['tp']}"
        )

    print("\n================================")
    print("V2 DELTA VS V1")
    print("================================")

    v1 = results[0]
    v2 = results[1]

    print(
        f"Precision: "
        f"{v1['precision']:.4f} → "
        f"{v2['precision']:.4f} "
        f"({v2['precision'] - v1['precision']:+.4f})"
    )

    print(
        f"Recall:    "
        f"{v1['recall']:.4f} → "
        f"{v2['recall']:.4f} "
        f"({v2['recall'] - v1['recall']:+.4f})"
    )

    print(
        f"F1:        "
        f"{v1['f1']:.4f} → "
        f"{v2['f1']:.4f} "
        f"({v2['f1'] - v1['f1']:+.4f})"
    )

    print(
        f"FPR:       "
        f"{v1['fpr']:.4f} → "
        f"{v2['fpr']:.4f} "
        f"({v2['fpr'] - v1['fpr']:+.4f})"
    )


if __name__ == "__main__":
    main()
