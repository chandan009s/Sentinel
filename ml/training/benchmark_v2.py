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
from ml.features.generate_transactions_v2 import generate_transactions
from ml.models.isolation_forest import AnomalyDetector


FEATURE_COLUMNS = [
    "velocity_ratio",
    "amount_ratio",
]

SEEDS = [1, 42, 100, 999, 2026]

FN_COST = 20
FP_COST = 1


def prepare_features(seed: int) -> pd.DataFrame:
    df = generate_transactions(
        duration_seconds=1800,
        seed=seed,
    )

    features = create_time_features(df)

    features = features.dropna(
        subset=[
            "transaction_count_baseline",
            "amount_baseline",
            "velocity_ratio",
            "amount_ratio",
        ]
    ).copy()

    return (
        features
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def evaluate_seed(seed: int) -> dict:
    features = prepare_features(seed)

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

    detector.fit(
        normal_train[FEATURE_COLUMNS]
    )

    train_scores = detector.anomaly_score(
        normal_train[FEATURE_COLUMNS]
    )

    validation_scores = detector.anomaly_score(
        validation[FEATURE_COLUMNS]
    )

    test_scores = detector.anomaly_score(
        test[FEATURE_COLUMNS]
    )

    best_cost = float("inf")
    best_percentile = None
    best_threshold = None

    for percentile in range(80, 100):
        threshold = np.percentile(
            train_scores,
            percentile,
        )

        predictions = (
            validation_scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            validation["fraud_spike"],
            predictions,
        ).ravel()

        cost = (
            fn * FN_COST
            + fp * FP_COST
        )

        if cost < best_cost:
            best_cost = cost
            best_percentile = percentile
            best_threshold = threshold

    predictions = (
        test_scores >= best_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        test["fraud_spike"],
        predictions,
    ).ravel()

    precision = precision_score(
        test["fraud_spike"],
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        test["fraud_spike"],
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        test["fraud_spike"],
        predictions,
        zero_division=0,
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "seed": seed,
        "rows": len(features),
        "train_fraud_rate": train["fraud_spike"].mean(),
        "validation_fraud_rate": validation["fraud_spike"].mean(),
        "test_fraud_rate": test["fraud_spike"].mean(),
        "percentile": best_percentile,
        "threshold": best_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_positives": fp,
        "false_negatives": fn,
        "validation_cost": best_cost,
    }


def main() -> None:
    print("\n================================")
    print("V2 MULTI-SEED BENCHMARK")
    print("================================")

    results = []

    for seed in SEEDS:
        print(f"\nRunning seed {seed}...")

        result = evaluate_seed(seed)

        results.append(result)

        print(
            f"  threshold={result['threshold']:.6f} "
            f"percentile={result['percentile']:.0f}"
        )
        print(
            f"  precision={result['precision']:.4f} "
            f"recall={result['recall']:.4f} "
            f"f1={result['f1']:.4f}"
        )
        print(
            f"  FPR={result['false_positive_rate']:.4f} "
            f"FP={result['false_positives']} "
            f"FN={result['false_negatives']}"
        )

    results_df = pd.DataFrame(results)

    print("\n================================")
    print("PER-SEED RESULTS")
    print("================================")

    print(
        results_df[
            [
                "seed",
                "train_fraud_rate",
                "validation_fraud_rate",
                "test_fraud_rate",
                "percentile",
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(
            index=False,
            formatters={
                "train_fraud_rate": "{:.4f}".format,
                "validation_fraud_rate": "{:.4f}".format,
                "test_fraud_rate": "{:.4f}".format,
                "threshold": "{:.6f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "false_positive_rate": "{:.4f}".format,
            },
        )
    )

    print("\n================================")
    print("MULTI-SEED SUMMARY")
    print("================================")

    for metric in [
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_positives",
        "false_negatives",
    ]:
        print(
            f"{metric:22s} "
            f"mean={results_df[metric].mean():.4f} "
            f"std={results_df[metric].std(ddof=0):.4f}"
        )


if __name__ == "__main__":
    main()
