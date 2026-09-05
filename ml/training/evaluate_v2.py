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


DATA_PATH = "data/raw/synthetic_transactions_v2.csv"

FEATURE_COLUMNS = [
    "velocity_ratio",
    "amount_ratio",
]

FN_COST = 20
FP_COST = 1


def prepare_features():
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

    return (
        features
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def main():
    features = prepare_features()

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

    # ---------------------------------------------------------
    # Threshold selection on validation only
    # ---------------------------------------------------------

    validation_results = []

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

        total_cost = (
            fn * FN_COST
            + fp * FP_COST
        )

        validation_results.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "false_positives": fp,
                "false_negatives": fn,
                "cost": total_cost,
                "precision": precision_score(
                    validation["fraud_spike"],
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    validation["fraud_spike"],
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    validation["fraud_spike"],
                    predictions,
                    zero_division=0,
                ),
            }
        )

    validation_results = pd.DataFrame(
        validation_results
    )

    best = validation_results.loc[
        validation_results["cost"].idxmin()
    ]

    threshold = float(best["threshold"])

    # ---------------------------------------------------------
    # Final evaluation on untouched test
    # ---------------------------------------------------------

    predictions = (
        test_scores >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        test["fraud_spike"],
        predictions,
    )

    tn, fp, fn, tp = matrix.ravel()

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

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    print("\n================================")
    print("V2 FINAL TEST EVALUATION")
    print("================================")

    print("\nDataset:")
    print(f"Rows: {len(features)}")

    print("\nSplit sizes:")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")

    print("\nClass distribution:")
    print(
        f"Train fraud rate: "
        f"{train['fraud_spike'].mean():.4f}"
    )
    print(
        f"Validation fraud rate: "
        f"{validation['fraud_spike'].mean():.4f}"
    )
    print(
        f"Test fraud rate: "
        f"{test['fraud_spike'].mean():.4f}"
    )

    print("\nLocked operating point:")
    print(
        f"Percentile: "
        f"{best['percentile']:.1f}"
    )
    print(
        f"Threshold:  "
        f"{threshold:.6f}"
    )

    print("\nValidation selection:")
    print(
        f"FP: {int(best['false_positives'])}"
    )
    print(
        f"FN: {int(best['false_negatives'])}"
    )
    print(
        f"Cost: {int(best['cost'])}"
    )

    print("\nValidation operating points:")
    print(
        validation_results
        .sort_values("cost")
        .head(10)
        .to_string(index=False)
    )

    print("\nTest metrics:")
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall:    {recall:.4f}"
    )
    print(
        f"F1 Score:  {f1:.4f}"
    )
    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.4f}"
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            test["fraud_spike"],
            predictions,
            zero_division=0,
        )
    )

    test["anomaly_score"] = test_scores
    test["prediction"] = predictions

    print("\nTest results by scenario:")

    scenario_rows = []

    for scenario, group in test.groupby(
        "scenario"
    ):
        scenario_predictions = group["prediction"]
        scenario_actual = group["fraud_spike"]

        scenario_rows.append(
            {
                "scenario": scenario,
                "rows": len(group),
                "actual_spikes": int(
                    scenario_actual.sum()
                ),
                "predicted_spikes": int(
                    scenario_predictions.sum()
                ),
                "mean_score": group[
                    "anomaly_score"
                ].mean(),
                "median_score": group[
                    "anomaly_score"
                ].median(),
            }
        )

    print(
        pd.DataFrame(scenario_rows)
        .sort_values(
            "mean_score",
            ascending=False,
        )
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # False negatives
    # ---------------------------------------------------------

    false_negatives = test[
        (test["fraud_spike"] == 1)
        & (test["prediction"] == 0)
    ].copy()

    print("\nFalse negatives:")
    print(
        f"Count: {len(false_negatives)}"
    )

    if not false_negatives.empty:
        false_negatives[
            "distance_below_threshold"
        ] = (
            threshold
            - false_negatives[
                "anomaly_score"
            ]
        )

        print(
            false_negatives[
                [
                    "merchant_id",
                    "scenario",
                    "timestamp",
                    "velocity_ratio",
                    "amount_ratio",
                    "anomaly_score",
                    "distance_below_threshold",
                ]
            ]
            .sort_values(
                "distance_below_threshold"
            )
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
