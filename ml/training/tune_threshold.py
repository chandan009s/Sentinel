from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector


DATA_PATH = "data/raw/synthetic_transactions_v2.csv"
MODEL_PATH = "ml/models/isolation_forest.pkl"
THRESHOLD_PATH = "models/threshold.json"

TRAIN_END = 707.06
VALIDATION_END = 1210.30

FEATURE_COLUMNS = [
    "velocity_ratio",
    "amount_ratio",
    "velocity_acceleration_1m",
    "amount_acceleration_1m",
]

FN_COST = 20
FP_COST = 1


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df = df.sort_values(
        ["timestamp", "merchant_id"]
    ).reset_index(drop=True)

    # Compute temporal features over the complete timeline
    # so validation retains the historical context required
    # by rolling and lagged features.
    features = create_time_features(df)

    features = features.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    train = features[
        features["timestamp"] < TRAIN_END
    ].copy()

    validation = features[
        (features["timestamp"] >= TRAIN_END)
        & (features["timestamp"] < VALIDATION_END)
    ].copy()

    if train.empty:
        raise RuntimeError(
            "Training set is empty."
        )

    if validation.empty:
        raise RuntimeError(
            "Validation set is empty."
        )

    normal_train = train[
        train["fraud_spike"] == 0
    ].copy()

    if normal_train.empty:
        raise RuntimeError(
            "No normal training samples available."
        )

    X_train = normal_train[
        FEATURE_COLUMNS
    ]

    X_validation = validation[
        FEATURE_COLUMNS
    ]

    y_validation = (
        validation["fraud_spike"]
        .astype(int)
        .to_numpy()
    )

    detector = AnomalyDetector(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )

    detector.fit(
        X_train.to_numpy()
    )

    train_scores = detector.anomaly_score(
        X_train.to_numpy()
    )

    validation_scores = detector.anomaly_score(
        X_validation.to_numpy()
    )

    results = []

    for percentile in range(80, 100):
        threshold = float(
            np.percentile(
                train_scores,
                percentile,
            )
        )

        predictions = (
            validation_scores >= threshold
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

        false_negatives = int(
            (
                (y_validation == 1)
                & (predictions == 0)
            ).sum()
        )

        false_positives = int(
            (
                (y_validation == 0)
                & (predictions == 1)
            ).sum()
        )

        total_cost = (
            FN_COST * false_negatives
            + FP_COST * false_positives
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

    Path("models").mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_payload = {
        "model": "isolation_forest",
        "feature_set": "v2",
        "features": FEATURE_COLUMNS,
        "train_end": TRAIN_END,
        "validation_end": VALIDATION_END,
        "percentile": float(
            best["percentile"]
        ),
        "threshold": float(
            best["threshold"]
        ),
        "false_negative_cost": FN_COST,
        "false_positive_cost": FP_COST,
    }

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            threshold_payload,
            f,
            indent=4,
        )

    print("\n================================")
    print("V2 THRESHOLD TUNING")
    print("================================")

    print(
        f"\nTraining window: "
        f"< {TRAIN_END:.2f}s"
    )

    print(
        f"Validation window: "
        f"{TRAIN_END:.2f}s–{VALIDATION_END:.2f}s"
    )

    print(
        f"\nTraining rows: "
        f"{len(train):,}"
    )

    print(
        f"Normal training samples: "
        f"{len(normal_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print("\nFeatures:")

    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")

    print("\nTop threshold candidates:")

    print(
        results_df.sort_values(
            [
                "total_cost",
                "false_negatives",
                "false_positives",
            ],
            ascending=True,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nBest validation threshold")

    print(
        f"Percentile:       "
        f"{best['percentile']:.0f}"
    )

    print(
        f"Threshold:        "
        f"{best['threshold']:.6f}"
    )

    print(
        f"False positives:  "
        f"{int(best['false_positives'])}"
    )

    print(
        f"False negatives:  "
        f"{int(best['false_negatives'])}"
    )

    print(
        f"Total cost:       "
        f"{int(best['total_cost'])}"
    )

    print(
        f"Precision:        "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall:           "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1:               "
        f"{best['f1']:.4f}"
    )

    print(
        f"\nSaved threshold: "
        f"{best['threshold']:.6f}"
    )

    print(
        f"Saved to: {THRESHOLD_PATH}"
    )


if __name__ == "__main__":
    main()
