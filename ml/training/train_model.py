from __future__ import annotations

import pandas as pd

from ml.features.feature_contract import (
    FEATURE_COLUMNS,
    validate_feature_frame,
)
from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector


DATA_PATH = "data/raw/synthetic_transactions_v2.csv"
MODEL_PATH = "ml/models/isolation_forest.pkl"

TRAIN_END = 707.06

def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df = df.sort_values(
        ["timestamp", "merchant_id"]
    ).reset_index(drop=True)

    # Build temporal features over the complete timeline so
    # rolling and lagged features retain their historical state.
    features = create_time_features(df)

    features = features.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    train = features[
        features["timestamp"] < TRAIN_END
    ].copy()

    normal_train = train[
        train["fraud_spike"] == 0
    ].copy()

    if normal_train.empty:
        raise RuntimeError(
            "No normal training samples were found."
        )

    validate_feature_frame(normal_train)

    X_train = normal_train[
        FEATURE_COLUMNS
    ]

    detector = AnomalyDetector(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )

    detector.fit(
        X_train
    )

    detector.save(MODEL_PATH)

    print("\n================================")
    print("V2 MODEL TRAINING")
    print("================================")

    print(
        f"\nTraining window: "
        f"< {TRAIN_END:.2f}s"
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Normal training samples: "
        f"{len(X_train):,}"
    )

    print("\nFeatures:")

    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
