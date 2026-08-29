import pandas as pd

from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector


DATA_PATH = "data/raw/synthetic_transactions.csv"
MODEL_PATH = "ml/models/isolation_forest.pkl"

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

    train_end = int(len(features) * 0.70)

    train = features.iloc[:train_end].copy()

    normal_train = train[
        train["fraud_spike"] == 0
    ].copy()

    X_train = normal_train[FEATURE_COLUMNS]

    detector = AnomalyDetector()

    detector.fit(X_train)

    detector.save(MODEL_PATH)

    print("\n================================")
    print("MODEL TRAINING")
    print("================================")

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )

if __name__ == "__main__":
    main()