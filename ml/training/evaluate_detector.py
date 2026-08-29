import pandas as pd

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

    detector = AnomalyDetector()
    detector.fit(X_train)

    validation["anomaly_score"] = (
        detector.anomaly_score(X_validation)
    )

    print("\nValidation score statistics:")
    print(
        validation.groupby("fraud_spike")[
            "anomaly_score"
        ].describe()
    )

    print("\nHighest validation anomaly scores:")

    print(
        validation[
            [
                "merchant_id",
                "timestamp",
                "fraud_spike",
                "velocity_ratio",
                "amount_ratio",
                "anomaly_score",
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