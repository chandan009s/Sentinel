import pandas as pd

from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector

DATA_PATH = "data/raw/synthetic_transactions.csv"

FEATURE_COLUMNS = [
    "transaction_count_1m",
    "total_amount_1m",
    "average_amount_1m",
    "transaction_count_5m",
    "total_amount_5m",
    "average_amount_5m",
    "transaction_count_1h",
    "total_amount_1h",
    "average_amount_1h",
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

    train = features.iloc[:train_end]
    validation = features.iloc[
        train_end:validation_end
    ]
    test = features.iloc[validation_end:].copy()

    detector = AnomalyDetector()
    detector.fit(train[FEATURE_COLUMNS])

    test["anomaly_score"] = detector.anomaly_score(
        test[FEATURE_COLUMNS]
    )

    m004_spike = test[
        (test["merchant_id"] == "M004")
        & (test["fraud_spike"] == 1)
    ]

    print("\nM004 TEST SPIKE:")
    print(
        m004_spike[
            [
                "timestamp",
                "amount",
                "transaction_count_1m",
                "transaction_count_5m",
                "transaction_count_1h",
                "velocity_ratio",
                "amount_ratio",
                "anomaly_score",
                "fraud_spike",
            ]
        ].to_string(index=False)
    )

    m004_normal = test[
        (test["merchant_id"] == "M004")
        & (test["fraud_spike"] == 0)
    ]

    print("\nM004 TEST NORMAL:")
    print(
        m004_normal[
            [
                "timestamp",
                "amount",
                "transaction_count_1m",
                "transaction_count_5m",
                "transaction_count_1h",
                "velocity_ratio",
                "amount_ratio",
                "anomaly_score",
                "fraud_spike",
            ]
        ].head(20).to_string(index=False)
    )

    m003_spike = validation[
        (validation["merchant_id"] == "M003")
        & (validation["fraud_spike"] == 1)
    ].copy()

    m003_spike["anomaly_score"] = detector.anomaly_score(
        m003_spike[FEATURE_COLUMNS]
    )

    print("\nM003 VALIDATION SPIKE:")
    print(
        m003_spike[
            [
                "timestamp",
                "amount",
                "transaction_count_1m",
                "transaction_count_5m",
                "transaction_count_1h",
                "velocity_ratio",
                "amount_ratio",
                "anomaly_score",
                "fraud_spike",
            ]
        ].head(20).to_string(index=False)
    )

if __name__ == "__main__":
    main()