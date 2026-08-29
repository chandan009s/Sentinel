import pandas as pd

from ml.features.feature_engineering import create_time_features


df = pd.read_csv(
    "data/raw/synthetic_transactions.csv"
)

features = create_time_features(df)

print("Feature columns:")
print(features.columns.tolist())

print("\nSample features:")
print(
    features[
        [
            "merchant_id",
            "timestamp",
            "amount",
            "fraud_spike",
            "transaction_count_1m",
            "transaction_count_5m",
            "velocity_ratio",
            "amount_ratio",
        ]
    ].head(30)
)