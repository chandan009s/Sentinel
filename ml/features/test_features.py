import pandas as pd

from feature_engineering import create_time_features

df = pd.read_csv("data/raw/creditcard.csv")

features = create_time_features(df)

print(features[
    [
        "Time",
        "Amount",
        "transaction_count_1m",
        "total_amount_1m",
        "average_amount_1m",
        "transaction_count_baseline",
        "amount_baseline",
        "velocity_ratio",
        "amount_ratio",
    ]
].iloc[1000:1030])

print("\nFeature Summary:")
print(
    features[
        [
            "transaction_count_1m",
            "total_amount_1m",
            "average_amount_1m",
            "velocity_ratio",
            "amount_ratio",
        ]
    ].describe()
)