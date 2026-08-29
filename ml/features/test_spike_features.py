import pandas as pd

from feature_engineering import create_time_features

df = pd.read_csv(
    "data/raw/synthetic_transactions.csv"
)
features = create_time_features(df)

m001 = features[
    (features["merchant_id"] == "M001")
    & (features["timestamp"] >= 430)
    & (features["timestamp"] <= 650)
]

print("\n========== M001 SPIKE ANALYSIS ==========\n")

print(
    m001[
        [
            "timestamp",
            "fraud_spike",
            "transaction_count_1m",
            "transaction_count_baseline",
            "velocity_ratio",
            "amount_ratio",
        ]
    ].to_string(index=False)
)

m003 = features[
    (features["merchant_id"] == "M003")
    & (features["timestamp"] >= 480)
    & (features["timestamp"] <= 600)
]

print("\n========== M003 DURING M001 SPIKE ==========\n")

print(
    m003[
        [
            "timestamp",
            "fraud_spike",
            "transaction_count_1m",
            "transaction_count_baseline",
            "velocity_ratio",
            "amount_ratio",
        ]
    ].to_string(index=False)
)

print("\n========== SUMMARY ==========\n")

print("M001 before spike:")
print(
    m001[
        (m001["timestamp"] >= 430)
        & (m001["timestamp"] < 480)
    ][
        ["velocity_ratio", "amount_ratio"]
    ].describe()
)

print("\nM001 during spike:")
print(
    m001[
        (m001["timestamp"] >= 480)
        & (m001["timestamp"] < 600)
    ][
        ["velocity_ratio", "amount_ratio"]
    ].describe()
)

print("\nM001 after spike:")
print(
    m001[
        (m001["timestamp"] >= 600)
        & (m001["timestamp"] <= 650)
    ][
        ["velocity_ratio", "amount_ratio"]
    ].describe()
)

print("\nM003 during M001 spike:")
print(
    m003[
        ["velocity_ratio", "amount_ratio"]
    ].describe()
)