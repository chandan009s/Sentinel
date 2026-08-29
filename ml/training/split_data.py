import pandas as pd

from ml.features.feature_engineering import create_time_features

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

    X = features[FEATURE_COLUMNS].copy()
    y = features["fraud_spike"].copy()

    n = len(features)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    X_train = X.iloc[:train_end].copy()
    X_validation = X.iloc[train_end:validation_end].copy()
    X_test = X.iloc[validation_end:].copy()

    y_train = y.iloc[:train_end].copy()
    y_validation = y.iloc[train_end:validation_end].copy()
    y_test = y.iloc[validation_end:].copy()

    print("\nDataset sizes:")
    print(f"Total:       {len(X)}")
    print(f"Train:       {len(X_train)}")
    print(f"Validation:  {len(X_validation)}")
    print(f"Test:        {len(X_test)}")

    print("\nTime ranges:")

    print(
        "Train:",
        features.iloc[:train_end]["timestamp"].min(),
        "→",
        features.iloc[:train_end]["timestamp"].max(),
    )

    print(
        "Validation:",
        features.iloc[train_end:validation_end]["timestamp"].min(),
        "→",
        features.iloc[train_end:validation_end]["timestamp"].max(),
    )

    print(
        "Test:",
        features.iloc[validation_end:]["timestamp"].min(),
        "→",
        features.iloc[validation_end:]["timestamp"].max(),
    )

    print("\nSpike distribution:")

    print("Train:")
    print(y_train.value_counts())

    print("\nValidation:")
    print(y_validation.value_counts())

    print("\nTest:")
    print(y_test.value_counts())

if __name__ == "__main__":
    main()