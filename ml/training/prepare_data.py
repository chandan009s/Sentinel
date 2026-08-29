import pandas as pd

from ml.features.feature_engineering import create_time_features

DATA_PATH = "data/raw/synthetic_transactions.csv"

FEATURE_COLUMNS = [
    "velocity_ratio",
    "amount_ratio",
]

def main():

    df = pd.read_csv(DATA_PATH)

    print(f"Raw transactions: {len(df)}")

    features = create_time_features(df)

    print(f"After feature engineering: {len(features)}")

    features = features.dropna(
        subset=[
            "transaction_count_baseline",
            "amount_baseline",
            "velocity_ratio",
            "amount_ratio",
        ]
    ).copy()

    print(f"After removing warm-up rows: {len(features)}")

    X = features[FEATURE_COLUMNS].copy()

    y = features["fraud_spike"].copy()

    print("\nX shape:")
    print(X.shape)

    print("\ny shape:")
    print(y.shape)

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nMissing values in X:")
    print(X.isna().sum())

    print("\nFeature preview:")
    print(X.head())

if __name__ == "__main__":
    main()