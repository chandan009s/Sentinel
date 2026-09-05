import pandas as pd

from ml.features.feature_engineering import create_time_features


def test_synthetic_features():
    df = pd.DataFrame(
        {
            "timestamp": [0.0, 20.0, 40.0, 60.0],
            "merchant_id": ["M001", "M001", "M001", "M001"],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "fraud_spike": [0, 0, 1, 1],
        }
    )

    features = create_time_features(df)

    expected_columns = [
        "merchant_id",
        "timestamp",
        "amount",
        "fraud_spike",
        "transaction_count_1m",
        "transaction_count_5m",
        "velocity_ratio",
        "amount_ratio",
    ]

    for column in expected_columns:
        assert column in features.columns

    assert len(features) == len(df)