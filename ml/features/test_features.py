import pandas as pd

from ml.features.feature_engineering import create_time_features


def test_create_time_features_produces_expected_columns():
    df = pd.DataFrame(
        {
            "Time": [0.0, 10.0, 20.0, 30.0],
            "Amount": [10.0, 20.0, 30.0, 40.0],
        }
    )

    features = create_time_features(df)

    expected_columns = [
        "transaction_count_1m",
        "total_amount_1m",
        "average_amount_1m",
        "transaction_count_baseline",
        "amount_baseline",
        "velocity_ratio",
        "amount_ratio",
    ]

    for column in expected_columns:
        assert column in features.columns


def test_create_time_features_preserves_source_columns():
    df = pd.DataFrame(
        {
            "Time": [0.0, 10.0, 20.0],
            "Amount": [10.0, 20.0, 30.0],
        }
    )

    features = create_time_features(df)

    assert list(features["Time"]) == [0.0, 10.0, 20.0]
    assert list(features["Amount"]) == [10.0, 20.0, 30.0]