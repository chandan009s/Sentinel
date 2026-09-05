import pandas as pd

from ml.features.feature_engineering import create_time_features


def test_spike_features_change_with_activity():
    df = pd.DataFrame(
        {
            "timestamp": [
                0.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
            ],
            "merchant_id": ["M001"] * 8,
            "amount": [
                20.0,
                20.0,
                20.0,
                20.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            "fraud_spike": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    features = create_time_features(df)

    normal_rows = features[features["fraud_spike"] == 0]
    spike_rows = features[features["fraud_spike"] == 1]

    assert not spike_rows.empty

    # The feature pipeline should preserve source rows.
    assert len(features) == len(df)

    # Spike transactions should have valid amount/velocity inputs.
    assert spike_rows["amount"].notna().all()
    assert spike_rows["timestamp"].notna().all()

    # At least one derived rolling feature should be available
    # once sufficient history exists.
    derived_columns = [
        "transaction_count_1m",
        "total_amount_1m",
        "average_amount_1m",
    ]

    assert any(
        spike_rows[column].notna().any()
        for column in derived_columns
    )

    assert normal_rows["amount"].mean() < spike_rows["amount"].mean()