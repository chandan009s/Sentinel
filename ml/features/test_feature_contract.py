import pandas as pd
import pytest

from ml.features.feature_contract import (
    FEATURE_COLUMNS,
    validate_feature_frame,
)


def test_feature_columns_are_canonical():
    assert FEATURE_COLUMNS == [
        "velocity_ratio",
        "amount_ratio",
        "velocity_acceleration_1m",
        "amount_acceleration_1m",
    ]


def test_valid_feature_frame_passes():
    frame = pd.DataFrame(
        {
            "velocity_ratio": [1.2],
            "amount_ratio": [1.1],
            "velocity_acceleration_1m": [0.2],
            "amount_acceleration_1m": [0.1],
        }
    )

    validate_feature_frame(frame)


def test_missing_feature_fails():
    frame = pd.DataFrame(
        {
            "velocity_ratio": [1.2],
            "amount_ratio": [1.1],
        }
    )

    with pytest.raises(ValueError):
        validate_feature_frame(frame)


def test_nan_feature_fails():
    frame = pd.DataFrame(
        {
            "velocity_ratio": [1.2],
            "amount_ratio": [1.1],
            "velocity_acceleration_1m": [None],
            "amount_acceleration_1m": [0.1],
        }
    )

    with pytest.raises(ValueError):
        validate_feature_frame(frame)