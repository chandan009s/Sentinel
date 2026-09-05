from __future__ import annotations

from typing import Final

import pandas as pd


FEATURE_COLUMNS: Final[list[str]] = [
    "velocity_ratio",
    "amount_ratio",
    "velocity_acceleration_1m",
    "amount_acceleration_1m",
]


REQUIRED_RAW_COLUMNS: Final[list[str]] = [
    "timestamp",
    "amount",
    "merchant_id",
]


def validate_feature_frame(
    features: pd.DataFrame,
) -> None:
    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing production features: {missing}"
        )

    if features[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "Production feature frame contains NaN values."
        )

    if not all(
        pd.api.types.is_numeric_dtype(features[column])
        for column in FEATURE_COLUMNS
    ):
        raise ValueError(
            "Production features must be numeric."
        )