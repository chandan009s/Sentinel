from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.features.feature_contract import FEATURE_COLUMNS
from ml.features.feature_engineering import create_time_features


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "synthetic_transactions_v2.csv"
)

METADATA_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "isolation_forest_v1_meta.json"
)

TRAIN_END = 707.06


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df = df.sort_values(
        ["timestamp", "merchant_id"]
    ).reset_index(drop=True)

    features = create_time_features(df)

    features = features.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    train = features[
        features["timestamp"] < TRAIN_END
    ].copy()

    normal_train = train[
        train["fraud_spike"] == 0
    ].copy()

    if normal_train.empty:
        raise RuntimeError(
            "No normal training samples found."
        )

    reference_stats = {}

    for feature in FEATURE_COLUMNS:
        series = normal_train[feature]

        reference_stats[feature] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    with METADATA_PATH.open() as f:
        metadata = json.load(f)

    metadata["reference_statistics"] = reference_stats

    with METADATA_PATH.open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )
        f.write("\n")

    print("Reference statistics updated.")

    for feature, stats in reference_stats.items():
        print(
            f"{feature}: "
            f"mean={stats['mean']:.6f}, "
            f"std={stats['std']:.6f}"
        )


if __name__ == "__main__":
    main()