import pandas as pd

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create merchant-aware time-based transaction features.

    Rolling windows:
        - 1 minute
        - 5 minutes
        - 1 hour

    Baseline:
        Calculated from previous completed 1-minute buckets.
    """

    data = df.copy()

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_numeric(
            data["timestamp"],
            errors="coerce",
        )
    elif "Time" in data.columns:
        data["timestamp"] = pd.to_numeric(
            data["Time"],
            errors="coerce",
        )
    else:
        raise ValueError(
            "Input data must contain either "
            "'timestamp' or 'Time'."
        )

    if "amount" in data.columns:
        data["amount"] = pd.to_numeric(
            data["amount"],
            errors="coerce",
        )
    elif "Amount" in data.columns:
        data["amount"] = pd.to_numeric(
            data["Amount"],
            errors="coerce",
        )
    else:
        raise ValueError(
            "Input data must contain either "
            "'amount' or 'Amount'."
        )

    if "merchant_id" not in data.columns:
        data["merchant_id"] = "GLOBAL"

    data = data.sort_values(
        ["merchant_id", "timestamp"]
    ).reset_index(drop=True)

    processed_groups = []

    for merchant_id, group in data.groupby(
        "merchant_id"
    ):

        group = group.copy()

        group["datetime"] = pd.to_timedelta(
            group["timestamp"],
            unit="s",
        )

        group = group.set_index("datetime")

        group["transaction_count_1m"] = (
            group["amount"]
            .rolling("1min")
            .count()
        )

        group["total_amount_1m"] = (
            group["amount"]
            .rolling("1min")
            .sum()
        )

        group["average_amount_1m"] = (
            group["amount"]
            .rolling("1min")
            .mean()
        )

        group["transaction_count_5m"] = (
            group["amount"]
            .rolling("5min")
            .count()
        )

        group["total_amount_5m"] = (
            group["amount"]
            .rolling("5min")
            .sum()
        )

        group["average_amount_5m"] = (
            group["amount"]
            .rolling("5min")
            .mean()
        )

        group["transaction_count_1h"] = (
            group["amount"]
            .rolling("1h")
            .count()
        )

        group["total_amount_1h"] = (
            group["amount"]
            .rolling("1h")
            .sum()
        )

        group["average_amount_1h"] = (
            group["amount"]
            .rolling("1h")
            .mean()
        )

        group["minute_bucket"] = (
            group["timestamp"] // 60
        ).astype(int)

        bucket = (
            group.reset_index()
            .groupby("minute_bucket")
            .agg(
                transaction_count_1m=(
                    "amount",
                    "count",
                ),
                total_amount_1m=(
                    "amount",
                    "sum",
                ),
            )
        )

        full_minute_index = pd.RangeIndex(
            bucket.index.min(),
            bucket.index.max() + 1,
        )

        bucket = bucket.reindex(
            full_minute_index,
            fill_value=0,
        )

        bucket_time = pd.to_timedelta(
            bucket.index * 60,
            unit="s",
        )

        transaction_counts = pd.Series(
            bucket["transaction_count_1m"].to_numpy(),
            index=bucket_time,
        )

        total_amounts = pd.Series(
            bucket["total_amount_1m"].to_numpy(),
            index=bucket_time,
        )

        transaction_baseline = (
            transaction_counts
            .shift(5)
            .rolling(
                "10min",
                min_periods=5,
            )
            .median()
        )

        amount_baseline = (
            total_amounts
            .shift(5)
            .rolling(
                "10min",
                min_periods=5,
            )
            .median()
        )

        bucket["transaction_count_baseline"] = (
            transaction_baseline.to_numpy()
        )

        bucket["amount_baseline"] = (
            amount_baseline.to_numpy()
        )

        group = group.reset_index()

        group = group.merge(
            bucket[
                [
                    "transaction_count_baseline",
                    "amount_baseline",
                ]
            ],
            left_on="minute_bucket",
            right_index=True,
            how="left",
        )

        group["velocity_ratio"] = (
            group["transaction_count_1m"]
            / group["transaction_count_baseline"]
        )

        group["amount_ratio"] = (
            group["total_amount_1m"]
            / group["amount_baseline"]
        )

        group["merchant_id"] = merchant_id

        processed_groups.append(group)

    data = pd.concat(
        processed_groups,
        ignore_index=True,
    )

    data = data.sort_values(
        ["timestamp", "merchant_id"]
    ).reset_index(drop=True)

    return data