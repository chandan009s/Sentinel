import numpy as np
import pandas as pd

MERCHANTS = {
    "M001": {
        "rate_per_minute": 30,
        "average_amount": 40,
    },
    "M002": {
        "rate_per_minute": 80,
        "average_amount": 75,
    },
    "M003": {
        "rate_per_minute": 180,
        "average_amount": 50,
    },
    "M004": {
        "rate_per_minute": 50,
        "average_amount": 150,
    },
}

SPIKE_EVENTS = [
    ("M001", 180, 60, 3.0),
    ("M002", 300, 60, 5.0),
    ("M003", 420, 60, 2.5),
    ("M004", 540, 60, 4.0),

    ("M001", 650, 60, 4.5),
    ("M002", 720, 60, 3.2),
    ("M003", 780, 30, 5.0),
    ("M003", 850, 30, 5.0),
]

def generate_poisson_transactions(
    merchant_id: str,
    start_time: float,
    duration_seconds: float,
    rate_per_minute: float,
    average_amount: float,
    fraud_spike: int,
    rng: np.random.Generator,
) -> list[dict]:

    rate_per_second = rate_per_minute / 60.0

    transactions = []
    current_time = start_time
    end_time = start_time + duration_seconds

    while current_time < end_time:

        inter_arrival = rng.exponential(
            1 / rate_per_second
        )

        current_time += inter_arrival

        if current_time >= end_time:
            break

        amount = rng.lognormal(
            mean=np.log(average_amount),
            sigma=0.5,
        )

        transactions.append(
            {
                "merchant_id": merchant_id,
                "timestamp": current_time,
                "amount": round(amount, 2),
                "fraud_spike": fraud_spike,
            }
        )

    return transactions

def generate_transactions(
    duration_seconds: int = 900,
    seed: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    all_transactions = []

    for merchant_id, config in MERCHANTS.items():

        merchant_events = [
            event
            for event in SPIKE_EVENTS
            if event[0] == merchant_id
        ]

        current_time = 0

        for _, spike_start, spike_duration, spike_multiplier in merchant_events:
            if spike_start > current_time:

                normal_transactions = (
                    generate_poisson_transactions(
                        merchant_id,
                        current_time,
                        spike_start - current_time,
                        config["rate_per_minute"],
                        config["average_amount"],
                        0,
                        rng,
                    )
                )

                all_transactions.extend(
                    normal_transactions
                )

            spike_transactions = (
                generate_poisson_transactions(
                    merchant_id,
                    spike_start,
                    spike_duration,
                    config["rate_per_minute"]
                    * spike_multiplier,
                    config["average_amount"],
                    1,
                    rng,
                )
            )

            all_transactions.extend(
                spike_transactions
            )

            current_time = (
                spike_start + spike_duration
            )

        if current_time < duration_seconds:

            normal_transactions = (
                generate_poisson_transactions(
                    merchant_id,
                    current_time,
                    duration_seconds - current_time,
                    config["rate_per_minute"],
                    config["average_amount"],
                    0,
                    rng,
                )
            )

            all_transactions.extend(
                normal_transactions
            )

    data = pd.DataFrame(all_transactions)

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return data

if __name__ == "__main__":

    data = generate_transactions()

    output_path = (
        "data/raw/synthetic_transactions.csv"
    )

    data.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved synthetic data to: {output_path}"
    )

    print("\nGenerated transactions:")
    print(data.head(20))

    print("\nShape:")
    print(data.shape)

    print("\nTransactions per merchant:")
    print(
        data["merchant_id"].value_counts()
    )

    print("\nSpike labels:")
    print(
        data["fraud_spike"].value_counts()
    )

    print("\nSpike distribution by merchant:")
    print(
        data[data["fraud_spike"] == 1]
        .groupby("merchant_id")["timestamp"]
        .agg(["min", "max", "count"])
    )