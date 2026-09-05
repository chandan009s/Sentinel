from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_seconds: float
    rate_multiplier: float
    amount_multiplier: float
    fraud_spike: int


SCENARIOS = (
    Scenario(
        name="NORMAL",
        duration_seconds=60,
        rate_multiplier=1.0,
        amount_multiplier=1.0,
        fraud_spike=0,
    ),
    Scenario(
        name="LEGITIMATE_BURST",
        duration_seconds=30,
        rate_multiplier=1.8,
        amount_multiplier=1.0,
        fraud_spike=0,
    ),
    Scenario(
        name="VELOCITY_SPIKE",
        duration_seconds=30,
        rate_multiplier=3.0,
        amount_multiplier=1.0,
        fraud_spike=1,
    ),
    Scenario(
        name="AMOUNT_SPIKE",
        duration_seconds=30,
        rate_multiplier=1.0,
        amount_multiplier=2.5,
        fraud_spike=1,
    ),
    Scenario(
        name="COMBINED_SPIKE",
        duration_seconds=30,
        rate_multiplier=2.5,
        amount_multiplier=2.5,
        fraud_spike=1,
    ),
    Scenario(
        name="GRADUAL_SPIKE",
        duration_seconds=60,
        rate_multiplier=2.5,
        amount_multiplier=2.0,
        fraud_spike=1,
    ),
    Scenario(
        name="SUSTAINED_SPIKE",
        duration_seconds=60,
        rate_multiplier=2.5,
        amount_multiplier=2.5,
        fraud_spike=1,
    ),
    Scenario(
        name="RECOVERY",
        duration_seconds=30,
        rate_multiplier=1.2,
        amount_multiplier=1.1,
        fraud_spike=0,
    ),
)


def generate_transactions_for_scenario(
    merchant_id: str,
    start_time: float,
    scenario: Scenario,
    base_rate_per_minute: float,
    base_average_amount: float,
    rng: np.random.Generator,
) -> list[dict]:
    rate_per_second = (
        base_rate_per_minute
        * scenario.rate_multiplier
        / 60.0
    )

    end_time = start_time + scenario.duration_seconds
    current_time = start_time
    transactions: list[dict] = []

    while current_time < end_time:
        inter_arrival = rng.exponential(
            1.0 / rate_per_second
        )

        current_time += inter_arrival

        if current_time >= end_time:
            break

        amount = rng.lognormal(
            mean=np.log(
                base_average_amount
                * scenario.amount_multiplier
            ),
            sigma=0.5,
        )

        transactions.append(
            {
                "merchant_id": merchant_id,
                "timestamp": current_time,
                "amount": round(float(amount), 2),
                "fraud_spike": scenario.fraud_spike,
                "scenario": scenario.name,
            }
        )

    return transactions


def build_scenario_sequence(
    rng: np.random.Generator,
    duration_seconds: int,
    merchant_id: str,
) -> list[Scenario]:
    scenarios_by_name = {
        scenario.name: scenario
        for scenario in SCENARIOS
    }

    normal = scenarios_by_name["NORMAL"]
    legitimate_burst = scenarios_by_name["LEGITIMATE_BURST"]
    recovery = scenarios_by_name["RECOVERY"]
    sustained = scenarios_by_name["SUSTAINED_SPIKE"]

    incident_plan = {
        "M001": [
            (180, "VELOCITY_SPIKE", False),
            (1080, "COMBINED_SPIKE", True),
        ],
        "M002": [
            (420, "AMOUNT_SPIKE", False),
            (1320, "GRADUAL_SPIKE", False),
        ],
        "M003": [
            (660, "COMBINED_SPIKE", False),
            (1500, "VELOCITY_SPIKE", True),
        ],
        "M004": [
            (300, "AMOUNT_SPIKE", False),
            (1140, "GRADUAL_SPIKE", False),
        ],
    }

    if merchant_id not in incident_plan:
        raise ValueError(
            f"No incident plan for merchant: {merchant_id}"
        )

    incidents = incident_plan[merchant_id]

    # Apply small timing jitter while preserving chronological order.
    planned_incidents = []

    for incident_start, attack_name, escalate in incidents:
        jittered_start = float(
            incident_start
            + rng.uniform(-20, 20)
        )

        planned_incidents.append(
            (
                jittered_start,
                attack_name,
                escalate,
            )
        )

    planned_incidents.sort(
        key=lambda item: item[0]
    )

    sequence: list[Scenario] = []
    cursor = 0.0

    def add(
        scenario: Scenario,
        duration: float,
    ) -> None:
        if duration <= 0:
            return

        sequence.append(
            Scenario(
                name=scenario.name,
                duration_seconds=float(duration),
                rate_multiplier=scenario.rate_multiplier,
                amount_multiplier=scenario.amount_multiplier,
                fraud_spike=scenario.fraud_spike,
            )
        )

    for incident_start, attack_name, escalate in planned_incidents:
        incident_start = max(
            incident_start,
            cursor,
        )

        # Normal background before the incident.
        add(
            normal,
            incident_start - cursor,
        )

        cursor = incident_start

        # Initial attack.
        attack = scenarios_by_name[attack_name]

        attack_duration = min(
            attack.duration_seconds,
            duration_seconds - cursor,
        )

        add(
            attack,
            attack_duration,
        )

        cursor += attack_duration

        if cursor >= duration_seconds:
            break

        # Optional sustained escalation.
        if (
            escalate
            and attack_name != "SUSTAINED_SPIKE"
        ):
            sustained_duration = min(
                sustained.duration_seconds,
                duration_seconds - cursor,
            )

            add(
                sustained,
                sustained_duration,
            )

            cursor += sustained_duration

            if cursor >= duration_seconds:
                break

        # Incident recovery.
        recovery_duration = min(
            recovery.duration_seconds,
            duration_seconds - cursor,
        )

        add(
            recovery,
            recovery_duration,
        )

        cursor += recovery_duration

        if cursor >= duration_seconds:
            break

    # Normal background after the final incident.
    if cursor < duration_seconds:
        add(
            normal,
            duration_seconds - cursor,
        )

    # Add a legitimate burst to one sufficiently long normal period.
    normal_indices = [
        index
        for index, scenario in enumerate(sequence)
        if (
            scenario.name == "NORMAL"
            and scenario.duration_seconds >= 50
        )
    ]

    if normal_indices:
        index = int(
            rng.choice(normal_indices)
        )

        original = sequence[index]

        burst_duration = min(
            legitimate_burst.duration_seconds,
            25.0,
            original.duration_seconds - 10.0,
        )

        if burst_duration > 0:
            sequence[index:index + 1] = [
                Scenario(
                    name=legitimate_burst.name,
                    duration_seconds=burst_duration,
                    rate_multiplier=legitimate_burst.rate_multiplier,
                    amount_multiplier=legitimate_burst.amount_multiplier,
                    fraud_spike=legitimate_burst.fraud_spike,
                ),
                Scenario(
                    name=normal.name,
                    duration_seconds=(
                        original.duration_seconds
                        - burst_duration
                    ),
                    rate_multiplier=normal.rate_multiplier,
                    amount_multiplier=normal.amount_multiplier,
                    fraud_spike=normal.fraud_spike,
                ),
            ]

    return sequence


def generate_transactions(
    duration_seconds: int = 1800,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    all_transactions: list[dict] = []

    for merchant_id, config in MERCHANTS.items():
        current_time = 0.0

        scenarios = build_scenario_sequence(
            rng,
            duration_seconds,
            merchant_id,
        )

        for scenario in scenarios:
            if current_time >= duration_seconds:
                break

            remaining = (
                duration_seconds - current_time
            )

            actual_duration = min(
                scenario.duration_seconds,
                remaining,
            )

            actual_scenario = Scenario(
                name=scenario.name,
                duration_seconds=actual_duration,
                rate_multiplier=scenario.rate_multiplier,
                amount_multiplier=scenario.amount_multiplier,
                fraud_spike=scenario.fraud_spike,
            )

            transactions = (
                generate_transactions_for_scenario(
                    merchant_id=merchant_id,
                    start_time=current_time,
                    scenario=actual_scenario,
                    base_rate_per_minute=config[
                        "rate_per_minute"
                    ],
                    base_average_amount=config[
                        "average_amount"
                    ],
                    rng=rng,
                )
            )

            all_transactions.extend(transactions)

            current_time += actual_duration

    data = pd.DataFrame(all_transactions)

    if data.empty:
        raise RuntimeError(
            "No transactions were generated."
        )

    return (
        data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def main() -> None:
    output_path = Path(
        "data/raw/synthetic_transactions_v2.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = generate_transactions()

    data.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved synthetic V2 data to: "
        f"{output_path}"
    )

    print("\nShape:")
    print(data.shape)

    print("\nTransactions per merchant:")
    print(
        data["merchant_id"]
        .value_counts()
        .sort_index()
    )

    print("\nFraud labels:")
    print(
        data["fraud_spike"]
        .value_counts()
        .sort_index()
    )

    print("\nScenario distribution:")
    print(
        data["scenario"]
        .value_counts()
    )

    print("\nScenario × fraud label:")
    print(
        pd.crosstab(
            data["scenario"],
            data["fraud_spike"],
        )
    )

    print("\nTime range:")
    print(
        f"{data['timestamp'].min():.3f}"
        f" -> "
        f"{data['timestamp'].max():.3f}"
    )


if __name__ == "__main__":
    main()
