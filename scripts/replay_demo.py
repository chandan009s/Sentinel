import argparse
import json
import time
import urllib.request
import uuid


API_URL = "http://localhost:8080/predict"


SCENARIOS = {
    "normal": [
        (1.0, 1.0, 0.05, 0.05),
        (1.05, 1.0, 0.05, 0.05),
        (1.1, 1.05, 0.05, 0.05),
        (1.0, 1.0, 0.05, 0.05),
    ],
    "gradual": [
        (1.1, 1.1, 0.10, 0.10),
        (1.3, 1.2, 0.15, 0.15),
        (1.5, 1.4, 0.20, 0.20),
        (1.8, 1.7, 0.25, 0.25),
        (2.2, 2.1, 0.30, 0.30),
    ],
    "velocity": [
        (1.5, 1.0, 0.20, 0.05),
        (2.0, 1.0, 0.25, 0.05),
        (2.5, 1.0, 0.30, 0.05),
        (3.0, 1.0, 0.35, 0.05),
    ],
    "amount": [
        (1.0, 1.5, 0.05, 0.20),
        (1.0, 2.0, 0.05, 0.30),
        (1.0, 2.5, 0.05, 0.40),
        (1.0, 3.0, 0.05, 0.50),
    ],
    "combined": [
        (1.8, 1.8, 0.20, 0.20),
        (2.2, 2.2, 0.25, 0.25),
        (2.8, 2.8, 0.30, 0.30),
        (3.2, 3.2, 0.35, 0.35),
    ],
    "recovery": [
        (2.5, 2.5, 0.30, 0.30),
        (3.0, 3.0, 0.35, 0.35),
        (3.5, 3.5, 0.40, 0.40),
        (1.5, 1.5, 0.05, 0.05),
        (1.1, 1.1, 0.02, 0.02),
        (1.0, 1.0, 0.01, 0.01),
    ],
}


def predict(event):
    payload = json.dumps(event).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def run_scenario(name, events, delay):
    merchant_id = f"SIM-{name.upper()}-{uuid.uuid4().hex[:8]}"
    base_timestamp = time.time()
    simulated_step = 10.0

    results = []

    print()
    print("=" * 80)
    print(f"SCENARIO: {name.upper()}")
    print(f"Merchant : {merchant_id}")
    print("=" * 80)

    for index, (velocity, amount, velocity_accel, amount_accel) in enumerate(events):
        event = {
            "event_id": f"{merchant_id}-{index + 1}",
            "merchant_id": merchant_id,
            "timestamp": base_timestamp + (index * simulated_step),
            "velocity_ratio": velocity,
            "amount_ratio": amount,
            "velocity_acceleration_1m": velocity_accel,
            "amount_acceleration_1m": amount_accel,
        }

        result = predict(event)
        results.append(result)

        print(
            f"{index + 1:02d} "
            f"velocity={velocity:<4.2f} "
            f"amount={amount:<4.2f} "
            f"score={result['anomaly_score']:<8.4f} "
            f"risk={result['risk_level']:<6} "
            f"state={result['event_state']:<9} "
            f"decision={result['decision']:<6}"
        )

        time.sleep(delay)

    activations = sum(
        result["event_state"] == "ACTIVE"
        for result in results
    )

    alerts = sum(
        result["prediction"] == 1
        for result in results
    )

    print("-" * 80)
    print(f"Predictions : {len(results)}")
    print(f"Raw alerts  : {alerts}")
    print(f"Activations : {activations}")
    print(f"Final state : {results[-1]['event_state']}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Scenario-driven AI Risk Manager simulator"
    )

    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS.keys(), "all"],
        default="all",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    print("=" * 80)
    print("AI RISK MANAGER - SIMULATION PLATFORM")
    print("=" * 80)

    scenarios = (
        SCENARIOS.items()
        if args.scenario == "all"
        else [(args.scenario, SCENARIOS[args.scenario])]
    )

    total_predictions = 0
    total_alerts = 0
    total_activations = 0

    for name, events in scenarios:
        results = run_scenario(name, events, args.delay)

        total_predictions += len(results)
        total_alerts += sum(
            result["prediction"] == 1
            for result in results
        )
        total_activations += sum(
            result["event_state"] == "ACTIVE"
            for result in results
        )

    print()
    print("=" * 80)
    print("SIMULATION SUMMARY")
    print("=" * 80)
    print(f"Predictions : {total_predictions}")
    print(f"Raw alerts  : {total_alerts}")
    print(f"Activations : {total_activations}")
    print("=" * 80)


if __name__ == "__main__":
    main()