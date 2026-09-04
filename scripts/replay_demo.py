import json
import time
import urllib.request


API_URL = "http://localhost:8080/predict"


EVENTS = [
    {
        "name": "Normal traffic",
        "velocity_ratio": 1.0,
        "amount_ratio": 1.0,
    },
    {
        "name": "Normal traffic",
        "velocity_ratio": 1.1,
        "amount_ratio": 1.0,
    },
    {
        "name": "Elevated traffic",
        "velocity_ratio": 2.0,
        "amount_ratio": 2.0,
    },
    {
        "name": "Elevated traffic",
        "velocity_ratio": 2.2,
        "amount_ratio": 2.1,
    },
    {
        "name": "Spike",
        "velocity_ratio": 4.0,
        "amount_ratio": 4.0,
    },
    {
        "name": "Spike",
        "velocity_ratio": 5.0,
        "amount_ratio": 5.0,
    },
    {
        "name": "Spike",
        "velocity_ratio": 6.0,
        "amount_ratio": 6.0,
    },
]


def predict(event):
    payload = json.dumps(
        {
            "velocity_ratio": event["velocity_ratio"],
            "amount_ratio": event["amount_ratio"],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    print("=" * 70)
    print("AI RISK MANAGER - LIVE REPLAY")
    print("=" * 70)

    for event in EVENTS:
        result = predict(event)

        print(
            f"{event['name']:<20} "
            f"velocity={event['velocity_ratio']:<4} "
            f"amount={event['amount_ratio']:<4} "
            f"score={result['anomaly_score']:.4f} "
            f"risk={result['risk_level']}"
        )

        time.sleep(1)

    print("=" * 70)
    print("Replay complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
