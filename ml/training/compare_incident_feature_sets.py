from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector


DATA_PATH = "data/raw/synthetic_transactions_v2.csv"

TRAIN_END = 707.06
VALIDATION_END = 1210.30

V1_FEATURES = [
    "velocity_ratio",
    "amount_ratio",
]

V2_FEATURES = [
    "velocity_ratio",
    "amount_ratio",
    "velocity_acceleration_1m",
    "amount_acceleration_1m",
]


def choose_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    candidates = np.quantile(
        scores,
        np.linspace(0.80, 0.99, 100),
    )

    best_threshold = None
    best_cost = float("inf")

    for threshold in candidates:
        predictions = (
            scores >= threshold
        ).astype(int)

        fp = int(
            ((predictions == 1) & (labels == 0)).sum()
        )

        fn = int(
            ((predictions == 0) & (labels == 1)).sum()
        )

        cost = 20 * fn + fp

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    if best_threshold is None:
        raise RuntimeError(
            "Unable to select threshold."
        )

    return best_threshold


def extract_test_incidents(
    test: pd.DataFrame,
) -> list[dict]:

    rows = []

    for merchant_id, group in test.groupby(
        "merchant_id",
        sort=True,
    ):
        group = group.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        in_incident = False
        start_time = None
        start_scenario = None

        for _, row in group.iterrows():
            fraud = int(row["fraud_spike"]) == 1

            if fraud and not in_incident:
                in_incident = True
                start_time = float(
                    row["timestamp"]
                )
                start_scenario = row["scenario"]

            elif not fraud and in_incident:
                end_time = float(
                    group.loc[
                        group["timestamp"] <= row["timestamp"],
                        "timestamp",
                    ].max()
                )

                incident_rows = group[
                    (group["timestamp"] >= start_time)
                    & (group["timestamp"] <= end_time)
                ]

                rows.append(
                    {
                        "merchant_id": merchant_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "scenario": start_scenario,
                        "rows": incident_rows,
                    }
                )

                in_incident = False

        if in_incident:
            incident_rows = group[
                group["timestamp"] >= start_time
            ]

            rows.append(
                {
                    "merchant_id": merchant_id,
                    "start_time": start_time,
                    "end_time": float(
                        incident_rows["timestamp"].max()
                    ),
                    "scenario": start_scenario,
                    "rows": incident_rows,
                }
            )

    return rows


def evaluate(
    featured: pd.DataFrame,
    feature_columns: list[str],
) -> dict:

    train = featured[
        featured["timestamp"] < TRAIN_END
    ].dropna(
        subset=feature_columns
    ).copy()

    validation = featured[
        (featured["timestamp"] >= TRAIN_END)
        & (featured["timestamp"] < VALIDATION_END)
    ].dropna(
        subset=feature_columns
    ).copy()

    test = featured[
        featured["timestamp"] >= VALIDATION_END
    ].dropna(
        subset=feature_columns
    ).copy()

    train_normal = train[
        train["fraud_spike"] == 0
    ]

    detector = AnomalyDetector(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )

    detector.fit(
        train_normal[
            feature_columns
        ].to_numpy()
    )

    validation_scores = detector.anomaly_score(
        validation[
            feature_columns
        ].to_numpy()
    )

    test_scores = detector.anomaly_score(
        test[
            feature_columns
        ].to_numpy()
    )

    threshold = choose_threshold(
        validation_scores,
        validation["fraud_spike"]
        .astype(int)
        .to_numpy(),
    )

    test["prediction"] = (
        test_scores >= threshold
    ).astype(int)

    incidents = extract_test_incidents(
        test,
    )

    detected = 0
    detection_times = []

    incident_results = []

    for incident in incidents:
        incident_rows = incident["rows"]

        positive = incident_rows[
            incident_rows["prediction"] == 1
        ]

        is_detected = not positive.empty

        ttd = None

        if is_detected:
            detected += 1
            detection_time = float(
                positive["timestamp"].min()
            )

            ttd = (
                detection_time
                - incident["start_time"]
            )

            detection_times.append(ttd)

        incident_results.append(
            {
                "merchant": incident["merchant_id"],
                "scenario": incident["scenario"],
                "detected": is_detected,
                "ttd": ttd,
            }
        )

    incident_recall = (
        detected / len(incidents)
        if incidents
        else 0.0
    )

    return {
        "threshold": threshold,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "incident_count": len(incidents),
        "detected_incidents": detected,
        "incident_recall": incident_recall,
        "median_ttd": (
            float(np.median(detection_times))
            if detection_times
            else None
        ),
        "p95_ttd": (
            float(np.percentile(detection_times, 95))
            if detection_times
            else None
        ),
        "incidents": incident_results,
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df = df.sort_values(
        ["timestamp", "merchant_id"]
    ).reset_index(drop=True)

    featured = create_time_features(df)

    print("\n================================")
    print("INCIDENT-LEVEL V1 vs V2")
    print("================================")

    results = {}

    for name, features in [
        ("V1", V1_FEATURES),
        ("V2", V2_FEATURES),
    ]:

        result = evaluate(
            featured,
            features,
        )

        results[name] = result

        print(f"\n{name}")
        print("-" * 30)

        print(
            f"Threshold:        "
            f"{result['threshold']:.6f}"
        )

        print(
            f"Train rows:       "
            f"{result['train_rows']:,}"
        )

        print(
            f"Incident count:   "
            f"{result['incident_count']}"
        )

        print(
            f"Detected:         "
            f"{result['detected_incidents']}"
        )

        print(
            f"Incident recall:  "
            f"{result['incident_recall']:.4f}"
        )

        if result["median_ttd"] is not None:
            print(
                f"Median TTD:       "
                f"{result['median_ttd']:.3f}s"
            )

        if result["p95_ttd"] is not None:
            print(
                f"P95 TTD:          "
                f"{result['p95_ttd']:.3f}s"
            )

        for incident in result["incidents"]:
            print(
                f"  {incident['merchant']} "
                f"{incident['scenario']}: "
                + (
                    f"{incident['ttd']:.3f}s"
                    if incident["detected"]
                    else "MISSED"
                )
            )

    print("\n================================")
    print("V2 INCIDENT DELTA")
    print("================================")

    v1 = results["V1"]
    v2 = results["V2"]

    print(
        f"Incident recall: "
        f"{v1['incident_recall']:.4f} → "
        f"{v2['incident_recall']:.4f}"
    )

    if (
        v1["median_ttd"] is not None
        and v2["median_ttd"] is not None
    ):
        print(
            f"Median TTD: "
            f"{v1['median_ttd']:.3f}s → "
            f"{v2['median_ttd']:.3f}s"
        )

    if (
        v1["p95_ttd"] is not None
        and v2["p95_ttd"] is not None
    ):
        print(
            f"P95 TTD: "
            f"{v1['p95_ttd']:.3f}s → "
            f"{v2['p95_ttd']:.3f}s"
        )


if __name__ == "__main__":
    main()
