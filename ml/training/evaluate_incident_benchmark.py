from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml.features.feature_engineering import create_time_features
from ml.models.isolation_forest import AnomalyDetector


DATA_PATH = Path("data/raw/synthetic_transactions_v2.csv")
THRESHOLD_PATH = Path("models/threshold.json")

TRAIN_END = 707.06
VALIDATION_END = 1210.30

FN_COST = 20.0
FP_COST = 1.0


def choose_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
) -> tuple[float, dict]:
    best = None

    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)

        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())

        cost = FN_COST * fn + FP_COST * fp

        result = {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "cost": float(cost),
        }

        if best is None or cost < best["cost"]:
            best = result

    if best is None:
        raise RuntimeError("No threshold candidate was evaluated.")

    return best["threshold"], best


def detect_incidents(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> list[dict]:
    """
    Detect each contiguous fraud episode and measure
    whether at least one prediction occurred during it.
    """
    work = test_df.copy()
    work["prediction"] = predictions

    work = work.sort_values(
        ["merchant_id", "timestamp"]
    ).reset_index(drop=True)

    work["previous_fraud"] = (
        work.groupby("merchant_id")["fraud_spike"]
        .shift(1)
        .fillna(0)
    )

    work["previous_prediction"] = (
        work.groupby("merchant_id")["prediction"]
        .shift(1)
        .fillna(0)
    )

    starts = work[
        (work["fraud_spike"] == 1)
        & (work["previous_fraud"] == 0)
    ]

    incidents = []

    for incident_id, (_, start_row) in enumerate(
        starts.iterrows(),
        start=1,
    ):
        merchant = start_row["merchant_id"]
        start_time = float(start_row["timestamp"])

        merchant_rows = work[
            work["merchant_id"] == merchant
        ]

        after_start = merchant_rows[
            (merchant_rows["timestamp"] >= start_time)
            & (merchant_rows["fraud_spike"] == 1)
        ]

        if after_start.empty:
            continue

        end_time = float(after_start["timestamp"].max())

        incident_rows = merchant_rows[
            (merchant_rows["timestamp"] >= start_time)
            & (merchant_rows["timestamp"] <= end_time)
        ]

        positive_rows = incident_rows[
            incident_rows["prediction"] == 1
        ]

        detected = not positive_rows.empty

        detection_time = None
        time_to_detection = None

        if detected:
            detection_time = float(
                positive_rows["timestamp"].min()
            )
            time_to_detection = (
                detection_time - start_time
            )

        scenario_values = (
            incident_rows["scenario"]
            .drop_duplicates()
            .tolist()
        )

        incidents.append(
            {
                "incident_id": incident_id,
                "merchant_id": merchant,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "scenarios": scenario_values,
                "detected": detected,
                "detection_time": detection_time,
                "time_to_detection": time_to_detection,
                "transactions": int(len(incident_rows)),
            }
        )

    return incidents


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {
        "merchant_id",
        "timestamp",
        "scenario",
        "fraud_spike",
        "amount",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.sort_values(
        ["timestamp", "merchant_id"]
    ).reset_index(drop=True)

    # Build temporal features on the COMPLETE timeline first.
    #
    # This is critical because rolling/lagged baselines need
    # historical context across train/validation/test boundaries.
    featured = create_time_features(df).dropna()

    train = featured[
        featured["timestamp"] < TRAIN_END
    ].copy()

    validation = featured[
        (featured["timestamp"] >= TRAIN_END)
        & (featured["timestamp"] < VALIDATION_END)
    ].copy()

    test = featured[
        featured["timestamp"] >= VALIDATION_END
    ].copy()

    feature_columns = [
        "velocity_ratio",
        "amount_ratio",
    ]

    train_normal = train[
        train["fraud_spike"] == 0
    ]

    validation_labels = (
        validation["fraud_spike"]
        .astype(int)
        .to_numpy()
    )

    test_labels = (
        test["fraud_spike"]
        .astype(int)
        .to_numpy()
    )

    detector = AnomalyDetector(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )

    detector.fit(
        train_normal[feature_columns]
        .to_numpy()
    )

    train_scores = detector.anomaly_score(
        train_normal[feature_columns].to_numpy()
    )

    validation_scores = detector.anomaly_score(
        validation[feature_columns].to_numpy()
    )

    test_scores = detector.anomaly_score(
        test[feature_columns].to_numpy()
    )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Locked threshold not found: {THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        threshold_config = json.load(f)

    threshold = float(
        threshold_config["threshold"]
    )

    validation_predictions = (
        validation_scores >= threshold
    ).astype(int)

    test_predictions = (
        test_scores >= threshold
    ).astype(int)

    threshold_result = {
        "threshold": threshold,
        "percentile": threshold_config.get(
            "percentile"
        ),
        "source": str(THRESHOLD_PATH),
    }

    precision = precision_score(
        test_labels,
        test_predictions,
        zero_division=0,
    )

    recall = recall_score(
        test_labels,
        test_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        test_labels,
        test_predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        test_labels,
        test_predictions,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    incidents = detect_incidents(
        test,
        test_predictions,
    )

    detected_incidents = [
        incident
        for incident in incidents
        if incident["detected"]
    ]

    incident_recall = (
        len(detected_incidents) / len(incidents)
        if incidents
        else 0.0
    )

    detection_times = [
        incident["time_to_detection"]
        for incident in detected_incidents
        if incident["time_to_detection"] is not None
    ]

    median_ttd = (
        float(np.median(detection_times))
        if detection_times
        else None
    )

    p95_ttd = (
        float(np.percentile(detection_times, 95))
        if detection_times
        else None
    )

    output = {
        "dataset": {
            "rows": int(len(df)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        "time_boundaries": {
            "train_end": TRAIN_END,
            "validation_end": VALIDATION_END,
        },
        "threshold": {
            "value": float(threshold),
            "validation": threshold_result,
        },
        "transaction_metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_positive_rate": float(fpr),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "incident_metrics": {
            "total_incidents": len(incidents),
            "detected_incidents": len(detected_incidents),
            "missed_incidents": (
                len(incidents) - len(detected_incidents)
            ),
            "incident_recall": float(incident_recall),
            "median_time_to_detection": median_ttd,
            "p95_time_to_detection": p95_ttd,
        },
        "incidents": incidents,
    }

    print("\n================================")
    print("INCIDENT-AWARE V2 BENCHMARK")
    print("================================")

    print(
        f"Train:      < {TRAIN_END:.0f}s "
        f"({len(train):,} rows)"
    )

    print(
        f"Validation: {TRAIN_END:.0f}s–{VALIDATION_END:.0f}s "
        f"({len(validation):,} rows)"
    )

    print(
        f"Test:       >= {VALIDATION_END:.0f}s "
        f"({len(test):,} rows)"
    )

    print("\nThreshold")
    print(
        f"  {threshold:.6f}"
    )

    print("\nTransaction metrics")
    print(
        f"  Precision: {precision:.4f}"
    )
    print(
        f"  Recall:    {recall:.4f}"
    )
    print(
        f"  F1:        {f1:.4f}"
    )
    print(
        f"  FPR:       {fpr:.4f}"
    )

    print("\nIncident metrics")
    print(
        f"  Total:     {len(incidents)}"
    )
    print(
        f"  Detected:  {len(detected_incidents)}"
    )
    print(
        f"  Missed:    {len(incidents) - len(detected_incidents)}"
    )
    print(
        f"  Recall:    {incident_recall:.4f}"
    )

    if median_ttd is not None:
        print(
            f"  Median TTD: {median_ttd:.3f}s"
        )

    if p95_ttd is not None:
        print(
            f"  P95 TTD:    {p95_ttd:.3f}s"
        )

    print("\nTest incidents")

    for incident in incidents:
        scenarios = "+".join(
            incident["scenarios"]
        )

        if incident["detected"]:
            detection = (
                f"detected @ "
                f"{incident['time_to_detection']:.3f}s"
            )
        else:
            detection = "MISSED"

        print(
            f"  {incident['merchant_id']} "
            f"{scenarios}: {detection}"
        )

    output_path = Path(
        "data/reports/v2_incident_benchmark.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print(
        f"\nSaved report to: {output_path}"
    )


if __name__ == "__main__":
    main()
