import numpy as np
import pandas as pd

from ml.models.isolation_forest import AnomalyDetector


FEATURE_COLUMNS = [
    "velocity_ratio",
    "amount_ratio",
]


def test_isolation_forest_fit_and_score():
    X_train = pd.DataFrame(
        {
            "velocity_ratio": [
                0.9,
                1.0,
                1.1,
                1.0,
                0.95,
                1.05,
                1.0,
                1.1,
                0.9,
                1.0,
            ],
            "amount_ratio": [
                0.95,
                1.0,
                1.05,
                1.1,
                0.9,
                1.0,
                1.05,
                0.95,
                1.1,
                1.0,
            ],
        }
    )

    detector = AnomalyDetector(
        n_estimators=50,
        random_state=42,
    )

    detector.fit(X_train)

    scores = detector.anomaly_score(X_train)

    assert len(scores) == len(X_train)
    assert np.isfinite(scores).all()


def test_isolation_forest_predict_returns_expected_shape():
    X_train = pd.DataFrame(
        {
            "velocity_ratio": [0.9, 1.0, 1.1, 1.0, 0.95, 1.05],
            "amount_ratio": [1.0, 0.95, 1.05, 1.1, 0.9, 1.0],
        }
    )

    detector = AnomalyDetector(
        n_estimators=50,
        random_state=42,
    )

    detector.fit(X_train)

    predictions = detector.predict(X_train)

    assert len(predictions) == len(X_train)
    assert set(predictions).issubset({-1, 1})
