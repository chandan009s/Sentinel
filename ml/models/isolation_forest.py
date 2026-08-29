from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib

class AnomalyDetector:
    """
    Isolation Forest based anomaly detector.

    The model is trained without using fraud labels.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: str = "auto",
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame):
        """
        Train the Isolation Forest using only feature data.
        """

        self.model.fit(X)
        return self

    def predict(self, X: pd.DataFrame):
        """
        Return Isolation Forest predictions.

        1  -> normal
        -1 -> anomaly
        """

        return self.model.predict(X)

    def anomaly_score(self, X: pd.DataFrame):
        """
        Return anomaly scores.

        Higher values indicate more anomalous observations.
        """

        return -self.model.score_samples(X)

    def save(self, path: str):
        """
        Save the trained model to disk.
        """

        joblib.dump(self.model, path)

    def load(self, path: str):
        """
        Load a trained model from disk.
        """

        self.model = joblib.load(path)
        return self