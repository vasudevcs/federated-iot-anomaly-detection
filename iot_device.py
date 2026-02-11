"""
iot_device.py — Simulated IoT Edge Device
==========================================
Each device holds its own Isolation Forest model and trains on LOCAL data only.
Raw data never leaves the device — only the trained model object is shared
with the federation server.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IoTDevice:
    """
    Represents one IoT edge device in the federated system.

    Responsibilities:
      - Hold local training data (private — never shared)
      - Train a local Isolation Forest anomaly detector
      - Export / import the trained model (for federation)
      - Detect anomalies on new data
    """

    def __init__(self, device_id: str, contamination: float = 0.1, seed: int = 42):
        """
        Args:
            device_id:     Unique name for this device (e.g. "sensor_01").
            contamination: Expected proportion of anomalies (used by Isolation Forest).
            seed:          Random state for reproducibility.
        """
        self.device_id = device_id
        self.contamination = contamination
        self.seed = seed
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=seed,
        )
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training (local only — data never leaves the device)
    # ------------------------------------------------------------------

    def train_local(self, normal_data: pd.DataFrame) -> None:
        """
        Train the local model on normal traffic data.

        Args:
            normal_data: DataFrame of normal IoT traffic features.
                         This data is PRIVATE and stays on the device.
        """
        self.model.fit(normal_data)
        self._is_trained = True

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict whether each sample is normal (1) or anomalous (-1).

        Returns:
            numpy array of predictions: 1 = normal, -1 = anomaly.
        """
        if not self._is_trained:
            raise RuntimeError(
                f"Device {self.device_id}: model not trained yet. "
                "Call train_local() first."
            )
        return self.model.predict(data)

    def anomaly_scores(self, data: pd.DataFrame) -> np.ndarray:
        """
        Return anomaly scores (lower = more anomalous).

        The Isolation Forest score_samples() method returns negative scores;
        more negative = more anomalous.
        """
        if not self._is_trained:
            raise RuntimeError(
                f"Device {self.device_id}: model not trained yet."
            )
        return self.model.score_samples(data)

    # ------------------------------------------------------------------
    # Federated model exchange
    # ------------------------------------------------------------------

    def get_model_params(self) -> bytes:
        """
        Serialize the trained model into bytes for sharing with the
        federation server.

        IMPORTANT: Only the MODEL is shared — never raw data.
        """
        if not self._is_trained:
            raise RuntimeError(
                f"Device {self.device_id}: cannot export untrained model."
            )
        return pickle.dumps(self.model)

    def set_model_params(self, model_bytes: bytes) -> None:
        """
        Replace the local model with a model received from the
        federation server (the aggregated global model).
        """
        self.model = pickle.loads(model_bytes)
        self._is_trained = True

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return f"IoTDevice(id={self.device_id!r}, status={status})"


if __name__ == "__main__":
    # Quick self-test
    from data_generator import generate_normal, generate_attack

    device = IoTDevice("test_device_01")
    normal = generate_normal(200)
    device.train_local(normal)

    attack = generate_attack("dos_flood", 20)
    preds = device.detect(attack)
    detected = (preds == -1).sum()
    print(f"Device: {device}")
    print(f"Attacks detected: {detected}/{len(attack)}")
