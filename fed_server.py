"""
fed_server.py — Federation Server
===================================
Collects model updates from all registered IoT devices, aggregates them
into a global model, and distributes the improved model back.

Aggregation strategy:
  Since Isolation Forest is tree-based (not a simple weight vector), we use
  an ENSEMBLE approach: the global model scores a data point by averaging
  the anomaly scores from ALL devices' models. This captures the benefit
  of federation (collective intelligence) without requiring weight-level
  averaging.

  For redistribution, we pick the model whose decision boundary is closest
  to the ensemble average as the new global model.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from iot_device import IoTDevice


class FederationServer:
    """
    Central server that coordinates federated learning across IoT devices.

    The server NEVER sees raw device data — only serialized model objects.
    """

    def __init__(self):
        self.devices: list[IoTDevice] = []
        self.collected_models: list[IsolationForest] = []
        self.global_model_bytes: bytes | None = None
        self.round_number: int = 0

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def register_device(self, device: IoTDevice) -> None:
        """Register an IoT device with the federation server."""
        self.devices.append(device)

    # ------------------------------------------------------------------
    # Federated learning rounds
    # ------------------------------------------------------------------

    def collect_updates(self) -> int:
        """
        Collect trained model parameters from all registered devices.

        Returns:
            Number of models collected.
        """
        self.collected_models = []
        for device in self.devices:
            model_bytes = device.get_model_params()
            model = pickle.loads(model_bytes)
            self.collected_models.append(model)
        return len(self.collected_models)

    def aggregate(self, reference_data: pd.DataFrame | None = None) -> bytes:
        """
        Aggregate collected models into a global model.

        Strategy: Ensemble scoring.
        We keep ALL collected models and create a wrapper that averages
        their anomaly scores. For simplicity in redistribution, we select
        the model whose average anomaly score on a small reference set is
        closest to the ensemble median.

        If no reference_data is provided, we simply select the first model
        as the global model (round-robin baseline).

        Returns:
            Serialized global model (bytes).
        """
        if not self.collected_models:
            raise RuntimeError("No models collected. Call collect_updates() first.")

        self.round_number += 1

        if reference_data is not None and len(self.collected_models) > 1:
            # Score the reference data with each model
            all_scores = []
            for model in self.collected_models:
                scores = model.score_samples(reference_data)
                all_scores.append(scores)

            all_scores = np.array(all_scores)  # shape: (n_models, n_samples)

            # Ensemble average score per sample
            ensemble_avg = np.mean(all_scores, axis=0)

            # Pick the model closest to ensemble average (L2 distance)
            distances = [
                np.linalg.norm(all_scores[i] - ensemble_avg)
                for i in range(len(self.collected_models))
            ]
            best_idx = int(np.argmin(distances))
            global_model = self.collected_models[best_idx]
        else:
            # Fallback: use the first model
            global_model = self.collected_models[0]

        self.global_model_bytes = pickle.dumps(global_model)
        return self.global_model_bytes

    def distribute_global_model(self) -> None:
        """
        Send the aggregated global model to all registered devices.

        Each device replaces its local model with the global model.
        """
        if self.global_model_bytes is None:
            raise RuntimeError("No global model to distribute. Call aggregate() first.")

        for device in self.devices:
            device.set_model_params(self.global_model_bytes)

    def ensemble_predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Use ALL collected models to make an ensemble prediction.

        A data point is labeled as anomalous if the average anomaly score
        across all models falls below a threshold.

        Returns:
            numpy array: 1 = normal, -1 = anomaly
        """
        if not self.collected_models:
            raise RuntimeError("No models available for ensemble prediction.")

        all_scores = np.array([
            model.score_samples(data) for model in self.collected_models
        ])

        avg_scores = np.mean(all_scores, axis=0)

        # Use a threshold of 0 (scikit-learn convention: negative = anomaly)
        predictions = np.where(avg_scores < 0, -1, 1)
        return predictions


if __name__ == "__main__":
    # Quick self-test
    from data_generator import generate_normal, generate_attack

    server = FederationServer()

    # Create and register 3 devices
    for i in range(3):
        device = IoTDevice(f"device_{i}", seed=i * 10)
        normal = generate_normal(200, seed=i * 100)
        device.train_local(normal)
        server.register_device(device)

    # Federated round
    n = server.collect_updates()
    print(f"Collected {n} model updates")

    ref_data = generate_normal(50, seed=999)
    server.aggregate(reference_data=ref_data)
    server.distribute_global_model()

    # Test detection
    attack = generate_attack("port_scan", 30)
    preds = server.devices[0].detect(attack)
    detected = (preds == -1).sum()
    print(f"After federation — attacks detected: {detected}/{len(attack)}")
