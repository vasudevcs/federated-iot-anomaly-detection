"""
data_generator.py — Synthetic IoT Traffic Generator
=====================================================
Generates realistic normal IoT traffic and simulated attack traffic.
Each record has 5 features:
  - packet_size       (bytes)
  - packet_rate       (packets/sec)
  - connection_duration (seconds)
  - unique_destinations (count)
  - protocol_type     (0=TCP, 1=UDP, 2=ICMP, 3=Other)
"""

import numpy as np
import pandas as pd

FEATURES = [
    "packet_size",
    "packet_rate",
    "connection_duration",
    "unique_destinations",
    "protocol_type",
]

ATTACK_TYPES = ["port_scan", "dos_flood", "data_exfiltration", "spoofing"]


def generate_normal(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate normal IoT traffic.

    Normal traffic has:
      - packet_size:          64 – 1500 bytes
      - packet_rate:          1 – 100   packets/sec
      - connection_duration:  0.1 – 300 seconds
      - unique_destinations:  1 – 20
      - protocol_type:        0, 1, 2, or 3
    """
    rng = np.random.RandomState(seed)

    data = pd.DataFrame(
        {
            "packet_size": rng.uniform(64, 1500, n_samples),
            "packet_rate": rng.uniform(1, 100, n_samples),
            "connection_duration": rng.uniform(0.1, 300, n_samples),
            "unique_destinations": rng.randint(1, 21, n_samples).astype(float),
            "protocol_type": rng.randint(0, 4, n_samples).astype(float),
        }
    )
    return data


def generate_attack(attack_type: str, n_samples: int = 50, seed: int = 99) -> pd.DataFrame:
    """
    Generate attack traffic for a specific attack type.

    Each attack type produces feature values far outside the normal range
    so that an anomaly detector can learn to flag them.
    """
    rng = np.random.RandomState(seed)

    if attack_type == "port_scan":
        # Many destinations, very short connections
        data = pd.DataFrame(
            {
                "packet_size": rng.uniform(40, 120, n_samples),
                "packet_rate": rng.uniform(200, 1000, n_samples),
                "connection_duration": rng.uniform(0.001, 0.05, n_samples),
                "unique_destinations": rng.uniform(50, 200, n_samples),
                "protocol_type": rng.choice([0, 2], n_samples).astype(float),
            }
        )

    elif attack_type == "dos_flood":
        # Extremely high packet rate, tiny packets
        data = pd.DataFrame(
            {
                "packet_size": rng.uniform(20, 64, n_samples),
                "packet_rate": rng.uniform(5000, 50000, n_samples),
                "connection_duration": rng.uniform(0.001, 0.1, n_samples),
                "unique_destinations": rng.randint(1, 3, n_samples).astype(float),
                "protocol_type": rng.choice([0, 1], n_samples).astype(float),
            }
        )

    elif attack_type == "data_exfiltration":
        # Very large packets, long connections, few destinations
        data = pd.DataFrame(
            {
                "packet_size": rng.uniform(5000, 50000, n_samples),
                "packet_rate": rng.uniform(1, 10, n_samples),
                "connection_duration": rng.uniform(500, 3600, n_samples),
                "unique_destinations": rng.randint(1, 3, n_samples).astype(float),
                "protocol_type": np.zeros(n_samples),
            }
        )

    elif attack_type == "spoofing":
        # Invalid protocol values, abnormal packet sizes
        data = pd.DataFrame(
            {
                "packet_size": rng.uniform(0, 30, n_samples),
                "packet_rate": rng.uniform(50, 500, n_samples),
                "connection_duration": rng.uniform(0.01, 1, n_samples),
                "unique_destinations": rng.randint(1, 5, n_samples).astype(float),
                "protocol_type": rng.uniform(10, 20, n_samples),  # impossible values
            }
        )

    else:
        raise ValueError(
            f"Unknown attack type: {attack_type}. "
            f"Choose from: {ATTACK_TYPES}"
        )

    return data


def generate_mixed_test_set(
    n_normal: int = 150,
    n_attack_per_type: int = 15,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a mixed test set with labels.

    Returns:
        data:   DataFrame of features
        labels: Series with 0 = normal, 1 = attack
    """
    normal = generate_normal(n_normal, seed=seed)
    normal_labels = pd.Series([0] * n_normal)

    attack_frames = []
    attack_labels = []
    for i, attack_type in enumerate(ATTACK_TYPES):
        attack = generate_attack(attack_type, n_attack_per_type, seed=seed + i + 1)
        attack_frames.append(attack)
        attack_labels.extend([1] * n_attack_per_type)

    all_data = pd.concat([normal] + attack_frames, ignore_index=True)
    all_labels = pd.concat(
        [normal_labels, pd.Series(attack_labels)], ignore_index=True
    )

    # Shuffle
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_data))
    all_data = all_data.iloc[idx].reset_index(drop=True)
    all_labels = all_labels.iloc[idx].reset_index(drop=True)

    return all_data, all_labels


if __name__ == "__main__":
    print("=== Normal Traffic Sample ===")
    print(generate_normal(5))
    print()
    for attack in ATTACK_TYPES:
        print(f"=== {attack.upper()} Attack Sample ===")
        print(generate_attack(attack, 3))
        print()
