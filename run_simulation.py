"""
run_simulation.py — Federated IoT Anomaly Detection Simulation
================================================================
Orchestrates the full federated learning pipeline:

  1. Create N simulated IoT devices
  2. For each federation round:
     a. Generate local training data (normal) for each device
     b. Train each device's local model
     c. Collect model updates at the server
     d. Aggregate into a global model
     e. Distribute the global model back to devices
     f. Evaluate detection on a mixed test set (normal + attacks)
  3. Print per-round and final summary metrics

Usage:
    python run_simulation.py
"""

import sys
import numpy as np
import pandas as pd

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    # Fallback if colorama is not installed
    class _Dummy:
        def __getattr__(self, _):
            return ""
    Fore = _Dummy()
    Style = _Dummy()

from data_generator import generate_normal, generate_attack, generate_mixed_test_set, ATTACK_TYPES
from iot_device import IoTDevice
from fed_server import FederationServer


# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────

NUM_DEVICES = 5          # Number of simulated IoT edge devices
NUM_ROUNDS = 4           # Number of federation rounds
NORMAL_SAMPLES = 200     # Normal training samples per device per round
TEST_NORMAL = 150        # Normal samples in test set
TEST_ATTACK_PER_TYPE = 15  # Attack samples per attack type in test set


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute detection metrics.

    Args:
        y_true: Ground truth (0 = normal, 1 = attack)
        y_pred: Predictions from Isolation Forest (1 = normal, -1 = anomaly)

    Returns:
        dict with accuracy, detection_rate (recall for attacks),
        and false_positive_rate.
    """
    # Convert Isolation Forest output: -1 (anomaly) → 1 (attack), 1 (normal) → 0
    pred_labels = np.where(y_pred == -1, 1, 0)

    total = len(y_true)
    correct = (pred_labels == y_true).sum()
    accuracy = correct / total if total > 0 else 0

    # Detection rate (recall): of actual attacks, how many did we catch?
    true_attacks = (y_true == 1).sum()
    detected_attacks = ((y_true == 1) & (pred_labels == 1)).sum()
    detection_rate = detected_attacks / true_attacks if true_attacks > 0 else 0

    # False positive rate: of actual normal, how many did we flag?
    true_normal = (y_true == 0).sum()
    false_positives = ((y_true == 0) & (pred_labels == 1)).sum()
    fpr = false_positives / true_normal if true_normal > 0 else 0

    return {
        "accuracy": accuracy,
        "detection_rate": detection_rate,
        "false_positive_rate": fpr,
    }


def print_header():
    """Print the simulation banner."""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   {Fore.WHITE}🛡️  FEDERATED IoT ANOMALY DETECTION SIMULATION  🛡️{Fore.CYAN}             ║
║                                                                  ║
║   {Fore.YELLOW}Devices: {NUM_DEVICES}  |  Rounds: {NUM_ROUNDS}  |  Attacks: {len(ATTACK_TYPES)} types{Fore.CYAN}            ║
║   {Fore.GREEN}Privacy: Raw data NEVER leaves the device{Fore.CYAN}                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def print_round_header(round_num: int):
    """Print the header for a federation round."""
    print(f"\n{Fore.CYAN}{'='*64}")
    print(f"  📡  FEDERATION ROUND {round_num}/{NUM_ROUNDS}")
    print(f"{'='*64}{Style.RESET_ALL}\n")


def print_device_result(device_id: str, metrics: dict, phase: str = ""):
    """Print detection results for a single device."""
    acc = metrics["accuracy"] * 100
    det = metrics["detection_rate"] * 100
    fpr = metrics["false_positive_rate"] * 100

    # Color code detection rate
    det_color = Fore.GREEN if det >= 80 else (Fore.YELLOW if det >= 50 else Fore.RED)
    fpr_color = Fore.GREEN if fpr <= 15 else (Fore.YELLOW if fpr <= 30 else Fore.RED)

    prefix = f"  [{phase}]" if phase else "  "
    print(
        f"{prefix} {Fore.WHITE}{device_id:<12}{Style.RESET_ALL} │ "
        f"Accuracy: {Fore.WHITE}{acc:5.1f}%{Style.RESET_ALL} │ "
        f"Detection: {det_color}{det:5.1f}%{Style.RESET_ALL} │ "
        f"FPR: {fpr_color}{fpr:5.1f}%{Style.RESET_ALL}"
    )


def run_simulation():
    """Main simulation loop."""
    print_header()

    # ── Step 1: Create devices and federation server ──
    print(f"{Fore.YELLOW}[INIT]{Style.RESET_ALL} Creating {NUM_DEVICES} simulated IoT devices...\n")

    server = FederationServer()
    devices = []
    for i in range(NUM_DEVICES):
        device = IoTDevice(
            device_id=f"sensor_{i+1:02d}",
            contamination=0.1,
            seed=i * 42,
        )
        devices.append(device)
        server.register_device(device)
        print(f"  ✅ Registered {Fore.GREEN}{device.device_id}{Style.RESET_ALL}")

    # Track metrics across rounds for final summary
    round_metrics = []

    # ── Step 2: Federation rounds ──
    for round_num in range(1, NUM_ROUNDS + 1):
        print_round_header(round_num)
        round_device_metrics = []

        # ── 2a: Generate local data and train ──
        print(f"  {Fore.YELLOW}[TRAIN]{Style.RESET_ALL} Local training on {NORMAL_SAMPLES} normal samples per device...")
        print(f"  {Fore.MAGENTA}(Raw data stays on each device — NEVER shared){Style.RESET_ALL}\n")

        for i, device in enumerate(devices):
            # Each device gets slightly different data (non-IID simulation)
            local_seed = round_num * 1000 + i * 100
            normal_data = generate_normal(NORMAL_SAMPLES, seed=local_seed)
            device.train_local(normal_data)

        # ── 2b: Evaluate BEFORE federation (local-only model) ──
        print(f"  {Fore.YELLOW}[EVAL]{Style.RESET_ALL} Detection BEFORE federation (local models only):\n")

        test_data, test_labels = generate_mixed_test_set(
            n_normal=TEST_NORMAL,
            n_attack_per_type=TEST_ATTACK_PER_TYPE,
            seed=round_num * 7777,
        )

        pre_fed_metrics = []
        for device in devices:
            preds = device.detect(test_data)
            metrics = compute_metrics(test_labels.values, preds)
            pre_fed_metrics.append(metrics)
            print_device_result(device.device_id, metrics, phase="LOCAL")

        avg_pre = {
            k: np.mean([m[k] for m in pre_fed_metrics])
            for k in pre_fed_metrics[0]
        }
        print(f"\n  {Fore.WHITE}  Average (local):{Style.RESET_ALL} "
              f"Accuracy={avg_pre['accuracy']*100:.1f}% | "
              f"Detection={avg_pre['detection_rate']*100:.1f}% | "
              f"FPR={avg_pre['false_positive_rate']*100:.1f}%")

        # ── 2c: Collect, aggregate, distribute ──
        print(f"\n  {Fore.YELLOW}[FED]{Style.RESET_ALL}  Collecting model updates from {len(devices)} devices...")
        n_collected = server.collect_updates()
        print(f"  {Fore.YELLOW}[FED]{Style.RESET_ALL}  Aggregating {n_collected} models (ensemble averaging)...")

        ref_data = generate_normal(50, seed=round_num * 5555)
        server.aggregate(reference_data=ref_data)

        print(f"  {Fore.YELLOW}[FED]{Style.RESET_ALL}  Distributing global model to all devices...")
        server.distribute_global_model()
        print(f"  {Fore.GREEN}[OK]{Style.RESET_ALL}   Global model distributed.\n")

        # ── 2d: Evaluate AFTER federation ──
        print(f"  {Fore.YELLOW}[EVAL]{Style.RESET_ALL} Detection AFTER federation (global model):\n")

        post_fed_metrics = []
        for device in devices:
            preds = device.detect(test_data)
            metrics = compute_metrics(test_labels.values, preds)
            post_fed_metrics.append(metrics)
            print_device_result(device.device_id, metrics, phase="GLOBAL")

        avg_post = {
            k: np.mean([m[k] for m in post_fed_metrics])
            for k in post_fed_metrics[0]
        }
        print(f"\n  {Fore.WHITE}  Average (global):{Style.RESET_ALL} "
              f"Accuracy={avg_post['accuracy']*100:.1f}% | "
              f"Detection={avg_post['detection_rate']*100:.1f}% | "
              f"FPR={avg_post['false_positive_rate']*100:.1f}%")

        # Track improvement
        improvement = avg_post["detection_rate"] - avg_pre["detection_rate"]
        if improvement > 0:
            print(f"\n  {Fore.GREEN}  ⬆ Detection improved by {improvement*100:.1f}% after federation{Style.RESET_ALL}")
        elif improvement < 0:
            print(f"\n  {Fore.RED}  ⬇ Detection decreased by {abs(improvement)*100:.1f}% after federation{Style.RESET_ALL}")
        else:
            print(f"\n  {Fore.YELLOW}  ── Detection unchanged after federation{Style.RESET_ALL}")

        round_metrics.append({
            "round": round_num,
            "pre_accuracy": avg_pre["accuracy"],
            "pre_detection": avg_pre["detection_rate"],
            "pre_fpr": avg_pre["false_positive_rate"],
            "post_accuracy": avg_post["accuracy"],
            "post_detection": avg_post["detection_rate"],
            "post_fpr": avg_post["false_positive_rate"],
        })

    # ── Step 3: Final summary ──
    print(f"\n\n{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗")
    print(f"║                    📊  FINAL SUMMARY                            ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")

    print(f"  {'Round':<8} {'Pre-Fed Acc':>12} {'Post-Fed Acc':>14} {'Pre-Fed Det':>13} {'Post-Fed Det':>14} {'Post-FPR':>10}")
    print(f"  {'─'*8} {'─'*12} {'─'*14} {'─'*13} {'─'*14} {'─'*10}")

    for rm in round_metrics:
        print(
            f"  {rm['round']:<8} "
            f"{rm['pre_accuracy']*100:>11.1f}% "
            f"{rm['post_accuracy']*100:>13.1f}% "
            f"{rm['pre_detection']*100:>12.1f}% "
            f"{rm['post_detection']*100:>13.1f}% "
            f"{rm['post_fpr']*100:>9.1f}%"
        )

    # Overall averages
    avg_final_det = np.mean([rm["post_detection"] for rm in round_metrics]) * 100
    avg_final_fpr = np.mean([rm["post_fpr"] for rm in round_metrics]) * 100
    avg_final_acc = np.mean([rm["post_accuracy"] for rm in round_metrics]) * 100

    print(f"\n  {Fore.GREEN}{'─'*75}")
    print(f"  Overall Average (Post-Federation):")
    print(f"    Accuracy:        {avg_final_acc:.1f}%")
    print(f"    Detection Rate:  {avg_final_det:.1f}%")
    print(f"    False Positive:  {avg_final_fpr:.1f}%{Style.RESET_ALL}")

    print(f"\n  {Fore.MAGENTA}🔒 Privacy: Raw data NEVER left any device during this simulation.")
    print(f"  📡 Only model parameters were exchanged between devices and server.{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}{'='*64}")
    print(f"  Simulation complete. {NUM_DEVICES} devices × {NUM_ROUNDS} rounds.")
    print(f"{'='*64}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    run_simulation()
