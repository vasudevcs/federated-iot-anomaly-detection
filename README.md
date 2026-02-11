# Federated IoT Anomaly Detection — Simulation

A privacy-preserving anomaly detection system for generic IoT devices using **federated learning**. Each simulated device trains locally and shares only model parameters — raw data never leaves the device.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full simulation
python run_simulation.py
```

## What You'll See

The simulation runs **4 federation rounds** across **5 simulated IoT devices**:

- Each round shows detection metrics **before** and **after** federation
- Per-device accuracy, detection rate, and false-positive rate
- A final summary table comparing all rounds

## Project Structure

| File | Purpose |
|------|---------|
| `FEDERATED_IOT_GUIDE.md` | Comprehensive guide (9 topics) — start here! |
| `data_generator.py` | Generates synthetic normal + attack IoT traffic |
| `iot_device.py` | Simulated IoT edge device with local Isolation Forest |
| `fed_server.py` | Federation server — aggregates and distributes models |
| `run_simulation.py` | Main entry point — runs the simulation |
| `requirements.txt` | Python dependencies |

## Attack Types Simulated

- **Port Scan** — high destination count, rapid short connections
- **DoS Flood** — extreme packet rate, tiny packets
- **Data Exfiltration** — oversized packets, long connections
- **Spoofing/Replay** — invalid protocol values, abnormal packet sizes

## Key Privacy Feature

```
✅ Model parameters travel over the network
❌ Raw data NEVER leaves the device
```

## Requirements

- Python 3.10+
- No GPU required
- No real hardware — fully simulated

## Learn More

Read the full guide: [FEDERATED_IOT_GUIDE.md](FEDERATED_IOT_GUIDE.md)
