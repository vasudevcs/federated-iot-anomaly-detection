# Federated IoT Anomaly Detection

A privacy-preserving anomaly detection system for simulated IoT devices using Isolation Forest and federated learning.

## Overview

This project simulates multiple IoT devices that:

- Train locally on normal traffic
- Share only model parameters (not raw data)
- Improve detection performance through federated aggregation

The goal is to demonstrate privacy-preserving anomaly detection in distributed IoT environments.

## Features

- Isolation Forest for unsupervised anomaly detection
- Federated model aggregation (ensemble-based strategy)
- Simulated non-IID device behavior
- Evaluation metrics: Accuracy, Detection Rate, False Positive Rate

## How It Works

1. Each device trains on normal traffic.
2. Devices send serialized models to a central server.
3. The server aggregates models and distributes a global model.
4. Devices evaluate detection performance before and after federation.

## Installation

```bash
pip install -r requirements.txt
Run Simulation
python run_simulation.py

Requirements

Python 3.10+

No GPU required
