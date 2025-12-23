# Continuous-Time Temporal Graph Neural Networks for Encrypted Traffic Analysis in Zero-Trust Architectures

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper "Continuous-Time Temporal Graph Neural Networks for Encrypted Traffic Analysis in Zero-Trust Architectures" submitted to IEEE Transactions on Information Forensics and Security.

**Authors:** Roger Nick Anaedevha, Alexander Gennadevich Trofimov, Yuri Vladimirovich Borodachev
**Affiliation:** National Research Nuclear University MEPhI (Moscow Engineering Physics Institute)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Model Architecture](#model-architecture)
- [Baseline Implementations](#baseline-implementations)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository provides a complete implementation of **CT-TGNN (Continuous-Time Temporal Graph Neural Networks)**, a novel framework for detecting network intrusions in encrypted traffic within zero-trust microservices architectures. The approach combines:

- **Neural Ordinary Differential Equations (ODEs)** for continuous-time security state evolution
- **Temporal Graph Neural Networks** for capturing network topology and attack propagation
- **Temporal Adaptive Batch Normalization** for stable training of deep continuous networks
- **Encrypted Edge Feature Encoding** for traffic analysis without payload inspection
- **Zero-Trust Integration** with continuous authentication and policy enforcement

### Key Results

| Dataset | Accuracy | Precision | Recall | F1-Score | Latency (P50) |
|---------|----------|-----------|--------|----------|---------------|
| **Microservices Trace** | 98.3% | 97.6% | 96.8% | 97.2% | 47 ms |
| **IoT-23 Encrypted** | 96.8% | 95.2% | 94.3% | 94.7% | 52 ms |
| **UNSW-NB15 Temporal** | 91.1% | 89.4% | 88.2% | 88.8% | 38 ms |

---

## Key Features

### Model Innovations

- **Continuous-Time Graph Dynamics:** Models security state evolution at arbitrary temporal resolution without discrete sampling artifacts
- **Multi-Scale Temporal Modeling:** Captures patterns across 8 orders of magnitude (microseconds to hours)
- **Graph ODE Formulation:** Extends Neural ODEs to graph-structured data with stability guarantees
- **Encrypted Traffic Analysis:** Extracts discriminative features from TLS timing and metadata without decryption
- **Zero-Trust Policy Enforcement:** Real-time threat mitigation through continuous authentication

### Implementation Features

- **Modular Architecture:** Clean separation of models, data, training, and evaluation
- **Multiple Baselines:** Implementations of 7 state-of-the-art baseline methods
- **Comprehensive Evaluation:** All metrics and experiments from the paper
- **Federated Learning:** Privacy-preserving distributed training with differential privacy
- **Extensive Logging:** Detailed experiment tracking with TensorBoard and Weights & Biases integration
- **Docker Support:** Containerized environment for reproducible experiments
- **GPU Optimization:** Efficient CUDA kernels for sparse graph operations

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.8 or higher (for GPU support)
- 16GB+ RAM recommended
- 100GB+ disk space for datasets

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/rogerpanel/CV.git
cd CV/CT-TGNN

# Create conda environment
conda create -n ct-tgnn python=3.9
conda activate ct-tgnn

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install CT-TGNN package in development mode
pip install -e .
```

### Option 2: Using Docker

```bash
# Build Docker image
docker build -t ct-tgnn:latest .

# Run container with GPU support
docker run --gpus all -it -v $(pwd):/workspace ct-tgnn:latest

# Inside container, run experiments
python experiments/run_all_experiments.py
```

### Option 3: Manual Installation

```bash
# Install core dependencies
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install torch-geometric==2.3.1
pip install torchdiffeq==0.2.3
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorboard wandb
pip install pyyaml tqdm h5py

# Install package
pip install -e .
```

---

## Quick Start

### 1. Download and Prepare Datasets

```bash
# Download all datasets (requires ~350GB disk space)
python data/download_datasets.py --all

# Or download individual datasets
python data/download_datasets.py --dataset microservices
python data/download_datasets.py --dataset iot23
python data/download_datasets.py --dataset unsw

# Preprocess datasets
python data/preprocess_microservices.py
python data/preprocess_iot23.py
python data/preprocess_unsw.py
```

### 2. Train CT-TGNN Model

```bash
# Train on Microservices dataset
python training/trainer.py --config config/ct_tgnn_config.yaml --dataset microservices

# Train on IoT-23 dataset
python training/trainer.py --config config/ct_tgnn_config.yaml --dataset iot23

# Train on UNSW-NB15 dataset
python training/trainer.py --config config/ct_tgnn_config.yaml --dataset unsw
```

### 3. Evaluate Model

```bash
# Evaluate trained model
python evaluation/evaluator.py --checkpoint checkpoints/ct_tgnn_microservices_best.pt --dataset microservices

# Generate all evaluation metrics and plots
python evaluation/evaluator.py --checkpoint checkpoints/ct_tgnn_microservices_best.pt --dataset microservices --generate-plots
```

### 4. Run Complete Experiments

```bash
# Reproduce all paper results
python experiments/run_all_experiments.py

# Run specific experiments
python experiments/run_microservices.py
python experiments/run_iot23.py
python experiments/run_unsw.py
python experiments/run_ablations.py
```

---

## Datasets

### 1. Microservices Trace Dataset

**Description:** Production Kubernetes cluster running e-commerce application with 847 microservices over 72 hours.

- **Size:** 15.3 million encrypted API calls
- **Format:** HDF5 files with graph structure and temporal sequences
- **Labels:** 1,247 simulated attack scenarios (lateral movement, privilege escalation, data exfiltration, reconnaissance)
- **Download:** `python data/download_datasets.py --dataset microservices`

**Data Structure:**
```python
{
    'graphs': List[nx.DiGraph],  # Service communication graphs
    'node_features': np.ndarray,  # Shape: (n_nodes, n_features)
    'edge_features': np.ndarray,  # Shape: (n_edges, edge_feat_dim)
    'timestamps': np.ndarray,     # Event timestamps
    'labels': np.ndarray,         # Attack labels
    'attack_types': List[str]     # Attack type annotations
}
```

### 2. IoT-23 Encrypted Dataset

**Description:** Packet captures from 23 malware-infected IoT devices with encrypted traffic.

- **Size:** 325 GB, 42 million flows
- **Format:** Preprocessed flow graphs with encrypted features
- **Labels:** Malicious C&C, data exfiltration, scanning, benign
- **Download:** `python data/download_datasets.py --dataset iot23`

**Citation:**
```
Garcia, S., et al. "IoT-23: A labeled dataset with malicious and benign IoT network traffic."
Stratosphere Laboratory, 2020.
```

### 3. UNSW-NB15 Temporal Dataset

**Description:** Temporal splits of UNSW-NB15 intrusion detection benchmark.

- **Size:** 2.5 million network flows over multiple days
- **Format:** Temporal graphs with flow relationships
- **Labels:** 9 attack categories + benign
- **Download:** `python data/download_datasets.py --dataset unsw`

**Citation:**
```
Moustafa, N., and Slay, J. "UNSW-NB15: A comprehensive data set for network intrusion detection systems."
Military Communications and Information Systems Conference (MilCIS), 2015.
```

### Dataset Statistics

| Dataset | Nodes | Edges | Time Span | Attack % | Encrypted % |
|---------|-------|-------|-----------|----------|-------------|
| Microservices | 847 | 15.3M | 72 hours | 8.2% | 100% |
| IoT-23 | 23 devices | 42M | Variable | 31.4% | 87% |
| UNSW-NB15 | ~2000 IPs | 2.5M | 7 days | 44.9% | 15% |

---

## Reproducing Paper Results

All experiments from the paper can be reproduced using the provided scripts. Results will be saved in `results/` directory with timestamps.

### Table 1: Microservices Lateral Movement Detection

```bash
python experiments/run_microservices.py --mode full --save-results
```

**Expected Output:**
```
CT-TGNN: Accuracy=98.3%, Precision=97.6%, Recall=96.8%, F1=97.2%
StrGNN: Accuracy=89.2%, Precision=87.4%, Recall=85.6%, F1=86.5%
OCTGAT: Accuracy=91.7%, Precision=89.8%, Recall=88.2%, F1=89.0%
...
```

### Table 2: IoT-23 Encrypted Traffic Classification

```bash
python experiments/run_iot23.py --mode full --save-results
```

### Table 3: UNSW-NB15 Temporal Generalization

```bash
python experiments/run_unsw.py --mode temporal-split --save-results
```

### Table 4: Ablation Studies

```bash
python experiments/run_ablations.py --dataset microservices
```

**Ablation Components:**
- Remove continuous-time (discrete snapshots)
- Remove graph structure (treat flows independently)
- Remove multi-scale temporal
- Remove temporal adaptive normalization
- Remove point process integration
- Remove encrypted edge features

### Table 5: Computational Performance

```bash
python evaluation/benchmark_latency.py --models all --batch-sizes 32,64,128
```

### Table 6: Zero-Trust Operational Metrics

```bash
python experiments/run_zerotrust_evaluation.py --attack-scenarios lateral_movement
```

---

## Model Architecture

### CT-TGNN Components

#### 1. Graph ODE Formulation

```python
from models.ct_tgnn import CTTGNN

model = CTTGNN(
    node_feat_dim=256,
    edge_feat_dim=87,
    hidden_dim=256,
    num_ode_blocks=2,
    num_scales=4,
    time_constants=[1e-6, 1e-3, 1.0, 3600.0],
    solver='dopri5',
    rtol=1e-3,
    atol=1e-4
)
```

#### 2. Temporal Adaptive Batch Normalization

```python
from models.temporal_adaptive_bn import TemporalAdaptiveBatchNorm

tabn = TemporalAdaptiveBatchNorm(
    num_features=256,
    time_encoding_dim=64,
    use_periodic=True,
    omega=2*np.pi/86400  # Daily periodicity
)
```

#### 3. Point Process Integration

```python
from models.point_process import TransformerPointProcess

tpp = TransformerPointProcess(
    hidden_dim=256,
    num_layers=4,
    num_heads=8,
    num_event_types=10,
    max_seq_len=1024
)
```

### Architecture Diagram

```
Input: Encrypted Traffic Graph G(t)
          ↓
[Feature Extraction]
  - Packet timing patterns
  - TLS handshake metadata
  - Flow statistics
          ↓
[Multi-Scale Temporal Encoding]
  - Scale 1: microseconds
  - Scale 2: milliseconds
  - Scale 3: seconds
  - Scale 4: hours
          ↓
[Graph ODE Integration]
  - TA-BN-ODE Block 1
  - TA-BN-ODE Block 2
  - Adjoint gradient computation
          ↓
[Point Process Modeling]
  - Transformer encoder
  - Event intensity prediction
          ↓
[Classification Head]
          ↓
Output: Threat predictions
```

---

## Baseline Implementations

All baseline methods are implemented for fair comparison:

### 1. StrGNN (Structural Temporal Graph Neural Network)

```bash
python training/trainer.py --model strgnn --config config/baseline_configs.yaml
```

**Reference:** NEC Labs, "Structural Temporal Graph Neural Networks for Security," 2024.

### 2. OCTGAT (One-Class Temporal Graph Attention)

```bash
python training/trainer.py --model octgat --config config/baseline_configs.yaml
```

### 3. Neural CDE (Neural Controlled Differential Equations)

```bash
python training/trainer.py --model neural_cde --config config/baseline_configs.yaml
```

**Reference:** Kidger et al., "Neural Controlled Differential Equations for Irregular Time Series," NeurIPS 2020.

### 4. Graph NODE (Graph Neural ODE)

```bash
python training/trainer.py --model graph_node --config config/baseline_configs.yaml
```

### 5. CNN-LSTM

```bash
python training/trainer.py --model cnn_lstm --config config/baseline_configs.yaml
```

### 6. TransECA-Net (Transformer with Efficient Channel Attention)

```bash
python training/trainer.py --model transeca --config config/baseline_configs.yaml
```

### 7. HP-LSTM (Hawkes Process LSTM)

```bash
python training/trainer.py --model hp_lstm --config config/baseline_configs.yaml
```

---

## Configuration

### Main Configuration File: `config/ct_tgnn_config.yaml`

```yaml
# Model Configuration
model:
  name: "CT-TGNN"
  node_feat_dim: 256
  edge_feat_dim: 87
  hidden_dim: 256
  num_ode_blocks: 2
  num_gnn_layers: 2
  num_scales: 4
  time_constants: [1.0e-6, 1.0e-3, 1.0, 3600.0]

  # ODE Solver Settings
  solver: "dopri5"
  rtol: 1.0e-3
  atol: 1.0e-4
  adjoint: true

  # Temporal Adaptive BatchNorm
  tabn:
    time_encoding_dim: 64
    use_periodic: true
    omega: 7.27e-5  # 2*pi / 86400 (daily)

  # Point Process
  point_process:
    num_layers: 4
    num_heads: 8
    num_event_types: 10
    max_seq_len: 1024

# Training Configuration
training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  lr_scheduler: "cosine"
  warmup_epochs: 5

  # Loss Weights
  loss_weights:
    classification: 1.0
    point_process: 0.5
    lipschitz_reg: 1.0e-3
    sparse_reg: 1.0e-4

  # Class Weights (for imbalanced data)
  use_class_weights: true

  # Gradient Clipping
  grad_clip: 1.0

# Data Configuration
data:
  dataset: "microservices"
  data_dir: "./data/processed"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  num_workers: 8
  pin_memory: true

# Evaluation Configuration
evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - auroc
    - auprc
  save_predictions: true
  generate_plots: true

# Zero-Trust Configuration
zero_trust:
  enable_continuous_auth: true
  threat_threshold: 0.5
  enable_predictive: true
  lookahead_horizon: 300  # seconds
  mitigation_actions:
    - revoke_access
    - rate_limit
    - isolate_service

# Federated Learning Configuration
federated:
  enable: false
  num_clients: 5
  local_epochs: 3
  aggregation: "fedavg"
  differential_privacy:
    enable: true
    epsilon: 1.0
    delta: 1.0e-5
    clip_norm: 1.0

# Logging Configuration
logging:
  log_dir: "./logs"
  tensorboard: true
  wandb:
    enable: false
    project: "ct-tgnn"
    entity: "your-username"

  # Checkpoint Configuration
  checkpoint_dir: "./checkpoints"
  save_every_n_epochs: 5
  keep_best_k: 3

# Reproducibility
seed: 42
deterministic: true
```

---

## Evaluation Metrics

### Classification Metrics

- **Accuracy:** Overall correctness
- **Precision:** Fraction of predicted attacks that are true attacks
- **Recall:** Fraction of actual attacks detected
- **F1-Score:** Harmonic mean of precision and recall
- **AUROC:** Area Under Receiver Operating Characteristic curve
- **AUPRC:** Area Under Precision-Recall curve

### Latency Metrics

- **P50 (Median):** 50th percentile processing time
- **P95:** 95th percentile processing time
- **P99:** 99th percentile processing time
- **Throughput:** Events processed per second

### Zero-Trust Metrics

- **MTTD:** Mean Time To Detection
- **MTTC:** Mean Time To Containment
- **False Positive Rate:** During normal operations
- **Services Compromised:** Average per incident

### Usage

```python
from evaluation.metrics import compute_all_metrics

metrics = compute_all_metrics(
    y_true=ground_truth,
    y_pred=predictions,
    y_scores=probability_scores,
    compute_latency=True,
    timestamps=event_timestamps
)

print(metrics)
# {
#     'accuracy': 0.983,
#     'precision': 0.976,
#     'recall': 0.968,
#     'f1_score': 0.972,
#     'auroc': 0.994,
#     'auprc': 0.989,
#     'latency_p50': 47.2,
#     'latency_p95': 93.1,
#     'latency_p99': 127.4,
#     'throughput': 8.7e6
# }
```

---

## Reproducibility

### Random Seed Control

All experiments use fixed random seeds for reproducibility:

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Hardware Specifications

All reported results were obtained on:

- **GPU:** NVIDIA A100 (40GB)
- **CPU:** Dual Intel Xeon Platinum 8358 (64 cores)
- **RAM:** 512GB
- **Storage:** NVMe SSD (2TB)
- **CUDA:** 11.8
- **PyTorch:** 2.0.1

### Software Versions

See `requirements.txt` for exact package versions.

### Experiment Logging

All experiments are automatically logged with:

- Configuration parameters
- Random seeds
- Hardware information
- Training curves
- Evaluation metrics
- Model checkpoints
- System resource usage

Results are saved in timestamped directories: `results/YYYY-MM-DD_HH-MM-SS/`

---

## Citation

If you use this code or our method in your research, please cite:

```bibtex
@article{anaedevha2025cttgnn,
  title={Continuous-Time Temporal Graph Neural Networks for Encrypted Traffic Analysis in Zero-Trust Architectures},
  author={Anaedevha, Roger Nick and Trofimov, Alexander Gennadevich and Borodachev, Yuri Vladimirovich},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  publisher={IEEE}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please:

1. Open an issue on GitHub
2. Contact: ar006@campus.mephi.ru

---

## Acknowledgments

This research was supported by:

- National Research Nuclear University MEPhI (Moscow Engineering Physics Institute)
- Artificial Intelligence Research Center, MEPhI

We thank the creators of the IoT-23 and UNSW-NB15 datasets for making their data publicly available.

---

## Project Status

- [x] Core CT-TGNN implementation
- [x] All baseline implementations
- [x] Dataset loaders and preprocessing
- [x] Training and evaluation pipelines
- [x] Experiment reproduction scripts
- [x] Comprehensive documentation
- [x] Docker support
- [ ] Pre-trained model weights (coming soon)
- [ ] Interactive demo (coming soon)
- [ ] Web API for deployment (coming soon)

---

**Last Updated:** December 2025
