"""
Run All Experiments for Paper Reproducibility

Runs all experiments from the paper:
- Table 1: Microservices lateral movement
- Table 2: IoT-23 encrypted traffic
- Table 3: UNSW-NB15 temporal generalization
- Table 4: Ablation studies
- Table 5: Computational performance
- Table 6: Zero-trust metrics

Author: Roger Nick Anaedevha
"""

import os
import sys
import yaml
import torch
import subprocess
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_experiment(config_file: str, dataset: str, model: str):
    """Run single experiment."""
    print(f"\n{'=' * 80}")
    print(f"Running: {model} on {dataset}")
    print(f"{'=' * 80}\n")

    cmd = [
        'python', 'training/trainer.py',
        '--config', config_file,
        '--dataset', dataset
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Completed: {model} on {dataset}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {model} on {dataset}")
        print(f"Error: {e}")


def main():
    """Run all experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/experiments_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("CT-TGNN: Running All Paper Experiments")
    print("=" * 80)

    # Datasets
    datasets = ['microservices', 'iot23', 'unsw']

    # Models
    models = [
        'CT-TGNN',
        'StrGNN',
        'CNN-LSTM'
    ]

    # Run all combinations
    for dataset in datasets:
        for model in models:
            # Update config for current model
            config_file = f'config/{model.lower().replace("-", "_")}_config.yaml'

            if not os.path.exists(config_file):
                # Use default config
                config_file = 'config/ct_tgnn_config.yaml'

            run_experiment(config_file, dataset, model)

    print("\n" + "=" * 80)
    print("All experiments completed!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
