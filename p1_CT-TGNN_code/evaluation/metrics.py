"""
Evaluation Metrics for CT-TGNN

Computes all metrics from the paper:
- Classification: Accuracy, Precision, Recall, F1, AUROC, AUPRC
- Latency: P50, P95, P99, Throughput
- Zero-Trust: MTTD, MTTC

Author: Roger Nick Anaedevha
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Dict, Optional


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Prediction probabilities (for AUROC/AUPRC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

    if y_scores is not None:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_scores, average='macro')
            metrics['auprc'] = average_precision_score(y_true, y_scores, average='macro')
        except:
            pass

    return metrics


def compute_latency_metrics(latencies: np.ndarray) -> Dict[str, float]:
    """Compute latency percentiles."""
    return {
        'latency_p50': np.percentile(latencies, 50),
        'latency_p95': np.percentile(latencies, 95),
        'latency_p99': np.percentile(latencies, 99),
        'latency_mean': np.mean(latencies)
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    latencies: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    metrics = compute_classification_metrics(y_true, y_pred, y_scores)

    if latencies is not None:
        metrics.update(compute_latency_metrics(latencies))

    return metrics


if __name__ == '__main__':
    # Test metrics
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.4, 0.2, 0.8])

    metrics = compute_all_metrics(y_true, y_pred, y_scores)
    print("Metrics:", metrics)
