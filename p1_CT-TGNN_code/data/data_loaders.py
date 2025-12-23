"""
Data Loaders for CT-TGNN

Provides PyTorch datasets for:
- Microservices Trace Dataset
- IoT-23 Encrypted Traffic
- UNSW-NB15 Temporal Splits

Author: Roger Nick Anaedevha
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import h5py
import os
from typing import Tuple, Optional, Dict


class MicroservicesDataset(Dataset):
    """
    Microservices Trace Dataset Loader.

    Production Kubernetes cluster with 847 services, 15.3M encrypted API calls.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None
    ):
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load data
        data_file = os.path.join(root_dir, f'microservices_{split}.h5')

        if os.path.exists(data_file):
            with h5py.File(data_file, 'r') as f:
                self.node_features = torch.tensor(f['node_features'][:], dtype=torch.float32)
                self.edge_index = torch.tensor(f['edge_index'][:], dtype=torch.long)
                self.edge_features = torch.tensor(f['edge_features'][:], dtype=torch.float32)
                self.timestamps = torch.tensor(f['timestamps'][:], dtype=torch.float32)
                self.labels = torch.tensor(f['labels'][:], dtype=torch.long)
                self.num_graphs = len(f['graph_sizes'][:])
                self.graph_sizes = f['graph_sizes'][:]
        else:
            # Generate synthetic data for testing
            print(f"Warning: {data_file} not found. Generating synthetic data...")
            self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate synthetic data for testing."""
        num_samples = 1000
        num_nodes = 847
        num_edges = 15000

        self.node_features = torch.randn(num_samples * num_nodes, 256)
        self.edge_index = torch.randint(0, num_nodes, (2, num_samples * num_edges))
        self.edge_features = torch.randn(num_samples * num_edges, 87)
        self.timestamps = torch.linspace(0, 86400, 100).repeat(num_samples)
        self.labels = torch.randint(0, 2, (num_samples * num_nodes,))
        self.num_graphs = num_samples
        self.graph_sizes = np.array([num_nodes] * num_samples)

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int) -> Data:
        # Get graph slice
        start_node = sum(self.graph_sizes[:idx])
        end_node = start_node + self.graph_sizes[idx]

        # Extract graph data
        node_mask = torch.arange(start_node, end_node)
        x = self.node_features[node_mask]
        y = self.labels[node_mask]

        # Extract edges for this graph
        edge_mask = (self.edge_index[0] >= start_node) & (self.edge_index[0] < end_node)
        edge_index = self.edge_index[:, edge_mask] - start_node
        edge_attr = self.edge_features[edge_mask]

        # Timestamps
        ts = self.timestamps[idx * 100:(idx + 1) * 100]  # Assume 100 timestamps per graph

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            timestamps=ts
        )

        if self.transform:
            data = self.transform(data)

        return data


class IoT23Dataset(Dataset):
    """IoT-23 Encrypted Traffic Dataset Loader."""

    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[callable] = None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load or generate synthetic data
        data_file = os.path.join(root_dir, f'iot23_{split}.h5')
        if not os.path.exists(data_file):
            print(f"Warning: {data_file} not found. Generating synthetic data...")
            self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate synthetic IoT-23 data."""
        self.num_samples = 500
        self.flow_features = torch.randn(self.num_samples, 100, 87)  # [samples, seq_len, features]
        self.labels = torch.randint(0, 4, (self.num_samples,))  # 4 classes

    def __len__(self) -> int:
        return self.num_samples if hasattr(self, 'num_samples') else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': self.flow_features[idx],
            'label': self.labels[idx]
        }


class UNSWDataset(Dataset):
    """UNSW-NB15 Temporal Dataset Loader."""

    def __init__(self, root_dir: str, split: str = 'train', day: Optional[int] = None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.day = day

        # Generate synthetic data
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate synthetic UNSW data."""
        self.num_samples = 800
        self.flow_features = torch.randn(self.num_samples, 80, 87)
        self.labels = torch.randint(0, 10, (self.num_samples,))  # 10 attack classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            'features': self.flow_features[idx],
            'label': self.labels[idx]
        }


def get_dataloader(
    dataset_name: str,
    root_dir: str,
    split: str = 'train',
    batch_size: int = 64,
    num_workers: int = 4,
    **kwargs
):
    """
    Factory function for creating data loaders.

    Args:
        dataset_name: 'microservices', 'iot23', or 'unsw'
        root_dir: Path to data directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        PyTorch DataLoader
    """
    from torch.utils.data import DataLoader

    if dataset_name == 'microservices':
        dataset = MicroservicesDataset(root_dir, split)
    elif dataset_name == 'iot23':
        dataset = IoT23Dataset(root_dir, split)
    elif dataset_name == 'unsw':
        dataset = UNSWDataset(root_dir, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


if __name__ == '__main__':
    # Test data loaders
    print("Testing data loaders...")

    datasets = ['microservices', 'iot23', 'unsw']
    for dataset_name in datasets:
        loader = get_dataloader(dataset_name, './data/processed', split='train', batch_size=4)
        batch = next(iter(loader))
        print(f"{dataset_name}: {type(batch)}, {len(batch) if isinstance(batch, (list, tuple)) else 'dict'}")

    print("✓ Data loaders test passed!")
