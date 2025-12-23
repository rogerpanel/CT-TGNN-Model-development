"""
Structural Temporal Graph Neural Network (StrGNN)

Baseline: Discrete temporal snapshots with GNN + GRU

Reference: NEC Labs, 2024
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class StrGNN(nn.Module):
    """Structural Temporal GNN with discrete snapshots."""

    def __init__(self, node_feat_dim, hidden_dim=256, num_classes=2):
        super().__init__()

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, timestamps, batch=None):
        h = self.node_encoder(x)
        h = torch.relu(self.gcn1(h, edge_index))
        h = torch.relu(self.gcn2(h, edge_index))

        # Reshape for GRU
        h = h.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        h, _ = self.gru(h)
        h = h.squeeze(0)

        logits = self.classifier(h)
        return {'logits': logits, 'embeddings': h}
