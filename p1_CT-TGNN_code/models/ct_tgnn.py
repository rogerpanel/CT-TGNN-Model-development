"""
Continuous-Time Temporal Graph Neural Network (CT-TGNN)

Main model implementation combining:
- Graph Neural ODEs for continuous-time dynamics
- Temporal Adaptive Batch Normalization for stability
- Multi-scale temporal modeling
- Point process integration for event modeling
- Zero-trust policy enforcement

Author: Roger Nick Anaedevha
Affiliation: National Research Nuclear University MEPhI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torchdiffeq import odeint_adjoint as odeint
from typing import Optional, Tuple, List, Dict
import numpy as np

from .graph_ode import GraphODEFunc, ODEBlock
from .temporal_adaptive_bn import TemporalAdaptiveBatchNorm
from .point_process import TransformerPointProcess


class MultiScaleTemporalEncoding(nn.Module):
    """
    Multi-scale temporal encoding with learned time constants.

    Captures patterns across 8 orders of magnitude:
    - tau_1 = 1e-6 seconds (microsecond timing attacks)
    - tau_2 = 1e-3 seconds (millisecond bursts)
    - tau_3 = 1.0 seconds (typical request-response)
    - tau_4 = 3600 seconds (hour-scale reconnaissance)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_scales: int = 4,
        time_constants: Optional[List[float]] = None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        # Default time constants spanning 8 orders of magnitude
        if time_constants is None:
            time_constants = [1e-6, 1e-3, 1.0, 3600.0]
        self.register_buffer('time_constants', torch.tensor(time_constants))

        # Scale-specific transformations
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_scales)
        ])

        # Attention for scale weighting
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node embeddings [num_nodes, hidden_dim]
            t: Current time scalar

        Returns:
            Multi-scale encoded features [num_nodes, hidden_dim]
        """
        # Compute scale-specific features
        scale_features = []
        for s, (tau, transform) in enumerate(zip(self.time_constants, self.scale_transforms)):
            # Normalize time by time constant
            t_scaled = t / tau
            scale_feat = transform(h) * torch.sin(t_scaled).item()
            scale_features.append(scale_feat)

        scale_features = torch.stack(scale_features, dim=-1)  # [num_nodes, hidden_dim, num_scales]

        # Compute attention weights
        alpha = self.scale_attention(h)  # [num_nodes, num_scales]
        alpha = alpha.unsqueeze(1)  # [num_nodes, 1, num_scales]

        # Weighted combination
        h_multi = (scale_features * alpha).sum(dim=-1)  # [num_nodes, hidden_dim]

        return h_multi


class EncryptedEdgeEncoder(nn.Module):
    """
    Encodes encrypted traffic edge features without payload inspection.

    Extracts features from:
    - Packet timing patterns (inter-arrival times)
    - TLS handshake metadata (cipher suites, versions)
    - Flow statistics (bytes, packets, duration)
    - Directional features (upstream/downstream ratios)
    """

    def __init__(self, edge_feat_dim: int, hidden_dim: int):
        super().__init__()

        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim

        # Timing feature encoder
        self.timing_encoder = nn.Sequential(
            nn.Linear(20, hidden_dim // 4),  # Inter-arrival time statistics
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 4)
        )

        # TLS metadata encoder
        self.tls_encoder = nn.Sequential(
            nn.Linear(30, hidden_dim // 4),  # Cipher suites, versions, cert features
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 4)
        )

        # Flow statistics encoder
        self.flow_encoder = nn.Sequential(
            nn.Linear(25, hidden_dim // 4),  # Size distributions, byte counts
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 4)
        )

        # Directional feature encoder
        self.direction_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim // 4),  # Upstream/downstream ratios
            nn.ELU(),
            nn.LayerNorm(hidden_dim // 4)
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: Raw edge features [num_edges, edge_feat_dim]

        Returns:
            Encoded edge features [num_edges, hidden_dim]
        """
        # Split features into components
        timing_feat = edge_features[:, :20]
        tls_feat = edge_features[:, 20:50]
        flow_feat = edge_features[:, 50:75]
        direction_feat = edge_features[:, 75:87]

        # Encode each component
        timing_enc = self.timing_encoder(timing_feat)
        tls_enc = self.tls_encoder(tls_feat)
        flow_enc = self.flow_encoder(flow_feat)
        direction_enc = self.direction_encoder(direction_feat)

        # Concatenate and fuse
        combined = torch.cat([timing_enc, tls_enc, flow_enc, direction_enc], dim=-1)
        edge_enc = self.fusion(combined)

        return edge_enc


class GraphAttentionAggregation(nn.Module):
    """
    Graph attention mechanism for neighbor aggregation.

    Learns to weight neighbor contributions based on:
    - Node states
    - Edge features
    - Temporal context
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Attention computation
        self.attn_weights = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),  # [h_i || h_j || e_ij]
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(0.2)
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, hidden_dim]

        Returns:
            Aggregated features [num_nodes, hidden_dim]
        """
        row, col = edge_index

        # Compute attention scores
        h_i = h[row]  # Source nodes
        h_j = h[col]  # Target nodes

        # Concatenate node and edge features
        attn_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        alpha = self.attn_weights(attn_input)  # [num_edges, 1]

        # Softmax over neighbors
        alpha = F.softmax(alpha, dim=0)

        # Aggregate
        h_agg = torch.zeros_like(h)
        h_agg.index_add_(0, row, alpha * h_j)

        return h_agg


class CTTGNN(nn.Module):
    """
    Continuous-Time Temporal Graph Neural Network for Encrypted Traffic Analysis.

    Main model combining:
    1. Graph ODE for continuous-time dynamics
    2. Temporal Adaptive Batch Normalization
    3. Multi-scale temporal encoding
    4. Point process integration
    5. Zero-trust policy enforcement

    Architecture:
        Input -> Edge Encoding -> Multi-Scale Encoding -> Graph ODE Blocks
        -> Point Process -> Classification Head -> Output
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 256,
        num_ode_blocks: int = 2,
        num_gnn_layers: int = 2,
        num_scales: int = 4,
        time_constants: Optional[List[float]] = None,
        num_classes: int = 2,
        solver: str = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4,
        use_point_process: bool = True,
        num_event_types: int = 10,
        **kwargs
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_ode_blocks = num_ode_blocks
        self.num_classes = num_classes
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_point_process = use_point_process

        # Input projection
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim)
        )

        # Encrypted edge feature encoder
        self.edge_encoder = EncryptedEdgeEncoder(edge_feat_dim, hidden_dim)

        # Multi-scale temporal encoding
        self.multi_scale = MultiScaleTemporalEncoding(hidden_dim, num_scales, time_constants)

        # Graph ODE blocks
        self.ode_blocks = nn.ModuleList([
            ODEBlock(
                GraphODEFunc(hidden_dim, num_gnn_layers),
                solver=solver,
                rtol=rtol,
                atol=atol
            ) for _ in range(num_ode_blocks)
        ])

        # Point process for event modeling
        if use_point_process:
            self.point_process = TransformerPointProcess(
                hidden_dim=hidden_dim,
                num_layers=4,
                num_heads=8,
                num_event_types=num_event_types,
                max_seq_len=1024
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Threat score for zero-trust
        self.threat_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        timestamps: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CT-TGNN.

        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            timestamps: Event timestamps [num_events]
            batch: Batch assignment for graphs [num_nodes]

        Returns:
            Dictionary containing:
                - logits: Classification logits [num_nodes, num_classes]
                - threat_scores: Threat scores for zero-trust [num_nodes, 1]
                - intensities: Event intensities if using point process [num_nodes, num_event_types]
                - embeddings: Final node embeddings [num_nodes, hidden_dim]
        """
        # Encode inputs
        h = self.node_encoder(x)  # [num_nodes, hidden_dim]
        edge_features = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]

        # Store embeddings for each timestamp
        h_trajectory = []

        # Integrate through time
        t_start = timestamps[0]
        for i, t_end in enumerate(timestamps[1:], 1):
            # Multi-scale temporal encoding
            h = self.multi_scale(h, t_end)

            # Graph ODE integration
            t_span = torch.tensor([t_start.item(), t_end.item()], device=h.device)

            for ode_block in self.ode_blocks:
                h = ode_block(h, edge_index, edge_features, t_span)

            h_trajectory.append(h)
            t_start = t_end

        # Use final embedding
        h_final = h_trajectory[-1] if h_trajectory else h

        # Classification
        logits = self.classifier(h_final)

        # Threat scoring for zero-trust
        threat_scores = self.threat_scorer(h_final)

        outputs = {
            'logits': logits,
            'threat_scores': threat_scores,
            'embeddings': h_final
        }

        # Point process intensities
        if self.use_point_process and len(h_trajectory) > 1:
            h_sequence = torch.stack(h_trajectory, dim=1)  # [num_nodes, seq_len, hidden_dim]
            intensities = self.point_process(h_sequence, timestamps[1:])
            outputs['intensities'] = intensities

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        event_labels: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            outputs: Model outputs from forward()
            labels: Ground truth labels [num_nodes]
            event_labels: Event type labels for point process [num_events]
            loss_weights: Weights for different loss components

        Returns:
            Dictionary of losses
        """
        if loss_weights is None:
            loss_weights = {
                'classification': 1.0,
                'point_process': 0.5,
                'lipschitz_reg': 1e-3
            }

        losses = {}

        # Classification loss
        logits = outputs['logits']
        loss_cls = F.cross_entropy(logits, labels)
        losses['classification'] = loss_cls

        # Point process loss
        if 'intensities' in outputs and event_labels is not None:
            intensities = outputs['intensities']
            loss_pp = self.point_process.compute_loss(intensities, event_labels)
            losses['point_process'] = loss_pp

        # Lipschitz regularization for stability
        loss_lip = 0.0
        for ode_block in self.ode_blocks:
            loss_lip += ode_block.ode_func.lipschitz_penalty()
        losses['lipschitz_reg'] = loss_lip

        # Total weighted loss
        total_loss = sum(loss_weights.get(k, 1.0) * v for k, v in losses.items())
        losses['total'] = total_loss

        return losses

    def predict_future_state(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        current_time: torch.Tensor,
        lookahead_horizon: float
    ) -> torch.Tensor:
        """
        Predict future security state for proactive mitigation.

        Args:
            x: Current node features [num_nodes, node_feat_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            current_time: Current timestamp
            lookahead_horizon: Time to predict ahead (seconds)

        Returns:
            Predicted threat scores [num_nodes, 1]
        """
        # Encode current state
        h = self.node_encoder(x)
        edge_features = self.edge_encoder(edge_attr)

        # Integrate to future time
        t_future = current_time + lookahead_horizon
        t_span = torch.tensor([current_time.item(), t_future.item()], device=h.device)

        # Multi-scale encoding
        h = self.multi_scale(h, t_future)

        # ODE integration
        for ode_block in self.ode_blocks:
            h = ode_block(h, edge_index, edge_features, t_span)

        # Predict threat scores
        future_threat_scores = self.threat_scorer(h)

        return future_threat_scores


if __name__ == '__main__':
    # Test model instantiation
    model = CTTGNN(
        node_feat_dim=128,
        edge_feat_dim=87,
        hidden_dim=256,
        num_ode_blocks=2,
        num_classes=2
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("CT-TGNN model created successfully!")
