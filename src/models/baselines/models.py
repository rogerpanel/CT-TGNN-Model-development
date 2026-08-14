"""
Baselines. Every one of these is a real, trainable implementation
evaluated under the SAME splits, features, and budget as CT-TGNN.

Included:
  EdgeBank        memorisation-only heuristic; NO learning. If a learned
                  model cannot beat this, the learning is doing nothing.
                  (Poursafaei et al., NeurIPS 2022)
  MLPFlow         per-flow MLP, no topology. Isolates the value of graph
                  structure.
  DiscreteGNN     snapshot GNN (StrGNN-style). Isolates the value of
                  continuous time.
  TemporalAttn    TGAT-style time-encoded attention.
  CNNLSTM         sequence baseline over flow features.

For the full TGN / TGAT / DyRep / JODIE / GraphMixer suite, run DyGLib
against the exported event stream; see scripts/export_dyglib.py. Using a
single harness is what makes "identical splits, features, hardware, and
tuning budget" true by construction rather than by assertion.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeBank:
    """
    Pure memorisation: a node is flagged if it appeared in a malicious
    edge during training (within an optional recency window). Not a
    neural model and has no parameters -- that is the point.
    """

    def __init__(self, window: float | None = None):
        self.window = window
        self.malicious_nodes: dict[int, float] = {}

    def fit(self, g):
        mal = g.flow_labels == 1
        t = g.interaction_times
        for k in np.flatnonzero(mal):
            for v in (int(g.interaction_src[k]), int(g.interaction_dst[k])):
                self.malicious_nodes[v] = max(
                    self.malicious_nodes.get(v, -np.inf), float(t[k]))
        return self

    def predict_nodes(self, node_idx: np.ndarray, t_now: float) -> np.ndarray:
        out = np.zeros(len(node_idx), dtype=np.float32)
        for i, v in enumerate(node_idx):
            last = self.malicious_nodes.get(int(v))
            if last is None:
                continue
            if self.window is None or (t_now - last) <= self.window:
                out[i] = 1.0
        return out


class MLPFlow(nn.Module):
    """Per-flow MLP aggregated to nodes. No topology, no time."""

    def __init__(self, edge_feat_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enc = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, n_nodes, edge_index, edge_attr, **kw):
        device = next(self.parameters()).device
        h = torch.zeros(n_nodes, self.hidden_dim, device=device)
        if edge_index.numel():
            e = self.enc(edge_attr)
            h = h.index_add(0, edge_index[1], e)
            h = h.index_add(0, edge_index[0], e)
            c = torch.zeros(n_nodes, 1, device=device)
            o = torch.ones(edge_index.size(1), 1, device=device)
            c = c.index_add(0, edge_index[1], o).index_add(0, edge_index[0], o)
            h = h / c.clamp(min=1.0)
        return {"logits": self.head(h).squeeze(-1), "embedding": h,
                "intensity": None, "nfe": 0}


class DiscreteGNN(nn.Module):
    """Snapshot message passing (StrGNN-style). Discrete time."""

    def __init__(self, edge_feat_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, n_nodes, edge_index, edge_attr, **kw):
        device = next(self.parameters()).device
        h = torch.zeros(n_nodes, self.hidden_dim, device=device)
        if edge_index.numel():
            e = self.edge_enc(edge_attr)
            h = h.index_add(0, edge_index[1], e)
            c = torch.zeros(n_nodes, 1, device=device)
            o = torch.ones(edge_index.size(1), 1, device=device)
            c = c.index_add(0, edge_index[1], o)
            h = h / c.clamp(min=1.0)
        for lin in self.layers:
            agg = torch.zeros_like(h)
            if edge_index.numel():
                agg = agg.index_add(0, edge_index[1], h[edge_index[0]])
                c = torch.zeros(n_nodes, 1, device=device)
                o = torch.ones(edge_index.size(1), 1, device=device)
                c = c.index_add(0, edge_index[1], o)
                agg = agg / c.clamp(min=1.0)
            h = self.drop(F.relu(lin(torch.cat([h, agg], dim=-1))))
        return {"logits": self.head(h).squeeze(-1), "embedding": h,
                "intensity": None, "nfe": 0}


class TimeEncoder(nn.Module):
    """Bochner time encoding (TGAT)."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(1, dim)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.w(dt.unsqueeze(-1)))


class TemporalAttn(nn.Module):
    """TGAT-style attention with time encoding."""

    def __init__(self, edge_feat_dim: int, hidden_dim: int = 128,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU())
        self.time_enc = TimeEncoder(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, n_nodes, edge_index, edge_attr, edge_time=None, **kw):
        device = next(self.parameters()).device
        h = torch.zeros(n_nodes, self.hidden_dim, device=device)
        if edge_index.numel():
            e = self.edge_enc(edge_attr)
            if edge_time is not None:
                e = e + self.time_enc(edge_time)
            h = h.index_add(0, edge_index[1], e)
            c = torch.zeros(n_nodes, 1, device=device)
            o = torch.ones(edge_index.size(1), 1, device=device)
            c = c.index_add(0, edge_index[1], o)
            h = h / c.clamp(min=1.0)
        a, _ = self.attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        h = h + a.squeeze(0)
        return {"logits": self.head(h).squeeze(-1), "embedding": h,
                "intensity": None, "nfe": 0}


class CNNLSTM(nn.Module):
    """1D CNN + LSTM over per-node edge-feature sequences."""

    def __init__(self, edge_feat_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(edge_feat_dim, hidden_dim, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, n_nodes, edge_index, edge_attr, **kw):
        device = next(self.parameters()).device
        h = torch.zeros(n_nodes, self.hidden_dim, device=device)
        if edge_index.numel():
            x = edge_attr.t().unsqueeze(0)               # (1, F, E)
            c = self.conv(x).permute(0, 2, 1)            # (1, E, H)
            o, _ = self.lstm(c)
            o = o.squeeze(0)
            h = h.index_add(0, edge_index[1], o)
            cnt = torch.zeros(n_nodes, 1, device=device)
            ones = torch.ones(edge_index.size(1), 1, device=device)
            cnt = cnt.index_add(0, edge_index[1], ones)
            h = h / cnt.clamp(min=1.0)
        return {"logits": self.head(h).squeeze(-1), "embedding": h,
                "intensity": None, "nfe": 0}


REGISTRY = {
    "mlp_flow": MLPFlow,
    "discrete_gnn": DiscreteGNN,
    "temporal_attn": TemporalAttn,
    "cnn_lstm": CNNLSTM,
}


def build_baseline(name: str, edge_feat_dim: int, **kw) -> nn.Module:
    if name not in REGISTRY:
        raise ValueError(f"unknown baseline '{name}'. "
                         f"available: {list(REGISTRY)}")
    return REGISTRY[name](edge_feat_dim=edge_feat_dim, **kw)
