"""
CT-TGNN: continuous-time temporal graph neural network.

Cleaned and made runnable from the original implementation. The
architecture is unchanged in substance: multi-scale temporal encoding,
graph-coupled ODE dynamics with Temporal Adaptive Batch Normalisation,
a marked point-process head, and a node classification head.

Falls back to a fixed-step RK4 integrator when torchdiffeq is absent so
the pipeline runs anywhere; the solver actually used is recorded in the
run manifest, because it affects the latency numbers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint_adjoint, odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


# ---------------------------------------------------------------------
# Temporal Adaptive Batch Normalisation
# ---------------------------------------------------------------------
class TABN(nn.Module):
    """Batch norm with time-dependent scale and shift (Eq. 9)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Sequential(nn.Linear(1, 32), nn.Tanh(),
                                   nn.Linear(32, dim))
        self.beta = nn.Sequential(nn.Linear(1, 32), nn.Tanh(),
                                  nn.Linear(32, dim))

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mu = z.mean(0, keepdim=True)
        var = z.var(0, unbiased=False, keepdim=True)
        zn = (z - mu) / torch.sqrt(var + self.eps)
        tt = t.reshape(1, 1).to(z.dtype)
        return zn * (1.0 + self.gamma(tt)) + self.beta(tt)


# ---------------------------------------------------------------------
# Multi-scale temporal encoding
# ---------------------------------------------------------------------
class MultiScaleTemporalEncoding(nn.Module):
    """
    Four parallel branches at fixed time constants, fused by learned
    attention (Eq. 10). Scales are configurable so the sensitivity
    analysis can sweep them.
    """

    def __init__(self, dim: int, taus=(1e-6, 1e-3, 1.0, 3600.0)):
        super().__init__()
        self.register_buffer("taus", torch.tensor(list(taus),
                                                  dtype=torch.float32))
        self.S = len(taus)
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim + 2, dim), nn.ELU(),
                           nn.Linear(dim, dim)) for _ in range(self.S)]
        )
        self.attn = nn.Linear(dim, self.S)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        outs = []
        for s in range(self.S):
            phase = t / self.taus[s]
            enc = torch.stack([torch.sin(phase), torch.cos(phase)])
            enc = enc.reshape(1, 2).expand(h.size(0), 2).to(h.dtype)
            outs.append(self.branches[s](torch.cat([h, enc], dim=-1)))
        stacked = torch.stack(outs, dim=1)                # (N, S, D)
        w = torch.softmax(self.attn(h), dim=-1).unsqueeze(-1)  # (N, S, 1)
        return (stacked * w).sum(dim=1)


# ---------------------------------------------------------------------
# Graph-coupled ODE vector field
# ---------------------------------------------------------------------
class GraphODEFunc(nn.Module):
    """
    dh_i/dt = sigma(W_self h_i + sum_j alpha_ij/|N_i| W_neigh h_j)   (Eq. 2)

    Topology is held FIXED during integration -- this is the whole point
    of the hybrid formulation. Jumps are applied between intervals.
    """

    def __init__(self, dim: int, taus, spectral_norm: bool = True):
        super().__init__()
        lin = (lambda a, b: nn.utils.parametrizations.spectral_norm(
            nn.Linear(a, b)) if spectral_norm else nn.Linear(a, b))
        self.w_self = lin(dim, dim)
        self.w_neigh = lin(dim, dim)
        self.attn = nn.Linear(2 * dim, 1)
        self.mste = MultiScaleTemporalEncoding(dim, taus)
        self.tabn = TABN(dim)
        self.edge_index = None
        self.nfe = 0

    def set_graph(self, edge_index: torch.Tensor):
        self.edge_index = edge_index

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        msg = torch.zeros_like(h)
        ei = self.edge_index
        if ei is not None and ei.numel() > 0:
            s, d = ei[0], ei[1]
            a = torch.sigmoid(self.attn(torch.cat([h[s], h[d]], dim=-1)))
            contrib = a * self.w_neigh(h[s])
            msg = msg.index_add(0, d, contrib)
            deg = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
            deg = deg.index_add(
                0, d, torch.ones(d.size(0), 1, device=h.device, dtype=h.dtype))
            msg = msg / deg.clamp(min=1.0)
        out = F.elu(self.w_self(h) + msg)
        out = self.mste(out, t)
        return self.tabn(out, t)


def rk4(func, h0, t0, t1, steps: int = 4):
    """Fixed-step RK4 fallback."""
    h, dt = h0, (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        k1 = func(t, h)
        k2 = func(t + dt / 2, h + dt * k1 / 2)
        k3 = func(t + dt / 2, h + dt * k2 / 2)
        k4 = func(t + dt, h + dt * k3)
        h = h + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + dt
    return h


# ---------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------
class CTTGNN(nn.Module):
    def __init__(self, edge_feat_dim: int, hidden_dim: int = 128,
                 n_blocks: int = 2, taus=(1e-6, 1e-3, 1.0, 3600.0),
                 dropout: float = 0.1, solver: str = "dopri5",
                 rtol: float = 1e-5, atol: float = 1e-5,
                 use_adjoint: bool = True, spectral_norm: bool = True,
                 point_process: bool = True, rk4_steps: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.rk4_steps = rk4_steps
        self.rtol, self.atol = rtol, atol
        self.use_adjoint = use_adjoint and HAS_TORCHDIFFEQ
        self.use_pp = point_process

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.node_init = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.odefuncs = nn.ModuleList(
            [GraphODEFunc(hidden_dim, taus, spectral_norm)
             for _ in range(n_blocks)]
        )
        self.jump = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.intensity = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1),
            nn.Softplus(),
        ) if point_process else None

    def aggregate_edges(self, n_nodes, edge_index, edge_attr, device):
        """Scatter encoded edge features onto their endpoint nodes."""
        h = torch.zeros(n_nodes, self.hidden_dim, device=device)
        if edge_index.numel() == 0:
            return h
        e = self.edge_encoder(edge_attr)
        h = h.index_add(0, edge_index[1], e)
        h = h.index_add(0, edge_index[0], e)
        cnt = torch.zeros(n_nodes, 1, device=device)
        ones = torch.ones(edge_index.size(1), 1, device=device)
        cnt = cnt.index_add(0, edge_index[1], ones)
        cnt = cnt.index_add(0, edge_index[0], ones)
        return h / cnt.clamp(min=1.0)

    def integrate(self, func, h, t0, t1):
        if t1 <= t0:
            return h
        # 'rk4' selects the internal fixed-step integrator: far cheaper on
        # CPU and used for smoke tests. The solver actually used is recorded
        # in the run manifest, since it affects both accuracy and latency.
        if self.solver == "rk4":
            return rk4(func, h, torch.tensor(t0, device=h.device),
                       torch.tensor(t1, device=h.device),
                       steps=self.rk4_steps)
        if self.use_adjoint or HAS_TORCHDIFFEQ:
            tt = torch.tensor([t0, t1], device=h.device, dtype=h.dtype)
            fn = odeint_adjoint if self.use_adjoint else odeint
            out = fn(func, h, tt, method=self.solver,
                     rtol=self.rtol, atol=self.atol)
            return out[-1]
        return rk4(func, h, torch.tensor(t0, device=h.device),
                   torch.tensor(t1, device=h.device))

    def forward(self, n_nodes, edge_index, edge_attr, t_span=(0.0, 1.0),
                apply_jump: bool = True, **unused):
        device = edge_attr.device if edge_attr.numel() else next(
            self.parameters()).device
        h = self.node_init(
            self.aggregate_edges(n_nodes, edge_index, edge_attr, device))

        total_nfe = 0
        for func in self.odefuncs:
            func.set_graph(edge_index)
            func.nfe = 0
            h = h + self.integrate(func, h, float(t_span[0]), float(t_span[1]))
            total_nfe += func.nfe
        if apply_jump:
            h = h + self.jump(h)

        logits = self.classifier(h).squeeze(-1)
        lam = (self.intensity(h).squeeze(-1) if self.use_pp else None)
        return {"logits": logits, "embedding": h, "intensity": lam,
                "nfe": total_nfe}

    def loss(self, out, y, lambda_event: float = 0.1,
             pos_weight: torch.Tensor | None = None):
        l_node = F.binary_cross_entropy_with_logits(
            out["logits"], y.float(), pos_weight=pos_weight)
        total = l_node
        parts = {"node": float(l_node.detach())}
        if out.get("intensity") is not None:
            lam = out["intensity"].clamp(min=1e-6)
            l_evt = (lam - torch.log(lam) * y.float()).mean()
            total = total + lambda_event * l_evt
            parts["event"] = float(l_evt.detach())
        parts["total"] = float(total.detach())
        return total, parts
