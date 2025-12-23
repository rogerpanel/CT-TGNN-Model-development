"""
Graph Neural Ordinary Differential Equations

Implements continuous-time dynamics on graph-structured data with:
- Graph-coupled ODE vector fields
- Temporal Adaptive Batch Normalization
- Lipschitz regularization for gradient stability
- Adjoint-based training for memory efficiency

Author: Roger Nick Anaedevha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torchdiffeq import odeint_adjoint as odeint
from typing import Optional, Tuple

from .temporal_adaptive_bn import TemporalAdaptiveBatchNorm


class GraphConvolution(MessagePassing):
    """
    Graph convolutional layer for continuous-time dynamics.

    Uses message passing with learned transformations.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')  # Sum aggregation

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Learnable weight matrices
        self.W_self = nn.Linear(in_channels, out_channels, bias=False)
        self.W_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.W_neigh.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Add self-loops
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=x.size(0)
        )

        # Normalize by degree
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Self-transformation
        out_self = self.W_self(x)

        # Message passing
        out_neigh = self.propagate(
            edge_index, x=x, norm=norm, edge_weight=edge_weight
        )

        return out_self + out_neigh + self.bias

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """
        Construct messages from neighbors.

        Args:
            x_j: Neighbor node features [num_edges, in_channels]
            norm: Normalization coefficients [num_edges]

        Returns:
            Messages [num_edges, in_channels]
        """
        return norm.view(-1, 1) * self.W_neigh(x_j)


class GraphODEFunc(nn.Module):
    """
    ODE function for graph neural networks.

    Defines the vector field: dh/dt = f(h, A, t)

    where h are node features, A is adjacency, t is time.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        use_tabn: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_tabn = use_tabn

        # Graph convolution layers
        self.graph_convs = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Temporal Adaptive Batch Normalization
        if use_tabn:
            self.tabn_layers = nn.ModuleList([
                TemporalAdaptiveBatchNorm(hidden_dim)
                for _ in range(num_layers)
            ])

        # Activation
        self.activation = nn.ELU()

        # Store graph structure
        self.edge_index = None
        self.edge_attr = None

    def set_graph(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ):
        """Set the graph structure for ODE integration."""
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt at time t.

        Args:
            t: Current time (scalar tensor)
            h: Node features [num_nodes, hidden_dim]

        Returns:
            Time derivative dh/dt [num_nodes, hidden_dim]
        """
        assert self.edge_index is not None, "Must call set_graph() before forward()"

        # Graph convolutions with temporal adaptation
        for i, conv in enumerate(self.graph_convs):
            h = conv(h, self.edge_index)

            if self.use_tabn:
                h = self.tabn_layers[i](h, t)

            if i < len(self.graph_convs) - 1:
                h = self.activation(h)

        return h

    def lipschitz_penalty(self) -> torch.Tensor:
        """
        Compute Lipschitz regularization to ensure gradient stability.

        Penalizes large weight norms to bound Lipschitz constant.
        """
        penalty = 0.0

        for conv in self.graph_convs:
            penalty += torch.norm(conv.W_self.weight, p='fro') ** 2
            penalty += torch.norm(conv.W_neigh.weight, p='fro') ** 2

        return penalty


class ODEBlock(nn.Module):
    """
    ODE block with adjoint-based training.

    Integrates graph dynamics from t0 to t1 using adaptive ODE solvers.
    """

    def __init__(
        self,
        ode_func: GraphODEFunc,
        solver: str = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = True
    ):
        super().__init__()

        self.ode_func = ode_func
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint

        # ODE integration options
        self.options = {
            'method': solver,
            'rtol': rtol,
            'atol': atol
        }

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate graph dynamics from t_span[0] to t_span[1].

        Args:
            h: Initial node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            t_span: Integration time span [t_start, t_end]

        Returns:
            Final node features [num_nodes, hidden_dim]
        """
        # Set graph structure
        self.ode_func.set_graph(edge_index, edge_attr)

        # Integrate
        if self.adjoint:
            h_out = odeint(
                self.ode_func,
                h,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver
            )
        else:
            from torchdiffeq import odeint as odeint_standard
            h_out = odeint_standard(
                self.ode_func,
                h,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver
            )

        # Return final state
        return h_out[-1]

    def trajectory(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """
        Get full ODE trajectory (for visualization/analysis).

        Returns:
            Trajectory [num_times, num_nodes, hidden_dim]
        """
        self.ode_func.set_graph(edge_index, edge_attr)

        h_trajectory = odeint(
            self.ode_func,
            h,
            t_span,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver
        )

        return h_trajectory


class ContinuousGraphDynamics(nn.Module):
    """
    Wrapper for continuous graph dynamics with multiple ODE blocks.

    Stacks multiple ODE blocks for deeper continuous networks.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int = 2,
        num_gnn_layers: int = 2,
        solver: str = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Stack ODE blocks
        self.ode_blocks = nn.ModuleList([
            ODEBlock(
                GraphODEFunc(hidden_dim, num_gnn_layers),
                solver=solver,
                rtol=rtol,
                atol=atol
            ) for _ in range(num_blocks)
        ])

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """
        Pass through multiple ODE blocks sequentially.

        Args:
            h: Initial node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            t_span: Integration time span [t_start, t_end]

        Returns:
            Final node features [num_nodes, hidden_dim]
        """
        for ode_block in self.ode_blocks:
            h = ode_block(h, edge_index, edge_attr, t_span)

        return h

    def compute_lipschitz_penalty(self) -> torch.Tensor:
        """Compute total Lipschitz penalty across all blocks."""
        penalty = 0.0
        for ode_block in self.ode_blocks:
            penalty += ode_block.ode_func.lipschitz_penalty()
        return penalty


if __name__ == '__main__':
    # Test Graph ODE
    hidden_dim = 64
    num_nodes = 100
    num_edges = 500

    # Create test data
    h = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    t_span = torch.tensor([0.0, 1.0])

    # Create model
    ode_func = GraphODEFunc(hidden_dim)
    ode_block = ODEBlock(ode_func)

    # Forward pass
    h_out = ode_block(h, edge_index, None, t_span)

    print(f"Input shape: {h.shape}")
    print(f"Output shape: {h_out.shape}")
    print(f"Lipschitz penalty: {ode_func.lipschitz_penalty().item():.4f}")
    print("Graph ODE test passed!")
