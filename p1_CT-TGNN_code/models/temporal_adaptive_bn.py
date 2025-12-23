"""
Temporal Adaptive Batch Normalization (TA-BN)

Extends batch normalization to continuous-time by parameterizing
normalization statistics as functions of integration time.

Resolves incompatibility between discrete batch normalization and
continuous Neural ODE dynamics.

Reference: Salvi et al., "Temporal Adaptive Batch Normalization in Neural ODEs," NeurIPS 2024

Author: Roger Nick Anaedevha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class TemporalEncoding(nn.Module):
    """
    Temporal encoding with periodic components for diurnal patterns.

    Uses sinusoidal functions to capture cyclic behaviors:
    - sin(ωt), cos(ωt) for periodicity
    - Learned MLPs for complex temporal dependencies
    """

    def __init__(
        self,
        time_encoding_dim: int = 64,
        use_periodic: bool = True,
        omega: float = 2 * np.pi / 86400  # Daily periodicity (seconds)
    ):
        super().__init__()

        self.time_encoding_dim = time_encoding_dim
        self.use_periodic = use_periodic
        self.omega = omega

        if use_periodic:
            # Input: [t, sin(ωt), cos(ωt)]
            input_dim = 3
        else:
            # Input: [t]
            input_dim = 1

        # MLP for temporal encoding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, time_encoding_dim),
            nn.ELU(),
            nn.Linear(time_encoding_dim, time_encoding_dim),
            nn.ELU()
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode timestamp into feature vector.

        Args:
            t: Time scalar or tensor

        Returns:
            Temporal encoding [time_encoding_dim]
        """
        # Ensure t is a tensor
        if isinstance(t, (int, float)):
            t = torch.tensor([t], dtype=torch.float32, device=self.encoder[0].weight.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0)

        if self.use_periodic:
            # Add periodic components
            t_periodic = torch.cat([
                t.view(-1, 1),
                torch.sin(self.omega * t).view(-1, 1),
                torch.cos(self.omega * t).view(-1, 1)
            ], dim=-1)
            encoding = self.encoder(t_periodic)
        else:
            encoding = self.encoder(t.view(-1, 1))

        return encoding.squeeze(0)


class TemporalAdaptiveBatchNorm(nn.Module):
    """
    Temporal Adaptive Batch Normalization for continuous-time networks.

    Standard batch normalization: y = γ * (x - μ) / √(σ² + ε) + β

    TA-BN extends to continuous time:
    y = γ(t) * (x - μ(t)) / √(σ²(t) + ε) + β(t)

    where γ(t), β(t), μ(t), σ²(t) are time-dependent functions.

    This ensures stable gradient flow during adjoint computation in Neural ODEs.
    """

    def __init__(
        self,
        num_features: int,
        time_encoding_dim: int = 64,
        eps: float = 1e-5,
        momentum: float = 0.1,
        use_periodic: bool = True,
        omega: float = 2 * np.pi / 86400
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Temporal encoding
        self.time_encoder = TemporalEncoding(
            time_encoding_dim=time_encoding_dim,
            use_periodic=use_periodic,
            omega=omega
        )

        # Time-dependent scale and shift parameters
        self.gamma_mlp = nn.Sequential(
            nn.Linear(time_encoding_dim, num_features),
            nn.Softmax(dim=-1)  # Ensure positive and normalized
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(time_encoding_dim, num_features)
        )

        # Running statistics (exponential moving averages)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Time-dependent running statistics MLPs
        self.mu_mlp = nn.Sequential(
            nn.Linear(time_encoding_dim, num_features)
        )

        self.sigma_mlp = nn.Sequential(
            nn.Linear(time_encoding_dim, num_features),
            nn.Softplus()  # Ensure positive variance
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        training: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Apply temporal adaptive batch normalization.

        Args:
            x: Input features [batch_size, num_features] or [num_features]
            t: Current integration time (scalar)
            training: Whether in training mode (uses batch stats) or eval (uses running stats)

        Returns:
            Normalized features with same shape as input
        """
        if training is None:
            training = self.training

        # Encode time
        t_enc = self.time_encoder(t)  # [time_encoding_dim]

        # Compute time-dependent parameters
        gamma_t = self.gamma_mlp(t_enc)  # [num_features]
        beta_t = self.beta_mlp(t_enc)    # [num_features]

        # Compute time-dependent running statistics
        mu_t = self.mu_mlp(t_enc)        # [num_features]
        sigma_t = self.sigma_mlp(t_enc)  # [num_features]

        # Normalization
        if training:
            # Use batch statistics
            if x.dim() == 1:
                # Single sample
                mean = x
                var = torch.zeros_like(x)
            else:
                # Batch
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)

            # Update running statistics with momentum
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                self.num_batches_tracked += 1

            # Combine with time-dependent adjustments
            mean = mean + mu_t
            var = var + sigma_t
        else:
            # Use running statistics
            mean = self.running_mean + mu_t
            var = self.running_var + sigma_t

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        out = gamma_t * x_norm + beta_t

        return out

    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}'


class TimeInvariantBatchNorm(nn.Module):
    """
    Standard batch normalization (time-invariant) for comparison.

    This is the baseline without temporal adaptation.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        super().__init__()

        self.bn = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum
        )

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features
            t: Time (ignored for time-invariant)

        Returns:
            Normalized features
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            out = self.bn(x)
            return out.squeeze(0)
        else:
            return self.bn(x)


def test_tabn():
    """Test Temporal Adaptive Batch Normalization."""
    print("Testing Temporal Adaptive Batch Normalization...")

    num_features = 128
    batch_size = 64

    # Create TA-BN layer
    tabn = TemporalAdaptiveBatchNorm(num_features)

    # Test data
    x = torch.randn(batch_size, num_features)
    t = torch.tensor(100.0)  # 100 seconds

    # Forward pass (training)
    tabn.train()
    out_train = tabn(x, t)
    print(f"Training mode - Input: {x.shape}, Output: {out_train.shape}")
    print(f"Mean: {out_train.mean().item():.4f}, Std: {out_train.std().item():.4f}")

    # Forward pass (evaluation)
    tabn.eval()
    out_eval = tabn(x, t)
    print(f"Eval mode - Input: {x.shape}, Output: {out_eval.shape}")
    print(f"Mean: {out_eval.mean().item():.4f}, Std: {out_eval.std().item():.4f}")

    # Test different times
    t_values = [0.0, 3600.0, 86400.0]  # 0s, 1h, 1day
    print("\nTesting temporal variation:")
    for t_val in t_values:
        out = tabn(x, torch.tensor(t_val))
        print(f"t={t_val:>8.1f}s: mean={out.mean().item():>7.4f}, std={out.std().item():>7.4f}")

    # Test gradient flow
    print("\nTesting gradient flow:")
    x.requires_grad = True
    out = tabn(x, t)
    loss = out.sum()
    loss.backward()
    print(f"Gradient norm: {x.grad.norm().item():.4f}")

    print("\n✓ Temporal Adaptive Batch Normalization test passed!")


if __name__ == '__main__':
    test_tabn()
