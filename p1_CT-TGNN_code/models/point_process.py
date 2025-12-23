"""
Temporal Point Process Models

Implements transformer-enhanced marked temporal point processes for
modeling discrete security events with continuous-time graph dynamics.

Author: Roger Nick Anaedevha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        return x + self.pe[:x.size(0), :]


class TransformerPointProcess(nn.Module):
    """
    Transformer-based temporal point process.

    Models event intensities through multi-head self-attention over
    historical states from continuous graph dynamics.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        num_event_types: int = 10,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_event_types = num_event_types

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Intensity projection
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_event_types),
            nn.Softplus()  # Ensure positive intensity
        )

    def forward(
        self,
        h_sequence: torch.Tensor,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute event intensities.

        Args:
            h_sequence: Historical states [batch, seq_len, hidden_dim]
            timestamps: Event times [seq_len]
            mask: Attention mask [batch, seq_len]

        Returns:
            Intensities [batch, seq_len, num_event_types]
        """
        # Add positional encoding
        h = self.pos_encoder(h_sequence)

        # Transformer encoding
        h_encoded = self.transformer(h, src_key_padding_mask=mask)

        # Compute intensities
        intensities = self.intensity_head(h_encoded)

        return intensities

    def compute_loss(
        self,
        intensities: torch.Tensor,
        event_types: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            intensities: Predicted intensities [batch, seq_len, num_event_types]
            event_types: Observed event types [batch, seq_len]
            timestamps: Event times [seq_len]

        Returns:
            Negative log-likelihood
        """
        # Log-likelihood at observed events
        batch_size, seq_len, _ = intensities.shape

        # Gather intensities for observed event types
        event_types_expanded = event_types.unsqueeze(-1)
        observed_intensities = torch.gather(intensities, 2, event_types_expanded).squeeze(-1)

        ll_events = torch.log(observed_intensities + 1e-8).sum()

        # Survival integral (compensator)
        dt = timestamps[1:] - timestamps[:-1]
        dt = torch.cat([dt, dt[-1:]])  # Pad last interval

        survival_integral = (intensities.sum(dim=-1) * dt.unsqueeze(0)).sum()

        # Negative log-likelihood
        nll = -ll_events + survival_integral

        return nll / (batch_size * seq_len)


class MarkedPointProcess(nn.Module):
    """Marked point process with Hawkes-like self-excitation."""

    def __init__(self, num_event_types: int, hidden_dim: int):
        super().__init__()

        self.num_event_types = num_event_types
        self.hidden_dim = hidden_dim

        # Base intensity
        self.base_intensity = nn.Parameter(torch.ones(num_event_types) * 0.1)

        # Cross-excitation matrix
        self.alpha = nn.Parameter(torch.randn(num_event_types, num_event_types) * 0.01)
        self.beta = nn.Parameter(torch.ones(num_event_types, num_event_types))

    def forward(self, h: torch.Tensor, event_history: list) -> torch.Tensor:
        """
        Compute intensity given node state and event history.

        Args:
            h: Node embedding [hidden_dim]
            event_history: List of (time, type) tuples

        Returns:
            Intensity for each event type [num_event_types]
        """
        intensity = self.base_intensity.clone()

        current_time = event_history[-1][0] if event_history else 0.0

        for t_event, k_event in event_history[:-1]:
            dt = current_time - t_event
            if dt > 0:
                excitation = self.alpha[k_event, :] * torch.exp(-self.beta[k_event, :] * dt)
                intensity = intensity + excitation

        return F.softplus(intensity)


if __name__ == '__main__':
    # Test point process
    batch_size, seq_len, hidden_dim = 16, 100, 256
    num_event_types = 10

    h_seq = torch.randn(batch_size, seq_len, hidden_dim)
    timestamps = torch.linspace(0, 100, seq_len)
    event_types = torch.randint(0, num_event_types, (batch_size, seq_len))

    tpp = TransformerPointProcess(hidden_dim, num_event_types=num_event_types)

    intensities = tpp(h_seq, timestamps)
    loss = tpp.compute_loss(intensities, event_types, timestamps)

    print(f"Intensities shape: {intensities.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("Point process test passed!")
