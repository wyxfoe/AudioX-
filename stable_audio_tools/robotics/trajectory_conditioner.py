"""
Trajectory Conditioner module - replaces AudioX's audio encoder.

This module encodes robot proprioceptive state (joint angles) into conditioning
tokens for the Diffusion Transformer. It follows the design philosophy of
RDT-1B's action encoder: a lightweight MLP that projects physical quantities
into the transformer's embedding space.

Architecture:
  joint_angles (B, state_dim) -> MLP -> (B, num_tokens, embed_dim)

The proprioceptive state includes current joint positions and optionally
velocities for all arms.
"""

import torch
import torch.nn as nn
import typing as tp
from ..models.conditioners import Conditioner


class TrajectoryConditioner(Conditioner):
    """
    Encodes robot proprioceptive state (joint angles, gripper state) into
    conditioning embeddings for the diffusion transformer.

    Replaces the audio encoder (CLAP/PretransformConditioner) from AudioX.
    Following RDT-1B, this uses a simple MLP to encode the low-dimensional
    physical state into high-dimensional embeddings.

    Args:
        state_dim: dimension of the proprioceptive state vector
        output_dim: dimension of the output conditioning embeddings
        num_tokens: number of output tokens (sequence length of conditioning)
        hidden_dim: hidden dimension of the MLP encoder
        num_layers: number of MLP layers
        include_velocity: whether state includes velocity information
        history_length: number of past states to encode (1 = current only)
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        num_tokens: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 3,
        include_velocity: bool = False,
        history_length: int = 1,
        project_out: bool = False,
    ):
        input_dim = state_dim * (2 if include_velocity else 1) * history_length
        super().__init__(dim=output_dim, output_dim=output_dim, project_out=project_out)

        self.state_dim = state_dim
        self.num_tokens = num_tokens
        self.include_velocity = include_velocity
        self.history_length = history_length
        self.input_dim = input_dim

        # Build MLP encoder (following RDT's action encoder design)
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim

        # Final projection to num_tokens * output_dim
        layers.append(nn.Linear(hidden_dim, num_tokens * output_dim))
        self.encoder = nn.Sequential(*layers)

        # Learnable empty state embedding (for CFG dropout)
        self.empty_state_embed = nn.Parameter(
            torch.zeros(1, num_tokens, output_dim), requires_grad=True
        )
        nn.init.normal_(self.empty_state_embed, std=0.02)

    def forward(
        self,
        states: tp.Union[torch.Tensor, tp.List[torch.Tensor]],
        device: tp.Union[torch.device, str] = "cuda"
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode proprioceptive states into conditioning embeddings.

        Args:
            states: list of state tensors (one per batch element), each of
                    shape (state_dim,) or (history_length, state_dim)
                    OR a single tensor of shape (B, state_dim) or (B, history, state_dim)
            device: target device

        Returns:
            embeddings: (B, num_tokens, output_dim)
            mask: (B, num_tokens) attention mask (ones)
        """
        self.encoder.to(device)

        if isinstance(states, list):
            # Stack list of tensors into batch
            states_tensor = torch.stack([
                s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
                for s in states
            ]).to(device)
        else:
            states_tensor = states.to(device)

        batch_size = states_tensor.shape[0]

        # Detect zero/empty states for CFG
        is_zero = torch.all(states_tensor == 0, dim=tuple(range(1, states_tensor.ndim)))

        # Flatten history dimension if present
        if states_tensor.ndim == 3:
            states_tensor = states_tensor.reshape(batch_size, -1)  # (B, history * state_dim)
        elif states_tensor.ndim == 2:
            pass  # Already (B, state_dim)
        else:
            raise ValueError(f"Unexpected state shape: {states_tensor.shape}")

        # Encode through MLP
        encoded = self.encoder(states_tensor.float())  # (B, num_tokens * output_dim)
        encoded = encoded.reshape(batch_size, self.num_tokens, -1)  # (B, num_tokens, output_dim)

        # Replace zero states with learned empty embedding
        empty_embed = self.empty_state_embed.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1)
        encoded = torch.where(is_zero_expanded, empty_embed, encoded)

        # Project output
        encoded = self.proj_out(encoded)

        mask = torch.ones(batch_size, self.num_tokens, device=device)

        return encoded, mask


class TrajectoryHistoryConditioner(Conditioner):
    """
    Encodes a sequence of past robot states using a small temporal transformer.
    This is analogous to AudioX's CLIPConditioner which uses a temporal
    transformer over video frames.

    Args:
        state_dim: dimension of per-timestep state
        output_dim: output conditioning dimension
        history_length: number of past timesteps
        num_tokens: number of output conditioning tokens
        hidden_dim: hidden dimension
        num_heads: number of attention heads in temporal transformer
        num_layers: number of transformer layers
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        history_length: int = 10,
        num_tokens: int = 32,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        project_out: bool = False,
    ):
        super().__init__(dim=output_dim, output_dim=output_dim, project_out=project_out)

        self.state_dim = state_dim
        self.history_length = history_length
        self.num_tokens = num_tokens

        # Per-timestep state encoder
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Positional embedding for temporal ordering
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, history_length, hidden_dim) * 0.02
        )

        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Project to output tokens
        self.output_proj = nn.Linear(history_length * hidden_dim, num_tokens * output_dim)

        # Empty embedding for CFG
        self.empty_embed = nn.Parameter(
            torch.zeros(1, num_tokens, output_dim), requires_grad=True
        )
        nn.init.normal_(self.empty_embed, std=0.02)

    def forward(
        self,
        states: tp.Union[torch.Tensor, tp.List[torch.Tensor]],
        device: tp.Union[torch.device, str] = "cuda"
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (B, history_length, state_dim) or list of such tensors
        """
        if isinstance(states, list):
            states_tensor = torch.stack([
                s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
                for s in states
            ]).to(device)
        else:
            states_tensor = states.to(device)

        batch_size = states_tensor.shape[0]
        is_zero = torch.all(states_tensor == 0, dim=(1, 2))

        # Project each timestep
        h = self.state_proj(states_tensor.float())  # (B, T, hidden)
        h = h + self.temporal_pos_embed[:, :h.shape[1], :]

        # Temporal attention
        h = self.temporal_transformer(h)  # (B, T, hidden)

        # Flatten and project to output tokens
        h = h.reshape(batch_size, -1)
        h = self.output_proj(h)  # (B, num_tokens * output_dim)
        h = h.reshape(batch_size, self.num_tokens, -1)

        # Replace zero inputs with empty embedding
        empty = self.empty_embed.expand(batch_size, -1, -1)
        h = torch.where(is_zero.view(batch_size, 1, 1), empty, h)

        h = self.proj_out(h)
        mask = torch.ones(batch_size, self.num_tokens, device=device)

        return h, mask
