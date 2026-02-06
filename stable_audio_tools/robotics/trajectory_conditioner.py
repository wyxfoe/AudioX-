"""
Trajectory Conditioner module - replaces AudioX's audio encoder.

This module encodes robot proprioceptive state (joint angles) into conditioning
tokens for the Diffusion Transformer.

Key insight: 本体感知编码只需要提供当前状态信息给cross-attention，
它不需要被"解码"回关节角。扩散模型的输出直接就是关节角目标值。

Three encoding modes:
  - "direct": 每个关节一个token，单层线性映射 (最简洁，推荐)
  - "single": 所有关节映射到单个token
  - "mlp": 多层MLP (原始设计，更复杂)
"""

import torch
import torch.nn as nn
import typing as tp
from ..models.conditioners import Conditioner


class TrajectoryConditioner(Conditioner):
    """
    Encodes robot proprioceptive state into conditioning tokens.

    推荐使用 mode="direct"，最简洁且易于理解：
      - 每个关节角直接对应一个conditioning token
      - 单层线性映射: joint_i → Linear → token_i
      - 维度: (B, 16) → (B, 16, 768)

    Args:
        state_dim: dimension of proprioceptive state (e.g., 16 for dual-arm)
        output_dim: dimension of conditioning embeddings (e.g., 768)
        mode: encoding mode
            - "direct": 每个关节一个token，单层Linear (推荐)
            - "single": 所有关节映射到一个token
            - "mlp": 多层MLP，支持自定义token数量
        num_tokens: output token数量 (仅mode="mlp"时使用)
        hidden_dim: MLP隐藏维度 (仅mode="mlp"时使用)
        num_layers: MLP层数 (仅mode="mlp"时使用)
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        mode: str = "direct",  # "direct", "single", or "mlp"
        num_tokens: int = None,  # 仅mlp模式需要
        hidden_dim: int = 512,
        num_layers: int = 3,
        project_out: bool = False,
    ):
        super().__init__(dim=output_dim, output_dim=output_dim, project_out=project_out)

        self.state_dim = state_dim
        self.mode = mode

        if mode == "direct":
            # 最简洁：每个关节一个token，单层线性映射
            # (B, state_dim) → 每个维度单独映射 → (B, state_dim, output_dim)
            self.num_tokens = state_dim
            self.encoder = nn.Linear(1, output_dim)  # 每个关节独立映射

        elif mode == "single":
            # 所有关节映射到单个token
            # (B, state_dim) → Linear → (B, 1, output_dim)
            self.num_tokens = 1
            self.encoder = nn.Linear(state_dim, output_dim)

        elif mode == "mlp":
            # 多层MLP，支持自定义token数
            self.num_tokens = num_tokens if num_tokens is not None else state_dim
            layers = []
            in_dim = state_dim
            for _ in range(num_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ])
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, self.num_tokens * output_dim))
            self.encoder = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # CFG时的空状态embedding
        self.empty_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, output_dim), requires_grad=True
        )
        nn.init.normal_(self.empty_embed, std=0.02)

    def forward(
        self,
        states: tp.Union[torch.Tensor, tp.List[torch.Tensor]],
        device: tp.Union[torch.device, str] = "cuda"
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (B, state_dim) 或 list of (state_dim,) tensors
        Returns:
            embeddings: (B, num_tokens, output_dim)
            mask: (B, num_tokens)
        """
        self.encoder.to(device)
        self.empty_embed.to(device)

        # 处理输入格式
        if isinstance(states, list):
            states_tensor = torch.stack([
                s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
                for s in states
            ]).to(device)
        else:
            states_tensor = states.to(device)

        batch_size = states_tensor.shape[0]

        # Flatten if needed (history dimension)
        if states_tensor.ndim == 3:
            states_tensor = states_tensor.reshape(batch_size, -1)

        # 检测空状态 (用于CFG)
        is_zero = torch.all(states_tensor == 0, dim=-1)

        # 根据模式编码
        if self.mode == "direct":
            # 每个关节独立通过同一个Linear
            # (B, state_dim) → (B, state_dim, 1) → Linear → (B, state_dim, output_dim)
            x = states_tensor.unsqueeze(-1)  # (B, state_dim, 1)
            encoded = self.encoder(x)  # (B, state_dim, output_dim)

        elif self.mode == "single":
            # (B, state_dim) → Linear → (B, output_dim) → (B, 1, output_dim)
            encoded = self.encoder(states_tensor.float())
            encoded = encoded.unsqueeze(1)

        elif self.mode == "mlp":
            # (B, state_dim) → MLP → (B, num_tokens * output_dim) → reshape
            encoded = self.encoder(states_tensor.float())
            encoded = encoded.reshape(batch_size, self.num_tokens, -1)

        # CFG: 空状态替换为learned embedding
        empty = self.empty_embed.expand(batch_size, -1, -1)
        encoded = torch.where(is_zero.view(batch_size, 1, 1), empty, encoded)

        encoded = self.proj_out(encoded)
        mask = torch.ones(batch_size, self.num_tokens, device=device)

        return encoded, mask


class TrajectoryHistoryConditioner(Conditioner):
    """
    编码历史状态序列，使用小型temporal transformer。

    适用于需要历史上下文的场景，例如：
    - 从过去10步状态预测未来动作
    - 需要速度/加速度估计

    Args:
        state_dim: 每步状态维度
        output_dim: 输出embedding维度
        history_length: 历史长度
        num_tokens: 输出token数量
        hidden_dim: 隐藏维度
        num_heads: 注意力头数
        num_layers: transformer层数
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        history_length: int = 10,
        num_tokens: int = 16,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        project_out: bool = False,
    ):
        super().__init__(dim=output_dim, output_dim=output_dim, project_out=project_out)

        self.state_dim = state_dim
        self.history_length = history_length
        self.num_tokens = num_tokens

        # 每步状态的线性映射
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # 时序位置编码
        self.temporal_pos = nn.Parameter(torch.randn(1, history_length, hidden_dim) * 0.02)

        # 小型temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出映射
        self.output_proj = nn.Linear(history_length * hidden_dim, num_tokens * output_dim)

        # CFG空embedding
        self.empty_embed = nn.Parameter(torch.zeros(1, num_tokens, output_dim))
        nn.init.normal_(self.empty_embed, std=0.02)

    def forward(
        self,
        states: tp.Union[torch.Tensor, tp.List[torch.Tensor]],
        device: tp.Union[torch.device, str] = "cuda"
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (B, history_length, state_dim)
        Returns:
            embeddings: (B, num_tokens, output_dim)
            mask: (B, num_tokens)
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

        # 编码每步状态
        h = self.state_proj(states_tensor.float())  # (B, T, hidden)
        h = h + self.temporal_pos[:, :h.shape[1], :]

        # Temporal attention
        h = self.transformer(h)  # (B, T, hidden)

        # 输出映射
        h = h.reshape(batch_size, -1)
        h = self.output_proj(h)
        h = h.reshape(batch_size, self.num_tokens, -1)

        # CFG
        empty = self.empty_embed.expand(batch_size, -1, -1).to(device)
        h = torch.where(is_zero.view(batch_size, 1, 1), empty, h)

        h = self.proj_out(h)
        mask = torch.ones(batch_size, self.num_tokens, device=device)

        return h, mask
