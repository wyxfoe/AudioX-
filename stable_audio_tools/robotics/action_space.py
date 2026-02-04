"""
Unified Action Space utilities following RDT-1B's design.

RDT uses a physically interpretable unified action space with reserved slots
for different physical quantities. This module implements that concept adapted
for RoboTwin's dual-arm joint angle control.

Action vector layout (per arm, 7-DoF + 1 gripper = 8 dims):
  [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]

For dual-arm (RoboTwin default):
  Left arm:  slots [0:8]   = 7 joints + 1 gripper
  Right arm: slots [8:16]  = 7 joints + 1 gripper
  Total: 16 dimensions per timestep

For single-arm:
  Right arm only: slots [0:8] = 7 joints + 1 gripper
  Total: 8 dimensions per timestep
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class ActionSpaceConfig:
    """Configuration for the unified action space."""
    num_arms: int = 2                # 1 for single-arm, 2 for dual-arm
    joints_per_arm: int = 7          # Number of joint angles per arm
    gripper_dim: int = 1             # Gripper open/close dimension
    action_chunk_size: int = 64      # Number of future actions to predict (following RDT)
    control_frequency: int = 10     # Control frequency in Hz

    @property
    def action_dim_per_arm(self) -> int:
        return self.joints_per_arm + self.gripper_dim

    @property
    def total_action_dim(self) -> int:
        return self.action_dim_per_arm * self.num_arms

    @property
    def state_dim(self) -> int:
        """Proprioceptive state dimension (same as action dim)."""
        return self.total_action_dim


# Default configs for supported RoboTwin embodiments
ROBOTWIN_CONFIGS = {
    "franka_dual": ActionSpaceConfig(num_arms=2, joints_per_arm=7),
    "ur5_dual": ActionSpaceConfig(num_arms=2, joints_per_arm=6),
    "aloha": ActionSpaceConfig(num_arms=2, joints_per_arm=7),
    "arx_x5_dual": ActionSpaceConfig(num_arms=2, joints_per_arm=7),
    "piper_dual": ActionSpaceConfig(num_arms=2, joints_per_arm=6),
    # Single-arm variants
    "franka_single": ActionSpaceConfig(num_arms=1, joints_per_arm=7),
    "ur5_single": ActionSpaceConfig(num_arms=1, joints_per_arm=6),
}


def normalize_actions(
    actions: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    mode: str = "minmax"
) -> torch.Tensor:
    """
    Normalize actions to [-1, 1] range.

    Args:
        actions: (B, T, D) raw action values
        stats: dict with 'min', 'max', 'mean', 'std' tensors of shape (D,)
        mode: 'minmax' or 'zscore'
    """
    if mode == "minmax":
        action_min = stats["min"].to(actions.device)
        action_max = stats["max"].to(actions.device)
        # Clip to avoid outliers (following OpenVLA's quantile approach)
        actions = actions.clamp(action_min, action_max)
        normalized = 2.0 * (actions - action_min) / (action_max - action_min + 1e-8) - 1.0
    elif mode == "zscore":
        mean = stats["mean"].to(actions.device)
        std = stats["std"].to(actions.device)
        normalized = (actions - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return normalized


def denormalize_actions(
    normalized: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    mode: str = "minmax"
) -> torch.Tensor:
    """
    Denormalize actions from [-1, 1] back to original range.

    Args:
        normalized: (B, T, D) normalized action values
        stats: dict with 'min', 'max', 'mean', 'std' tensors
        mode: 'minmax' or 'zscore'
    """
    if mode == "minmax":
        action_min = stats["min"].to(normalized.device)
        action_max = stats["max"].to(normalized.device)
        actions = (normalized + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min
    elif mode == "zscore":
        mean = stats["mean"].to(normalized.device)
        std = stats["std"].to(normalized.device)
        actions = normalized * (std + 1e-8) + mean
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return actions


def compute_action_stats(
    all_actions: torch.Tensor,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99
) -> Dict[str, torch.Tensor]:
    """
    Compute normalization statistics from a dataset of actions.
    Uses quantiles (following OpenVLA) to handle outliers.

    Args:
        all_actions: (N, D) all action samples concatenated
        quantile_low: lower quantile for clipping
        quantile_high: upper quantile for clipping
    """
    q_low = torch.quantile(all_actions.float(), quantile_low, dim=0)
    q_high = torch.quantile(all_actions.float(), quantile_high, dim=0)

    return {
        "min": q_low,
        "max": q_high,
        "mean": all_actions.float().mean(dim=0),
        "std": all_actions.float().std(dim=0),
    }


def format_joint_to_unified(
    joint_angles: torch.Tensor,
    config: ActionSpaceConfig,
    arm: str = "right"
) -> torch.Tensor:
    """
    Convert robot-specific joint angles to the unified action space format.

    Args:
        joint_angles: (..., N) joint angle tensor
        config: action space configuration
        arm: 'left', 'right', or 'both' (for dual-arm)
    """
    if arm == "both":
        # Already in the correct format for dual-arm
        assert joint_angles.shape[-1] == config.total_action_dim
        return joint_angles

    unified = torch.zeros(
        *joint_angles.shape[:-1], config.total_action_dim,
        device=joint_angles.device, dtype=joint_angles.dtype
    )

    n_dims = joint_angles.shape[-1]
    if arm == "right":
        if config.num_arms == 2:
            unified[..., config.action_dim_per_arm:config.action_dim_per_arm + n_dims] = joint_angles
        else:
            unified[..., :n_dims] = joint_angles
    elif arm == "left":
        unified[..., :n_dims] = joint_angles

    return unified
