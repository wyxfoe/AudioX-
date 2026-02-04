"""
RobotX Model: Adapts AudioX's conditional diffusion architecture for robot
trajectory prediction.

Key modifications from AudioX:
  1. Audio output (io_channels for audio latents) -> Action output (action_dim * chunk_size)
  2. Audio encoder (CLAP) -> Trajectory encoder (MLP for joint angles)
  3. Audio autoencoder (pretransform) -> Removed (actions are low-dimensional)
  4. Vision encoder (CLIP) -> Retained for visual observation
  5. Text encoder (T5) -> Retained for language instructions

The diffusion process denoises random noise into action chunks, conditioned
on visual observations, language instructions, and current proprioceptive state.

Following RDT-1B:
  - Predicts chunks of 64 future actions
  - Uses the DiffusionTransformer backbone with cross-attention conditioning
  - Action space is continuous (no discretization, unlike OpenVLA)

Following OpenVLA:
  - Visual features from pretrained vision encoders
  - Language grounding via text encoder
"""

import torch
import torch.nn as nn
import typing as tp

from ..models.diffusion import (
    ConditionedDiffusionModel,
    ConditionedDiffusionModelWrapper,
)
from ..models.dit import DiffusionTransformer
from ..models.conditioners import (
    MultiConditioner,
    Conditioner,
    T5Conditioner,
    CLIPConditioner,
    create_multi_conditioner_from_conditioning_config,
)
from .trajectory_conditioner import TrajectoryConditioner, TrajectoryHistoryConditioner
from .action_space import ActionSpaceConfig


class RobotDiTWrapper(ConditionedDiffusionModel):
    """
    Wraps DiffusionTransformer for robot action prediction.

    The key difference from DiTWrapper is that io_channels now represents
    the action dimension, and the sequence length represents the action
    chunk size (number of future timesteps).

    Input/output shape: (B, action_dim, chunk_size)
      - action_dim: total action dimension (e.g., 16 for dual-arm 7-DoF)
      - chunk_size: number of future actions (e.g., 64 following RDT)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            supports_cross_attention=True,
            supports_global_cond=False,
            supports_input_concat=False,
            supports_prepend_cond=True,
        )
        self.model = DiffusionTransformer(*args, **kwargs)

        # Initialize with smaller weights for stability
        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        negative_input_concat_cond=None,
        global_cond=None,
        negative_global_cond=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob: float = 0.0,
        batch_cfg: bool = True,
        rescale_cfg: bool = False,
        scale_phi: float = 0.0,
        **kwargs,
    ):
        return self.model(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            global_embed=global_cond,
            **kwargs,
        )


class RobotDiffusionModelWrapper(nn.Module):
    """
    Complete robot diffusion model that wraps the conditioner and diffusion model.

    This is the robot equivalent of ConditionedDiffusionModelWrapper.
    Key differences:
      - No pretransform (actions are already low-dimensional)
      - Adds action denormalization utilities
      - Supports action chunking (predict multiple future actions)
    """

    def __init__(
        self,
        model: ConditionedDiffusionModel,
        conditioner: MultiConditioner,
        action_dim: int,
        action_chunk_size: int = 64,
        diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
        cross_attn_cond_ids: tp.List[str] = [],
        global_cond_ids: tp.List[str] = [],
        input_concat_ids: tp.List[str] = [],
        prepend_cond_ids: tp.List[str] = [],
    ):
        super().__init__()

        self.model = model
        self.conditioner = conditioner
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.diffusion_objective = diffusion_objective

        # No pretransform needed for low-dimensional actions
        self.pretransform = None
        self.io_channels = action_dim
        self.sample_rate = 1  # Not relevant for actions, kept for compatibility
        self.min_input_length = 1

        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.prepend_cond_ids = prepend_cond_ids

    def get_conditioning_inputs(
        self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False
    ):
        """Extract conditioning inputs from conditioner outputs."""
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None

        if len(self.cross_attn_cond_ids) > 0:
            cross_attention_input = []
            cross_attention_masks = []

            for key in self.cross_attn_cond_ids:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]

                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)

                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)

            cross_attention_input = torch.cat(cross_attention_input, dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]
                global_conds.append(global_cond_input)
            global_cond = torch.cat(global_conds, dim=-1)
            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if len(self.input_concat_ids) > 0:
            input_concat_cond = torch.cat(
                [conditioning_tensors[key][0] for key in self.input_concat_ids], dim=1
            )

        if len(self.prepend_cond_ids) > 0:
            prepend_conds = []
            prepend_cond_masks = []
            for key in self.prepend_cond_ids:
                prepend_cond_input, prepend_mask = conditioning_tensors[key]
                prepend_conds.append(prepend_cond_input)
                prepend_cond_masks.append(prepend_mask)
            prepend_cond = torch.cat(prepend_conds, dim=1)
            prepend_cond_mask = torch.cat(prepend_cond_masks, dim=1)

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
            }
        else:
            return {
                "cross_attn_cond": cross_attention_input,
                "cross_attn_mask": cross_attention_masks,
                "global_cond": global_cond,
                "input_concat_cond": input_concat_cond,
                "prepend_cond": prepend_cond,
                "prepend_cond_mask": prepend_cond_mask,
            }

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs
    ):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)


def create_robot_conditioner_from_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """
    Create a MultiConditioner for robot model from config.
    Extends AudioX's conditioner factory with trajectory conditioner types.
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        cid = conditioner_info["id"]
        cond_type = conditioner_info["type"]
        cond_config = {"output_dim": cond_dim}
        cond_config.update(conditioner_info["config"])

        if cond_type == "t5":
            conditioners[cid] = T5Conditioner(**cond_config)
        elif cond_type == "clip":
            conditioners[cid] = CLIPConditioner(**cond_config)
        elif cond_type == "trajectory":
            conditioners[cid] = TrajectoryConditioner(**cond_config)
        elif cond_type == "trajectory_history":
            conditioners[cid] = TrajectoryHistoryConditioner(**cond_config)
        else:
            raise ValueError(f"Unknown conditioner type for robot model: {cond_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)


def create_robot_model_from_config(config: tp.Dict[str, tp.Any]) -> RobotDiffusionModelWrapper:
    """
    Create a complete RobotX model from configuration dictionary.

    Config structure:
    {
        "model_type": "robot_diffusion",
        "action_dim": 16,
        "action_chunk_size": 64,
        "model": {
            "diffusion": {
                "type": "dit",
                "config": { ... DiffusionTransformer config ... },
                "diffusion_objective": "v",
                "cross_attention_cond_ids": ["prompt", "video", "proprio"],
                "prepend_cond_ids": [],
            },
            "conditioning": {
                "cond_dim": 768,
                "configs": [
                    {"id": "prompt", "type": "t5", "config": {...}},
                    {"id": "video", "type": "clip", "config": {...}},
                    {"id": "proprio", "type": "trajectory", "config": {...}},
                ]
            }
        }
    }
    """
    model_config = config["model"]
    action_dim = config["action_dim"]
    action_chunk_size = config.get("action_chunk_size", 64)

    # Create diffusion model
    diffusion_config = model_config["diffusion"]
    diffusion_model_config = diffusion_config["config"]

    # Remove video_fps if present (not relevant for robot model)
    diffusion_model_config.pop("video_fps", None)

    diffusion_model = RobotDiTWrapper(**diffusion_model_config)

    # Create conditioner
    conditioning_config = model_config.get("conditioning", None)
    conditioner = None
    if conditioning_config is not None:
        conditioner = create_robot_conditioner_from_config(conditioning_config)

    # Get conditioning routing
    cross_attention_ids = diffusion_config.get("cross_attention_cond_ids", [])
    global_cond_ids = diffusion_config.get("global_cond_ids", [])
    input_concat_ids = diffusion_config.get("input_concat_ids", [])
    prepend_cond_ids = diffusion_config.get("prepend_cond_ids", [])

    diffusion_objective = diffusion_config.get("diffusion_objective", "v")

    return RobotDiffusionModelWrapper(
        model=diffusion_model,
        conditioner=conditioner,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        diffusion_objective=diffusion_objective,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
    )
