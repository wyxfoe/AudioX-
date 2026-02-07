"""
AudioX (RobotX) policy deployment for RoboTwin evaluation.

Implements the standard RoboTwin policy interface:
  - encode_obs(observation) -> processed obs dict
  - get_model(usr_args) -> initialized model
  - eval(TASK_ENV, model, observation) -> execute policy in env
  - reset_model(model) -> reset between episodes

Reference: RoboTwin policy adapter pattern (DP, RDT, etc.)
"""

import os
import sys
import numpy as np

from pathlib import Path

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
project_root = str(Path(parent_directory).parent.parent)

sys.path.insert(0, parent_directory)
sys.path.insert(0, project_root)

from .audiox_model import AudioXPolicy


def encode_obs(observation):
    """
    Post-process raw observation from RoboTwin environment.

    Extracts camera images and joint positions into a standardized format.

    Args:
        observation: dict from TASK_ENV.get_obs() with structure:
            observation["observation"]["head_camera"]["rgb"]  -> (H, W, 3) uint8
            observation["observation"]["left_camera"]["rgb"]  -> (H, W, 3) uint8
            observation["observation"]["right_camera"]["rgb"] -> (H, W, 3) uint8
            observation["joint_action"]["vector"]             -> (action_dim,) float

    Returns:
        obs: dict with processed observation fields
    """
    obs = observation.copy()

    # Extract camera images (RGB numpy arrays)
    obs["head_cam"] = observation["observation"]["head_camera"]["rgb"]
    obs["right_cam"] = observation["observation"]["right_camera"]["rgb"]
    obs["left_cam"] = observation["observation"]["left_camera"]["rgb"]

    # Extract joint state
    obs["agent_pos"] = observation["joint_action"]["vector"]

    return obs


def get_model(usr_args):
    """
    Initialize AudioX policy model from configuration.

    Args:
        usr_args: dict from deploy_policy.yml + eval.sh overrides, containing:
            - task_name: task identifier
            - ckpt_setting: checkpoint/training config name
            - checkpoint_num: checkpoint step number
            - left_arm_dim: left arm DoF
            - right_arm_dim: right arm DoF
            - audiox_chunk_size: action chunk size (default 64)
            - cfg_scale: classifier-free guidance scale (default 3.0)
            - num_steps: diffusion sampling steps (default 50)

    Returns:
        AudioXPolicy instance
    """
    # Resolve paths
    policy_dir = parent_directory
    project_root_path = str(Path(policy_dir).parent.parent)

    config_path = usr_args.get(
        "config_path",
        os.path.join(project_root_path, "configs", "robotx_robotwin.json"),
    )

    # Build checkpoint path
    ckpt_path = usr_args.get("ckpt_path", None)
    if ckpt_path is None:
        task_name = usr_args["task_name"]
        ckpt_setting = usr_args["ckpt_setting"]
        checkpoint_num = usr_args.get("checkpoint_num", "latest")
        ckpt_path = os.path.join(
            policy_dir,
            "checkpoints",
            f"{task_name}-{ckpt_setting}",
            f"{checkpoint_num}.ckpt",
        )

    action_stats_path = usr_args.get(
        "action_stats_path",
        os.path.join(os.path.dirname(ckpt_path), "action_stats.pt"),
    )

    left_arm_dim = usr_args.get("left_arm_dim", 7)
    right_arm_dim = usr_args.get("right_arm_dim", 7)
    chunk_size = usr_args.get("audiox_chunk_size", 64)
    cfg_scale = usr_args.get("cfg_scale", 3.0)
    num_steps = usr_args.get("num_steps", 50)

    model = AudioXPolicy(
        config_path=config_path,
        ckpt_path=ckpt_path,
        action_stats_path=action_stats_path,
        left_arm_dim=left_arm_dim,
        right_arm_dim=right_arm_dim,
        action_chunk_size=chunk_size,
        cfg_scale=cfg_scale,
        num_steps=num_steps,
    )

    return model


def eval(TASK_ENV, model, observation):
    """
    Main evaluation loop: process observation, generate actions, execute in env.

    Follows the standard RoboTwin eval pattern:
      1. Encode observation
      2. Get language instruction
      3. On first frame, initialize observation window
      4. Generate action chunk
      5. Execute each action step, updating observations

    Args:
        TASK_ENV: RoboTwin task environment instance
        model: AudioXPolicy instance from get_model()
        observation: raw observation dict from TASK_ENV.get_obs()
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # Image arrays: [head, right, left] following RoboTwin camera convention
    input_rgb_arr = [obs["head_cam"], obs["right_cam"], obs["left_cam"]]
    input_state = obs["agent_pos"]

    # Initialize on first frame
    if model.observation_window is None:
        model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    # Generate action chunk
    actions = model.get_action()  # (chunk_size, action_dim)

    # Execute each action step
    for action in actions:
        # action format: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)

        input_rgb_arr = [obs["head_cam"], obs["right_cam"], obs["left_cam"]]
        input_state = obs["agent_pos"]
        model.update_observation_window(input_rgb_arr, input_state)

        # Check if task is done
        if TASK_ENV.eval_success:
            break


def reset_model(model):
    """
    Reset model state at the beginning of each evaluation episode.

    Clears observation window and cached language instruction.
    """
    model.reset()
