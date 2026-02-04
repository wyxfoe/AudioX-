"""
RobotX Inference Script.

Generates robot action trajectories conditioned on visual observations,
language instructions, and proprioceptive state.

Usage:
    # Single prediction
    python inference_robotx.py \
        --config configs/robotx_robotwin.json \
        --ckpt_path checkpoints/robotx/robotx_final.safetensors \
        --action_stats checkpoints/robotx/action_stats.pt \
        --image_path observation.png \
        --instruction "pick up the red cube" \
        --joint_state "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.0"

    # Evaluation on RoboTwin episodes
    python inference_robotx.py \
        --config configs/robotx_robotwin.json \
        --ckpt_path checkpoints/robotx/robotx_final.safetensors \
        --eval_dir /path/to/robotwin/test_episodes \
        --output_dir results/
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from stable_audio_tools.inference.sampling import (
    get_alphas_sigmas,
    sample,
    sample_discrete_euler,
)
from stable_audio_tools.robotics.robot_model import create_robot_model_from_config
from stable_audio_tools.robotics.action_space import denormalize_actions


def load_model(config_path: str, ckpt_path: str, device: str = "cuda"):
    """Load the RobotX model from config and checkpoint."""
    with open(config_path, "r") as f:
        config = json.load(f)

    model = create_robot_model_from_config(config)

    # Load weights
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    return model, config


def prepare_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for CLIP conditioning."""
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img)


@torch.no_grad()
def predict_actions(
    model,
    config: dict,
    image: torch.Tensor = None,
    instruction: str = "",
    joint_state: torch.Tensor = None,
    cfg_scale: float = 3.0,
    num_steps: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate an action trajectory prediction.

    Args:
        model: loaded RobotX model
        config: model config dictionary
        image: preprocessed image tensor (C, H, W) or (num_views, C, H, W)
        instruction: language instruction string
        joint_state: current joint angle state (action_dim,)
        cfg_scale: classifier-free guidance scale
        num_steps: number of diffusion sampling steps
        device: computation device

    Returns:
        actions: predicted action chunk (action_dim, chunk_size)
    """
    action_dim = config["action_dim"]
    chunk_size = config.get("action_chunk_size", 64)

    # Prepare metadata in AudioX-compatible format
    metadata = [{}]

    # Language instruction
    metadata[0]["prompt"] = instruction

    # Proprioceptive state
    if joint_state is not None:
        metadata[0]["proprio"] = joint_state.to(device)
    else:
        metadata[0]["proprio"] = torch.zeros(action_dim).to(device)

    # Visual observation
    if image is not None:
        if image.ndim == 3:
            image = image.unsqueeze(0)  # Add camera dim
        # Pad to expected number of frames for CLIP temporal transformer
        # AudioX CLIP expects (num_frames, C, H, W)
        metadata[0]["video"] = image.to(device)
    else:
        metadata[0]["video"] = torch.zeros(1, 3, 224, 224).to(device)

    # Get conditioning
    with torch.cuda.amp.autocast():
        conditioning = model.conditioner(metadata, device)

    cond_inputs = model.get_conditioning_inputs(conditioning)

    # Sample from diffusion
    noise = torch.randn(1, action_dim, chunk_size).to(device)

    diffusion_model = model.model

    with torch.cuda.amp.autocast():
        if model.diffusion_objective == "v":
            actions = sample(
                diffusion_model,
                noise,
                num_steps,
                0,
                **cond_inputs,
                cfg_scale=cfg_scale,
                batch_cfg=True,
            )
        elif model.diffusion_objective == "rectified_flow":
            actions = sample_discrete_euler(
                diffusion_model,
                noise,
                num_steps,
                **cond_inputs,
                cfg_scale=cfg_scale,
                batch_cfg=True,
            )

    return actions[0]  # Remove batch dim -> (action_dim, chunk_size)


def evaluate_on_episodes(
    model,
    config: dict,
    eval_dir: str,
    action_stats: dict,
    output_dir: str,
    cfg_scale: float = 3.0,
    num_steps: int = 50,
    device: str = "cuda",
):
    """Evaluate model on a directory of RoboTwin episodes."""
    from stable_audio_tools.robotics.robotwin_dataset import create_robotwin_dataloader

    os.makedirs(output_dir, exist_ok=True)

    dataloader, dataset = create_robotwin_dataloader(
        data_dir=eval_dir,
        batch_size=1,
        action_chunk_size=config.get("action_chunk_size", 64),
        image_size=config["dataset"].get("image_size", 224),
        embodiment=config["dataset"].get("embodiment", "franka_dual"),
        normalize=True,
        action_stats=action_stats,
        shuffle=False,
        augment=False,
        num_workers=0,
    )

    all_errors = []
    per_joint_errors = []

    for batch_idx, (actions, metadata) in enumerate(dataloader):
        actions = actions.to(device)

        with torch.cuda.amp.autocast():
            conditioning = model.conditioner(metadata, device)
        cond_inputs = model.get_conditioning_inputs(conditioning)

        noise = torch.randn_like(actions)

        with torch.cuda.amp.autocast():
            if model.diffusion_objective == "v":
                preds = sample(
                    model.model, noise, num_steps, 0,
                    **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True
                )
            else:
                preds = sample_discrete_euler(
                    model.model, noise, num_steps,
                    **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True
                )

        # Compute errors
        mse = torch.nn.functional.mse_loss(preds, actions).item()
        all_errors.append(mse)

        per_joint = torch.nn.functional.mse_loss(
            preds, actions, reduction="none"
        ).mean(dim=2)[0].cpu().numpy()
        per_joint_errors.append(per_joint)

        if batch_idx % 100 == 0:
            print(f"  Episode {batch_idx}: MSE = {mse:.6f}")

    # Summary
    all_errors = np.array(all_errors)
    per_joint_errors = np.stack(per_joint_errors, axis=0)

    results = {
        "mean_mse": float(all_errors.mean()),
        "std_mse": float(all_errors.std()),
        "median_mse": float(np.median(all_errors)),
        "per_joint_mean_mse": per_joint_errors.mean(axis=0).tolist(),
        "num_episodes": len(all_errors),
    }

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation Results ({len(all_errors)} episodes):")
    print(f"  Mean MSE: {results['mean_mse']:.6f}")
    print(f"  Std MSE:  {results['std_mse']:.6f}")
    print(f"  Median:   {results['median_mse']:.6f}")
    print(f"  Per-joint: {[f'{x:.6f}' for x in results['per_joint_mean_mse'][:8]]}")
    print(f"  Saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="RobotX Inference")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--action_stats", type=str, default=None,
                        help="Path to action normalization stats")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--num_steps", type=int, default=50)

    # Single prediction mode
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--joint_state", type=str, default=None,
                        help="Comma-separated joint angles")

    # Evaluation mode
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")

    args = parser.parse_args()

    # Load model
    model, config = load_model(args.config, args.ckpt_path, args.device)

    # Load action stats
    action_stats = None
    if args.action_stats and os.path.exists(args.action_stats):
        action_stats = torch.load(args.action_stats, map_location="cpu")

    if args.eval_dir:
        # Evaluation mode
        evaluate_on_episodes(
            model, config, args.eval_dir, action_stats,
            args.output_dir, args.cfg_scale, args.num_steps, args.device,
        )
    else:
        # Single prediction mode
        image = None
        if args.image_path:
            image = prepare_image(args.image_path)

        joint_state = None
        if args.joint_state:
            joint_state = torch.tensor(
                [float(x) for x in args.joint_state.split(",")],
                dtype=torch.float32,
            )

        actions = predict_actions(
            model, config, image=image, instruction=args.instruction,
            joint_state=joint_state, cfg_scale=args.cfg_scale,
            num_steps=args.num_steps, device=args.device,
        )

        # Denormalize if stats available
        if action_stats:
            # actions shape: (action_dim, chunk_size) -> (1, chunk_size, action_dim)
            actions_t = actions.unsqueeze(0).permute(0, 2, 1)
            actions_denorm = denormalize_actions(actions_t, action_stats)
            actions_np = actions_denorm[0].cpu().numpy()
        else:
            actions_np = actions.permute(1, 0).cpu().numpy()

        print(f"\nPredicted action trajectory: shape {actions_np.shape}")
        print(f"  (chunk_size={actions_np.shape[0]}, action_dim={actions_np.shape[1]})")
        print(f"\nFirst 5 timesteps:")
        for t in range(min(5, actions_np.shape[0])):
            print(f"  t={t}: {actions_np[t, :8].round(4)}")

        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "predicted_actions.npy")
        np.save(output_path, actions_np)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
