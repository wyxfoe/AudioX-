"""
RobotX Training Script.

Fine-tunes the AudioX-derived diffusion transformer for robot trajectory
prediction using RoboTwin data.

Usage:
    python train_robotx.py --config configs/robotx_robotwin.json \
                           --data_dir /path/to/robotwin/data \
                           --batch_size 32 \
                           --num_gpus 1

    # Resume from checkpoint:
    python train_robotx.py --config configs/robotx_robotwin.json \
                           --ckpt_path /path/to/checkpoint.ckpt

    # Fine-tune from AudioX pretrained weights:
    python train_robotx.py --config configs/robotx_robotwin.json \
                           --pretrained_audiox /path/to/audiox.ckpt
"""

import argparse
import json
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from stable_audio_tools.robotics.robot_model import create_robot_model_from_config
from stable_audio_tools.robotics.robot_training import (
    RobotDiffusionTrainingWrapper,
    RobotDemoCallback,
)
from stable_audio_tools.robotics.robotwin_dataset import (
    create_robotwin_dataloader,
)


def load_audiox_weights(model, ckpt_path: str, strict: bool = False):
    """
    Load pretrained AudioX weights into the RobotX model.

    This transfers weights from matching layers (DiffusionTransformer, T5, CLIP)
    while ignoring mismatched layers (audio encoder -> trajectory encoder,
    audio output channels -> action channels).
    """
    print(f"Loading pretrained AudioX weights from {ckpt_path}")

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

    # Filter out incompatible keys
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
        # Skip audio-specific layers
        if any(skip in key for skip in [
            "clap", "audio_autoencoder", "pretransform",
            "audio_branch", "empty_audio_feat",
        ]):
            skipped_keys.append(key)
            continue

        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state, strict=False)
    print(f"Loaded {len(loaded_keys)} parameters, skipped {len(skipped_keys)}")

    if skipped_keys:
        print(f"Skipped keys (first 20): {skipped_keys[:20]}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train RobotX model on RoboTwin data")

    # Config
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config JSON")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory from config")
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Validation data directory")

    # Training
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Checkpointing
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--pretrained_audiox", type=str, default=None,
                        help="Path to pretrained AudioX weights")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/robotx")
    parser.add_argument("--save_every", type=int, default=5000)

    # Logging
    parser.add_argument("--project_name", type=str, default="robotx")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--demo_every", type=int, default=5000)

    # Task filtering
    parser.add_argument("--task_name", type=str, default=None,
                        help="Filter for specific RoboTwin task")
    parser.add_argument("--embodiment", type=str, default=None,
                        help="Override robot embodiment from config")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit number of training episodes")

    args = parser.parse_args()

    # Set seed
    pl.seed_everything(args.seed)

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Override config with CLI args
    dataset_config = config["dataset"]
    if args.data_dir:
        dataset_config["data_dir"] = args.data_dir
    if args.batch_size:
        dataset_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        dataset_config["num_workers"] = args.num_workers
    if args.embodiment:
        dataset_config["embodiment"] = args.embodiment
    if args.task_name:
        dataset_config["task_name"] = args.task_name

    # Create model
    print("Creating RobotX model...")
    model = create_robot_model_from_config(config)

    # Load pretrained weights if specified
    if args.pretrained_audiox:
        model = load_audiox_weights(model, args.pretrained_audiox)

    # Create training wrapper
    training_config = config.get("training", {})
    training_wrapper = RobotDiffusionTrainingWrapper(
        model=model,
        **training_config,
    )

    # Create data loaders
    print("Creating data loaders...")
    train_dataloader, train_dataset = create_robotwin_dataloader(
        data_dir=dataset_config["data_dir"],
        batch_size=dataset_config.get("batch_size", 32),
        action_chunk_size=config.get("action_chunk_size", 64),
        image_size=dataset_config.get("image_size", 224),
        camera_names=dataset_config.get("camera_names"),
        embodiment=dataset_config.get("embodiment", "franka_dual"),
        num_workers=dataset_config.get("num_workers", 4),
        max_episodes=args.max_episodes,
        task_name=dataset_config.get("task_name"),
        normalize=dataset_config.get("normalize", True),
        augment=dataset_config.get("augment", True),
        shuffle=True,
    )

    val_dataloader = None
    if args.val_data_dir:
        val_dataloader, _ = create_robotwin_dataloader(
            data_dir=args.val_data_dir,
            batch_size=dataset_config.get("batch_size", 32),
            action_chunk_size=config.get("action_chunk_size", 64),
            image_size=dataset_config.get("image_size", 224),
            camera_names=dataset_config.get("camera_names"),
            embodiment=dataset_config.get("embodiment", "franka_dual"),
            num_workers=dataset_config.get("num_workers", 4),
            task_name=dataset_config.get("task_name"),
            normalize=True,
            action_stats=train_dataset.action_stats if hasattr(train_dataset, 'action_stats') else None,
            augment=False,
            shuffle=False,
        )

    # Save action stats for inference
    if hasattr(train_dataset, "action_stats") and train_dataset.action_stats:
        stats_path = os.path.join(args.save_dir, "action_stats.pt")
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(train_dataset.action_stats, stats_path)
        print(f"Saved action statistics to {stats_path}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="robotx-{step:08d}",
            every_n_train_steps=args.save_every,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
        RobotDemoCallback(
            demo_every=args.demo_every,
            num_demos=4,
            demo_steps=50,
            demo_cfg_scales=[1.0, 3.0],
        ),
    ]

    # Logger
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.run_name,
        save_dir=args.save_dir,
    )

    # Trainer
    trainer = pl.Trainer(
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto",
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=wandb_logger,
        max_steps=args.max_steps,
        log_every_n_steps=10,
        val_check_interval=args.save_every,
    )

    # Train
    print("Starting training...")
    trainer.fit(
        training_wrapper,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.ckpt_path,
    )

    # Export final model
    final_path = os.path.join(args.save_dir, "robotx_final.safetensors")
    training_wrapper.export_model(final_path, use_safetensors=True)
    print(f"Exported final model to {final_path}")


if __name__ == "__main__":
    main()
