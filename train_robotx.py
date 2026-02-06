"""
RobotX Training Script.

Fine-tunes the AudioX-derived diffusion transformer for robot trajectory
prediction using RoboTwin data.

Usage:
    # 训练 Aloha-AgileX 模型:
    python train_robotx.py --config configs/robotx_aloha_agilex.json

    # 指定数据目录和批次大小:
    python train_robotx.py --config configs/robotx_aloha_agilex.json \
                           --data_dir /path/to/data \
                           --batch_size 16

    # 从检查点恢复:
    python train_robotx.py --config configs/robotx_aloha_agilex.json \
                           --ckpt_path checkpoints/robotx-00010000.ckpt
"""

import argparse
import json
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_audio_tools.robotics.robot_model import create_robot_model_from_config
from stable_audio_tools.robotics.robot_training import (
    RobotDiffusionTrainingWrapper,
    RobotDemoCallback,
)
from stable_audio_tools.robotics.robotwin_dataset import (
    create_robotwin_dataloader,
)


def main():
    parser = argparse.ArgumentParser(description="Train RobotX model on RoboTwin data")

    # Config
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config JSON")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory from config")

    # Training
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed",
                        help="Training precision: 16-mixed, bf16-mixed, 32")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Checkpointing
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/robotx")
    parser.add_argument("--save_every", type=int, default=2000)

    # Logging
    parser.add_argument("--project_name", type=str, default="robotx")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--demo_every", type=int, default=2000)

    # Task
    parser.add_argument("--task_description", type=str, default=None,
                        help="Override task description")
    parser.add_argument("--max_episodes", type=int, default=None)

    args = parser.parse_args()

    # Set seed
    pl.seed_everything(args.seed)

    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)

    # Override config with CLI args
    dataset_config = config.get("dataset", {})
    if args.data_dir:
        dataset_config["data_dir"] = args.data_dir
    if args.batch_size:
        dataset_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        dataset_config["num_workers"] = args.num_workers
    if args.task_description:
        dataset_config["task_description"] = args.task_description

    # Create model
    print("Creating RobotX model...")
    print(f"  action_dim: {config['action_dim']}")
    print(f"  action_chunk_size: {config['action_chunk_size']}")

    model = create_robot_model_from_config(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Create training wrapper
    training_config = config.get("training", {})
    training_wrapper = RobotDiffusionTrainingWrapper(
        model=model,
        lr=training_config.get("learning_rate", None),
        use_ema=training_config.get("use_ema", True),
        log_loss_info=training_config.get("log_loss_info", True),
        cfg_dropout_prob=training_config.get("cfg_dropout_prob", 0.1),
        timestep_sampler=training_config.get("timestep_sampler", "logit_normal"),
        optimizer_configs=training_config.get("optimizer_configs", None),
    )

    # Create data loader
    print("\nCreating data loader...")
    data_dir = dataset_config.get("data_dir")
    if not data_dir:
        raise ValueError("data_dir must be specified in config or via --data_dir")

    print(f"  Data directory: {data_dir}")

    train_dataloader, train_dataset = create_robotwin_dataloader(
        data_dir=data_dir,
        batch_size=dataset_config.get("batch_size", 16),
        action_chunk_size=config.get("action_chunk_size", 50),
        image_size=dataset_config.get("image_size", 224),
        camera_names=dataset_config.get("camera_names", ["front_camera", "head_camera"]),
        task_description=dataset_config.get("task_description", "complete the task"),
        num_workers=dataset_config.get("num_workers", 4),
        max_episodes=args.max_episodes,
        normalize=dataset_config.get("normalize", True),
        augment=dataset_config.get("augment", True),
        shuffle=True,
    )

    print(f"  Loaded {len(train_dataset)} samples from {len(train_dataset.episodes)} episodes")

    # Save action stats
    os.makedirs(args.save_dir, exist_ok=True)
    if hasattr(train_dataset, "action_stats") and train_dataset.action_stats:
        stats_path = os.path.join(args.save_dir, "action_stats.pt")
        torch.save(train_dataset.action_stats, stats_path)
        print(f"  Saved action statistics to {stats_path}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="robotx-{step:08d}",
            every_n_train_steps=args.save_every,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Add demo callback
    demo_config = training_config.get("demo", {})
    callbacks.append(RobotDemoCallback(
        demo_every=demo_config.get("demo_every", args.demo_every),
        num_demos=demo_config.get("num_demos", 4),
        demo_steps=demo_config.get("demo_steps", 50),
        demo_cfg_scales=demo_config.get("demo_cfg_scales", [1.0, 3.0]),
    ))

    # Logger
    if args.wandb:
        logger = WandbLogger(
            project=args.project_name,
            name=args.run_name,
            save_dir=args.save_dir,
        )
    else:
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=args.save_dir,
            name="tensorboard",
        )

    # Trainer
    print("\nInitializing trainer...")
    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        max_steps=args.max_steps,
        log_every_n_steps=10,
        val_check_interval=args.save_every,
        enable_checkpointing=True,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.fit(
        training_wrapper,
        train_dataloaders=train_dataloader,
        ckpt_path=args.ckpt_path,
    )

    # Export final model
    final_path = os.path.join(args.save_dir, "robotx_final.pt")
    training_wrapper.export_model(final_path, use_safetensors=False)
    print(f"\nExported final model to {final_path}")


if __name__ == "__main__":
    main()
