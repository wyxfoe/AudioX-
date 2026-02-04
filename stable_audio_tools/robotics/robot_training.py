"""
RobotX Training Pipeline.

Adapts AudioX's DiffusionCondTrainingWrapper for robot trajectory prediction.
The training loop follows the same diffusion training paradigm:
  1. Get conditioning (vision, language, proprioception)
  2. Add noise to ground-truth action chunks
  3. Predict the noise/velocity with the diffusion transformer
  4. Compute MSE loss

Key differences from AudioX training:
  - No pretransform (audio encoder/decoder) - actions are low-dimensional
  - Action chunks replace audio latents as the diffusion target
  - Proprioceptive state replaces audio conditioning
  - Supports action chunking (predicting multiple future actions)
"""

import gc
import random
import typing as tp

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from torch import optim
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler
from ..training.losses import MSELoss, MultiLoss
from ..training.utils import create_optimizer_from_config, create_scheduler_from_config
from .robot_model import RobotDiffusionModelWrapper


class RobotDiffusionTrainingWrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper for training the robot diffusion model.

    Training follows AudioX's conditional diffusion training with these adaptations:
      - Input: action chunks (B, action_dim, chunk_size) instead of audio latents
      - Conditioning: vision (CLIP) + language (T5) + proprioception (trajectory encoder)
      - No pretransform encoding/decoding step
      - Supports temporal weighting of the action loss

    Args:
        model: RobotDiffusionModelWrapper instance
        lr: learning rate (deprecated, use optimizer_configs)
        use_ema: whether to use exponential moving average
        cfg_dropout_prob: probability of dropping conditioning for CFG
        timestep_sampler: how to sample diffusion timesteps
        optimizer_configs: optimizer and scheduler configuration
        action_loss_weights: optional per-timestep loss weights for action chunks
    """

    def __init__(
        self,
        model: RobotDiffusionModelWrapper,
        lr: float = None,
        use_ema: bool = True,
        log_loss_info: bool = True,
        cfg_dropout_prob: float = 0.1,
        timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform",
        optimizer_configs: dict = None,
        action_loss_weights: tp.Optional[tp.List[float]] = None,
    ):
        super().__init__()

        self.diffusion = model

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3 / 4,
                update_every=1,
                update_after_step=1,
                include_online_model=False,
            )
        else:
            self.diffusion_ema = None

        self.cfg_dropout_prob = cfg_dropout_prob
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.timestep_sampler = timestep_sampler
        self.diffusion_objective = model.diffusion_objective
        self.log_loss_info = log_loss_info

        # Loss: MSE between predicted and target velocity/noise
        self.loss_modules = [
            MSELoss(
                "output",
                "targets",
                weight=1.0,
                name="mse_loss",
            )
        ]
        self.losses = MultiLoss(self.loss_modules)

        # Optional temporal weighting for action chunks
        # (e.g., weight near-future actions more heavily)
        if action_loss_weights is not None:
            self.register_buffer(
                "action_loss_weights",
                torch.tensor(action_loss_weights, dtype=torch.float32),
            )
        else:
            self.action_loss_weights = None

        # Optimizer config
        assert (
            lr is not None or optimizer_configs is not None
        ), "Must specify either lr or optimizer_configs"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {"lr": lr, "weight_decay": 0.01},
                    }
                }
            }

        self.optimizer_configs = optimizer_configs

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs["diffusion"]
        opt_diff = create_optimizer_from_config(
            diffusion_opt_config["optimizer"], self.diffusion.parameters()
        )

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(
                diffusion_opt_config["scheduler"], opt_diff
            )
            sched_diff_config = {"scheduler": sched_diff, "interval": "step"}
            return [opt_diff], [sched_diff_config]

        return [opt_diff]

    def training_step(self, batch, batch_idx):
        actions, metadata = batch  # actions: (B, action_dim, chunk_size)

        if actions.ndim == 4 and actions.shape[0] == 1:
            actions = actions[0]

        # Actions are the diffusion target (no pretransform needed)
        diffusion_input = actions

        loss_info = {}

        # Get conditioning from all modalities
        with torch.cuda.amp.autocast():
            conditioning = self.diffusion.conditioner(metadata, self.device)

        # Sample timesteps
        if self.timestep_sampler == "uniform":
            t = self.rng.draw(actions.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(actions.shape[0], device=self.device))

        # Noise schedule
        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1 - t, t

        # Add noise to action chunks
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        # Compute targets
        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input

        # Forward pass
        with torch.cuda.amp.autocast():
            output = self.diffusion(
                noised_inputs,
                t,
                cond=conditioning,
                cfg_dropout_prob=self.cfg_dropout_prob,
            )

            # Apply temporal weighting if specified
            if self.action_loss_weights is not None:
                # action_loss_weights shape: (chunk_size,)
                weights = self.action_loss_weights.to(output.device)
                weights = weights[: output.shape[2]]  # Trim to actual size
                # Reshape for broadcasting: (1, 1, chunk_size)
                weighted_output = output * weights.unsqueeze(0).unsqueeze(0)
                weighted_targets = targets * weights.unsqueeze(0).unsqueeze(0)
                loss_info.update(
                    {"output": weighted_output, "targets": weighted_targets}
                )
            else:
                loss_info.update({"output": output, "targets": targets})

            loss, losses = self.losses(loss_info)

            # Additional loss logging
            if self.log_loss_info:
                num_loss_buckets = 10
                bucket_size = 1 / num_loss_buckets
                loss_all = F.mse_loss(output, targets, reduction="none")
                sigmas_flat = rearrange(
                    self.all_gather(sigmas), "b c n -> (b) c n"
                ).squeeze()
                loss_all = rearrange(
                    self.all_gather(loss_all), "b c n -> (b) c n"
                )
                loss_all = torch.stack(
                    [
                        loss_all[(sigmas_flat >= i) & (sigmas_flat < i + bucket_size)].mean()
                        for i in torch.arange(0, 1, bucket_size).to(self.device)
                    ]
                )
                debug_log_dict = {
                    f"model/loss_all_{i / num_loss_buckets:.1f}": loss_all[i].detach()
                    for i in range(num_loss_buckets)
                    if not torch.isnan(loss_all[i])
                }
                self.log_dict(debug_log_dict)

        # Per-joint loss logging
        with torch.no_grad():
            per_joint_loss = F.mse_loss(output, targets, reduction="none").mean(dim=2)
            for j in range(min(per_joint_loss.shape[1], 16)):
                self.log(
                    f"train/joint_{j}_loss",
                    per_joint_loss[:, j].mean().detach(),
                    prog_bar=False,
                )

        log_dict = {
            "train/loss": loss.detach(),
            "train/std_data": diffusion_input.std(),
            "train/lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        actions, metadata = batch

        if actions.ndim == 4 and actions.shape[0] == 1:
            actions = actions[0]

        diffusion_input = actions
        loss_info = {}

        with torch.cuda.amp.autocast():
            conditioning = self.diffusion.conditioner(metadata, self.device)

        if self.timestep_sampler == "uniform":
            t = self.rng.draw(actions.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(actions.shape[0], device=self.device))

        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1 - t, t

        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input

        with torch.cuda.amp.autocast():
            output = self.diffusion(
                noised_inputs,
                t,
                cond=conditioning,
                cfg_dropout_prob=0.0,  # No CFG dropout during validation
            )
            loss_info.update({"output": output, "targets": targets})
            loss, losses = self.losses(loss_info)

        log_dict = {
            "valid/loss": loss.detach(),
            "valid/std_data": diffusion_input.std(),
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"valid/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)


class RobotDemoCallback(pl.Callback):
    """
    Callback for generating demo action predictions during training.

    Logs predicted vs ground-truth action trajectories to wandb.
    """

    def __init__(
        self,
        demo_every: int = 5000,
        num_demos: int = 4,
        demo_steps: int = 50,
        demo_cfg_scales: tp.List[float] = [1.0, 3.0],
    ):
        super().__init__()
        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_steps = demo_steps
        self.demo_cfg_scales = demo_cfg_scales
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer,
        module: RobotDiffusionTrainingWrapper,
        outputs,
        batch,
        batch_idx,
    ):
        if (
            (trainer.global_step - 1) % self.demo_every != 0
            or self.last_demo_step == trainer.global_step
        ):
            return

        module.eval()
        self.last_demo_step = trainer.global_step

        try:
            actions, metadata = batch
            actions = actions[: self.num_demos].to(module.device)
            demo_metadata = metadata[: self.num_demos]

            with torch.cuda.amp.autocast():
                conditioning = module.diffusion.conditioner(
                    demo_metadata, module.device
                )

            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            # Generate noise
            noise = torch.randn_like(actions)

            log_dict = {}

            for cfg_scale in self.demo_cfg_scales:
                model = (
                    module.diffusion_ema.model
                    if module.diffusion_ema is not None
                    else module.diffusion.model
                )

                with torch.cuda.amp.autocast():
                    if module.diffusion_objective == "v":
                        preds = sample(
                            model,
                            noise,
                            self.demo_steps,
                            0,
                            **cond_inputs,
                            cfg_scale=cfg_scale,
                            batch_cfg=True,
                        )
                    elif module.diffusion_objective == "rectified_flow":
                        preds = sample_discrete_euler(
                            model,
                            noise,
                            self.demo_steps,
                            **cond_inputs,
                            cfg_scale=cfg_scale,
                            batch_cfg=True,
                        )

                # Compute prediction error
                pred_error = F.mse_loss(preds, actions).item()
                log_dict[f"demo/mse_cfg_{cfg_scale}"] = pred_error

                # Log trajectory plots
                for b in range(min(2, preds.shape[0])):
                    pred_traj = preds[b].cpu().numpy()  # (action_dim, chunk_size)
                    gt_traj = actions[b].cpu().numpy()

                    # Create matplotlib figure
                    try:
                        import matplotlib.pyplot as plt

                        fig, axes = plt.subplots(
                            min(4, pred_traj.shape[0]), 1,
                            figsize=(10, 8), sharex=True
                        )
                        if pred_traj.shape[0] == 1:
                            axes = [axes]

                        for j in range(min(4, pred_traj.shape[0])):
                            axes[j].plot(gt_traj[j], label="GT", alpha=0.7)
                            axes[j].plot(pred_traj[j], label="Pred", alpha=0.7, linestyle="--")
                            axes[j].set_ylabel(f"Joint {j}")
                            axes[j].legend(fontsize=6)
                        axes[-1].set_xlabel("Timestep")
                        fig.suptitle(f"CFG={cfg_scale}, MSE={pred_error:.4f}")
                        fig.tight_layout()

                        log_dict[
                            f"demo/traj_b{b}_cfg_{cfg_scale}"
                        ] = wandb.Image(fig)
                        plt.close(fig)
                    except ImportError:
                        pass

            trainer.logger.experiment.log(log_dict)

        except Exception as e:
            print(f"Demo callback error: {type(e).__name__}: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()
