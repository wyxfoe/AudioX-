"""
AudioX (RobotX) model wrapper for RoboTwin evaluation.

Wraps the RobotX diffusion-based policy to provide a consistent interface
for the RoboTwin eval framework, following the same pattern as DP and RDT.

Key components:
  - Observation window management (deque-based, similar to RDT)
  - Language instruction encoding via T5 conditioner
  - Visual observation encoding via CLIP conditioner
  - Proprioceptive state encoding via TrajectoryConditioner
  - Action generation via diffusion sampling
"""

import os
import sys
import json
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add project root to path
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_audio_tools.inference.sampling import sample, sample_discrete_euler
from stable_audio_tools.robotics.robot_model import create_robot_model_from_config
from stable_audio_tools.robotics.action_space import denormalize_actions


class AudioXPolicy:
    """
    AudioX-based robot policy for RoboTwin evaluation.

    Similar to RDT's model wrapper, this class manages:
      - Model loading and initialization
      - Observation window buffering
      - Language instruction caching
      - Action chunk generation via diffusion

    Args:
        config_path: path to model config JSON
        ckpt_path: path to model checkpoint
        action_stats_path: path to action normalization statistics
        left_arm_dim: left arm joint dimension
        right_arm_dim: right arm joint dimension
        action_chunk_size: number of action steps per chunk
        cfg_scale: classifier-free guidance scale
        num_steps: number of diffusion sampling steps
        device: torch device
    """

    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        action_stats_path: str = None,
        left_arm_dim: int = 7,
        right_arm_dim: int = 7,
        action_chunk_size: int = 64,
        cfg_scale: float = 3.0,
        num_steps: int = 50,
        device: str = "cuda:0",
    ):
        self.device = device
        self.cfg_scale = cfg_scale
        self.num_steps = num_steps
        self.left_arm_dim = left_arm_dim
        self.right_arm_dim = right_arm_dim
        self.action_chunk_size = action_chunk_size

        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.action_dim = self.config["action_dim"]

        # Create and load model
        self.model = self._load_model(config_path, ckpt_path)

        # Load action normalization stats
        self.action_stats = None
        if action_stats_path and os.path.exists(action_stats_path):
            self.action_stats = torch.load(action_stats_path, map_location="cpu")
            print(f"Loaded action stats from {action_stats_path}")

        # Observation window (similar to RDT's deque approach)
        self.observation_window = None
        self.instruction = None
        self.img_size = (224, 224)

        # CLIP-compatible image transform
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]
        self.image_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])

        print(f"AudioX Policy initialized:")
        print(f"  action_dim={self.action_dim}, chunk_size={self.action_chunk_size}")
        print(f"  cfg_scale={self.cfg_scale}, num_steps={self.num_steps}")
        print(f"  device={self.device}")

    def _load_model(self, config_path: str, ckpt_path: str):
        """Load the RobotX model from config and checkpoint."""
        print(f"Creating model from {config_path}")
        model = create_robot_model_from_config(self.config)

        print(f"Loading checkpoint from {ckpt_path}")
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
                # Remove PyTorch Lightning "diffusion." prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("diffusion."):
                        new_state_dict[k[len("diffusion."):]] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
            else:
                state_dict = ckpt

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

        model = model.to(self.device).eval()
        print("Model loaded successfully")
        return model

    def set_language_instruction(self, instruction: str):
        """Cache the language instruction for the current episode."""
        self.instruction = instruction

    def update_observation_window(self, img_arr, state):
        """
        Update the observation window with new images and state.

        Following RDT's pattern: JPEG encode/decode for alignment with training,
        maintain a window of recent observations.

        Args:
            img_arr: list of RGB images [head_cam, right_cam, left_cam] as numpy arrays
            state: current joint state as numpy array (action_dim,)
        """
        def jpeg_mapping(img):
            """JPEG encode/decode to match training data distribution."""
            if img is None:
                return None
            img = cv2.imencode(".jpg", img)[1].tobytes()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            return img

        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)
            # Append a dummy first frame (following RDT pattern)
            self.observation_window.append({
                "qpos": None,
                "images": [None] * len(img_arr),
            })

        # Process images: JPEG mapping for consistency with training
        processed_imgs = []
        for img in img_arr:
            if img is not None:
                img = jpeg_mapping(img)
            processed_imgs.append(img)

        # Store state as tensor
        if isinstance(state, np.ndarray):
            qpos = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            qpos = state.float().to(self.device)
        else:
            qpos = torch.tensor(state, dtype=torch.float32, device=self.device)

        self.observation_window.append({
            "qpos": qpos,
            "images": processed_imgs,
        })

    def _prepare_conditioning(self):
        """
        Prepare conditioning inputs from the observation window.

        Converts buffered observations into the metadata format expected
        by the RobotX MultiConditioner.
        """
        # Get latest observation
        latest = self.observation_window[-1]
        qpos = latest["qpos"]
        images = latest["images"]

        # Process images -> video tensor for CLIP
        video_frames = []
        for img in images:
            if img is not None:
                # Convert BGR (OpenCV) to RGB PIL
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                img_tensor = self.image_transform(pil_img)
                video_frames.append(img_tensor)

        if video_frames:
            video_tensor = torch.stack(video_frames, dim=0).to(self.device)
        else:
            video_tensor = torch.zeros(1, 3, *self.img_size).to(self.device)

        # Build metadata dict (compatible with AudioX's MultiConditioner)
        metadata = [{
            "prompt": self.instruction or "",
            "proprio": qpos,
            "video": video_tensor,
        }]

        return metadata

    @torch.no_grad()
    def get_action(self):
        """
        Generate action chunk via diffusion sampling.

        Returns:
            actions: numpy array of shape (chunk_size, action_dim)
                     in the format [left_arm + left_gripper + right_arm + right_gripper]
        """
        metadata = self._prepare_conditioning()

        # Get conditioning embeddings
        with torch.cuda.amp.autocast():
            conditioning = self.model.conditioner(metadata, self.device)
        cond_inputs = self.model.get_conditioning_inputs(conditioning)

        # Sample from noise
        noise = torch.randn(1, self.action_dim, self.action_chunk_size).to(self.device)

        with torch.cuda.amp.autocast():
            if self.model.diffusion_objective == "v":
                actions = sample(
                    self.model.model, noise, self.num_steps, 0,
                    **cond_inputs, cfg_scale=self.cfg_scale, batch_cfg=True,
                )
            else:
                actions = sample_discrete_euler(
                    self.model.model, noise, self.num_steps,
                    **cond_inputs, cfg_scale=self.cfg_scale, batch_cfg=True,
                )

        # actions shape: (1, action_dim, chunk_size)
        # Transpose to (1, chunk_size, action_dim)
        actions = actions.permute(0, 2, 1)

        # Denormalize if stats available
        if self.action_stats is not None:
            actions = denormalize_actions(actions, self.action_stats)

        # Convert to numpy (chunk_size, action_dim)
        actions_np = actions[0].cpu().numpy()

        return actions_np

    def reset(self):
        """Reset model state between episodes."""
        self.observation_window = None
        self.instruction = None
        print("AudioX Policy: reset observation window and instruction")
