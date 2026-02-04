"""
RoboTwin Dataset Loader for RobotX.

Loads trajectory data from RoboTwin simulation benchmark for fine-tuning.
RoboTwin provides dual-arm manipulation demonstrations with:
  - Joint angle trajectories (proprioception + action)
  - Multi-view camera images
  - Language instructions

Data format (per episode):
  - observations/joint_positions: (T, num_joints) current joint angles
  - actions: (T, num_joints) target joint angles
  - observations/images/{camera_name}: (T, H, W, 3) RGB images
  - language_instruction: str

This loader converts the data into the format expected by the RobotX
training pipeline, including action chunking and normalization.
"""

import os
import json
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple, Callable
from PIL import Image
from torchvision import transforms

from .action_space import (
    ActionSpaceConfig,
    normalize_actions,
    denormalize_actions,
    compute_action_stats,
    ROBOTWIN_CONFIGS,
)


class RoboTwinDataset(Dataset):
    """
    Dataset for loading RoboTwin trajectory data.

    Supports multiple data formats:
      1. HDF5/zarr format (standard RoboTwin output)
      2. NPZ format (preprocessed)
      3. Directory-based format with JSON metadata

    Args:
        data_dir: root directory containing episode data
        action_config: ActionSpaceConfig defining the action space
        action_chunk_size: number of future actions per sample
        image_size: resize images to (image_size, image_size)
        camera_names: list of camera views to use
        max_episodes: maximum number of episodes to load (None = all)
        task_name: specific task to filter for (None = all tasks)
        embodiment: robot embodiment name (e.g., 'franka_dual')
        normalize: whether to normalize actions
        action_stats: precomputed action statistics for normalization
        augment: whether to apply data augmentation
        language_instructions: list of language instructions per episode
    """

    def __init__(
        self,
        data_dir: str,
        action_config: ActionSpaceConfig = None,
        action_chunk_size: int = 64,
        image_size: int = 224,
        camera_names: List[str] = None,
        max_episodes: Optional[int] = None,
        task_name: Optional[str] = None,
        embodiment: str = "franka_dual",
        normalize: bool = True,
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
        augment: bool = True,
    ):
        super().__init__()

        if action_config is None:
            action_config = ROBOTWIN_CONFIGS.get(embodiment, ActionSpaceConfig())

        self.action_config = action_config
        self.action_chunk_size = action_chunk_size
        self.image_size = image_size
        self.camera_names = camera_names or ["front", "left_wrist", "right_wrist"]
        self.normalize = normalize
        self.augment = augment
        self.embodiment = embodiment

        # Image transforms (CLIP-compatible preprocessing)
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])

        self.augment_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ]) if augment else None

        # Load episodes
        self.episodes = self._load_episodes(data_dir, task_name, max_episodes)

        # Build index: (episode_idx, timestep) for each valid sample
        self.samples = self._build_sample_index()

        # Compute or load action statistics
        if self.normalize:
            if action_stats is not None:
                self.action_stats = action_stats
            else:
                self.action_stats = self._compute_stats()

        print(f"RoboTwinDataset: {len(self.episodes)} episodes, "
              f"{len(self.samples)} samples, "
              f"action_dim={action_config.total_action_dim}, "
              f"chunk_size={action_chunk_size}")

    def _load_episodes(
        self, data_dir: str, task_name: Optional[str], max_episodes: Optional[int]
    ) -> List[Dict]:
        """Load episode data from directory structure."""
        episodes = []

        # Try loading from different formats
        # Format 1: NPZ files
        npz_pattern = os.path.join(data_dir, "**", "*.npz")
        npz_files = sorted(glob.glob(npz_pattern, recursive=True))

        if npz_files:
            for npz_file in npz_files:
                if task_name and task_name not in npz_file:
                    continue
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    episode = {
                        "actions": torch.from_numpy(data["actions"]).float(),
                        "states": torch.from_numpy(data["joint_positions"]).float()
                            if "joint_positions" in data
                            else torch.from_numpy(data["states"]).float(),
                        "file_path": npz_file,
                    }

                    # Load images if available
                    for cam in self.camera_names:
                        key = f"images_{cam}"
                        if key in data:
                            episode[key] = data[key]  # Keep as numpy, load on demand

                    # Load language instruction
                    if "language_instruction" in data:
                        episode["language"] = str(data["language_instruction"])
                    else:
                        # Infer from directory name
                        task_dir = os.path.basename(os.path.dirname(npz_file))
                        episode["language"] = task_dir.replace("_", " ")

                    episodes.append(episode)
                except Exception as e:
                    print(f"Error loading {npz_file}: {e}")
                    continue

                if max_episodes and len(episodes) >= max_episodes:
                    break

        # Format 2: JSON metadata + binary files
        if not episodes:
            json_pattern = os.path.join(data_dir, "**", "metadata.json")
            json_files = sorted(glob.glob(json_pattern, recursive=True))

            for json_file in json_files:
                if task_name and task_name not in json_file:
                    continue
                try:
                    with open(json_file, "r") as f:
                        metadata = json.load(f)

                    ep_dir = os.path.dirname(json_file)

                    # Load trajectory data
                    actions_path = os.path.join(ep_dir, "actions.npy")
                    states_path = os.path.join(ep_dir, "states.npy")

                    if os.path.exists(actions_path) and os.path.exists(states_path):
                        episode = {
                            "actions": torch.from_numpy(np.load(actions_path)).float(),
                            "states": torch.from_numpy(np.load(states_path)).float(),
                            "language": metadata.get("language_instruction", ""),
                            "file_path": json_file,
                        }

                        # Store image directory for lazy loading
                        for cam in self.camera_names:
                            cam_dir = os.path.join(ep_dir, "images", cam)
                            if os.path.exists(cam_dir):
                                episode[f"image_dir_{cam}"] = cam_dir

                        episodes.append(episode)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
                    continue

                if max_episodes and len(episodes) >= max_episodes:
                    break

        # Format 3: HDF5 files
        if not episodes:
            try:
                import h5py
                hdf5_pattern = os.path.join(data_dir, "**", "*.hdf5")
                hdf5_files = sorted(glob.glob(hdf5_pattern, recursive=True))

                for hdf5_file in hdf5_files:
                    if task_name and task_name not in hdf5_file:
                        continue
                    try:
                        with h5py.File(hdf5_file, "r") as f:
                            episode = {
                                "actions": torch.from_numpy(f["actions"][()]).float(),
                                "states": torch.from_numpy(
                                    f["observations/joint_positions"][()]
                                ).float(),
                                "file_path": hdf5_file,
                            }
                            if "language_instruction" in f.attrs:
                                episode["language"] = f.attrs["language_instruction"]
                            else:
                                task_dir = os.path.basename(os.path.dirname(hdf5_file))
                                episode["language"] = task_dir.replace("_", " ")

                            for cam in self.camera_names:
                                key = f"observations/images/{cam}"
                                if key in f:
                                    episode[f"images_{cam}"] = f[key][()]

                        episodes.append(episode)
                    except Exception as e:
                        print(f"Error loading {hdf5_file}: {e}")
                        continue

                    if max_episodes and len(episodes) >= max_episodes:
                        break
            except ImportError:
                pass

        return episodes

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """Build (episode_idx, timestep) index for valid samples."""
        samples = []
        for ep_idx, episode in enumerate(self.episodes):
            T = episode["actions"].shape[0]
            # Each timestep where we can extract a full action chunk
            for t in range(T - self.action_chunk_size + 1):
                samples.append((ep_idx, t))
        return samples

    def _compute_stats(self) -> Dict[str, torch.Tensor]:
        """Compute action normalization statistics across the dataset."""
        all_actions = torch.cat([ep["actions"] for ep in self.episodes], dim=0)
        return compute_action_stats(all_actions)

    def _load_image(self, episode: Dict, cam: str, timestep: int) -> Optional[torch.Tensor]:
        """Load and preprocess a camera image."""
        # Try numpy array first
        key = f"images_{cam}"
        if key in episode:
            img_np = episode[key][timestep]
            img = Image.fromarray(img_np.astype(np.uint8))
            if self.augment and self.augment_transform is not None:
                img = self.augment_transform(img)
            return self.image_transform(img)

        # Try directory-based loading
        dir_key = f"image_dir_{cam}"
        if dir_key in episode:
            img_path = os.path.join(episode[dir_key], f"{timestep:06d}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(episode[dir_key], f"{timestep:06d}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                if self.augment and self.augment_transform is not None:
                    img = self.augment_transform(img)
                return self.image_transform(img)

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, timestep = self.samples[idx]
        episode = self.episodes[ep_idx]

        # Get action chunk: (chunk_size, action_dim)
        actions = episode["actions"][timestep: timestep + self.action_chunk_size]

        # Get current proprioceptive state
        state = episode["states"][timestep]  # (state_dim,)

        # Normalize actions
        if self.normalize:
            actions = normalize_actions(actions.unsqueeze(0), self.action_stats).squeeze(0)

        # Reshape actions for diffusion: (action_dim, chunk_size)
        # This matches AudioX's (channels, sequence_length) format
        actions = actions.permute(1, 0)  # (action_dim, chunk_size)

        # Load images for the current timestep
        video_frames = []
        for cam in self.camera_names:
            img = self._load_image(episode, cam, timestep)
            if img is not None:
                video_frames.append(img)

        # Stack camera views into video tensor
        # Shape: (num_cameras, C, H, W) - treated as temporal frames for CLIP
        if video_frames:
            video_tensor = torch.stack(video_frames, dim=0)  # (num_cams, 3, H, W)
        else:
            # Empty video placeholder
            video_tensor = torch.zeros(
                len(self.camera_names), 3, self.image_size, self.image_size
            )

        # Metadata dictionary (following AudioX's metadata format)
        info = {
            "prompt": episode.get("language", ""),
            "proprio": state,
            "video": video_tensor,
            "padding_mask": torch.ones(self.action_chunk_size),
            "episode_idx": ep_idx,
            "timestep": timestep,
        }

        return actions, info


def robotwin_collate_fn(samples):
    """Custom collation for RoboTwin dataset."""
    actions = torch.stack([s[0] for s in samples])
    metadata = [s[1] for s in samples]
    return actions, metadata


def create_robotwin_dataloader(
    data_dir: str,
    batch_size: int = 32,
    action_chunk_size: int = 64,
    image_size: int = 224,
    camera_names: List[str] = None,
    embodiment: str = "franka_dual",
    num_workers: int = 4,
    max_episodes: Optional[int] = None,
    task_name: Optional[str] = None,
    normalize: bool = True,
    action_stats: Optional[Dict] = None,
    shuffle: bool = True,
    augment: bool = True,
) -> Tuple[DataLoader, RoboTwinDataset]:
    """
    Create a DataLoader for RoboTwin data.

    Returns:
        dataloader: PyTorch DataLoader
        dataset: the dataset instance (for accessing action_stats etc.)
    """
    config = ROBOTWIN_CONFIGS.get(embodiment, ActionSpaceConfig())

    dataset = RoboTwinDataset(
        data_dir=data_dir,
        action_config=config,
        action_chunk_size=action_chunk_size,
        image_size=image_size,
        camera_names=camera_names,
        max_episodes=max_episodes,
        task_name=task_name,
        embodiment=embodiment,
        normalize=normalize,
        action_stats=action_stats,
        augment=augment,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
        collate_fn=robotwin_collate_fn,
    )

    return dataloader, dataset
