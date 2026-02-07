"""
RoboTwin Dataset Loader for RobotX.

Supports RoboTwin HDF5 format with the following structure:
  - joint_action/vector: (T, action_dim) combined joint actions
  - joint_action/left_arm: (T, 6) left arm joints
  - joint_action/right_arm: (T, 6) right arm joints
  - joint_action/left_gripper: (T,) left gripper
  - joint_action/right_gripper: (T,) right gripper
  - observation/{camera_name}/rgb: (T,) encoded RGB images
"""

import os
import io
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple
from PIL import Image
from torchvision import transforms

from .action_space import (
    ActionSpaceConfig,
    normalize_actions,
    denormalize_actions,
    compute_action_stats,
)


# Aloha-AgileX 配置: 每臂6 DoF + 1 gripper = 14维
ALOHA_AGILEX_CONFIG = ActionSpaceConfig(
    num_arms=2,           # 双臂
    joints_per_arm=6,     # 每臂6个关节
    gripper_dim=1,        # 每臂1个夹爪维度
)


class RoboTwinHDF5Dataset(Dataset):
    """
    Dataset for loading RoboTwin HDF5 trajectory data.

    Specifically designed for the Aloha-AgileX format with:
      - joint_action/vector as combined action
      - observation/{camera}/rgb as encoded images

    Args:
        data_dir: directory containing episode HDF5 files
        action_chunk_size: number of future actions per sample (default 64)
        image_size: resize images to (image_size, image_size)
        camera_names: list of camera views to use
        max_episodes: maximum number of episodes to load
        task_description: language instruction for this task
        normalize: whether to normalize actions
        augment: whether to apply data augmentation
    """

    def __init__(
        self,
        data_dir: str,
        action_chunk_size: int = 64,
        image_size: int = 224,
        camera_names: List[str] = None,
        max_episodes: Optional[int] = None,
        task_description: str = "complete the manipulation task",
        normalize: bool = True,
        augment: bool = True,
    ):
        super().__init__()

        self.action_chunk_size = action_chunk_size
        self.image_size = image_size
        self.camera_names = camera_names or ["front_camera", "head_camera"]
        self.task_description = task_description
        self.normalize = normalize
        self.augment = augment
        self.action_dim = 14  # Aloha-AgileX: 6+1+6+1=14

        # Image transforms (CLIP-compatible)
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])

        self.augment_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]) if augment else None

        # Find all HDF5 files
        self.hdf5_files = self._find_hdf5_files(data_dir, max_episodes)
        print(f"Found {len(self.hdf5_files)} HDF5 episode files")

        # Load all episodes into memory (for small datasets like 500 episodes)
        self.episodes = []
        self._load_all_episodes()

        # Build sample index
        self.samples = self._build_sample_index()
        print(f"Total samples: {len(self.samples)}")

        # Compute action statistics
        if self.normalize and len(self.episodes) > 0:
            self.action_stats = self._compute_stats()
            print(f"Action stats - mean: {self.action_stats['mean'][:4]}..., "
                  f"std: {self.action_stats['std'][:4]}...")
        else:
            self.action_stats = None

    def _find_hdf5_files(self, data_dir: str, max_episodes: Optional[int]) -> List[str]:
        """Find all HDF5 files in the data directory."""
        # Try multiple patterns
        patterns = [
            os.path.join(data_dir, "*.hdf5"),
            os.path.join(data_dir, "**", "*.hdf5"),
            os.path.join(data_dir, "data", "*.hdf5"),
        ]

        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))

        files = sorted(list(set(files)))  # Remove duplicates and sort

        if max_episodes and len(files) > max_episodes:
            files = files[:max_episodes]

        return files

    def _load_all_episodes(self):
        """Load all episodes into memory."""
        import h5py

        for hdf5_path in self.hdf5_files:
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    # Load actions - use vector if available, else concatenate
                    if 'joint_action/vector' in f:
                        actions = f['joint_action/vector'][:]
                    else:
                        # Concatenate individual components
                        left_arm = f['joint_action/left_arm'][:]
                        left_grip = f['joint_action/left_gripper'][:].reshape(-1, 1)
                        right_arm = f['joint_action/right_arm'][:]
                        right_grip = f['joint_action/right_gripper'][:].reshape(-1, 1)
                        actions = np.concatenate([left_arm, left_grip, right_arm, right_grip], axis=1)

                    # Actions are also the states (current joint positions)
                    states = actions.copy()

                    # Load images for each camera
                    images = {}
                    for cam in self.camera_names:
                        cam_key = f'observation/{cam}/rgb'
                        if cam_key in f:
                            # Images are stored as encoded byte strings
                            images[cam] = f[cam_key][:]

                    episode = {
                        'actions': torch.from_numpy(actions).float(),
                        'states': torch.from_numpy(states).float(),
                        'images': images,
                        'file_path': hdf5_path,
                        'length': actions.shape[0],
                    }
                    self.episodes.append(episode)

            except Exception as e:
                print(f"Error loading {hdf5_path}: {e}")
                continue

        print(f"Loaded {len(self.episodes)} episodes successfully")

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """Build (episode_idx, timestep) index for valid samples."""
        samples = []
        for ep_idx, episode in enumerate(self.episodes):
            T = episode['length']
            # Each timestep where we can extract a full action chunk
            for t in range(max(0, T - self.action_chunk_size + 1)):
                samples.append((ep_idx, t))
        return samples

    def _compute_stats(self) -> Dict[str, torch.Tensor]:
        """Compute action normalization statistics."""
        all_actions = torch.cat([ep['actions'] for ep in self.episodes], dim=0)
        return compute_action_stats(all_actions)

    def _decode_image(self, encoded_bytes) -> Image.Image:
        """Decode image from byte string."""
        if isinstance(encoded_bytes, bytes):
            img_bytes = encoded_bytes
        elif isinstance(encoded_bytes, np.ndarray):
            img_bytes = encoded_bytes.tobytes()
        else:
            img_bytes = bytes(encoded_bytes)

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            return img
        except Exception as e:
            # Return a blank image if decoding fails
            return Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, timestep = self.samples[idx]
        episode = self.episodes[ep_idx]

        # Get action chunk
        end_t = min(timestep + self.action_chunk_size, episode['length'])
        actions = episode['actions'][timestep:end_t].clone()

        # Pad if needed
        if actions.shape[0] < self.action_chunk_size:
            pad_size = self.action_chunk_size - actions.shape[0]
            # Repeat last action for padding
            last_action = actions[-1:].repeat(pad_size, 1)
            actions = torch.cat([actions, last_action], dim=0)

        # Get current state (proprioception)
        state = episode['states'][timestep].clone()

        # Normalize actions
        if self.normalize and self.action_stats is not None:
            actions = normalize_actions(actions.unsqueeze(0), self.action_stats).squeeze(0)

        # Reshape for diffusion: (action_dim, chunk_size)
        actions = actions.permute(1, 0)  # (14, 64)

        # Load and process images
        video_frames = []
        for cam in self.camera_names:
            if cam in episode['images'] and timestep < len(episode['images'][cam]):
                encoded = episode['images'][cam][timestep]
                img = self._decode_image(encoded)

                if self.augment and self.augment_transform is not None:
                    img = self.augment_transform(img)

                img_tensor = self.image_transform(img)
                video_frames.append(img_tensor)

        # Stack camera views
        if video_frames:
            video_tensor = torch.stack(video_frames, dim=0)
        else:
            video_tensor = torch.zeros(len(self.camera_names), 3, self.image_size, self.image_size)

        # Metadata
        info = {
            'prompt': self.task_description,
            'proprio': state,
            'video': video_tensor,
            'padding_mask': torch.ones(self.action_chunk_size),
            'episode_idx': ep_idx,
            'timestep': timestep,
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
    task_description: str = "complete the manipulation task",
    num_workers: int = 4,
    max_episodes: Optional[int] = None,
    normalize: bool = True,
    action_stats: Optional[Dict] = None,
    shuffle: bool = True,
    augment: bool = True,
    **kwargs,
) -> Tuple[DataLoader, RoboTwinHDF5Dataset]:
    """
    Create a DataLoader for RoboTwin HDF5 data.

    Args:
        data_dir: path to directory containing HDF5 files
        batch_size: batch size
        action_chunk_size: number of future actions to predict
        image_size: image resize dimension
        camera_names: list of camera names (e.g., ["front_camera", "head_camera"])
        task_description: language instruction
        num_workers: dataloader workers
        max_episodes: limit number of episodes
        normalize: normalize actions
        action_stats: precomputed stats (if None, computed from data)
        shuffle: shuffle data
        augment: apply augmentation

    Returns:
        dataloader, dataset
    """
    if camera_names is None:
        camera_names = ["front_camera", "head_camera"]

    dataset = RoboTwinHDF5Dataset(
        data_dir=data_dir,
        action_chunk_size=action_chunk_size,
        image_size=image_size,
        camera_names=camera_names,
        max_episodes=max_episodes,
        task_description=task_description,
        normalize=normalize,
        augment=augment,
    )

    # Use provided action_stats if given
    if action_stats is not None:
        dataset.action_stats = action_stats

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
