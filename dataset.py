"""
Dataset loader for robosuite bimanual demonstrations.
Compatible with ACT training with optional vision support.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import torch.nn.functional as F

# ImageNet normalization constants (used for ResNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class BimanualDataset(Dataset):
    """
    Dataset for bimanual robosuite demonstrations.

    Observation format (per arm, DoF-agnostic):
        - joint_pos: (N,) joint positions (N depends on robot, e.g., 6 for Piper, 7 for Panda)
        - joint_vel: (N,) joint velocities
        - eef_pos: (3,) end-effector position
        - eef_quat: (4,) end-effector quaternion
        - gripper_qpos: (2,) gripper position

    Total observation: 2 * (2*N + 9) dims (varies by robot)
    With object obs: +15 dims (cloth_corners: 12, cloth_center: 3)
    Action: Depends on controller (OSC_POSE = 14 dims for 2-arm)
    """
    
    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 50,
        use_images: bool = False,
        include_object_obs: bool = False,
        camera_names: List[str] = None,
        image_size: int = 224,
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.use_images = use_images
        self.include_object_obs = include_object_obs
        self.camera_names = camera_names or ["bimanual_view"]
        self.image_size = image_size
        
        # Find all episodes
        self.episode_files = sorted(self.data_dir.glob("episode_*.hdf5"))
        if not self.episode_files:
            raise ValueError(f"No episodes found in {data_dir}")
            
        print(f"Found {len(self.episode_files)} episodes")
        
        # Build index
        self.samples = []
        for ep_file in self.episode_files:
            with h5py.File(ep_file, 'r') as f:
                ep_len = f['metadata'].attrs['episode_length']
                for t in range(ep_len - chunk_size):
                    self.samples.append((ep_file, t))
                    
        print(f"Total samples: {len(self.samples)}")
        
        # Compute normalization stats
        self.stats = self._compute_stats()
        
    def _compute_stats(self) -> Dict[str, np.ndarray]:
        """Compute normalization statistics.

        For observations: Use mean/std normalization (z-score)
        For actions: Use fixed bounds [-1, 1] since OSC controller expects this range
        """
        all_obs = []

        for ep_file in self.episode_files[:min(50, len(self.episode_files))]:
            with h5py.File(ep_file, 'r') as f:
                obs = self._load_obs_array(f)
                all_obs.append(obs)

        all_obs = np.concatenate(all_obs)

        # Observation normalization: mean/std
        obs_mean = all_obs.mean(axis=0)
        obs_std = all_obs.std(axis=0) + 1e-6

        # Action normalization: fixed bounds [-1, 1] for OSC controller
        # This ensures training and deployment use the same normalization
        action_dim = 14  # OSC_POSE: 7 per arm × 2 arms
        action_low = np.full(action_dim, -1.0)
        action_high = np.full(action_dim, 1.0)

        return {
            'obs_mean': obs_mean,
            'obs_std': obs_std,
            # For actions, store bounds instead of mean/std
            'action_low': action_low,
            'action_high': action_high,
            # Also compute empirical stats for reference/debugging
            'action_mean': np.zeros(action_dim),  # Centered at 0
            'action_std': np.ones(action_dim),    # Unit scale
        }
        
    def _load_obs_array(self, f: h5py.File) -> np.ndarray:
        """Load observations as single array"""
        obs_grp = f['observations']

        # Concatenate all observation components
        # Now includes joint velocities for better temporal modeling
        components = [
            obs_grp['robot0_joint_pos'][:],
            obs_grp['robot0_joint_vel'][:],  # Added: joint velocities
            obs_grp['robot0_eef_pos'][:],
            obs_grp['robot0_eef_quat'][:],
            obs_grp['robot0_gripper_qpos'][:],
            obs_grp['robot1_joint_pos'][:],
            obs_grp['robot1_joint_vel'][:],  # Added: joint velocities
            obs_grp['robot1_eef_pos'][:],
            obs_grp['robot1_eef_quat'][:],
            obs_grp['robot1_gripper_qpos'][:],
        ]

        if self.include_object_obs:
            if 'cloth_corners' in obs_grp:
                components.append(obs_grp['cloth_corners'][:])
            if 'cloth_center' in obs_grp:
                components.append(obs_grp['cloth_center'][:])

        return np.concatenate(components, axis=1)
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_file, t = self.samples[idx]
        
        with h5py.File(ep_file, 'r') as f:
            # Load observation at time t
            obs = self._load_obs_array(f)[t]
            
            # Load action chunk
            actions = f['actions'][t:t + self.chunk_size]
            
            # Pad if needed
            if len(actions) < self.chunk_size:
                pad_len = self.chunk_size - len(actions)
                actions = np.pad(actions, ((0, pad_len), (0, 0)), mode='edge')
                
            # Load images if requested
            images_list = []
            if self.use_images:
                for cam in self.camera_names:
                    key = f"{cam}_image"
                    if key in f['observations']:
                        images_list.append(f['observations'][key][t])

        # Normalize observations (z-score)
        obs = (obs - self.stats['obs_mean']) / self.stats['obs_std']

        # Normalize actions to [-1, 1] using fixed bounds
        # Since OSC actions are already in [-1, 1], this is essentially identity
        # but we keep it explicit for consistency
        action_low = self.stats['action_low']
        action_high = self.stats['action_high']
        actions = 2 * (actions - action_low) / (action_high - action_low) - 1
        # Clip to ensure bounds (handles any outliers in data)
        actions = np.clip(actions, -1.0, 1.0)

        sample = {
            'obs': torch.FloatTensor(obs),
            'action': torch.FloatTensor(actions),
        }

        # Process images if using vision
        if self.use_images and images_list:
            processed_images = []
            for img in images_list:
                # Convert to float [0, 1]
                img = img.astype(np.float32) / 255.0
                # HWC -> CHW
                img = np.transpose(img, (2, 0, 1))
                processed_images.append(img)

            # Stack cameras: (N, C, H, W)
            images_tensor = torch.FloatTensor(np.stack(processed_images))

            # Resize to target size if needed
            if images_tensor.shape[-1] != self.image_size:
                images_tensor = F.interpolate(
                    images_tensor,
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False,
                )

            # If single camera, squeeze the camera dimension
            if len(self.camera_names) == 1:
                images_tensor = images_tensor.squeeze(0)  # (C, H, W)

            sample['images'] = images_tensor

        return sample
    
    def get_action_dim(self) -> int:
        """Get action dimension"""
        with h5py.File(self.episode_files[0], 'r') as f:
            return f['actions'].shape[1]
            
    def get_obs_dim(self) -> int:
        """Get observation dimension"""
        with h5py.File(self.episode_files[0], 'r') as f:
            return self._load_obs_array(f).shape[1]


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    chunk_size: int = 50,
    train_split: float = 0.9,
    num_workers: int = 4,
    include_object_obs: bool = False,
    use_images: bool = False,
    camera_names: List[str] = None,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train/val dataloaders.

    Args:
        data_dir: Path to demonstration data directory
        batch_size: Batch size for dataloaders
        chunk_size: Action chunk size
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of dataloader workers
        include_object_obs: Include object observations (cloth corners, etc.)
        use_images: Load camera images for vision-based training
        camera_names: List of camera names to load
        image_size: Target image size (for ResNet, typically 224)

    Returns:
        train_loader, val_loader, stats
    """
    dataset = BimanualDataset(
        data_dir,
        chunk_size=chunk_size,
        include_object_obs=include_object_obs,
        use_images=use_images,
        camera_names=camera_names,
        image_size=image_size,
    )
    
    # Split
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, dataset.stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/bimanual")
    args = parser.parse_args()
    
    dataset = BimanualDataset(args.data_dir)
    print(f"\nDataset size: {len(dataset)}")
    print(f"Observation dim: {dataset.get_obs_dim()}")
    print(f"Action dim: {dataset.get_action_dim()}")
    
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Obs shape: {sample['obs'].shape}")
    print(f"Action shape: {sample['action'].shape}")
