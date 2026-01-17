"""
Dataset loader for robosuite bimanual demonstrations.
Compatible with ACT training.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class BimanualDataset(Dataset):
    """
    Dataset for bimanual robosuite demonstrations.
    
    Observation format (per arm):
        - joint_pos: (7,) joint positions
        - joint_vel: (7,) joint velocities  
        - eef_pos: (3,) end-effector position
        - eef_quat: (4,) end-effector quaternion
        - gripper_qpos: (2,) gripper position
        
    Total observation: 46 dims (23 per arm)
    Action: Depends on controller (OSC_POSE = 14 dims)
    """
    
    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 50,
        use_images: bool = False,
        include_object_obs: bool = False,
        camera_names: List[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.use_images = use_images
        self.include_object_obs = include_object_obs
        self.camera_names = camera_names or ["agentview"]
        
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
        """Compute mean/std for normalization"""
        all_obs = []
        all_actions = []
        
        for ep_file in self.episode_files[:min(50, len(self.episode_files))]:
            with h5py.File(ep_file, 'r') as f:
                obs = self._load_obs_array(f)
                actions = f['actions'][:]
                all_obs.append(obs)
                all_actions.append(actions)
                
        all_obs = np.concatenate(all_obs)
        all_actions = np.concatenate(all_actions)
        
        return {
            'obs_mean': all_obs.mean(axis=0),
            'obs_std': all_obs.std(axis=0) + 1e-6,
            'action_mean': all_actions.mean(axis=0),
            'action_std': all_actions.std(axis=0) + 1e-6,
        }
        
    def _load_obs_array(self, f: h5py.File) -> np.ndarray:
        """Load observations as single array"""
        obs_grp = f['observations']
        
        # Concatenate all observation components
        components = [
            obs_grp['robot0_joint_pos'][:],
            obs_grp['robot0_eef_pos'][:],
            obs_grp['robot0_eef_quat'][:],
            obs_grp['robot0_gripper_qpos'][:],
            obs_grp['robot1_joint_pos'][:],
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
            images = {}
            if self.use_images:
                for cam in self.camera_names:
                    key = f"{cam}_image"
                    if key in f['observations']:
                        images[cam] = f['observations'][key][t]
                        
        # Normalize
        obs = (obs - self.stats['obs_mean']) / self.stats['obs_std']
        actions = (actions - self.stats['action_mean']) / self.stats['action_std']
        
        sample = {
            'obs': torch.FloatTensor(obs),
            'action': torch.FloatTensor(actions),
        }
        
        for cam, img in images.items():
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            sample[f'image_{cam}'] = torch.FloatTensor(img)
            
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
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train/val dataloaders"""
    
    dataset = BimanualDataset(data_dir, chunk_size=chunk_size, include_object_obs=include_object_obs)
    
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
