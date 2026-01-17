"""Mirror-augment bimanual datasets for symmetric training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from mirror_transform import MirrorConfig


def mirror_position(pos: np.ndarray, axis: str) -> np.ndarray:
    mirrored = np.array(pos, copy=True)
    if axis == "x":
        mirrored[..., 0] *= -1
    elif axis == "y":
        mirrored[..., 1] *= -1
    elif axis == "z":
        mirrored[..., 2] *= -1
    return mirrored


def mirror_quaternion(quat: np.ndarray, axis: str) -> np.ndarray:
    mirrored = np.array(quat, copy=True)
    if axis == "y":
        mirrored[..., 2] *= -1
        mirrored[..., 3] *= -1
    elif axis == "x":
        mirrored[..., 1] *= -1
        mirrored[..., 3] *= -1
    return mirrored


def mirror_arm_action(action: np.ndarray, axis: str) -> np.ndarray:
    mirrored = np.array(action, copy=True)
    if axis == "y":
        mirrored[..., 1] *= -1
        mirrored[..., 3] *= -1
        mirrored[..., 5] *= -1
    elif axis == "x":
        mirrored[..., 0] *= -1
        mirrored[..., 4] *= -1
        mirrored[..., 5] *= -1
    return mirrored


def mirror_arm_obs(obs: Dict[str, np.ndarray], config: MirrorConfig) -> Dict[str, np.ndarray]:
    out = {}
    if obs.get("joint_pos") is not None:
        out["joint_pos"] = obs["joint_pos"] * config.joint_flip_mask
    if obs.get("joint_vel") is not None:
        out["joint_vel"] = obs["joint_vel"] * config.joint_flip_mask
    if obs.get("eef_pos") is not None:
        out["eef_pos"] = mirror_position(obs["eef_pos"], config.symmetry_axis)
    if obs.get("eef_quat") is not None:
        out["eef_quat"] = mirror_quaternion(obs["eef_quat"], config.symmetry_axis)
    if obs.get("gripper_qpos") is not None:
        out["gripper_qpos"] = np.array(obs["gripper_qpos"], copy=True)
    return out


def mirror_bimanual_obs(
    obs0: Dict[str, np.ndarray],
    obs1: Dict[str, np.ndarray],
    config: MirrorConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    m0 = mirror_arm_obs(obs0, config)
    m1 = mirror_arm_obs(obs1, config)
    if config.swap_arms:
        return m1, m0
    return m0, m1


def next_episode_index(output_dir: Path) -> int:
    existing = list(output_dir.glob("episode_*.hdf5"))
    if not existing:
        return 0
    indices = []
    for path in existing:
        stem = path.stem
        try:
            indices.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    return max(indices) + 1 if indices else 0


def _maybe_flip_image(data: np.ndarray, flip: bool) -> np.ndarray:
    if not flip:
        return data
    return np.flip(data, axis=2)


def mirror_episode(
    input_path: Path,
    output_path: Path,
    config: MirrorConfig,
    flip_images: bool,
):
    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        meta = fout.create_group("metadata")
        for key, value in fin["metadata"].attrs.items():
            meta.attrs[key] = value
        meta.attrs["mirrored"] = True
        meta.attrs["source_episode"] = input_path.name

        obs_in = fin["observations"]
        obs_out = fout.create_group("observations")

        def read_obs(prefix: str, key: str) -> Optional[np.ndarray]:
            full_key = f"{prefix}_{key}"
            if full_key in obs_in:
                return obs_in[full_key][:]
            return None

        obs0 = {
            "joint_pos": read_obs("robot0", "joint_pos"),
            "joint_vel": read_obs("robot0", "joint_vel"),
            "eef_pos": read_obs("robot0", "eef_pos"),
            "eef_quat": read_obs("robot0", "eef_quat"),
            "gripper_qpos": read_obs("robot0", "gripper_qpos"),
        }
        obs1 = {
            "joint_pos": read_obs("robot1", "joint_pos"),
            "joint_vel": read_obs("robot1", "joint_vel"),
            "eef_pos": read_obs("robot1", "eef_pos"),
            "eef_quat": read_obs("robot1", "eef_quat"),
            "gripper_qpos": read_obs("robot1", "gripper_qpos"),
        }

        m0, m1 = mirror_bimanual_obs(obs0, obs1, config)

        for key, data in m0.items():
            if data is not None:
                obs_out.create_dataset(f"robot0_{key}", data=data)
        for key, data in m1.items():
            if data is not None:
                obs_out.create_dataset(f"robot1_{key}", data=data)

        handled_keys = set()
        if "cloth_corners" in obs_in:
            corners = obs_in["cloth_corners"][:]
            corners = corners.reshape(corners.shape[0], -1, 3)
            corners = mirror_position(corners, config.symmetry_axis)
            obs_out.create_dataset("cloth_corners", data=corners.reshape(corners.shape[0], -1))
            handled_keys.add("cloth_corners")
        if "cloth_center" in obs_in:
            centers = obs_in["cloth_center"][:]
            obs_out.create_dataset(
                "cloth_center", data=mirror_position(centers, config.symmetry_axis)
            )
            handled_keys.add("cloth_center")

        for key in obs_in.keys():
            if key.startswith("robot0_") or key.startswith("robot1_"):
                continue
            if key in handled_keys:
                continue
            data = obs_in[key][:]
            if "image" in key:
                data = _maybe_flip_image(data, flip_images)
                obs_out.create_dataset(key, data=data, compression="gzip")
            else:
                obs_out.create_dataset(key, data=data)

        actions = fin["actions"][:]
        arm0 = actions[:, :7]
        arm1 = actions[:, 7:]
        m_arm0 = mirror_arm_action(arm0, config.symmetry_axis)
        m_arm1 = mirror_arm_action(arm1, config.symmetry_axis)
        mirrored_actions = np.concatenate([m_arm1, m_arm0], axis=1) if config.swap_arms else np.concatenate(
            [m_arm0, m_arm1], axis=1
        )

        fout.create_dataset("actions", data=mirrored_actions)
        if "rewards" in fin:
            fout.create_dataset("rewards", data=fin["rewards"][:])


def main():
    parser = argparse.ArgumentParser(description="Mirror-augment bimanual dataset")
    parser.add_argument("--input_dir", type=str, default="data/bimanual")
    parser.add_argument("--output_dir", type=str, default="data/bimanual_mirrored")
    parser.add_argument("--symmetry_axis", type=str, default="y", choices=["x", "y", "z"])
    parser.add_argument("--swap_arms", action="store_true", default=True)
    parser.add_argument("--no_swap_arms", action="store_false", dest="swap_arms")
    parser.add_argument("--flip_images", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MirrorConfig(symmetry_axis=args.symmetry_axis, swap_arms=args.swap_arms)

    episode_files = sorted(input_dir.glob("episode_*.hdf5"))
    if not episode_files:
        raise ValueError(f"No episodes found in {input_dir}")

    next_index = next_episode_index(output_dir)
    for ep_file in episode_files:
        out_file = output_dir / f"episode_{next_index:04d}.hdf5"
        mirror_episode(ep_file, out_file, config, args.flip_images)
        print(f"Mirrored {ep_file.name} -> {out_file.name}")
        next_index += 1


if __name__ == "__main__":
    main()
