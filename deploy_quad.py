"""Deploy a bimanual ACT policy on a two-arm robosuite environment."""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import numpy as np
import torch

from collect_robosuite import create_bimanual_env
from train_act import load_policy


def flatten_bimanual_obs(obs: Dict) -> np.ndarray:
    components = [
        obs["robot0_joint_pos"],
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
        obs["robot1_joint_pos"],
        obs["robot1_eef_pos"],
        obs["robot1_eef_quat"],
        obs["robot1_gripper_qpos"],
    ]
    if "cloth_corners" in obs:
        components.append(obs["cloth_corners"])
    if "cloth_center" in obs:
        components.append(obs["cloth_center"])
    return np.concatenate([np.asarray(c) for c in components], axis=0)


class ObsActionNormalizer:
    def __init__(self, stats: Optional[Dict]):
        self.obs_mean = None
        self.obs_std = None
        self.action_mean = None
        self.action_std = None

        if stats:
            self.obs_mean = np.asarray(stats.get("obs_mean"))
            self.obs_std = np.asarray(stats.get("obs_std"))
            self.action_mean = np.asarray(stats.get("action_mean"))
            self.action_std = np.asarray(stats.get("action_std"))

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mean is None or self.obs_std is None:
            return obs
        return (obs - self.obs_mean) / self.obs_std

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mean is None or self.action_std is None:
            return action
        return action * self.action_std + self.action_mean


class ChunkedPolicyAdapter:
    def __init__(
        self,
        policy,
        normalizer: ObsActionNormalizer,
        device: str,
        expected_obs_dim: Optional[int] = None,
    ):
        self.policy = policy
        self.normalizer = normalizer
        self.device = device
        self.expected_obs_dim = expected_obs_dim
        self.reset()

    def reset(self):
        self.chunk = None
        self.chunk_idx = 0

    def predict(self, obs_dict: Dict) -> np.ndarray:
        obs_vec = flatten_bimanual_obs(obs_dict)
        if self.expected_obs_dim is not None:
            if obs_vec.size > self.expected_obs_dim:
                obs_vec = obs_vec[: self.expected_obs_dim]
            elif obs_vec.size < self.expected_obs_dim:
                raise ValueError(
                    f"Observation dim {obs_vec.size} is smaller than expected {self.expected_obs_dim}"
                )
        obs_vec = self.normalizer.normalize_obs(obs_vec)

        if self.chunk is None or self.chunk_idx >= len(self.chunk):
            obs_tensor = torch.as_tensor(
                obs_vec, dtype=torch.float32, device=self.device
            )
            self.chunk = self.policy.predict(obs_tensor)
            self.chunk = self.normalizer.unnormalize_action(self.chunk)
            self.chunk_idx = 0

        action = self.chunk[self.chunk_idx]
        self.chunk_idx += 1
        return action


def run_deploy(
    checkpoint: str,
    task: str,
    robot: str,
    episodes: int,
    max_steps: Optional[int],
    render: bool,
    input_ref_frame: Optional[str],
    two_arm_cloth: bool,
    env_configuration: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, stats = load_policy(checkpoint, device=device)
    normalizer = ObsActionNormalizer(stats)
    expected_obs_dim = None
    if stats and stats.get("obs_mean") is not None:
        expected_obs_dim = int(np.asarray(stats["obs_mean"]).shape[0])
    elif hasattr(policy, "config"):
        expected_obs_dim = int(getattr(policy.config, "obs_dim", 0)) or None

    if two_arm_cloth:
        from envs.two_arm_cloth_fold import TwoArmClothFold

        if env_configuration == "front-back":
            env_configuration = "opposed"
        robots_list = [robot, robot] if isinstance(robot, str) else list(robot)
        env = TwoArmClothFold(
            robots=robots_list,
            env_configuration=env_configuration,
            has_renderer=render,
            has_offscreen_renderer=render,
            use_camera_obs=False,
            use_object_obs=True,
            render_camera="bimanual_view",
        )
    else:
        env = create_bimanual_env(
            robots=robot,
            task=task,
            has_renderer=render,
            has_offscreen_renderer=render,
            use_camera_obs=False,
            input_ref_frame=input_ref_frame,
        )

    policy_adapter = ChunkedPolicyAdapter(
        policy, normalizer, device, expected_obs_dim=expected_obs_dim
    )

    for ep in range(episodes):
        obs = env.reset()
        policy_adapter.reset()
        done = False
        step = 0
        step_limit = max_steps or env.horizon
        info = {}

        while not done and step < step_limit:
            action = policy_adapter.predict(obs)
            obs, reward, done, info = env.step(action)

            if render:
                env.render()

            step += 1

        print(f"Episode {ep + 1}/{episodes} done")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Deploy bimanual ACT policy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--task", type=str, default="TwoArmLift")
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--input_ref_frame", type=str, default="world", choices=["base", "world"]
    )
    parser.add_argument(
        "--two_arm_cloth",
        action="store_true",
        help="Deploy in TwoArmClothFold environment",
    )
    parser.add_argument(
        "--env_configuration",
        type=str,
        default="parallel",
        help="Two-arm cloth layout (parallel/opposed/single-robot)",
    )

    args = parser.parse_args()

    run_deploy(
        checkpoint=args.checkpoint,
        task=args.task,
        robot=args.robot,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        input_ref_frame=args.input_ref_frame,
        two_arm_cloth=args.two_arm_cloth,
        env_configuration=args.env_configuration,
    )


if __name__ == "__main__":
    main()
