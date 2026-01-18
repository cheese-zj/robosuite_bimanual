"""Deploy a bimanual ACT policy on a two-arm robosuite environment."""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import numpy as np
import torch

from collect_robosuite import create_bimanual_env
from train_act import load_policy


def flatten_bimanual_obs(obs: Dict) -> np.ndarray:
    """Flatten bimanual observations into a single vector.

    Observation format (per arm, 30 dims each):
        - joint_pos: (7,) joint positions
        - joint_vel: (7,) joint velocities
        - eef_pos: (3,) end-effector position
        - eef_quat: (4,) end-effector quaternion
        - gripper_qpos: (2,) gripper position

    Total: 60 dims (30 per arm) + optional object obs
    """
    components = [
        obs["robot0_joint_pos"],
        obs["robot0_joint_vel"],  # Added: joint velocities
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
        obs["robot1_joint_pos"],
        obs["robot1_joint_vel"],  # Added: joint velocities
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
    """Handles observation and action normalization/unnormalization.

    Observations: z-score normalization (mean/std)
    Actions: fixed bounds [-1, 1] for OSC controller compatibility
    """

    def __init__(self, stats: Optional[Dict]):
        self.obs_mean = None
        self.obs_std = None
        self.action_low = None
        self.action_high = None

        if stats:
            self.obs_mean = np.asarray(stats.get("obs_mean"))
            self.obs_std = np.asarray(stats.get("obs_std"))
            # Use fixed bounds for actions
            self.action_low = np.asarray(stats.get("action_low", -1.0))
            self.action_high = np.asarray(stats.get("action_high", 1.0))

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using z-score."""
        if self.obs_mean is None or self.obs_std is None:
            return obs
        return (obs - self.obs_mean) / self.obs_std

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Unnormalize action from [-1, 1] to original bounds.

        Since OSC expects [-1, 1] and we normalize to [-1, 1],
        this is essentially identity but we keep it for consistency.
        """
        if self.action_low is None or self.action_high is None:
            return action
        # Convert from [-1, 1] to [action_low, action_high]
        action = (action + 1) / 2 * (self.action_high - self.action_low) + self.action_low
        # Clip to valid range for safety
        return np.clip(action, self.action_low, self.action_high)


class ChunkedPolicyAdapter:
    """Adapter for executing chunked action policies with receding horizon.

    Args:
        policy: ACT policy model
        normalizer: Observation/action normalizer
        device: Torch device
        expected_obs_dim: Expected observation dimension (for truncation)
        recompute_freq: How often to recompute chunks (receding horizon)
            - Set to chunk_size for full chunk execution
            - Set to smaller value (e.g., 10) for error correction
        camera_names: Camera names for vision-based policies
    """

    def __init__(
        self,
        policy,
        normalizer: ObsActionNormalizer,
        device: str,
        expected_obs_dim: Optional[int] = None,
        recompute_freq: Optional[int] = None,
        camera_names: Optional[list] = None,
    ):
        self.policy = policy
        self.normalizer = normalizer
        self.device = device
        self.expected_obs_dim = expected_obs_dim
        self.camera_names = camera_names or ["bimanual_view"]

        # Get chunk size from policy config
        self.chunk_size = policy.config.chunk_size
        self.uses_images = policy.config.use_images

        # Receding horizon: recompute every N steps (default: full chunk)
        self.recompute_freq = recompute_freq or self.chunk_size

        self.reset()

    def reset(self):
        """Reset action buffer for new episode."""
        self.chunk = None
        self.chunk_idx = 0

    def _process_images(self, obs_dict: Dict) -> Optional[torch.Tensor]:
        """Process camera images for vision-based policy."""
        if not self.uses_images:
            return None

        images = []
        for cam in self.camera_names:
            key = f"{cam}_image"
            if key in obs_dict:
                img = obs_dict[key].astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                images.append(img)

        if not images:
            return None

        images_tensor = torch.FloatTensor(np.stack(images)).to(self.device)
        if len(self.camera_names) == 1:
            images_tensor = images_tensor.squeeze(0)

        return images_tensor

    def predict(self, obs_dict: Dict) -> np.ndarray:
        """Predict action from observation.

        Uses receding horizon: recomputes action chunk every recompute_freq steps
        to correct for accumulated errors.
        """
        obs_vec = flatten_bimanual_obs(obs_dict)
        if self.expected_obs_dim is not None:
            if obs_vec.size > self.expected_obs_dim:
                obs_vec = obs_vec[: self.expected_obs_dim]
            elif obs_vec.size < self.expected_obs_dim:
                raise ValueError(
                    f"Observation dim {obs_vec.size} is smaller than expected {self.expected_obs_dim}"
                )
        obs_vec = self.normalizer.normalize_obs(obs_vec)

        # Recompute chunk using receding horizon strategy
        if self.chunk is None or self.chunk_idx >= self.recompute_freq:
            obs_tensor = torch.as_tensor(
                obs_vec, dtype=torch.float32, device=self.device
            )

            # Get images for vision-based policy
            images_tensor = self._process_images(obs_dict)

            self.chunk = self.policy.predict(obs_tensor, images_tensor)
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
    recompute_freq: Optional[int] = None,
    camera_names: Optional[list] = None,
):
    """Deploy a trained ACT policy in the environment.

    Args:
        checkpoint: Path to model checkpoint
        task: Task name
        robot: Robot type
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to render
        input_ref_frame: Input reference frame for controller
        two_arm_cloth: Use TwoArmClothFold environment
        env_configuration: Environment configuration
        recompute_freq: Receding horizon frequency (default: chunk_size)
        camera_names: Camera names for vision policy
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, stats = load_policy(checkpoint, device=device)
    normalizer = ObsActionNormalizer(stats)

    # Get expected observation dimension
    expected_obs_dim = None
    if stats and stats.get("obs_mean") is not None:
        expected_obs_dim = int(np.asarray(stats["obs_mean"]).shape[0])
    elif hasattr(policy, "config"):
        expected_obs_dim = int(getattr(policy.config, "obs_dim", 0)) or None

    # Check if policy uses vision
    uses_images = policy.config.use_images if hasattr(policy, "config") else False
    if camera_names is None:
        camera_names = ["bimanual_view"]

    if two_arm_cloth:
        from envs.two_arm_cloth_fold import TwoArmClothFold

        if env_configuration == "front-back":
            env_configuration = "opposed"
        robots_list = [robot, robot] if isinstance(robot, str) else list(robot)
        env = TwoArmClothFold(
            robots=robots_list,
            env_configuration=env_configuration,
            has_renderer=render,
            has_offscreen_renderer=render or uses_images,
            use_camera_obs=uses_images,
            use_object_obs=True,
            render_camera="bimanual_view" if render else None,
            camera_names=camera_names if uses_images else None,
        )
    else:
        env = create_bimanual_env(
            robots=robot,
            task=task,
            has_renderer=render,
            has_offscreen_renderer=render or uses_images,
            use_camera_obs=uses_images,
            input_ref_frame=input_ref_frame,
        )

    policy_adapter = ChunkedPolicyAdapter(
        policy,
        normalizer,
        device,
        expected_obs_dim=expected_obs_dim,
        recompute_freq=recompute_freq,
        camera_names=camera_names,
    )

    for ep in range(episodes):
        obs = env.reset()
        policy_adapter.reset()
        done = False
        step = 0
        step_limit = max_steps or env.horizon
        total_reward = 0
        info = {}

        while not done and step < step_limit:
            action = policy_adapter.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if render:
                env.render()

            step += 1

        success = info.get("success", False)
        status = "SUCCESS" if success else "DONE"
        print(f"Episode {ep + 1}/{episodes}: {status} | Steps: {step} | Reward: {total_reward:.2f}")

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
    parser.add_argument(
        "--recompute_freq",
        type=int,
        default=10,
        help="Receding horizon: recompute chunks every N steps (default: 10)",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["bimanual_view"],
        help="Camera names for vision policy",
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
        recompute_freq=args.recompute_freq,
        camera_names=args.camera_names,
    )


if __name__ == "__main__":
    main()
