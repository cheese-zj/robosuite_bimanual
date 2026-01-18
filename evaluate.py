"""
Policy evaluation script for ACT models.

Runs trained policies in the simulation environment and computes success metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import torch

from train_act import load_policy
from deploy_quad import flatten_bimanual_obs, ObsActionNormalizer


def evaluate_policy(
    checkpoint_path: str,
    task: str = "TwoArmClothFold",
    robot: str = "Panda",
    num_episodes: int = 20,
    max_steps: Optional[int] = None,
    render: bool = False,
    save_results: bool = True,
    results_dir: str = "eval_results",
    env_configuration: str = "parallel",
    use_images: bool = False,
    camera_names: List[str] = None,
    recompute_freq: int = 50,
) -> Dict:
    """
    Evaluate a trained ACT policy in the simulation environment.

    Args:
        checkpoint_path: Path to trained model checkpoint
        task: Task name (TwoArmClothFold, TwoArmLift, etc.)
        robot: Robot type (Panda, UR5e, etc.)
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode (None = use env default)
        render: Whether to render during evaluation
        save_results: Whether to save results to file
        results_dir: Directory to save results
        env_configuration: Environment configuration (parallel, opposed)
        use_images: Whether the policy uses vision
        camera_names: Camera names for vision policy
        recompute_freq: How often to recompute action chunks

    Returns:
        Dictionary with evaluation metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load policy
    print(f"Loading policy from {checkpoint_path}")
    policy, stats = load_policy(checkpoint_path, device=device)
    normalizer = ObsActionNormalizer(stats)

    # Get expected observation dimension
    expected_obs_dim = None
    if stats and stats.get("obs_mean") is not None:
        expected_obs_dim = int(np.asarray(stats["obs_mean"]).shape[0])

    # Get config from policy
    config = policy.config
    chunk_size = config.chunk_size
    policy_uses_images = config.use_images

    if camera_names is None:
        camera_names = ["bimanual_view"]

    # Create environment
    if task == "TwoArmClothFold":
        from envs.two_arm_cloth_fold import TwoArmClothFold

        if env_configuration == "front-back":
            env_configuration = "opposed"
        robots_list = [robot, robot]
        env = TwoArmClothFold(
            robots=robots_list,
            env_configuration=env_configuration,
            has_renderer=render,
            has_offscreen_renderer=render or policy_uses_images,
            use_camera_obs=policy_uses_images,
            use_object_obs=True,
            render_camera="bimanual_view" if render else None,
            camera_names=camera_names if policy_uses_images else None,
        )
    else:
        from collect_robosuite import create_bimanual_env

        env = create_bimanual_env(
            robots=robot,
            task=task,
            has_renderer=render,
            has_offscreen_renderer=render or policy_uses_images,
            use_camera_obs=policy_uses_images,
        )

    step_limit = max_steps or env.horizon

    # Evaluation metrics
    results = {
        "checkpoint": checkpoint_path,
        "task": task,
        "robot": robot,
        "num_episodes": num_episodes,
        "max_steps": step_limit,
        "chunk_size": chunk_size,
        "recompute_freq": recompute_freq,
        "episodes": [],
    }

    successes = 0
    total_rewards = []
    episode_lengths = []

    print(f"\nEvaluating {num_episodes} episodes...")
    print(f"Task: {task}, Robot: {robot}")
    print(f"Chunk size: {chunk_size}, Recompute freq: {recompute_freq}")

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        episode_reward = 0
        success = False

        # Action chunk buffer
        action_chunk = None
        chunk_idx = 0

        while not done and step < step_limit:
            # Get observation
            obs_vec = flatten_bimanual_obs(obs)
            if expected_obs_dim is not None:
                if obs_vec.size > expected_obs_dim:
                    obs_vec = obs_vec[:expected_obs_dim]
            obs_vec = normalizer.normalize_obs(obs_vec)
            obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=device)

            # Get images if policy uses vision
            images_tensor = None
            if policy_uses_images:
                images = []
                for cam in camera_names:
                    key = f"{cam}_image"
                    if key in obs:
                        img = obs[key].astype(np.float32) / 255.0
                        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                        images.append(img)
                if images:
                    images_tensor = torch.FloatTensor(np.stack(images)).to(device)
                    if len(camera_names) == 1:
                        images_tensor = images_tensor.squeeze(0)

            # Recompute action chunk if needed (receding horizon)
            if action_chunk is None or chunk_idx >= recompute_freq:
                action_chunk = policy.predict(obs_tensor, images_tensor)
                action_chunk = normalizer.unnormalize_action(action_chunk)
                chunk_idx = 0

            # Execute action
            action = action_chunk[chunk_idx]
            chunk_idx += 1

            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1

            if render:
                env.render()

            # Check for success
            if info.get("success", False):
                success = True

        # Record episode results
        episode_result = {
            "episode": ep,
            "steps": step,
            "reward": episode_reward,
            "success": success,
        }
        results["episodes"].append(episode_result)

        if success:
            successes += 1
        total_rewards.append(episode_reward)
        episode_lengths.append(step)

        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep + 1}/{num_episodes}: {status} | Steps: {step} | Reward: {episode_reward:.2f}")

    env.close()

    # Compute aggregate metrics
    success_rate = successes / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)

    results["success_rate"] = success_rate
    results["avg_reward"] = avg_reward
    results["avg_episode_length"] = avg_length
    results["std_reward"] = np.std(total_rewards)
    results["std_length"] = np.std(episode_lengths)

    print(f"\n{'=' * 50}")
    print(f"EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"Success Rate: {success_rate * 100:.1f}% ({successes}/{num_episodes})")
    print(f"Avg Reward: {avg_reward:.2f} (+/- {results['std_reward']:.2f})")
    print(f"Avg Episode Length: {avg_length:.1f} (+/- {results['std_length']:.1f})")
    print(f"{'=' * 50}")

    # Save results
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        checkpoint_name = Path(checkpoint_path).stem
        results_file = results_path / f"{checkpoint_name}_eval.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="TwoArmClothFold",
        help="Task name",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="Panda",
        help="Robot type",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during evaluation",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to file",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="eval_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--env_configuration",
        type=str,
        default="parallel",
        help="Environment configuration (parallel, opposed)",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["bimanual_view"],
        help="Camera names for vision policy",
    )
    parser.add_argument(
        "--recompute_freq",
        type=int,
        default=10,
        help="How often to recompute action chunks (receding horizon)",
    )

    args = parser.parse_args()

    evaluate_policy(
        checkpoint_path=args.checkpoint,
        task=args.task,
        robot=args.robot,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_results=not args.no_save,
        results_dir=args.results_dir,
        env_configuration=args.env_configuration,
        camera_names=args.camera_names,
        recompute_freq=args.recompute_freq,
    )


if __name__ == "__main__":
    main()
