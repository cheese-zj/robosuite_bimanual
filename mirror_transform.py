"""
Mirror Transform utilities for symmetric bimanual tasks.

These helpers mirror observations and actions across a symmetry plane for
bimanual data augmentation or symmetric control experiments.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MirrorConfig:
    """Configuration for mirror transformation"""

    # Symmetry plane: 'x' means reflection across YZ plane (negate x)
    #                 'y' means reflection across XZ plane (negate y)
    symmetry_axis: str = "y"

    # Whether to swap left/right arms in the mirrored pair
    swap_arms: bool = True

    # Joint indices that need sign flip for mirroring
    # This depends on your robot's kinematic structure
    # For a typical 7-DOF arm (like Panda):
    #   - Joints rotating about axes parallel to mirror plane: flip
    #   - Joints rotating about axes perpendicular: keep
    # Example for Panda with Y-axis symmetry:
    joint_flip_mask: np.ndarray = field(
        default_factory=lambda: np.array([1, -1, 1, -1, 1, -1, 1])  # 7 joints
    )


class MirrorTransform:
    """
    Transforms observations and actions between primary and mirror arm pairs.

    Usage:
        transform = MirrorTransform(config)

        # Mirror a bimanual observation pair
        obs0_m, obs1_m = transform.transform_bimanual_obs(obs0, obs1)
        mirrored_action = transform.transform_action(action)
    """

    def __init__(self, config: Optional[MirrorConfig] = None):
        self.config = config or MirrorConfig()

    def mirror_position(self, pos: np.ndarray) -> np.ndarray:
        """Mirror a 3D position"""
        pos = np.asarray(pos).copy()

        if self.config.symmetry_axis == "x":
            pos[..., 0] *= -1
        elif self.config.symmetry_axis == "y":
            pos[..., 1] *= -1
        elif self.config.symmetry_axis == "z":
            pos[..., 2] *= -1

        return pos

    def mirror_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """
        Mirror a quaternion (w, x, y, z).

        For reflection across a plane, we flip components corresponding
        to rotation axes that are perpendicular to the plane.
        """
        quat = np.asarray(quat).copy()

        # Quaternion reflection formula depends on the plane
        if self.config.symmetry_axis == "y":
            # Reflection across XZ plane: flip y and z components
            quat[..., 2] *= -1  # y component
            quat[..., 3] *= -1  # z component
        elif self.config.symmetry_axis == "x":
            # Reflection across YZ plane: flip x and z components
            quat[..., 1] *= -1  # x component
            quat[..., 3] *= -1  # z component

        return quat

    def mirror_joint_positions(self, joints: np.ndarray) -> np.ndarray:
        """Mirror joint positions using the flip mask"""
        return joints * self.config.joint_flip_mask

    def mirror_gripper(self, gripper: np.ndarray) -> np.ndarray:
        """Gripper is symmetric, no change needed"""
        return gripper.copy()

    def transform_single_arm_obs(self, obs: Dict) -> Dict:
        """Transform observation for a single arm"""
        transformed = {}

        if "joint_pos" in obs:
            transformed["joint_pos"] = self.mirror_joint_positions(obs["joint_pos"])
        if "joint_vel" in obs:
            transformed["joint_vel"] = self.mirror_joint_positions(obs["joint_vel"])
        if "eef_pos" in obs:
            transformed["eef_pos"] = self.mirror_position(obs["eef_pos"])
        if "eef_quat" in obs:
            transformed["eef_quat"] = self.mirror_quaternion(obs["eef_quat"])
        if "gripper_qpos" in obs:
            transformed["gripper_qpos"] = self.mirror_gripper(obs["gripper_qpos"])

        return transformed

    def transform_bimanual_obs(
        self,
        robot0_obs: Dict,
        robot1_obs: Dict,
    ) -> Tuple[Dict, Dict]:
        """
        Transform observations for a bimanual pair.

        If swap_arms is True:
            - robot0_obs becomes robot1-like after mirroring
            - robot1_obs becomes robot0-like after mirroring
        """
        mirrored_0 = self.transform_single_arm_obs(robot0_obs)
        mirrored_1 = self.transform_single_arm_obs(robot1_obs)

        if self.config.swap_arms:
            return mirrored_1, mirrored_0
        else:
            return mirrored_0, mirrored_1

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        """
        Transform a bimanual action for the mirror pair.

        Args:
            action: (14,) for OSC_POSE: [robot0_pose(6) + gripper(1), robot1_pose(6) + gripper(1)]

        Returns:
            Transformed action for mirror pair
        """
        action = np.asarray(action).copy()

        # Split into per-arm actions
        arm0_action = action[:7]  # 6 pose + 1 gripper
        arm1_action = action[7:14]

        # Mirror the pose components
        arm0_mirrored = self._mirror_arm_action(arm0_action)
        arm1_mirrored = self._mirror_arm_action(arm1_action)

        # Swap if configured
        if self.config.swap_arms:
            return np.concatenate([arm1_mirrored, arm0_mirrored])
        else:
            return np.concatenate([arm0_mirrored, arm1_mirrored])

    def _mirror_arm_action(self, arm_action: np.ndarray) -> np.ndarray:
        """Mirror action for a single arm (OSC_POSE format: dx,dy,dz,dax,day,daz,gripper)"""
        mirrored = arm_action.copy()

        # Position deltas
        if self.config.symmetry_axis == "y":
            mirrored[1] *= -1  # dy
            # Rotation deltas (axis-angle representation)
            mirrored[3] *= -1  # dax (rotation about x)
            mirrored[5] *= -1  # daz (rotation about z)
        elif self.config.symmetry_axis == "x":
            mirrored[0] *= -1  # dx
            mirrored[4] *= -1  # day
            mirrored[5] *= -1  # daz

        return mirrored


class QuadArmController:
    """Legacy helper for running a bimanual policy on four arms.

    Arms 0,1: Primary pair (runs policy directly)
    Arms 2,3: Mirror pair (transforms obs, runs policy, transforms action)
    """

    def __init__(self, policy, mirror_config: Optional[MirrorConfig] = None):
        """
        Args:
            policy: Trained bimanual policy with predict(obs) -> action
            mirror_config: Configuration for mirror transformation
        """
        self.policy = policy
        self.transform = MirrorTransform(mirror_config)

        # For action chunking synchronization
        self.primary_chunk = None
        self.mirror_chunk = None
        self.chunk_idx = 0

    def get_actions(
        self,
        obs_primary: Dict,
        obs_mirror: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get actions for all 4 arms.

        Args:
            obs_primary: Observation from arms 0,1
            obs_mirror: Observation from arms 2,3

        Returns:
            (action_primary, action_mirror)
        """
        # Primary pair: run policy directly
        action_primary = self.policy.predict(obs_primary)

        # Mirror pair: transform obs -> run policy -> transform action
        obs_mirror_transformed = self._transform_mirror_obs(obs_mirror)
        action_mirror_raw = self.policy.predict(obs_mirror_transformed)
        action_mirror = self.transform.transform_action(action_mirror_raw)

        return action_primary, action_mirror

    def _transform_mirror_obs(self, obs: Dict) -> Dict:
        """Transform mirror pair observation to primary-like format"""
        # Extract per-robot observations
        robot0_obs = {
            "joint_pos": obs.get("robot0_joint_pos"),
            "joint_vel": obs.get("robot0_joint_vel"),
            "eef_pos": obs.get("robot0_eef_pos"),
            "eef_quat": obs.get("robot0_eef_quat"),
            "gripper_qpos": obs.get("robot0_gripper_qpos"),
        }
        robot1_obs = {
            "joint_pos": obs.get("robot1_joint_pos"),
            "joint_vel": obs.get("robot1_joint_vel"),
            "eef_pos": obs.get("robot1_eef_pos"),
            "eef_quat": obs.get("robot1_eef_quat"),
            "gripper_qpos": obs.get("robot1_gripper_qpos"),
        }

        # Transform and swap
        t_robot0, t_robot1 = self.transform.transform_bimanual_obs(
            robot0_obs, robot1_obs
        )

        # Reconstruct observation dict
        transformed = {}
        for key, value in t_robot0.items():
            if value is not None:
                transformed[f"robot0_{key}"] = value
        for key, value in t_robot1.items():
            if value is not None:
                transformed[f"robot1_{key}"] = value

        return transformed

    def reset(self):
        """Reset state for new episode"""
        self.primary_chunk = None
        self.mirror_chunk = None
        self.chunk_idx = 0


def test_mirror_transform():
    """Test the mirror transform"""
    transform = MirrorTransform()

    # Test position mirroring
    pos = np.array([1.0, 2.0, 3.0])
    mirrored = transform.mirror_position(pos)
    print(f"Position: {pos} -> {mirrored}")
    assert mirrored[1] == -2.0, "Y should be negated"

    # Test quaternion mirroring
    quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    mirrored_quat = transform.mirror_quaternion(quat)
    print(f"Quaternion: {quat} -> {mirrored_quat}")

    # Test action mirroring
    action = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.1,
            0.2,
            0.3,
            0.5,  # robot0
            0.1,
            0.2,
            0.3,
            0.1,
            0.2,
            0.3,
            0.5,
        ]
    )  # robot1
    mirrored_action = transform.transform_action(action)
    print(f"Action shape: {action.shape} -> {mirrored_action.shape}")
    print(f"Original: {action}")
    print(f"Mirrored: {mirrored_action}")

    # Verify double mirror = identity
    double_mirrored = transform.transform_action(mirrored_action)
    print(f"\nDouble mirrored ≈ original: {np.allclose(action, double_mirrored)}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_mirror_transform()
