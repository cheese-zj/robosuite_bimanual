"""
Piper robot wrapper for the bimanual project.

Based on URDF from: https://github.com/agilexrobotics/piper_isaac_sim

Piper Specs:
- 6 arm joints (revolute)
- 2 gripper joints (joint7 and joint8, prismatic, coupled via equality constraint)
- Total: 8 controllable joints per arm (but joint8 is mirrored from joint7)
- DOF: 6 (arm) + 1 (effective gripper DOF, since fingers are coupled)

Note: The Piper gripper has two finger joints that move in opposite directions.
An equality constraint in the MJCF couples them so controlling joint7 automatically
moves joint8 in the opposite direction.
"""

import numpy as np
from pathlib import Path

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import register_gripper


@register_gripper
class PiperGripper(GripperModel):
    """Piper integrated gripper - 0 DOF (controlled via grasp_assist).

    The Piper robot has an integrated parallel jaw gripper where joint7 controls
    the finger opening. The geometry is in the arm's MJCF (not the gripper model).

    For cloth manipulation, grasp_assist is used to pin cloth vertices to the
    gripper. The gripper collision is enabled (contype=1) so the grippers visually
    touch the cloth before the grasp_assist pins it.

    Note: The gripper actuators (torq_j7, torq_j8) are in the arm MJCF, not
    in the gripper model file, so robosuite's GRIP controller cannot control them.
    Future work could add a custom gripper XML with proper actuator definitions.
    """

    def __init__(self, idn=0):
        # Use null_gripper.xml as base (no geometry, no actuators)
        from robosuite.utils.mjcf_utils import xml_path_completion

        super().__init__(xml_path_completion("grippers/null_gripper.xml"), idn=idn)

    @property
    def dof(self):
        """Piper gripper uses grasp_assist, so 0 DOF for controller."""
        return 0

    @property
    def init_qpos(self):
        """Gripper starts open (0 = fully open)."""
        return np.array([0.0])

    def format_action(self, action):
        """No gripper control - actions are ignored (grasp_assist handles grasping)."""
        return np.array([])


class PiperRobot:
    """Piper 6-DoF arm with integrated parallel gripper."""

    # Robot metadata
    NAME = "Piper"

    # Joint configuration
    ARM_JOINTS = 6
    GRIPPER_JOINTS = (
        2  # joint7 + joint8 (coupled via equality constraint, 1 effective DOF)
    )
    TOTAL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS  # 8

    # Observation dimensions (arm only, gripper is separate)
    OBS_DIM_PER_ARM = (
        ARM_JOINTS  # joint_pos
        + ARM_JOINTS  # joint_vel
        + 3  # eef_pos
        + 4  # eef_quat
        + GRIPPER_JOINTS  # gripper_qpos
    )  # = 6 + 6 + 3 + 4 + 1 = 20

    # For bimanual: 2 arms
    OBS_DIM_BIMANUAL = OBS_DIM_PER_ARM * 2  # 40

    # Action dimensions (OSC_POSE = 6 arm + 1 gripper per arm)
    ACTION_DIM_PER_ARM = 7  # 6 pose + 1 gripper
    ACTION_DIM_BIMANUAL = ACTION_DIM_PER_ARM * 2  # 14

    @classmethod
    def get_default_obs_dims(cls, include_object_obs: bool = False) -> int:
        """Get total observation dimension for bimanual setup."""
        dims = cls.OBS_DIM_BIMANUAL
        if include_object_obs:
            dims += 12 + 3  # cloth_corners (12) + cloth_center (3)
        return dims

    @classmethod
    def get_action_dim(cls) -> int:
        """Get action dimension for bimanual setup."""
        return cls.ACTION_DIM_BIMANUAL

    @classmethod
    def get_arm_dof(cls) -> int:
        """Get arm degrees of freedom (excluding gripper)."""
        return cls.ARM_JOINTS

    @classmethod
    def get_gripper_dof(cls) -> int:
        """Get gripper degrees of freedom."""
        return cls.GRIPPER_JOINTS

    @classmethod
    def get_urdf_path(cls) -> str:
        """Get path to URDF file."""
        return str(
            Path(__file__).parent.parent
            / "assets"
            / "urdf"
            / "piper"
            / "piper_description.urdf"
        )

    @classmethod
    def get_mjcf_path(cls) -> str:
        """Get path to MJCF XML file."""
        return str(
            Path(__file__).parent.parent
            / "assets"
            / "mjcf"
            / "piper"
            / "piper_description.xml"
        )


# Robot comparison table for reference
ROBOT_SPECS = {
    "Panda": {
        "arm_dof": 7,
        "gripper_dof": 2,
        "obs_per_arm": 23,  # 7+7+3+4+2
        "action_per_arm": 7,
    },
    "Piper": {
        "arm_dof": 6,
        "gripper_dof": 2,  # 2 physical joints (coupled), 1 effective DOF
        "obs_per_arm": 20,  # 6+6+3+4+1
        "action_per_arm": 7,
    },
    "UR5e": {
        "arm_dof": 6,
        "gripper_dof": 2,
        "obs_per_arm": 21,
        "action_per_arm": 7,
    },
}
