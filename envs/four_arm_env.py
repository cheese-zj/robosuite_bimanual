"""Four-arm robosuite environment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from copy import deepcopy

import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
import robosuite.utils.transform_utils as T

PRIMARY_PAIR = (0, 1)
MIRROR_PAIR = (2, 3)


@dataclass
class FourArmEnvConfig:
    """Configuration for four-arm robot base placement."""

    env_configuration: str = "front-back"
    pair_offset: float = 0.25
    base_positions: Optional[Sequence[Sequence[float]]] = None
    base_rotations: Optional[Sequence[Sequence[float]]] = None


def _resolve_controller_name(controller: Optional[str]) -> Optional[str]:
    if controller is None or controller == "" or controller == "default":
        return None
    if controller in {"OSC_POSE", "OSC_POSITION", "JOINT_VELOCITY", "JOINT_POSITION", "IK_POSE"}:
        return "BASIC"
    return controller


def _apply_input_ref_frame(controller_config, input_ref_frame: Optional[str]):
    if input_ref_frame is None:
        return controller_config
    config = deepcopy(controller_config)
    body_parts = config.get("body_parts", {})
    for part_cfg in body_parts.values():
        if isinstance(part_cfg, dict) and "input_ref_frame" in part_cfg:
            part_cfg["input_ref_frame"] = input_ref_frame
    return config


def extract_pair_obs(obs: Dict, pair: Tuple[int, int] = PRIMARY_PAIR) -> Dict:
    """Extract a bimanual observation dict for a specific arm pair.

    Robot-specific keys are remapped to robot0_ / robot1_ to match training format.
    """
    if obs is None:
        return {}

    pair = tuple(pair)
    index_map = {pair[0]: 0, pair[1]: 1}
    output = {}

    for key, value in obs.items():
        if not key.startswith("robot"):
            output[key] = value
            continue

        mapped = False
        for src_idx, dst_idx in index_map.items():
            prefix = f"robot{src_idx}_"
            if key.startswith(prefix):
                output[f"robot{dst_idx}_{key[len(prefix):]}"] = value
                mapped = True
                break
        if not mapped:
            continue

    return output


def combine_pair_actions(
    action_primary: np.ndarray,
    action_mirror: np.ndarray,
    primary_pair: Tuple[int, int] = PRIMARY_PAIR,
    mirror_pair: Tuple[int, int] = MIRROR_PAIR,
) -> np.ndarray:
    """Combine two bimanual actions into a single 4-arm action."""
    action_primary = np.asarray(action_primary)
    action_mirror = np.asarray(action_mirror)

    if action_primary.ndim == 1:
        if action_primary.size % 2 != 0:
            raise ValueError("Primary action length must be divisible by 2")
        if action_mirror.size != action_primary.size:
            raise ValueError("Primary and mirror actions must have same length")

        per_arm_dim = action_primary.size // 2
        full_action = np.zeros(per_arm_dim * 4, dtype=action_primary.dtype)

        primary_indices = [primary_pair[0], primary_pair[1]]
        mirror_indices = [mirror_pair[0], mirror_pair[1]]
        for i, arm_idx in enumerate(primary_indices):
            start = arm_idx * per_arm_dim
            full_action[start : start + per_arm_dim] = action_primary[i * per_arm_dim : (i + 1) * per_arm_dim]
        for i, arm_idx in enumerate(mirror_indices):
            start = arm_idx * per_arm_dim
            full_action[start : start + per_arm_dim] = action_mirror[i * per_arm_dim : (i + 1) * per_arm_dim]

        return full_action

    if action_primary.ndim == 2:
        if action_primary.shape[1] % 2 != 0:
            raise ValueError("Primary action length must be divisible by 2")
        if action_mirror.shape != action_primary.shape:
            raise ValueError("Primary and mirror actions must have same shape")

        per_arm_dim = action_primary.shape[1] // 2
        full_action = np.zeros((action_primary.shape[0], per_arm_dim * 4), dtype=action_primary.dtype)

        primary_indices = [primary_pair[0], primary_pair[1]]
        mirror_indices = [mirror_pair[0], mirror_pair[1]]
        for i, arm_idx in enumerate(primary_indices):
            start = arm_idx * per_arm_dim
            full_action[:, start : start + per_arm_dim] = action_primary[:, i * per_arm_dim : (i + 1) * per_arm_dim]
        for i, arm_idx in enumerate(mirror_indices):
            start = arm_idx * per_arm_dim
            full_action[:, start : start + per_arm_dim] = action_mirror[:, i * per_arm_dim : (i + 1) * per_arm_dim]

        return full_action

    raise ValueError("Action arrays must be 1D or 2D")


class FourArmEnv(ManipulationEnv):
    """Base environment for four single-arm robots."""

    _CONFIG_ALIASES = {
        "default": "front-back",
        "front_back": "front-back",
        "opposed": "front-back",
        "quad": "square",
    }

    def __init__(
        self,
        robots,
        env_configuration: str = "front-back",
        controller_configs=None,
        base_types="default",
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs: bool = True,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        render_camera: str = "birdview",
        render_collision_mesh: bool = False,
        render_visual_mesh: bool = True,
        render_gpu_device_id: int = -1,
        control_freq: int = 20,
        lite_physics: bool = True,
        horizon: int = 1000,
        ignore_done: bool = False,
        hard_reset: bool = True,
        camera_names: Union[str, Iterable[str]] = ["birdview", "agentview"],
        camera_heights: Union[int, Iterable[int]] = 256,
        camera_widths: Union[int, Iterable[int]] = 256,
        camera_depths: Union[bool, Iterable[bool]] = False,
        camera_segmentations=None,
        renderer: str = "mjviewer",
        renderer_config=None,
        seed: Optional[int] = None,
        four_arm_config: Optional[FourArmEnvConfig] = None,
    ):
        self.four_arm_config = four_arm_config or FourArmEnvConfig()
        if env_configuration in self._CONFIG_ALIASES:
            env_configuration = self._CONFIG_ALIASES[env_configuration]
        self.four_arm_config.env_configuration = env_configuration

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def _check_robot_configuration(self, robots):
        robots = robots if isinstance(robots, (list, tuple)) else [robots]
        if len(robots) != 4:
            raise ValueError("FourArmEnv requires exactly 4 robots")

        env_config = self._CONFIG_ALIASES.get(self.env_configuration, self.env_configuration)
        if env_config not in {"front-back", "square"}:
            raise ValueError(
                f"Unknown env_configuration: {self.env_configuration}. "
                "Valid options: front-back, square."
            )
        self.env_configuration = env_config

    def _set_robot_base_poses(self, pair_offset: Optional[float] = None):
        """Set base pose for all four robots according to the env configuration."""
        pair_offset = self.four_arm_config.pair_offset if pair_offset is None else pair_offset
        base_positions = self.four_arm_config.base_positions
        base_rotations = self.four_arm_config.base_rotations

        if base_positions is not None:
            if len(base_positions) != 4:
                raise ValueError("base_positions must be length 4")
            if base_rotations is not None and len(base_rotations) != 4:
                raise ValueError("base_rotations must be length 4 if provided")

            for idx, robot in enumerate(self.robots):
                robot.robot_model.set_base_xpos(base_positions[idx])
                if base_rotations is not None:
                    robot.robot_model.set_base_ori(base_rotations[idx])
            return

        if self.env_configuration == "square":
            rotations = [0.0, np.pi / 2, np.pi, -np.pi / 2]
            offsets = [0.0, 0.0, 0.0, 0.0]
        else:
            rotations = [np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2]
            offsets = [-pair_offset, pair_offset, -pair_offset, pair_offset]

        for robot, rotation, lateral in zip(self.robots, rotations, offsets):
            base = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
            rot = np.array((0.0, 0.0, rotation))
            xpos = T.euler2mat(rot) @ np.array(base)
            xpos = np.array(xpos) + np.array((lateral, 0.0, 0.0))
            robot.robot_model.set_base_xpos(xpos)
            robot.robot_model.set_base_ori(rot)

    def _gripper0_to_target(self, target, target_type="body", return_distance=False):
        return self._gripper_to_target(self.robots[0].gripper, target, target_type, return_distance)

    def _gripper1_to_target(self, target, target_type="body", return_distance=False):
        return self._gripper_to_target(self.robots[1].gripper, target, target_type, return_distance)

    def _default_arm(self, robot) -> str:
        return robot.arms[0] if hasattr(robot, "arms") and robot.arms else "right"

    @property
    def _eef0_xpos(self):
        arm = self._default_arm(self.robots[0])
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])

    @property
    def _eef1_xpos(self):
        arm = self._default_arm(self.robots[1])
        return np.array(self.sim.data.site_xpos[self.robots[1].eef_site_id[arm]])

    @property
    def _eef0_xmat(self):
        arm = self._default_arm(self.robots[0])
        gripper = self.robots[0].gripper
        if isinstance(gripper, dict):
            gripper = gripper[arm]
        pf = gripper.naming_prefix
        return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "grip_site")]).reshape(3, 3)

    @property
    def _eef0_xquat(self):
        return T.mat2quat(self._eef0_xmat)


def create_four_arm_env(
    robots: str = "Panda",
    task: str = "FourArmLift",
    controller: str = "BASIC",
    input_ref_frame: Optional[str] = None,
    has_renderer: bool = False,
    has_offscreen_renderer: bool = True,
    use_camera_obs: bool = True,
    use_object_obs: bool = True,
    camera_names: Optional[List[str]] = None,
    env_configuration: str = "front-back",
):
    """Create a four-arm robosuite environment."""
    controller_name = _resolve_controller_name(controller)
    controller_config = load_composite_controller_config(
        controller=controller_name,
        robot=robots if isinstance(robots, str) else robots[0],
    )
    controller_config = _apply_input_ref_frame(controller_config, input_ref_frame)

    if camera_names is None and use_camera_obs:
        # Use training_pair_view for cloth tasks, birdview for others
        if "Cloth" in task:
            camera_names = ["training_pair_view", "agentview"]
        else:
            camera_names = ["birdview", "agentview"]

    robot_list = [robots] * 4 if isinstance(robots, str) else list(robots)

    if task == "FourArmLift":
        from .four_arm_lift import FourArmLift

        return FourArmLift(
            robots=robot_list,
            env_configuration=env_configuration,
            controller_configs=controller_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            camera_names=camera_names,
            camera_heights=480,
            camera_widths=640,
            control_freq=20,
            horizon=500,
        )
    if task == "FourArmClothFold":
        from .four_arm_cloth_fold import FourArmClothFold

        return FourArmClothFold(
            robots=robot_list,
            env_configuration=env_configuration,
            controller_configs=controller_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            camera_names=camera_names,
            camera_heights=480,
            camera_widths=640,
            control_freq=20,
            horizon=500,
        )

    return suite.make(
        env_name=task,
        robots=robot_list,
        env_configuration=env_configuration,
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        use_camera_obs=use_camera_obs,
        use_object_obs=use_object_obs,
        camera_names=camera_names,
        camera_heights=480,
        camera_widths=640,
        control_freq=20,
        horizon=500,
    )
