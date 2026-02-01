"""Two-arm cloth folding environment using MuJoCo flexcloth."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
import tempfile

import mujoco
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.base import MujocoXML
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import array_to_string

from .cloth_config import ClothConfig, generate_cloth_xml, get_cloth_config


class TwoArmClothFold(TwoArmEnv):
    """Cloth folding task with two robot arms and a flex cloth sheet.

    This environment is designed for bimanual data collection and evaluation in
    parallel configuration.
    """

    def __init__(
        self,
        robots,
        env_configuration="parallel",  # Two robots side-by-side on -x side
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera: Optional[str] = "bimanual_view",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=500,
        ignore_done=False,
        hard_reset=True,
        camera_names=["bimanual_view"],
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        # Cloth-specific parameters
        cloth_preset: str = "medium",
        cloth_config: Optional[ClothConfig] = None,
        cloth_offset: float = 0.01,
        cloth_x_offset: float = -0.15,  # Shift cloth toward robots for reachability
        grasp_assist: bool = True,
        assist_radius: float = 0.18,  # Increased from 0.15 for better vertex capture with randomization
        assist_max_verts: int = 12,  # Increased from 8 for more stable grip with randomization
        assist_action_threshold: float = 0.3,
        assist_strength: float = 0.5,  # Blend factor for soft vertex attachment
        assist_velocity_damping: float = 0.0,  # Damping on attached vertex velocities
        render_two_sided: bool = True,  # Disable backface culling for cloth visuals
        # Noise/randomization parameters
        cloth_noise: bool = False,  # Randomize cloth position on reset
        cloth_noise_std: float = 0.03,  # Standard deviation for cloth position noise (meters)
        cloth_rotation_noise: bool = False,  # Randomize cloth rotation (yaw) on reset
        cloth_rotation_noise_max: float = 0.2618,  # Max rotation in radians (~15 degrees)
        cloth_reach_margin: float = 0.05,
        cloth_randomize_max_tries: int = 10,
        robot_noise: bool = False,  # Use robot initialization noise (joint position randomization)
    ):
        # Cloth and grasp assist parameters
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.use_object_obs = use_object_obs
        self.cloth_offset = cloth_offset
        self.cloth_x_offset = cloth_x_offset
        self.grasp_assist = grasp_assist
        self.assist_radius = assist_radius
        self.assist_max_verts = assist_max_verts
        self.assist_action_threshold = assist_action_threshold
        self.assist_strength = float(np.clip(assist_strength, 0.0, 1.0))
        self.assist_velocity_damping = float(
            np.clip(assist_velocity_damping, 0.0, 1.0)
        )
        self.render_two_sided = render_two_sided

        # Noise/randomization parameters
        self.cloth_noise = cloth_noise
        self.cloth_noise_std = cloth_noise_std
        self.cloth_rotation_noise = cloth_rotation_noise
        self.cloth_rotation_noise_max = cloth_rotation_noise_max
        self.cloth_reach_margin = max(0.0, cloth_reach_margin)
        self.cloth_randomize_max_tries = max(1, int(cloth_randomize_max_tries))
        self.robot_noise = robot_noise

        # Reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # Resolve cloth configuration
        if cloth_config is not None:
            self.cloth_config = cloth_config
        else:
            self.cloth_config = get_cloth_config(cloth_preset)

        # Auto-scale assist_max_verts based on cloth resolution if using default
        if assist_max_verts == 5 and self.cloth_config.vertex_count > 100:
            # Scale up for higher resolution cloth
            density = self.cloth_config.vertex_count / (
                self.cloth_config.cloth_size[0] * self.cloth_config.cloth_size[1]
            )
            self.assist_max_verts = max(5, min(15, int(density * 0.003)))

        # Cloth simulation state
        self._cloth_flex_id = None
        self._cloth_vert_slice = None
        self._cloth_body_ids = None
        self._cloth_body_pos_base = None
        self._gripper_site_ids = []
        self._assist_vertices = {}
        self._assist_offsets = {}
        self._reach_gripper_positions = None
        self._reach_dist_thresholds = None

        # Handle robot initialization noise flag
        # If robot_noise is False, disable initialization noise; otherwise use default
        if not robot_noise:
            initialization_noise = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
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
        self._apply_render_flags()

    def _load_model(self):
        """Load model and set up robot base poses."""
        super()._load_model()

        # Adjust robot base poses for parallel configuration
        # Robots positioned side-by-side on the -x side of the table
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](
                        self.table_full_size[0]
                    )
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting (default for cloth folding)
                # Set up robots parallel to each other but offset from the center
                # Both robots on the -x side, offset along y-axis
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](
                        self.table_full_size[0]
                    )
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # Create table arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Create manipulation task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=None,
        )

        # Generate cloth XML from configuration
        cloth_pos = (
            self.cloth_x_offset,
            0.0,
            self.table_offset[2] + self.cloth_offset,
        )
        cloth_xml_str = generate_cloth_xml(self.cloth_config, base_pos=cloth_pos)

        # Write to temporary file for MujocoXML to load
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False
        ) as tmp_file:
            tmp_file.write(cloth_xml_str)
            tmp_path = tmp_file.name

        cloth_model = MujocoXML(tmp_path)
        self.model.merge(cloth_model)

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        # Add custom camera for bimanual view
        camera_xml_path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "cameras"
            / "two_arm_birdview.xml"
        )
        if camera_xml_path.exists():
            camera_model = MujocoXML(str(camera_xml_path))
            self.model.merge(camera_model)

    def _setup_references(self):
        """Set up references to cloth and gripper elements."""
        super()._setup_references()

        # Find cloth flex component
        if getattr(self.sim.model, "nflex", 0) > 0:
            self._cloth_flex_id = 0
            if hasattr(self.sim.model, "flex_name2id"):
                try:
                    self._cloth_flex_id = self.sim.model.flex_name2id("cloth")
                except Exception:
                    self._cloth_flex_id = 0

            # Get vertex slice for this flex component
            if hasattr(self.sim.model, "flex_vertadr") and hasattr(
                self.sim.model, "flex_vertnum"
            ):
                start = int(self.sim.model.flex_vertadr[self._cloth_flex_id])
                count = int(self.sim.model.flex_vertnum[self._cloth_flex_id])
                self._cloth_vert_slice = slice(start, start + count)
                if hasattr(self.sim.model, "flex_vertbodyid"):
                    vert_body_ids = np.asarray(
                        self.sim.model.flex_vertbodyid, dtype=int
                    )
                    if vert_body_ids.size > 0:
                        self._cloth_body_ids = vert_body_ids
                        self._cloth_body_pos_base = np.array(
                            self.sim.model.body_pos[vert_body_ids], copy=True
                        )

        # Find gripper site IDs for grasp assist
        self._gripper_site_ids = []
        for robot in self.robots:
            arm = robot.arms[0] if hasattr(robot, "arms") and robot.arms else "right"
            gripper = robot.gripper
            if isinstance(gripper, dict):
                gripper = gripper[arm]
            site_name = gripper.naming_prefix + "grip_site"
            self._gripper_site_ids.append(self.sim.model.site_name2id(site_name))

    def _get_cloth_vertices(self) -> Optional[np.ndarray]:
        """Get all cloth vertex positions."""
        if getattr(self.sim.model, "nflex", 0) <= 0:
            return None
        if not hasattr(self.sim.data, "flexvert_xpos"):
            return None
        verts = self.sim.data.flexvert_xpos
        if self._cloth_vert_slice is not None:
            verts = verts[self._cloth_vert_slice]
        return np.array(verts, copy=True)

    def _get_gripper_positions(self) -> Optional[np.ndarray]:
        """Get positions of all gripper sites."""
        if not self._gripper_site_ids:
            return None
        return np.array(
            [self.sim.data.site_xpos[sid] for sid in self._gripper_site_ids], copy=True
        )

    def _find_nearest_vertices(self, gripper_pos: np.ndarray) -> np.ndarray:
        """Find cloth vertices within assist radius of gripper."""
        if getattr(self.sim.model, "nflex", 0) <= 0 or not hasattr(
            self.sim.data, "flexvert_xpos"
        ):
            return np.array([], dtype=int)
        verts = self.sim.data.flexvert_xpos
        offset = 0
        if self._cloth_vert_slice is not None:
            offset = self._cloth_vert_slice.start
            verts = verts[self._cloth_vert_slice]

        dists = np.linalg.norm(verts - gripper_pos[None, :], axis=1)
        candidates = np.where(dists <= self.assist_radius)[0]
        if candidates.size == 0:
            return np.array([], dtype=int)
        order = candidates[np.argsort(dists[candidates])]
        chosen = order[: self.assist_max_verts]
        return chosen + offset

    def _update_grasp_assist(self, action):
        """Update grasp assist: pin nearby cloth vertices to closing grippers.

        Uses hysteresis for stable grip: vertices attach when grip > attach_threshold,
        but only release when grip < release_threshold (a separate, lower value).
        This prevents accidental release during manipulation.
        """
        if not self.grasp_assist:
            return
        if getattr(self.sim.model, "nflex", 0) <= 0 or not hasattr(
            self.sim.data, "flexvert_xpos"
        ):
            return
        if action is None:
            return

        action = np.asarray(action).flatten()
        if action.size == 0:
            return

        per_arm_dim = action.size // len(self.robots)
        if per_arm_dim < 1:
            return

        gripper_positions = self._get_gripper_positions()
        if gripper_positions is None:
            return

        # Hysteresis thresholds for stable grip
        attach_threshold = self.assist_action_threshold  # Default 0.3
        release_threshold = -0.3  # Must open gripper significantly to release

        # Check each gripper for closing/opening
        for idx, gripper_pos in enumerate(gripper_positions):
            grip_cmd = action[idx * per_arm_dim + (per_arm_dim - 1)]

            if grip_cmd > attach_threshold and idx not in self._assist_vertices:
                # Gripper is closing - find nearby vertices to attach
                vertices = self._find_nearest_vertices(gripper_pos)
                if vertices.size == 0:
                    continue
                self._assist_vertices[idx] = vertices
                offsets = self.sim.data.flexvert_xpos[vertices] - gripper_pos[None, :]
                self._assist_offsets[idx] = offsets
            elif grip_cmd < release_threshold and idx in self._assist_vertices:
                # Gripper is opening significantly - release vertices
                self._assist_vertices.pop(idx, None)
                self._assist_offsets.pop(idx, None)

        # Update positions of attached vertices
        for idx, vertices in self._assist_vertices.items():
            offsets = self._assist_offsets.get(idx)
            if offsets is None:
                continue
            gripper_pos = gripper_positions[idx]
            self._apply_grasp_blend(vertices, gripper_pos, offsets)

    def _apply_grasp_positions(self):
        """Apply stored grasp assist positions to attached vertices.

        Called after sim.step() to counteract physics-induced vertex movement.
        This ensures cloth stays attached to grippers even when physics forces
        (elasticity, tension) try to pull vertices away during fold motion.
        """
        if not self.grasp_assist:
            return
        if not hasattr(self.sim.data, "flexvert_xpos"):
            return

        gripper_positions = self._get_gripper_positions()
        if gripper_positions is None:
            return

        for idx, vertices in self._assist_vertices.items():
            offsets = self._assist_offsets.get(idx)
            if offsets is None or idx >= len(gripper_positions):
                continue
            gripper_pos = gripper_positions[idx]
            self._apply_grasp_blend(vertices, gripper_pos, offsets)

    def _apply_grasp_blend(
        self, vertices: np.ndarray, gripper_pos: np.ndarray, offsets: np.ndarray
    ):
        """Softly pull attached vertices toward the gripper to avoid frozen cloth."""
        if self.assist_strength <= 0.0:
            return
        target = gripper_pos[None, :] + offsets
        current = self.sim.data.flexvert_xpos[vertices]
        if self.assist_strength >= 1.0:
            blended = target
        else:
            blended = current + self.assist_strength * (target - current)
        self.sim.data.flexvert_xpos[vertices] = blended
        if hasattr(self.sim.data, "flexvert_xvel") and self.assist_velocity_damping > 0.0:
            self.sim.data.flexvert_xvel[vertices] *= 1.0 - self.assist_velocity_damping

    def _apply_render_flags(self):
        """Disable backface culling so cloth renders from both sides."""
        if not self.render_two_sided:
            return
        if getattr(self, "sim", None) is not None:
            render_context = getattr(self.sim, "_render_context_offscreen", None)
            if render_context is not None:
                render_context.scn.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0

        viewer_wrapper = getattr(self, "viewer", None)
        viewer = getattr(viewer_wrapper, "viewer", None) if viewer_wrapper else None
        if viewer is not None and hasattr(viewer, "scn"):
            viewer.scn.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0

    @staticmethod
    def _extract_corners(vertices: np.ndarray) -> np.ndarray:
        """Extract the 4 corners of the cloth from all vertices.

        Returns corners in order: bottom-left, top-left, bottom-right, top-right.
        """
        if vertices is None or vertices.size == 0:
            return np.zeros((4, 3))
        if vertices.shape[0] < 4:
            mean = vertices.mean(axis=0)
            return np.tile(mean, (4, 1))

        sums = vertices[:, 0] + vertices[:, 1]
        diffs = vertices[:, 0] - vertices[:, 1]

        idx_bl = int(np.argmin(sums))
        idx_tr = int(np.argmax(sums))
        idx_br = int(np.argmax(diffs))
        idx_tl = int(np.argmin(diffs))

        return np.stack(
            [vertices[idx_bl], vertices[idx_tl], vertices[idx_br], vertices[idx_tr]],
            axis=0,
        )

    def _pick_grasp_corners(
        self, corners: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pick the two corners to grasp based on layout configuration."""
        if corners.shape != (4, 3):
            raise ValueError("cloth_corners must be shape (4, 3)")

        if self.env_configuration in {"opposed", "front-back"}:
            ys = corners[:, 1]
            front_indices = np.argsort(ys)[-2:]
            front = corners[front_indices]
            corner0 = front[np.argmin(front[:, 0])]
            corner1 = front[np.argmax(front[:, 0])]
            return corner0, corner1

        xs = corners[:, 0]
        left_indices = np.argsort(xs)[:2]
        left = corners[left_indices]
        corner0 = left[np.argmin(left[:, 1])]
        corner1 = left[np.argmax(left[:, 1])]
        return corner0, corner1

    def _cache_reachability(self):
        """Cache reachability thresholds for cloth grasping."""
        gripper_positions = self._get_gripper_positions()
        verts = self._cloth_body_pos_base
        if verts is None:
            verts = self._get_cloth_vertices()
        if gripper_positions is None or verts is None or verts.size == 0:
            self._reach_gripper_positions = None
            self._reach_dist_thresholds = None
            return

        corners = self._extract_corners(verts)
        try:
            corner0, corner1 = self._pick_grasp_corners(corners)
        except ValueError:
            self._reach_gripper_positions = None
            self._reach_dist_thresholds = None
            return

        dist0 = np.linalg.norm(corner0 - gripper_positions[0])
        if gripper_positions.shape[0] > 1:
            dist1 = np.linalg.norm(corner1 - gripper_positions[1])
        else:
            dist1 = dist0

        self._reach_gripper_positions = gripper_positions
        self._reach_dist_thresholds = (
            dist0 + self.cloth_reach_margin,
            dist1 + self.cloth_reach_margin,
        )

    def _setup_observables(self):
        """Set up observables for this environment."""
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cloth_corners(obs_cache):
                """4 corner positions of the cloth (12-dim)."""
                verts = self._get_cloth_vertices()
                corners = self._extract_corners(verts)
                return corners.reshape(-1)

            @sensor(modality=modality)
            def cloth_center(obs_cache):
                """Center position of the cloth (3-dim)."""
                verts = self._get_cloth_vertices()
                if verts is None or verts.size == 0:
                    return np.zeros(3)
                return verts.mean(axis=0)

            sensors = [cloth_corners, cloth_center]
            for s in sensors:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def reward(self, action=None):
        """Compute reward for current state.

        For now, returns 0.0. Can be extended to include fold quality metrics.
        """
        return 0.0

    def _check_success(self):
        """Check if cloth folding task is successful.

        Always returns False - episode termination is controlled by the
        scripted policy's done flag in collect_scripted.py.
        """
        return False

    def _pre_action(self, action, policy_step=False):
        """Called before applying action to update grasp assist."""
        super()._pre_action(action, policy_step)
        self._update_grasp_assist(action)

    def step(self, action):
        """Override step to apply grasp assist after each physics step.

        This ensures cloth vertices stay attached to grippers even when
        physics forces (elasticity, tension) try to pull them away during
        fold motion. The base class only calls _pre_action before sim.step(),
        but physics can move vertices afterward. We fix this by also applying
        grasp positions after each sim.step().
        """
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1
        policy_step = True

        # Core simulation loop - same as base class but with post-step grasp assist
        for i in range(int(self.control_timestep / self.model_timestep)):
            if self.lite_physics:
                self.sim.step1()
            else:
                self.sim.forward()
            self._pre_action(action, policy_step)
            if self.lite_physics:
                self.sim.step2()
            else:
                self.sim.step()

            # KEY FIX: Apply grasp positions AFTER physics step to counteract
            # physics-induced vertex movement during fold motion
            self._apply_grasp_positions()

            self._update_observables()
            policy_step = False

        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()
            self._apply_render_flags()

        if self.viewer is not None and self.renderer == "mujoco":
            if self.viewer.is_running() is False:
                done = True

        observations = (
            self.viewer._get_observations()
            if self.viewer_get_obs
            else self._get_observations()
        )
        return observations, reward, done, info

    def _reset_internal(self):
        """Reset internal state variables."""
        super()._reset_internal()
        self._assist_vertices = {}
        self._assist_offsets = {}
        self._reach_gripper_positions = None
        self._reach_dist_thresholds = None

    def reset(self):
        """Reset the environment and apply cloth randomization.

        Cloth randomization must happen after the base reset completes because
        reset() calls sim.forward() which recomputes flex vertex positions from
        model.body_pos. We need to modify body_pos after this forward() call.
        """
        # Call base reset which handles sim reset, _reset_internal, and sim.forward()
        obs = super().reset()
        self._apply_render_flags()

        # Apply cloth randomization after base reset completes
        if (
            self.cloth_noise or self.cloth_rotation_noise
        ) and self._cloth_vert_slice is not None:
            self._cache_reachability()
            self._apply_cloth_randomization()
            # Force update observables to get fresh values after cloth position change
            obs = self._get_observations(force_update=True)

        return obs

    def _apply_cloth_randomization(self):
        """Apply position and rotation noise to cloth by modifying vertex body positions.

        MuJoCo flex components have each vertex as a separate body. To properly
        randomize cloth position, we must modify model.body_pos for all vertex
        bodies and call sim.forward() to update the simulation state.
        """
        if not hasattr(self.sim.model, "flex_vertbodyid"):
            return

        # Get vertex body IDs
        vert_body_ids = self._cloth_body_ids
        if vert_body_ids is None or vert_body_ids.size == 0:
            return

        if self._cloth_body_pos_base is None:
            base_positions = np.array(self.sim.model.body_pos[vert_body_ids], copy=True)
        else:
            base_positions = np.array(self._cloth_body_pos_base, copy=True)
        cloth_center = base_positions.mean(axis=0)
        centered_xy = base_positions[:, :2] - cloth_center[:2]

        best_positions = None
        best_score = np.inf

        for _ in range(self.cloth_randomize_max_tries):
            xy_offset = np.zeros(2)
            if self.cloth_noise:
                xy_offset = np.random.normal(0, self.cloth_noise_std, size=2)

            yaw = 0.0
            if self.cloth_rotation_noise:
                yaw = np.random.uniform(
                    -self.cloth_rotation_noise_max, self.cloth_rotation_noise_max
                )

            if yaw != 0.0:
                cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                rotated_xy = centered_xy @ rot.T
            else:
                rotated_xy = centered_xy

            candidate_positions = base_positions.copy()
            candidate_positions[:, :2] = rotated_xy + cloth_center[:2] + xy_offset

            if (
                self._reach_gripper_positions is None
                or self._reach_dist_thresholds is None
            ):
                best_positions = candidate_positions
                break

            corners = self._extract_corners(candidate_positions)
            try:
                corner0, corner1 = self._pick_grasp_corners(corners)
            except ValueError:
                continue

            dist0 = np.linalg.norm(corner0 - self._reach_gripper_positions[0])
            if self._reach_gripper_positions.shape[0] > 1:
                dist1 = np.linalg.norm(corner1 - self._reach_gripper_positions[1])
            else:
                dist1 = dist0

            score = max(dist0, dist1)
            if score < best_score:
                best_score = score
                best_positions = candidate_positions

            if (
                dist0 <= self._reach_dist_thresholds[0]
                and dist1 <= self._reach_dist_thresholds[1]
            ):
                best_positions = candidate_positions
                break

        if best_positions is None:
            best_positions = base_positions

        self.sim.model.body_pos[vert_body_ids] = best_positions
        self.sim.forward()
        if (
            hasattr(self.sim.data, "flexvert_xvel")
            and self._cloth_vert_slice is not None
        ):
            self.sim.data.flexvert_xvel[self._cloth_vert_slice] = 0.0

    def _setup_free_camera(self):
        """Enable free camera mode for interactive navigation with mouse."""
        if not self.has_renderer:
            return

        try:
            # Access the camera through various possible viewer structures
            cam = None

            # Try method 1: viewer.viewer.cam (MuJoCo 2.x with robosuite wrapper)
            if hasattr(self, "viewer") and self.viewer is not None:
                if hasattr(self.viewer, "viewer") and self.viewer.viewer is not None:
                    cam = self.viewer.viewer.cam
                # Try method 2: viewer.cam (direct access)
                elif hasattr(self.viewer, "cam"):
                    cam = self.viewer.cam
                # Try method 3: sim.render_contexts[0].cam
                elif hasattr(self, "sim") and hasattr(self.sim, "render_contexts"):
                    if len(self.sim.render_contexts) > 0:
                        cam = self.sim.render_contexts[0].cam

            # Try method 4: access through sim model directly
            if cam is None and hasattr(self, "sim"):
                if hasattr(self.sim, "viewer") and self.sim.viewer is not None:
                    if hasattr(self.sim.viewer, "cam"):
                        cam = self.sim.viewer.cam

            # If we found a camera, configure it for free control
            if cam is not None:
                cam.type = 0  # mjCAMERA_FREE - enables mouse control

                # Position camera centered over table
                # For parallel config, look at center of table
                cam.distance = 1.5
                cam.elevation = -70  # Top-down view
                cam.azimuth = 90  # Looking from above
                try:
                    # Look at center of table/cloth
                    cam.lookat[:] = [0.0, 0.0, 0.85]
                except:
                    # Some versions might have different lookat API
                    pass
        except Exception as e:
            # Silently fail - the camera will just use default settings
            pass

    def render(self):
        """Override render to ensure free camera is set up."""
        # Set up free camera on first render
        if self.has_renderer and not hasattr(self, "_free_camera_initialized"):
            if str(getattr(self, "render_camera", "")).lower() in {
                "free_camera",
                "free",
            }:
                self._setup_free_camera()
            self._free_camera_initialized = True
        return super().render()
