"""Two-arm cloth folding environment using MuJoCo flexcloth."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import tempfile

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
        cloth_x_offset: float = -0.05,  # Shift cloth toward robots for reachability
        grasp_assist: bool = False,  # Disabled by default - rely on physics-based friction grasping
        assist_radius: float = 0.15,
        assist_max_verts: int = 12,  # Increased for more stable cloth grip
        assist_action_threshold: float = 0.3,
        assist_strict: bool = False,  # Use strict detection (tight tolerances)
        assist_z_tolerance: float = 0.02,  # Vertical tolerance for strict mode (2cm)
        assist_xy_radius: float = 0.03,  # Horizontal radius for strict mode (3cm)
        # Noise/randomization parameters
        cloth_noise: bool = False,  # Randomize cloth position on reset
        cloth_noise_std: float = 0.03,  # Standard deviation for cloth position noise (meters)
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
        self.assist_strict = assist_strict
        self.assist_z_tolerance = assist_z_tolerance
        self.assist_xy_radius = assist_xy_radius

        # Noise/randomization parameters
        self.cloth_noise = cloth_noise
        self.cloth_noise_std = cloth_noise_std
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
        self._gripper_site_ids = []
        self._assist_vertices = {}
        self._assist_offsets = {}
        self._assist_initial_xpos = {}  # Store initial vertex xpos for qpos calculation
        self._cloth_qpos_start = None  # Starting index of cloth vertices in qpos
        # External gripper signals for 0-DOF grippers (like Piper)
        # Set via env.gripper_signals = [signal0, signal1] before step()
        self.gripper_signals = None

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
                # Piper EEF starts ~0.02m outward from base, can reach ~0.02m inward
                # Cloth corners at Y=±0.15, targets at Y=±0.14 after inward offset
                # With bases at Y=±0.16, EEF starts at ~Y=±0.14 (exactly at targets)
                y_offset = 0.16
                for robot, sign in zip(self.robots, (-1, 1)):
                    xpos = robot.robot_model.base_xpos_offset["table"](
                        self.table_full_size[0]
                    )
                    xpos = np.array(xpos) + np.array((0, sign * y_offset, 0))
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

        # Find gripper site IDs for grasp assist
        # For Piper robots, use link7 (finger) body position instead of grip_site
        # because the null_gripper's grip_site is at link6, not the actual fingers
        self._gripper_site_ids = []
        self._gripper_body_ids = []  # For Piper: use body positions instead
        for idx, robot in enumerate(self.robots):
            arm = robot.arms[0] if hasattr(robot, "arms") and robot.arms else "right"
            gripper = robot.gripper
            if isinstance(gripper, dict):
                gripper = gripper[arm]

            # Check if this is a Piper robot by looking for link7 body
            piper_link7_name = f"robot{idx}_link7"
            try:
                body_id = self.sim.model.body_name2id(piper_link7_name)
                self._gripper_body_ids.append(body_id)
                self._gripper_site_ids.append(None)  # Placeholder
            except Exception:
                # Not Piper - use standard grip_site
                site_name = gripper.naming_prefix + "grip_site"
                self._gripper_site_ids.append(self.sim.model.site_name2id(site_name))
                self._gripper_body_ids.append(None)

        # Find the starting qpos index for cloth vertices
        # Cloth bodies come after robot joints in qpos
        # Each cloth vertex has 3 DOF (x, y, z offset from initial position)
        robot_joints = sum(len(robot.robot_joints) for robot in self.robots)
        self._cloth_qpos_start = robot_joints

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
        """Get positions of all gripper sites (or finger bodies for Piper)."""
        if not self._gripper_site_ids and not self._gripper_body_ids:
            return None
        positions = []
        for idx in range(len(self.robots)):
            if self._gripper_body_ids and self._gripper_body_ids[idx] is not None:
                # Piper: use link7 body position (actual finger position)
                positions.append(self.sim.data.body_xpos[self._gripper_body_ids[idx]])
            elif self._gripper_site_ids and self._gripper_site_ids[idx] is not None:
                # Standard: use grip_site position
                positions.append(self.sim.data.site_xpos[self._gripper_site_ids[idx]])
            else:
                return None
        return np.array(positions, copy=True)

    def _find_nearest_vertices(self, gripper_pos: np.ndarray, radius: float = None) -> np.ndarray:
        """Find cloth vertices within radius of gripper.

        Args:
            gripper_pos: Gripper position (3D)
            radius: Search radius, defaults to self.assist_radius

        Returns:
            Array of vertex indices within radius
        """
        if radius is None:
            radius = self.assist_radius

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
        candidates = np.where(dists <= radius)[0]
        if candidates.size == 0:
            return np.array([], dtype=int)
        # Return all candidates (caller will filter/sort as needed)
        return candidates + offset

    def _update_grasp_assist(self, action):
        """Update grasp assist: pin cloth vertices to closing grippers.

        IMPORTANT: Only pins vertices when gripper is physically at cloth level.
        This ensures grippers visually touch the cloth before picking it up.

        For 0-DOF grippers (like Piper), set env.gripper_signals = [signal0, signal1]
        before calling step(). Values > assist_action_threshold (default 0.3) trigger closing.
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

        # Get cloth Z level for proximity check
        if hasattr(self, '_cloth_vert_slice') and self._cloth_vert_slice is not None:
            cloth_z = self.sim.data.flexvert_xpos[self._cloth_vert_slice, 2].mean()
        else:
            cloth_z = self.table_offset[2] + 0.01  # Fallback to table height + cloth offset

        # Check each gripper for closing/opening
        for idx, gripper_pos in enumerate(gripper_positions):
            # Use external gripper signals if provided (for 0-DOF grippers like Piper)
            # Otherwise, extract from action (last element of each arm's action)
            if self.gripper_signals is not None and idx < len(self.gripper_signals):
                grip_cmd = self.gripper_signals[idx]
            else:
                grip_cmd = action[idx * per_arm_dim + (per_arm_dim - 1)]
            closing = grip_cmd > self.assist_action_threshold


            if closing and idx not in self._assist_vertices:
                # Use strict or relaxed tolerances based on config
                if self.assist_strict:
                    z_tolerance = self.assist_z_tolerance   # Default 2cm
                    xy_radius = self.assist_xy_radius       # Default 3cm
                else:
                    z_tolerance = 0.05   # Legacy tolerance
                    xy_radius = 0.10     # Legacy radius

                # Only pin if gripper is at cloth level
                gripper_z = gripper_pos[2]
                z_distance = abs(gripper_z - cloth_z)
                if z_distance > z_tolerance:
                    # Gripper not at cloth level yet - don't pin
                    continue

                # Gripper is at cloth level - find nearby vertices to attach
                vertices = self._find_nearest_vertices(gripper_pos, radius=xy_radius)
                if vertices.size == 0:
                    continue

                # Sort by distance and take closest ones
                vertex_positions = self.sim.data.flexvert_xpos[vertices]
                distances = np.linalg.norm(vertex_positions - gripper_pos[None, :], axis=1)
                sorted_indices = np.argsort(distances)
                # Take up to assist_max_verts closest vertices
                num_to_pin = min(self.assist_max_verts, len(sorted_indices))
                vertices = vertices[sorted_indices[:num_to_pin]]
                if vertices.size == 0:
                    continue

                self._assist_vertices[idx] = vertices
                offsets = self.sim.data.flexvert_xpos[vertices] - gripper_pos[None, :]
                self._assist_offsets[idx] = offsets
                # Store initial xpos for qpos-based manipulation
                # flexvert_xpos = initial_xpos + qpos_offset, so we need initial_xpos
                # to compute correct qpos values
                self._assist_initial_xpos[idx] = self.sim.data.flexvert_xpos[vertices].copy()

            elif not closing and idx in self._assist_vertices:
                # Gripper is opening - release vertices
                self._assist_vertices.pop(idx, None)
                self._assist_offsets.pop(idx, None)
                self._assist_initial_xpos.pop(idx, None)

    def _enforce_grasp_assist(self):
        """Enforce grasp assist constraints after physics step.

        This is called after the MuJoCo simulation step to override any
        physics-computed vertex positions for pinned vertices.

        Uses qpos-based manipulation since directly setting flexvert_xpos
        is overwritten by mj_forward. The relationship is:
            flexvert_xpos = initial_xpos + qpos_offset
        where initial_xpos is the position at qpos=0 (from model definition).
        """
        if not self.grasp_assist:
            return
        if not self._assist_vertices:
            return
        if self._cloth_qpos_start is None:
            return

        gripper_positions = self._get_gripper_positions()
        if gripper_positions is None:
            return

        for idx, vertices in self._assist_vertices.items():
            offsets = self._assist_offsets.get(idx)
            initial_xpos = self._assist_initial_xpos.get(idx)
            if offsets is None or initial_xpos is None or idx >= len(gripper_positions):
                continue

            gripper_pos = gripper_positions[idx]
            # Compute desired vertex positions (gripper_pos + offset from when grasped)
            desired_xpos = gripper_pos[None, :] + offsets

            # Get current qpos for these vertices to determine initial model positions
            # The initial model position is: initial_xpos_model = flexvert_xpos - qpos_offset
            # But we stored initial_xpos when we first pinned, which was the flexvert_xpos at that time
            # Since we want to move vertices to follow gripper, we need to set qpos such that:
            #   flexvert_xpos = initial_model_xpos + qpos = desired_xpos
            #   qpos = desired_xpos - initial_model_xpos
            # The initial_model_xpos (xpos when qpos=0) can be computed from:
            #   initial_xpos = initial_model_xpos + qpos_at_pin_time
            # But since we pinned when cloth was near rest (qpos~0), initial_xpos ≈ initial_model_xpos

            # For each vertex, set its qpos to move it to desired position
            for i, vert_idx in enumerate(vertices):
                # Cloth vertex vert_idx maps to qpos starting at _cloth_qpos_start
                # Each vertex has 3 DOF (x, y, z)
                local_idx = vert_idx
                if self._cloth_vert_slice is not None:
                    local_idx = vert_idx - self._cloth_vert_slice.start
                qpos_idx = self._cloth_qpos_start + local_idx * 3

                # Compute qpos offset to achieve desired position
                # qpos = desired_xpos - initial_model_xpos
                # Using initial_xpos as approximation of initial_model_xpos
                qpos_offset = desired_xpos[i] - initial_xpos[i]
                self.sim.data.qpos[qpos_idx:qpos_idx+3] = qpos_offset

                # Also zero the velocity
                if self.sim.model.nv > qpos_idx + 2:
                    self.sim.data.qvel[qpos_idx:qpos_idx+3] = 0.0

        # Call forward to apply the qpos changes to the simulation state
        # This updates flexvert_xpos to reflect the new vertex positions
        if self._assist_vertices:
            self.sim.forward()

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

    def _post_action(self, action):
        """Called after simulation step to enforce grasp assist constraints."""
        result = super()._post_action(action)
        # Re-apply vertex pinning after physics step to ensure cloth follows gripper
        self._enforce_grasp_assist()
        return result

    def _reset_internal(self):
        """Reset internal state variables."""
        super()._reset_internal()
        self._assist_vertices = {}
        self._assist_offsets = {}
        self._assist_initial_xpos = {}


        if self.env_configuration == "parallel":
            joint1_rotations = [0, 0]  # Robot1 needs more rotation to match Robot0
            for i, rotation in enumerate(joint1_rotations):
                joint_name = f"robot{i}_joint1"
                try:
                    addr = self.sim.model.get_joint_qpos_addr(joint_name)
                    if isinstance(addr, tuple):
                        addr = addr[0]
                    self.sim.data.qpos[addr] = rotation
                except (ValueError, KeyError):
                    pass  # Joint not found, skip
            self.sim.forward()

        # Apply cloth position noise if enabled
        if self.cloth_noise and hasattr(self.sim.data, "flexvert_xpos"):
            if self._cloth_vert_slice is not None:
                # Generate random XY offset (same for all vertices to move cloth as a whole)
                xy_noise = np.random.normal(0, self.cloth_noise_std, size=2)
                # Apply to all cloth vertices
                self.sim.data.flexvert_xpos[self._cloth_vert_slice, 0] += xy_noise[0]
                self.sim.data.flexvert_xpos[self._cloth_vert_slice, 1] += xy_noise[1]
                # Zero velocities after position change
                if hasattr(self.sim.data, "flexvert_xvel"):
                    self.sim.data.flexvert_xvel[self._cloth_vert_slice] = 0.0

    def _post_reset(self):
        """Called after reset to set up free camera."""
        super()._post_reset()

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
