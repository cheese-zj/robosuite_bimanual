"""Four-arm cloth folding environment using MuJoCo flexcloth."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import tempfile

import numpy as np

from robosuite.models.arenas import TableArena
from robosuite.models.base import MujocoXML
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import array_to_string

from .cloth_config import ClothConfig, generate_cloth_xml, get_cloth_config
from .four_arm_env import FourArmEnv


class FourArmClothFold(FourArmEnv):
    """Cloth folding task with four arms and a flex cloth sheet."""

    def __init__(
        self,
        robots,
        env_configuration="front-back",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="training_pair_view",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=500,
        ignore_done=False,
        hard_reset=True,
        camera_names=["training_pair_view", "agentview"],
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        cloth_preset: str = "medium",
        cloth_config: Optional[ClothConfig] = None,
        cloth_offset: float = 0.01,
        grasp_assist: bool = True,
        assist_radius: float = 0.10,
        assist_max_verts: int = 5,
        assist_action_threshold: float = 0.3,
        # Noise/randomization parameters
        cloth_noise: bool = False,  # Randomize cloth position on reset
        cloth_noise_std: float = 0.03,  # Standard deviation for cloth position noise (meters)
        robot_noise: bool = False,  # Use robot initialization noise (joint position randomization)
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.use_object_obs = use_object_obs
        self.cloth_offset = cloth_offset
        self.grasp_assist = grasp_assist
        self.assist_radius = assist_radius
        self.assist_max_verts = assist_max_verts
        self.assist_action_threshold = assist_action_threshold

        # Noise/randomization parameters
        self.cloth_noise = cloth_noise
        self.cloth_noise_std = cloth_noise_std
        self.robot_noise = robot_noise

        # Resolve cloth configuration
        if cloth_config is not None:
            self.cloth_config = cloth_config
        else:
            self.cloth_config = get_cloth_config(cloth_preset)

        # Auto-scale assist_max_verts based on cloth resolution if using default
        if assist_max_verts == 5 and self.cloth_config.vertex_count > 100:
            density = self.cloth_config.vertex_count / (
                self.cloth_config.cloth_size[0] * self.cloth_config.cloth_size[1]
            )
            self.assist_max_verts = max(5, min(15, int(density * 0.003)))

        self._cloth_flex_id = None
        self._cloth_vert_slice = None
        self._gripper_site_ids = []
        self._assist_vertices = {}
        self._assist_offsets = {}

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
        super()._load_model()

        self._set_robot_base_poses()

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=None,
        )

        # Generate cloth XML from configuration
        cloth_pos = (0.0, 0.0, self.table_offset[2] + self.cloth_offset)
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

        # Add custom camera for training pair view
        camera_xml_path = Path(__file__).resolve().parents[1] / "assets" / "cameras" / "training_pair_birdview.xml"
        if camera_xml_path.exists():
            camera_model = MujocoXML(str(camera_xml_path))
            self.model.merge(camera_model)

    def _setup_references(self):
        super()._setup_references()

        if getattr(self.sim.model, "nflex", 0) > 0:
            self._cloth_flex_id = 0
            if hasattr(self.sim.model, "flex_name2id"):
                try:
                    self._cloth_flex_id = self.sim.model.flex_name2id("cloth")
                except Exception:
                    self._cloth_flex_id = 0

            if hasattr(self.sim.model, "flex_vertadr") and hasattr(self.sim.model, "flex_vertnum"):
                start = int(self.sim.model.flex_vertadr[self._cloth_flex_id])
                count = int(self.sim.model.flex_vertnum[self._cloth_flex_id])
                self._cloth_vert_slice = slice(start, start + count)

        self._gripper_site_ids = []
        for robot in self.robots:
            arm = robot.arms[0] if hasattr(robot, "arms") and robot.arms else "right"
            gripper = robot.gripper
            if isinstance(gripper, dict):
                gripper = gripper[arm]
            site_name = gripper.naming_prefix + "grip_site"
            self._gripper_site_ids.append(self.sim.model.site_name2id(site_name))

    def _get_cloth_vertices(self) -> Optional[np.ndarray]:
        if getattr(self.sim.model, "nflex", 0) <= 0:
            return None
        if not hasattr(self.sim.data, "flexvert_xpos"):
            return None
        verts = self.sim.data.flexvert_xpos
        if self._cloth_vert_slice is not None:
            verts = verts[self._cloth_vert_slice]
        return np.array(verts, copy=True)

    def _get_gripper_positions(self) -> Optional[np.ndarray]:
        if not self._gripper_site_ids:
            return None
        return np.array([self.sim.data.site_xpos[sid] for sid in self._gripper_site_ids], copy=True)

    def _find_nearest_vertices(self, gripper_pos: np.ndarray) -> np.ndarray:
        if getattr(self.sim.model, "nflex", 0) <= 0 or not hasattr(self.sim.data, "flexvert_xpos"):
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
        if not self.grasp_assist:
            return
        if getattr(self.sim.model, "nflex", 0) <= 0 or not hasattr(self.sim.data, "flexvert_xpos"):
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

        for idx, gripper_pos in enumerate(gripper_positions):
            grip_cmd = action[idx * per_arm_dim + (per_arm_dim - 1)]
            closing = grip_cmd > self.assist_action_threshold

            if closing and idx not in self._assist_vertices:
                vertices = self._find_nearest_vertices(gripper_pos)
                if vertices.size == 0:
                    continue
                self._assist_vertices[idx] = vertices
                offsets = self.sim.data.flexvert_xpos[vertices] - gripper_pos[None, :]
                self._assist_offsets[idx] = offsets
            elif not closing and idx in self._assist_vertices:
                self._assist_vertices.pop(idx, None)
                self._assist_offsets.pop(idx, None)

        for idx, vertices in self._assist_vertices.items():
            offsets = self._assist_offsets.get(idx)
            if offsets is None:
                continue
            gripper_pos = gripper_positions[idx]
            self.sim.data.flexvert_xpos[vertices] = gripper_pos[None, :] + offsets
            if hasattr(self.sim.data, "flexvert_xvel"):
                self.sim.data.flexvert_xvel[vertices] = 0.0

    @staticmethod
    def _extract_corners(vertices: np.ndarray) -> np.ndarray:
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
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cloth_corners(obs_cache):
                verts = self._get_cloth_vertices()
                corners = self._extract_corners(verts)
                return corners.reshape(-1)

            @sensor(modality=modality)
            def cloth_center(obs_cache):
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
        return 0.0

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        self._update_grasp_assist(action)

    def _reset_internal(self):
        super()._reset_internal()
        self._assist_vertices = {}
        self._assist_offsets = {}

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
        self._setup_free_camera()

    def _setup_free_camera(self):
        """Enable free camera mode for interactive navigation with mouse."""
        if not self.has_renderer:
            return

        try:
            # Access the camera through various possible viewer structures
            cam = None

            # Try method 1: viewer.viewer.cam (MuJoCo 2.x with robosuite wrapper)
            if hasattr(self, 'viewer') and self.viewer is not None:
                if hasattr(self.viewer, 'viewer') and self.viewer.viewer is not None:
                    cam = self.viewer.viewer.cam
                # Try method 2: viewer.cam (direct access)
                elif hasattr(self.viewer, 'cam'):
                    cam = self.viewer.cam
                # Try method 3: sim.render_contexts[0].cam
                elif hasattr(self, 'sim') and hasattr(self.sim, 'render_contexts'):
                    if len(self.sim.render_contexts) > 0:
                        cam = self.sim.render_contexts[0].cam

            # Try method 4: access through sim model directly
            if cam is None and hasattr(self, 'sim'):
                # Some robosuite versions access camera differently
                if hasattr(self.sim, 'viewer') and self.sim.viewer is not None:
                    if hasattr(self.sim.viewer, 'cam'):
                        cam = self.sim.viewer.cam

            # If we found a camera, configure it for free control
            if cam is not None:
                cam.type = 0  # mjCAMERA_FREE - enables mouse control

                # Position camera to look at training pair (arms 0 and 1)
                # For front-back config, robots 0 and 1 are at y=+0.25
                cam.distance = 1.5
                cam.elevation = -60  # More top-down view
                cam.azimuth = 90     # Looking from side
                try:
                    # Look at center point between training pair robots
                    cam.lookat[:] = [0.0, 0.25, 0.85]  # Center on training pair
                except:
                    # Some versions might have different lookat API
                    pass
        except Exception as e:
            # Silently fail - the camera will just use default settings
            pass

    def render(self):
        """Override render to ensure free camera is set up."""
        # Set up free camera on first render
        if self.has_renderer and not hasattr(self, '_free_camera_initialized'):
            self._setup_free_camera()
            self._free_camera_initialized = True
        return super().render()
