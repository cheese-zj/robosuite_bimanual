"""Four-arm variant of the robosuite TwoArmLift task."""

from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.arenas import TableArena
from robosuite.models.objects import PotWithHandlesObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

from .four_arm_env import FourArmEnv


class FourArmLift(FourArmEnv):
    """Lift task with four single-arm robots.

    Primary pair is robots 0 and 1; robots 2 and 3 are available for mirroring.
    """

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
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="birdview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=["birdview", "agentview"],
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

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

    def reward(self, action=None):
        reward = 0

        mat = T.quat2mat(self._pot_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_30 = np.cos(np.pi / 6)
        direction_coef = 1 if cos_z >= cos_30 else 0

        if self._check_success():
            reward = 3.0 * direction_coef
        elif self.reward_shaping:
            pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
            table_height = self.sim.data.site_xpos[self.table_top_id][2]
            elevation = pot_bottom_height - table_height
            r_lift = min(max(elevation - 0.05, 0), 0.15)
            reward += 10.0 * direction_coef * r_lift

            _gripper0_to_handle0 = self._gripper0_to_handle0
            _gripper1_to_handle1 = self._gripper1_to_handle1

            (g0, g1) = (self.robots[0].gripper, self.robots[1].gripper)

            _g0h_dist = np.linalg.norm(_gripper0_to_handle0)
            _g1h_dist = np.linalg.norm(_gripper1_to_handle1)

            if self._check_grasp(gripper=g0, object_geoms=self.pot.handle0_geoms):
                reward += 0.25
            reward += 0.5 * (1 - np.tanh(10.0 * _g0h_dist))

            if self._check_grasp(gripper=g1, object_geoms=self.pot.handle1_geoms):
                reward += 0.25
            reward += 0.5 * (1 - np.tanh(10.0 * _g1h_dist))

        if self.reward_scale is not None:
            reward *= self.reward_scale / 3.0

        return reward

    def _load_model(self):
        super()._load_model()

        self._set_robot_base_poses()

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.pot = PotWithHandlesObject(name="pot")

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.pot)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.pot,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                rotation=(np.pi + -np.pi / 3, np.pi + np.pi / 3),
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.pot,
        )

    def _setup_references(self):
        super()._setup_references()

        self.pot_body_id = self.sim.model.body_name2id(self.pot.root_body)
        self.handle0_site_id = self.sim.model.site_name2id(self.pot.important_sites["handle0"])
        self.handle1_site_id = self.sim.model.site_name2id(self.pot.important_sites["handle1"])
        self.table_top_id = self.sim.model.site_name2id("table_top")
        self.pot_center_id = self.sim.model.site_name2id(self.pot.important_sites["center"])

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def pot_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.pot_body_id])

            @sensor(modality=modality)
            def pot_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.pot_body_id], to="xyzw")

            @sensor(modality=modality)
            def handle0_xpos(obs_cache):
                return np.array(self._handle0_xpos)

            @sensor(modality=modality)
            def handle1_xpos(obs_cache):
                return np.array(self._handle1_xpos)

            sensors = [pot_pos, pot_quat, handle0_xpos, handle1_xpos]
            names = [s.__name__ for s in sensors]

            arm_sensor_fns = []
            robot_arm_prefixes = [self._get_arm_prefixes(robot, include_robot_name=False) for robot in self.robots[:2]]
            robot_full_prefixes = [self._get_arm_prefixes(robot, include_robot_name=True) for robot in self.robots[:2]]
            for i, (arm_prefixes, full_prefixes) in enumerate(zip(robot_arm_prefixes, robot_full_prefixes)):
                arm_sensor_fns += [
                    self._get_obj_eef_sensor(
                        full_pf, f"handle{i}_xpos", f"{arm_pf}gripper{i}_to_handle{i}", modality
                    )
                    for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                ]

            sensors += arm_sensor_fns
            names += [s.__name__ for s in arm_sensor_fns]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)

        if vis_settings.get("grippers"):
            handles = [self.pot.important_sites[f"handle{i}"] for i in range(2)]
            grippers = [robot.gripper for robot in self.robots[:2]]
            for gripper, handle in zip(grippers, handles):
                self._visualize_gripper_to_target(gripper=gripper, target=handle, target_type="site")

    def _check_success(self):
        pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
        table_height = self.sim.data.site_xpos[self.table_top_id][2]
        return pot_bottom_height > table_height + 0.10

    @property
    def _handle0_xpos(self):
        return self.sim.data.site_xpos[self.handle0_site_id]

    @property
    def _handle1_xpos(self):
        return self.sim.data.site_xpos[self.handle1_site_id]

    @property
    def _pot_quat(self):
        return T.convert_quat(self.sim.data.body_xquat[self.pot_body_id], to="xyzw")

    @property
    def _gripper0_to_handle0(self):
        return self._gripper0_to_target(self.pot.important_sites["handle0"], target_type="site")

    @property
    def _gripper1_to_handle1(self):
        return self._gripper1_to_target(self.pot.important_sites["handle1"], target_type="site")
