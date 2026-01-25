"""
Bimanual Data Collection using Robosuite

This uses robosuite's built-in:
- Robot models (Panda, UR5e, etc.)
- OSC controllers (much better than raw joint control)
- Teleoperation devices
- Data collection utilities

Collect data with two arms, then train ACT policies on bimanual demonstrations.

Robosuite supports: Panda, Sawyer, UR5e, Kinova, IIWA, Jaco
"""

import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody


CameraNames = Optional[Union[str, Sequence[str]]]
FREE_CAMERA_NAME = "free_camera"
FREE_CAMERA_ALIASES = {FREE_CAMERA_NAME, "free"}


def _is_free_camera_name(camera_name: str) -> bool:
    return str(camera_name).lower() in FREE_CAMERA_ALIASES


def _resolve_controller_name(controller):
    if controller is None or controller == "" or controller == "default":
        return None
    if controller in {
        "OSC_POSE",
        "OSC_POSITION",
        "JOINT_VELOCITY",
        "JOINT_POSITION",
        "IK_POSE",
    }:
        return "BASIC"
    return controller


def _parse_camera_names(camera_names: CameraNames) -> Optional[list[str]]:
    if camera_names is None:
        return None
    if isinstance(camera_names, (list, tuple)):
        names = list(camera_names)
    else:
        names = [name.strip() for name in str(camera_names).split(",") if name.strip()]
    names = [name for name in names if not _is_free_camera_name(name)]
    return names or None


def _apply_input_ref_frame(controller_config, input_ref_frame: Optional[str]):
    if input_ref_frame is None:
        return controller_config
    config = deepcopy(controller_config)
    body_parts = config.get("body_parts", {})
    for part_cfg in body_parts.values():
        if isinstance(part_cfg, dict) and "input_ref_frame" in part_cfg:
            part_cfg["input_ref_frame"] = input_ref_frame
    return config


def _init_prev_gripper_actions(env):
    return [
        {
            f"{arm}_gripper": np.repeat([0], robot.gripper[arm].dof)
            for arm in robot.arms
            if robot.gripper[arm].dof > 0
        }
        for robot in env.robots
    ]


def _normalize_action_dict(robot, input_ac_dict):
    action_dict = deepcopy(input_ac_dict)
    for arm in robot.arms:
        if isinstance(robot.composite_controller, WholeBody):
            controller_input_type = (
                robot.composite_controller.joint_action_policy.input_type
            )
        else:
            controller_input_type = robot.part_controllers[arm].input_type
        if controller_input_type == "delta":
            action_dict[arm] = input_ac_dict[f"{arm}_delta"]
        elif controller_input_type == "absolute":
            action_dict[arm] = input_ac_dict[f"{arm}_abs"]
        else:
            raise ValueError(
                f"Unsupported controller input type: {controller_input_type}"
            )
    return action_dict


def _build_env_action(env, robot_action_dicts, prev_gripper_actions):
    env_action = [
        robot.create_action_vector(prev_gripper_actions[i])
        for i, robot in enumerate(env.robots)
    ]
    for robot_idx, action_dict in robot_action_dicts.items():
        env_action[robot_idx] = env.robots[robot_idx].create_action_vector(action_dict)
        for gripper_key in prev_gripper_actions[robot_idx]:
            prev_gripper_actions[robot_idx][gripper_key] = action_dict[gripper_key]
    return np.concatenate(env_action)


def _get_viewer_cam(env):
    if hasattr(env, "viewer") and env.viewer is not None:
        if hasattr(env.viewer, "viewer") and env.viewer.viewer is not None:
            if hasattr(env.viewer.viewer, "cam"):
                return env.viewer.viewer.cam
        if hasattr(env.viewer, "cam"):
            return env.viewer.cam
    if hasattr(env, "sim") and hasattr(env.sim, "render_contexts"):
        if env.sim.render_contexts:
            cam = getattr(env.sim.render_contexts[0], "cam", None)
            if cam is not None:
                return cam
    if (
        hasattr(env, "sim")
        and hasattr(env.sim, "viewer")
        and env.sim.viewer is not None
    ):
        if hasattr(env.sim.viewer, "cam"):
            return env.sim.viewer.cam
    return None


def _camera_exists(env, camera_name: str) -> bool:
    try:
        env.sim.model.camera_name2id(camera_name)
    except Exception:
        return False
    return True


def _set_render_camera(env, camera_name: str) -> bool:
    cam = _get_viewer_cam(env)
    if cam is None:
        return False
    if _is_free_camera_name(camera_name):
        if hasattr(env, "_setup_free_camera"):
            try:
                env._setup_free_camera()
            except Exception:
                pass
        cam.type = 0  # mjCAMERA_FREE
        return True
    try:
        cam_id = env.sim.model.camera_name2id(camera_name)
    except Exception:
        return False
    cam.type = 2  # mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    if hasattr(env, "render_camera"):
        env.render_camera = [camera_name]
    return True


def _get_render_camera_cycle(env) -> list[str]:
    cycle = []
    if _camera_exists(env, "bimanual_view"):
        cycle.append("bimanual_view")
    if FREE_CAMERA_NAME not in cycle:
        cycle.append(FREE_CAMERA_NAME)
    return cycle


class RecordingControls:
    def __init__(self):
        from pynput.keyboard import Key, Listener

        self._key = Key
        self._actions = deque()
        self._listener = Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key):
        if key == self._key.enter:
            self._actions.append("toggle_record")
        elif key == self._key.backspace:
            self._actions.append("discard")
        elif key == self._key.esc:
            self._actions.append("quit")
        else:
            try:
                if hasattr(key, "char") and key.char == "l":
                    self._actions.append("toggle_camera")
            except Exception:
                pass

    def pop_action(self):
        if self._actions:
            return self._actions.popleft()
        return None

    def stop(self):
        self._listener.stop()


def create_bimanual_env(
    robots: str = "Panda",
    task: str = "TwoArmLift",
    controller: str = "BASIC",
    input_ref_frame: Optional[str] = None,
    has_renderer: bool = True,
    has_offscreen_renderer: bool = True,
    use_camera_obs: bool = True,
    ignore_done: bool = False,
    camera_names: CameraNames = None,
    render_camera: Optional[str] = None,
    cloth_preset: str = "medium",
):
    """
    Create a bimanual robosuite environment.

    Args:
        robots: Robot type ("Panda", "UR5e", "Sawyer", etc.)
        task: Task name (see AVAILABLE_TASKS below)
        controller: Composite controller name or config path (e.g., "BASIC")
        has_renderer: Enable on-screen rendering (viewer window)
        has_offscreen_renderer: Enable off-screen rendering (for camera obs)
        use_camera_obs: Include camera observations in the observation dict
        camera_names: List of cameras to include in observations

    Returns:
        Robosuite environment
    """
    if task == "TwoArmClothFold":
        from envs.two_arm_cloth_fold import TwoArmClothFold

        robots_list = [robots, robots] if isinstance(robots, str) else robots
        camera_names = _parse_camera_names(camera_names)
        if camera_names is None and use_camera_obs:
            camera_names = ["bimanual_view"]
        if isinstance(render_camera, (list, tuple)):
            render_camera = render_camera[0] if render_camera else None
        if render_camera is None and has_renderer and use_camera_obs:
            render_camera = "bimanual_view"
        if render_camera == FREE_CAMERA_NAME:
            render_camera = None
        return TwoArmClothFold(
            robots=robots_list,
            env_configuration="parallel",
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            use_object_obs=True,
            camera_names=camera_names,
            render_camera=render_camera,
            horizon=500,
            ignore_done=ignore_done,
            cloth_preset=cloth_preset,
        )

    # Load composite controller config (robosuite >= 1.5)
    controller_name = _resolve_controller_name(controller)
    controller_config = load_composite_controller_config(
        controller=controller_name,
        robot=robots if isinstance(robots, str) else robots[0],
    )
    controller_config = _apply_input_ref_frame(controller_config, input_ref_frame)

    # Default cameras
    camera_names = _parse_camera_names(camera_names)
    if camera_names is None and use_camera_obs:
        camera_names = ["birdview", "agentview", "frontview"]
    if isinstance(render_camera, (list, tuple)):
        render_camera = render_camera[0] if render_camera else None
    free_camera = render_camera == FREE_CAMERA_NAME
    if free_camera:
        render_camera = None
    if render_camera is None and has_renderer and not free_camera:
        render_camera = "frontview"

    # Create environment
    env = suite.make(
        env_name=task,
        robots=[robots, robots],  # Two robots for bimanual
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        use_camera_obs=use_camera_obs,
        camera_names=camera_names,
        camera_heights=480,
        camera_widths=640,
        render_camera=render_camera,
        control_freq=20,  # 20Hz control
        horizon=500,  # Max episode length
        ignore_done=ignore_done,
    )

    return env


# Available bimanual tasks in robosuite
AVAILABLE_TASKS = {
    "TwoArmLift": "Two arms cooperatively lift an object (pot)",
    "TwoArmPegInHole": "Two arms insert peg into hole",
    "TwoArmHandover": "Pass object between two arms",
    "TwoArmTransport": "Transport object with two arms",
    "TwoArmClothFold": "Flexcloth folding with two arms (custom env)",
}


class BimanualDataCollector:
    """
    Collects demonstration data from a bimanual robosuite environment.
    Saves in ACT-compatible HDF5 format.
    """

    def __init__(
        self,
        env,
        save_dir: str = "data/bimanual",
        camera_names: CameraNames = None,
    ):
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if camera_names is None:
            camera_names = getattr(env, "camera_names", None)
        self.camera_names = _parse_camera_names(camera_names) or ["agentview"]

        # Episode tracking
        self.episode_count = self._count_existing()
        self.current_episode = None
        self.recording = False

        print(f"Save directory: {self.save_dir}")
        print(f"Existing episodes: {self.episode_count}")

    def _count_existing(self) -> int:
        return len(list(self.save_dir.glob("episode_*.hdf5")))

    def start_episode(self):
        """Start recording a new episode"""
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        self.recording = True
        print(f"\n🔴 Recording episode {self.episode_count}")

    def record_step(self, obs: dict, action: np.ndarray, reward: float, done: bool):
        """Record a single step"""
        if not self.recording:
            return
        if self.current_episode is None:
            return

        # Store observation
        obs_data = {
            "robot0_joint_pos": obs.get("robot0_joint_pos", np.zeros(7)),
            "robot0_joint_vel": obs.get("robot0_joint_vel", np.zeros(7)),
            "robot0_eef_pos": obs.get("robot0_eef_pos", np.zeros(3)),
            "robot0_eef_quat": obs.get("robot0_eef_quat", np.zeros(4)),
            "robot0_gripper_qpos": obs.get("robot0_gripper_qpos", np.zeros(2)),
            "robot1_joint_pos": obs.get("robot1_joint_pos", np.zeros(7)),
            "robot1_joint_vel": obs.get("robot1_joint_vel", np.zeros(7)),
            "robot1_eef_pos": obs.get("robot1_eef_pos", np.zeros(3)),
            "robot1_eef_quat": obs.get("robot1_eef_quat", np.zeros(4)),
            "robot1_gripper_qpos": obs.get("robot1_gripper_qpos", np.zeros(2)),
        }

        # Add camera images
        for cam in self.camera_names:
            key = f"{cam}_image"
            if key in obs:
                obs_data[key] = obs[key]

        if "cloth_corners" in obs:
            obs_data["cloth_corners"] = obs["cloth_corners"]
        if "cloth_center" in obs:
            obs_data["cloth_center"] = obs["cloth_center"]

        self.current_episode["observations"].append(obs_data)
        self.current_episode["actions"].append(action)
        self.current_episode["rewards"].append(reward)
        self.current_episode["dones"].append(done)

    def save_episode(self, success: bool = True):
        """Save current episode to HDF5"""
        if not self.recording or self.current_episode is None:
            return

        self.recording = False
        n_steps = len(self.current_episode["actions"])

        if n_steps < 10:
            print(f"❌ Episode too short ({n_steps} steps), discarding")
            return

        filename = self.save_dir / f"episode_{self.episode_count:04d}.hdf5"

        with h5py.File(filename, "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.attrs["timestamp"] = datetime.now().isoformat()
            meta.attrs["success"] = success
            meta.attrs["episode_length"] = n_steps

            # Observations
            obs_grp = f.create_group("observations")

            # Aggregate observations
            for key in self.current_episode["observations"][0].keys():
                data = np.array([o[key] for o in self.current_episode["observations"]])
                if "image" in key:
                    obs_grp.create_dataset(key, data=data, compression="gzip")
                else:
                    obs_grp.create_dataset(key, data=data)

            # Actions
            f.create_dataset("actions", data=np.array(self.current_episode["actions"]))

            # Rewards
            f.create_dataset("rewards", data=np.array(self.current_episode["rewards"]))

        print(f"✅ Saved episode {self.episode_count} ({n_steps} steps)")
        self.episode_count += 1

    def discard_episode(self):
        """Discard current episode"""
        self.recording = False
        self.current_episode = None
        print("❌ Episode discarded")


def run_keyboard_collection(
    task: str = "TwoArmLift",
    robot: str = "Panda",
    save_dir: str = "data/bimanual",
    num_episodes: int = 10,
    camera_names: CameraNames = None,
    use_camera_obs: bool = True,
    render_camera: Optional[str] = "bimanual_view",
    practice_mode: bool = False,
    cloth_preset: str = "medium",
):
    """
    Run interactive data collection with keyboard control.

    Controls:
        Robosuite prints the default keyboard mapping on start.

        Recording:
        ENTER: Start/stop recording
        BACKSPACE: Discard episode
        ESC: Quit
        s: Switch active arm
        =: Switch active robot
        l: Toggle render camera
    """
    print("\n" + "=" * 60)
    print("BIMANUAL DATA COLLECTION")
    print("=" * 60)
    print(f"Task: {task}")
    print(f"Robot: {robot}")
    print(f"Target episodes: {num_episodes}")
    print("=" * 60)

    free_camera = render_camera == FREE_CAMERA_NAME
    if practice_mode:
        print("[Practice Mode] Viewer stays open after saving episodes.")

    # Create environment
    env = create_bimanual_env(
        robots=robot,
        task=task,
        has_renderer=True,
        has_offscreen_renderer=use_camera_obs,
        use_camera_obs=use_camera_obs,
        ignore_done=practice_mode,
        camera_names=camera_names,
        render_camera=render_camera,
        cloth_preset=cloth_preset,
    )

    # Create data collector
    collector = BimanualDataCollector(env, save_dir=save_dir, camera_names=camera_names)

    from robosuite.devices.keyboard import Keyboard

    device = Keyboard(env=env, pos_sensitivity=0.05, rot_sensitivity=0.1)
    controls = RecordingControls()

    obs = env.reset()
    env.render()
    device.start_control()
    prev_gripper_actions = _init_prev_gripper_actions(env)
    camera_cycle = _get_render_camera_cycle(env)
    camera_index = 0

    print("\nCONTROLS:")
    print("  Enter: Start/stop recording | Backspace: Discard | Esc: Quit")
    print("  s: Switch active arm | =: Switch active robot")
    print("  l: Toggle render camera")
    print("  (Movement keys are printed by robosuite on start)\n")
    if camera_cycle:
        printable_cycle = [
            "Free" if _is_free_camera_name(name) else name for name in camera_cycle
        ]
        print(f"  Render camera cycle: {', '.join(printable_cycle)}\n")

    while collector.episode_count < num_episodes or practice_mode:
        action = controls.pop_action()

        if action == "toggle_record":
            if collector.recording:
                collector.save_episode(success=True)
                if practice_mode:
                    print("[Practice] Episode saved. Continuing without exiting.")
            else:
                collector.start_episode()
        elif action == "discard":
            collector.discard_episode()
        elif action == "quit":
            break
        elif action == "toggle_camera" and camera_cycle:
            camera_index = (camera_index + 1) % len(camera_cycle)
            _set_render_camera(env, camera_cycle[camera_index])

        input_ac_dict = device.input2action()
        if input_ac_dict is None:
            if collector.recording:
                collector.discard_episode()
            obs = env.reset()
            env.render()
            device.start_control()
            prev_gripper_actions = _init_prev_gripper_actions(env)
            camera_cycle = _get_render_camera_cycle(env)
            camera_index = 0
            continue

        robot = env.robots[device.active_robot]
        action_dict = _normalize_action_dict(robot, input_ac_dict)
        env_action = _build_env_action(
            env,
            {device.active_robot: action_dict},
            prev_gripper_actions,
        )

        obs, reward, done, info = env.step(env_action)
        collector.record_step(obs, env_action, reward, done)
        env.render()

        if done:
            if collector.recording:
                collector.save_episode(success=info.get("success", False))
                if practice_mode:
                    print("[Practice] Episode saved. Continuing without exiting.")
            obs = env.reset()
            env.render()
            device.start_control()
            prev_gripper_actions = _init_prev_gripper_actions(env)
            if not practice_mode and collector.episode_count >= num_episodes:
                break

    env.close()
    controls.stop()


def run_spacemouse_collection(
    task: str = "TwoArmLift",
    robot: str = "Panda",
    save_dir: str = "data/bimanual",
    num_episodes: int = 10,
    camera_names: CameraNames = None,
    use_camera_obs: bool = True,
    render_camera: Optional[str] = "bimanual_view",
    practice_mode: bool = False,
    cloth_preset: str = "medium",
):
    """
    Run data collection with SpaceMouse (if available).
    Much more intuitive than keyboard!
    """
    try:
        from robosuite.devices.spacemouse import SpaceMouse
    except Exception as e:
        print(f"SpaceMouse import error: {e}")
        print("Falling back to keyboard control...")
        return run_keyboard_collection(
            task,
            robot,
            save_dir,
            num_episodes,
            camera_names=camera_names,
            use_camera_obs=use_camera_obs,
            render_camera=render_camera,
            practice_mode=practice_mode,
            cloth_preset=cloth_preset,
        )

    print("\n" + "=" * 60)
    print("BIMANUAL DATA COLLECTION (SpaceMouse)")
    print("=" * 60)
    print("Enter: Start/stop recording | Backspace: Discard | Esc: Quit")
    print("l: Toggle render camera")
    print("Press '=' to switch active robot if using a single SpaceMouse.")
    if practice_mode:
        print("[Practice Mode] Viewer stays open after saving episodes.")

    env = create_bimanual_env(
        robots=robot,
        task=task,
        has_renderer=True,
        has_offscreen_renderer=use_camera_obs,
        use_camera_obs=use_camera_obs,
        ignore_done=practice_mode,
        camera_names=camera_names,
        render_camera=render_camera,
        cloth_preset=cloth_preset,
    )

    collector = BimanualDataCollector(env, save_dir=save_dir, camera_names=camera_names)
    controls = RecordingControls()

    devices = []
    try:
        device0 = SpaceMouse(env=env, pos_sensitivity=1.0, rot_sensitivity=1.0)
        device0.start_control()
        devices.append((device0, 0))
        try:
            device1 = SpaceMouse(
                env=env,
                pos_sensitivity=1.0,
                rot_sensitivity=1.0,
                vendor_id=9583,
                product_id=50741,
            )
            device1.start_control()
            devices.append((device1, 1))
            print("Two SpaceMouse devices found")
        except Exception as e:
            print(f"Second SpaceMouse not found ({e}); using single device.")
            print("Press '=' to switch active robot.")
            devices = [(device0, None)]
    except Exception as e:
        print(f"SpaceMouse error: {e}")
        print("Falling back to keyboard control...")
        env.close()
        controls.stop()
        return run_keyboard_collection(
            task,
            robot,
            save_dir,
            num_episodes,
            camera_names=camera_names,
            use_camera_obs=use_camera_obs,
            render_camera=render_camera,
            practice_mode=practice_mode,
            cloth_preset=cloth_preset,
        )

    obs = env.reset()
    env.render()
    prev_gripper_actions = _init_prev_gripper_actions(env)
    camera_cycle = _get_render_camera_cycle(env)
    camera_index = 0
    if camera_cycle:
        printable_cycle = [
            "Free" if _is_free_camera_name(name) else name for name in camera_cycle
        ]
        print(f"Render camera cycle: {', '.join(printable_cycle)}")

    while collector.episode_count < num_episodes or practice_mode:
        action = controls.pop_action()
        if action == "toggle_record":
            if collector.recording:
                collector.save_episode(success=True)
                if practice_mode:
                    print("[Practice] Episode saved. Continuing without exiting.")
            else:
                collector.start_episode()
        elif action == "discard":
            collector.discard_episode()
        elif action == "quit":
            break
        elif action == "toggle_camera" and camera_cycle:
            camera_index = (camera_index + 1) % len(camera_cycle)
            _set_render_camera(env, camera_cycle[camera_index])

        robot_action_dicts = {}
        reset = False
        for device, fixed_robot_idx in devices:
            if fixed_robot_idx is not None:
                device.active_robot = fixed_robot_idx
                robot_idx = fixed_robot_idx
            else:
                robot_idx = device.active_robot

            input_ac_dict = device.input2action()
            if input_ac_dict is None:
                reset = True
                break

            action_dict = _normalize_action_dict(env.robots[robot_idx], input_ac_dict)
            robot_action_dicts[robot_idx] = action_dict

        if reset:
            if collector.recording:
                collector.discard_episode()
            obs = env.reset()
            env.render()
            for device, _ in devices:
                device.start_control()
            prev_gripper_actions = _init_prev_gripper_actions(env)
            camera_cycle = _get_render_camera_cycle(env)
            camera_index = 0
            continue

        env_action = _build_env_action(env, robot_action_dicts, prev_gripper_actions)
        obs, reward, done, info = env.step(env_action)
        collector.record_step(obs, env_action, reward, done)
        env.render()

        if done:
            if collector.recording:
                collector.save_episode(success=info.get("success", False))
                if practice_mode:
                    print("[Practice] Episode saved. Continuing without exiting.")
            obs = env.reset()
            env.render()
            for device, _ in devices:
                device.start_control()
            prev_gripper_actions = _init_prev_gripper_actions(env)
            if not practice_mode and collector.episode_count >= num_episodes:
                break

    env.close()
    controls.stop()


def simple_demo(task: str = "TwoArmLift", robot: str = "Panda", use_cv2: bool = False):
    """Just view the environment without recording"""
    print(f"\nViewing {task} with {robot} robots...")

    if use_cv2:
        import cv2

        env = create_bimanual_env(
            robots=robot,
            task=task,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            render_camera="agentview",
        )

        obs = env.reset()
        print("\nPress 'q' in the OpenCV window to exit.")
        print("The robot will perform random actions.\n")

        for _ in range(1000):
            env_action = np.random.randn(env.action_dim or 0) * 0.1
            obs, reward, done, info = env.step(env_action)

            frame = env.sim.render(width=640, height=480, camera_name="agentview")
            frame = frame[::-1]  # Flip vertically (MuJoCo convention)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"{task} Demo", frame)

            if cv2.waitKey(20) & 0xFF == ord("q"):
                break

            if done:
                obs = env.reset()

        cv2.destroyAllWindows()
        env.close()
    else:
        env = create_bimanual_env(
            robots=robot,
            task=task,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            render_camera=None,
        )

        obs = env.reset()
        env.render()
        controls = RecordingControls()
        camera_cycle = _get_render_camera_cycle(env)
        camera_index = 0

        print("\nClose the viewer window to exit.")
        print("The robot will perform random actions.\n")
        if camera_cycle:
            printable_cycle = [
                "Free" if _is_free_camera_name(name) else name for name in camera_cycle
            ]
            print(f"Render camera cycle: {', '.join(printable_cycle)}")

        for _ in range(1000):
            action = controls.pop_action()
            if action == "toggle_camera" and camera_cycle:
                camera_index = (camera_index + 1) % len(camera_cycle)
                _set_render_camera(env, camera_cycle[camera_index])

            env_action = np.random.randn(env.action_dim or 0) * 0.1
            obs, reward, done, info = env.step(env_action)
            env.render()

            if done:
                obs = env.reset()

        controls.stop()
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Bimanual data collection with robosuite"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="TwoArmLift",
        choices=list(AVAILABLE_TASKS.keys()),
        help="Task environment",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="Panda",
        choices=["Panda", "UR5e", "Sawyer", "IIWA", "Jaco", "Kinova3"],
        help="Robot type",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/bimanual",
        help="Directory to save demonstrations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
        help="Input device",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just view environment without recording"
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        default=None,
        help="Comma-separated camera names (e.g., birdview,frontview)",
    )
    parser.add_argument(
        "--camera_obs",
        action="store_true",
        default=True,
        help="Enable camera observations (default)",
    )
    parser.add_argument(
        "--no_camera_obs",
        action="store_false",
        dest="camera_obs",
        help="Disable camera observations",
    )
    parser.add_argument(
        "--render_camera",
        type=str,
        default=None,
        help="Viewer camera name (default: task-dependent)",
    )
    parser.add_argument(
        "--free_camera",
        action="store_true",
        help="Start viewer in free camera mode (implies no fixed camera)",
    )
    parser.add_argument(
        "--practice",
        action="store_true",
        help="Keep viewer open after saving episodes",
    )
    parser.add_argument(
        "--ignore_done",
        action="store_true",
        help="Ignore task termination and keep stepping",
    )
    parser.add_argument(
        "--cloth_preset",
        type=str,
        default="medium",
        choices=["fast", "medium", "realistic", "legacy"],
        help="Cloth simulation preset for TwoArmClothFold (fast=9x9, medium=15x15, realistic=21x21)",
    )
    parser.add_argument(
        "--cv2",
        action="store_true",
        help="Use OpenCV for rendering (workaround for macOS mjpython issue)",
    )

    args = parser.parse_args()

    print("\nAvailable tasks:")
    for name, desc in AVAILABLE_TASKS.items():
        print(f"  {name}: {desc}")
    print()

    camera_names = args.camera_names

    render_camera = args.render_camera
    if args.free_camera:
        render_camera = "free_camera"

    practice_mode = args.practice or args.ignore_done

    if args.demo:
        simple_demo(args.task, args.robot, use_cv2=args.cv2)
    elif args.device == "spacemouse":
        run_spacemouse_collection(
            args.task,
            args.robot,
            args.save_dir,
            args.episodes,
            camera_names=camera_names,
            use_camera_obs=args.camera_obs,
            render_camera=render_camera,
            practice_mode=practice_mode,
            cloth_preset=args.cloth_preset,
        )
    else:
        run_keyboard_collection(
            args.task,
            args.robot,
            args.save_dir,
            args.episodes,
            camera_names=camera_names,
            use_camera_obs=args.camera_obs,
            render_camera=render_camera,
            practice_mode=practice_mode,
            cloth_preset=args.cloth_preset,
        )


if __name__ == "__main__":
    main()
