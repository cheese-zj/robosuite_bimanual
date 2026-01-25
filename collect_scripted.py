"""Scripted bimanual data collection for robosuite tasks."""

import argparse
from typing import Optional

from collect_robosuite import (
    BimanualDataCollector,
    _parse_camera_names,
    create_bimanual_env,
)
from scripted_policy import (
    ScriptedBimanualPolicy,
    ScriptedClothFoldPolicy,
    ScriptedPolicyConfig,
    ClothFoldPolicyConfig,
)


def run_scripted_collection(
    task: str = "TwoArmLift",
    robot: str = "Panda",
    save_dir: str = "data/bimanual",
    num_episodes: int = 10,
    max_steps: Optional[int] = None,
    render: bool = False,
    use_camera_obs: bool = False,
    camera_names: Optional[str] = None,
    two_arm_cloth: bool = False,
    env_configuration: str = "parallel",
    input_ref_frame: Optional[str] = "world",
    policy_name: Optional[str] = None,
    debug: bool = False,
    demo: bool = False,
    cloth_preset: str = "medium",
    cloth_noise: bool = False,
    robot_noise: bool = False,
    use_cv2: bool = False,
):
    if policy_name is None:
        policy_name = "cloth_fold" if two_arm_cloth or "Cloth" in task else "lift"

    if policy_name == "cloth_fold":
        two_arm_cloth = True

    if two_arm_cloth and env_configuration == "front-back":
        env_configuration = "opposed"

    if policy_name == "cloth_fold":
        layout = "parallel"
        if env_configuration == "opposed":
            layout = "front-back"
        policy = ScriptedClothFoldPolicy(
            ClothFoldPolicyConfig(layout=layout), debug=debug
        )
        needs_object_obs = True
    else:
        policy = ScriptedBimanualPolicy(ScriptedPolicyConfig())
        needs_object_obs = False

    if use_cv2 and render:
        import cv2
    else:
        cv2 = None

    # When using cv2, use offscreen renderer instead of native viewer
    use_native_renderer = render and not use_cv2
    use_offscreen = use_camera_obs or render

    if debug:
        print(f"[Debug Mode] Task: {task}, Robot: {robot}, Policy: {policy_name}")
        if two_arm_cloth:
            print(
                f"[Debug Mode] Two-arm cloth folding, Env config: {env_configuration}"
            )

    camera_list = _parse_camera_names(camera_names)

    if two_arm_cloth and camera_list is None:
        camera_list = ["bimanual_view"]

    if two_arm_cloth:
        from envs.two_arm_cloth_fold import TwoArmClothFold

        robots_list = [robot, robot] if isinstance(robot, str) else robot
        env = TwoArmClothFold(
            robots=robots_list,
            env_configuration=env_configuration,
            has_renderer=use_native_renderer,
            has_offscreen_renderer=use_offscreen,
            use_camera_obs=use_camera_obs,
            use_object_obs=needs_object_obs,
            camera_names=camera_list,
            render_camera="bimanual_view",
            horizon=max_steps or 500,
            cloth_preset=cloth_preset,
            cloth_noise=cloth_noise,
            robot_noise=robot_noise,
        )
    else:
        env = create_bimanual_env(
            robots=robot,
            task=task,
            has_renderer=use_native_renderer,
            has_offscreen_renderer=use_offscreen,
            use_camera_obs=use_camera_obs,
            input_ref_frame=input_ref_frame,
            camera_names=camera_list,
        )

    # Skip data collection if demo mode
    if demo:
        collector = None
        print("\n[Demo Mode] Running without data collection. Close window to exit.\n")
    else:
        collector = BimanualDataCollector(
            env, save_dir=save_dir, camera_names=camera_list
        )

    for episode in range(num_episodes):
        obs = env.reset()
        policy.reset()
        if collector:
            collector.start_episode()

        done = False
        steps = 0
        step_limit = max_steps or env.horizon

        if debug:
            print(f"\n[Episode {episode + 1}/{num_episodes}] Starting...")

        info = {}
        while not done and steps < step_limit:
            action = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            if collector:
                collector.record_step(obs, action, reward, done)

            if render:
                if cv2 is not None:
                    cv2_camera = "bimanual_view" if two_arm_cloth else "agentview"
                    frame = env.sim.render(width=640, height=480, camera_name=cv2_camera)
                    frame = frame[::-1]  # Flip vertically (MuJoCo convention)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"{task} Scripted Collection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    env.render()

            steps += 1

            # Check if scripted policy has completed all phases
            if hasattr(policy, 'done') and policy.done:
                done = True
                info["success"] = True

        if collector:
            collector.save_episode(success=info.get("success", False))

        if debug or demo:
            print(
                f"Episode {episode + 1}/{num_episodes} complete (steps: {steps}, success: {info.get('success', False)})"
            )

    if cv2 is not None:
        cv2.destroyAllWindows()
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Scripted bimanual data collection",
        epilog="Camera controls: Left-click drag to rotate, right-click drag to pan, scroll to zoom",
    )
    parser.add_argument("--task", type=str, default="TwoArmLift")
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="data/bimanual")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--render", action="store_true", help="Show visualization window"
    )
    parser.add_argument("--camera_obs", action="store_true")
    parser.add_argument("--camera_names", type=str, default=None)
    parser.add_argument(
        "--two_arm_cloth",
        action="store_true",
        help="Use TwoArmClothFold environment for cloth demos",
    )
    parser.add_argument(
        "--env_configuration",
        type=str,
        default="parallel",
        help="Two-arm cloth layout (parallel/opposed/single-robot)",
    )
    parser.add_argument(
        "--input_ref_frame", type=str, default="world", choices=["base", "world"]
    )
    parser.add_argument(
        "--policy", type=str, choices=["lift", "cloth_fold"], default=None
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output (prints policy state)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode: visualization only, no data collection",
    )
    parser.add_argument(
        "--cloth_preset",
        type=str,
        default="medium",
        choices=["fast", "medium", "realistic", "legacy"],
        help="Cloth simulation preset (fast=9x9, medium=15x15, realistic=21x21)",
    )
    parser.add_argument(
        "--cloth_noise",
        action="store_true",
        help="Enable cloth position randomization on reset",
    )
    parser.add_argument(
        "--robot_noise",
        action="store_true",
        help="Enable robot joint initialization noise",
    )
    parser.add_argument(
        "--cv2",
        action="store_true",
        help="Use OpenCV for rendering (workaround for macOS mjpython issue)",
    )

    args = parser.parse_args()

    run_scripted_collection(
        task=args.task,
        robot=args.robot,
        save_dir=args.save_dir,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render or args.demo,
        use_camera_obs=args.camera_obs or args.two_arm_cloth,
        camera_names=args.camera_names,
        two_arm_cloth=args.two_arm_cloth,
        env_configuration=args.env_configuration,
        input_ref_frame=args.input_ref_frame,
        policy_name=args.policy,
        debug=args.debug,
        demo=args.demo,
        cloth_preset=args.cloth_preset,
        cloth_noise=args.cloth_noise,
        robot_noise=args.robot_noise,
        use_cv2=args.cv2,
    )


if __name__ == "__main__":
    main()
