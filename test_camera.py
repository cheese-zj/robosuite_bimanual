#!/usr/bin/env python3
"""
Quick test script to verify camera setup.
"""

from envs.two_arm_cloth_fold import TwoArmClothFold
import time

print("Creating environment...")
env = TwoArmClothFold(
    robots=["Panda", "Panda"],
    env_configuration="parallel",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    render_camera="bimanual_view",
)

print("Resetting environment...")
obs = env.reset()

print("\nViewer information:")
print(f"  has_renderer: {env.has_renderer}")
print(f"  viewer: {env.viewer}")
print(f"  viewer type: {type(env.viewer)}")

if env.viewer:
    print(f"  viewer attributes: {dir(env.viewer)}")

    if hasattr(env.viewer, "viewer"):
        print(f"  viewer.viewer: {env.viewer.viewer}")
        if env.viewer.viewer:
            print(f"  viewer.viewer.cam: {env.viewer.viewer.cam}")
            cam = env.viewer.viewer.cam
            print(f"  Camera type: {cam.type}")
            print(f"  Camera distance: {cam.distance}")
            print(f"  Camera elevation: {cam.elevation}")
            print(f"  Camera azimuth: {cam.azimuth}")

    if hasattr(env.viewer, "cam"):
        print(f"  viewer.cam: {env.viewer.cam}")
        cam = env.viewer.cam
        print(f"  Camera type: {cam.type}")

print("\nRendering for 10 seconds...")
print("Try using mouse to control camera:")
print("  - Left-click drag: rotate")
print("  - Right-click drag: pan")
print("  - Scroll: zoom")
print("  - Double-click: select body (if not in free mode)")

for i in range(200):  # 10 seconds at 20Hz
    action = env.action_space.sample() * 0.0  # zero actions
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.05)

    if done:
        obs = env.reset()

env.close()
print("\nTest complete!")
