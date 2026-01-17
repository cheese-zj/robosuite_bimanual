#!/usr/bin/env python3
"""
Diagnostic script to identify the correct camera access method for your Robosuite version.
Run this to help debug the free camera issue.
"""

from envs.two_arm_cloth_fold import TwoArmClothFold
import time

print("=" * 70)
print("CAMERA DIAGNOSTICS")
print("=" * 70)

print("\n[1] Creating environment with renderer...")
try:
    env = TwoArmClothFold(
        robots=["Panda", "Panda"],
        env_configuration="parallel",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        render_camera="bimanual_view",
    )
    print("✓ Environment created successfully")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    exit(1)

print("\n[2] Resetting environment...")
try:
    obs = env.reset()
    print("✓ Environment reset successfully")
except Exception as e:
    print(f"✗ Failed to reset: {e}")
    env.close()
    exit(1)

print("\n[3] Rendering once to initialize viewer...")
try:
    env.render()
    print("✓ First render completed")
except Exception as e:
    print(f"✗ Failed to render: {e}")

print("\n[4] Analyzing viewer structure...")
print(f"  env.has_renderer = {env.has_renderer}")
print(f"  env.viewer exists = {hasattr(env, 'viewer')}")
print(f"  env.viewer = {env.viewer}")

camera_found = False

if hasattr(env, "viewer") and env.viewer is not None:
    print(f"\n  [4a] env.viewer type = {type(env.viewer)}")
    print(f"  [4a] env.viewer.__class__.__name__ = {env.viewer.__class__.__name__}")

    # Check viewer.viewer
    if hasattr(env.viewer, "viewer"):
        print(f"  [4b] env.viewer.viewer exists = {env.viewer.viewer is not None}")
        if env.viewer.viewer is not None:
            print(f"  [4b] env.viewer.viewer type = {type(env.viewer.viewer)}")
            if hasattr(env.viewer.viewer, "cam"):
                print(f"  [4c] ✓ FOUND: env.viewer.viewer.cam")
                cam = env.viewer.viewer.cam
                print(f"       Camera type: {cam.type}")
                print(f"       Camera distance: {cam.distance}")
                print(f"       Camera elevation: {cam.elevation}")
                print(f"       Camera azimuth: {cam.azimuth}")
                camera_found = True

    # Check viewer.cam
    if hasattr(env.viewer, "cam"):
        print(f"  [4d] ✓ FOUND: env.viewer.cam")
        cam = env.viewer.cam
        print(f"       Camera type: {cam.type}")
        camera_found = True

# Check sim-based access
if hasattr(env, "sim"):
    print(f"\n  [4e] env.sim exists")

    if hasattr(env.sim, "viewer"):
        print(f"  [4f] env.sim.viewer exists = {env.sim.viewer is not None}")
        if env.sim.viewer is not None and hasattr(env.sim.viewer, "cam"):
            print(f"  [4g] ✓ FOUND: env.sim.viewer.cam")
            camera_found = True

    if hasattr(env.sim, "render_contexts"):
        print(
            f"  [4h] env.sim.render_contexts exists, length = {len(env.sim.render_contexts)}"
        )
        if len(env.sim.render_contexts) > 0:
            if hasattr(env.sim.render_contexts[0], "cam"):
                print(f"  [4i] ✓ FOUND: env.sim.render_contexts[0].cam")
                camera_found = True

print("\n[5] Testing camera modification...")
if camera_found:
    print("  Attempting to set camera to free mode...")
    try:
        # Try the method that worked
        if (
            hasattr(env, "viewer")
            and env.viewer
            and hasattr(env.viewer, "viewer")
            and env.viewer.viewer
        ):
            if hasattr(env.viewer.viewer, "cam"):
                cam = env.viewer.viewer.cam
                original_type = cam.type
                cam.type = 0  # mjCAMERA_FREE
                cam.distance = 2.0
                cam.elevation = -45
                cam.azimuth = 90
                cam.lookat[:] = [0.0, 0.0, 0.85]
                print(f"  ✓ Camera modified successfully!")
                print(f"    Original type: {original_type} -> New type: {cam.type}")
    except Exception as e:
        print(f"  ✗ Failed to modify camera: {e}")
else:
    print("  ✗ No camera access method found!")

print("\n[6] Rendering for 5 seconds...")
print("  Try to control the camera with your mouse:")
print("    - Left-click drag: rotate")
print("    - Right-click drag: pan")
print("    - Scroll: zoom")
print("  If the camera doesn't respond, it's not in free mode.")

for i in range(100):
    action = env.action_space.sample() * 0.0
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.05)
    if done:
        obs = env.reset()

env.close()

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
print("\nIf camera control didn't work, please share this output!")
