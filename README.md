# Bimanual Robot Learning with Robosuite

A clean framework for:
1. **Collecting bimanual demonstrations** using robosuite
2. **Training ACT policies** on bimanual data
3. **Deploying bimanual policies** in robosuite

## Your Use Case

Bimanual tasks where two arms collaborate on the same object (e.g., lifting or cloth folding).
Use the custom `TwoArmClothFold` environment for flexcloth demos and training.

## Installation

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install robosuite numpy h5py torch

# Or with pip
pip install robosuite numpy h5py torch
```

## Quick Start

### 1. View the environment
```bash
python collect_robosuite.py --demo --task TwoArmLift --robot Panda
```

### 2. Collect demonstrations
```bash
# Keyboard control
python collect_robosuite.py --task TwoArmLift --episodes 50

# SpaceMouse (if available)
python collect_robosuite.py --task TwoArmLift --episodes 50 --device spacemouse

# Cloth folding (TwoArmClothFold)
python collect_robosuite.py --task TwoArmClothFold --episodes 50

# Faster teleop (no camera obs)
python collect_robosuite.py --task TwoArmClothFold --episodes 50 --no_camera_obs

# Practice mode (keep viewer open)
python collect_robosuite.py --task TwoArmClothFold --episodes 1 --no_camera_obs --practice

# Practice mode (ignore task termination)
python collect_robosuite.py --task TwoArmClothFold --episodes 1 --no_camera_obs --practice --ignore_done
```

### 2b. Collect scripted demonstrations
```bash
python collect_scripted.py --task TwoArmLift --episodes 50
```

To run the scripted cloth policy in `TwoArmClothFold`:

```bash
python collect_scripted.py --two_arm_cloth --policy cloth_fold --episodes 50 --render
```

Scripted collection uses world-frame OSC deltas by default. If you want base-frame control, add `--input_ref_frame base`.

### 3. Train ACT policy
```bash
# State-only training
python train_act.py --data_dir data/bimanual --epochs 500

# Vision-based training (recommended for cloth folding)
python train_act.py --data_dir data/bimanual --epochs 500 --use_images --camera_names bimanual_view

# Include cloth observations (corners, center)
python train_act.py --data_dir data/bimanual --epochs 500 --include_object_obs
```

### 4. Deploy policy (see below)

## Keyboard Controls

When collecting data, use these controls:

Robosuite prints its default keyboard mapping on start. Recording hotkeys:

| Key | Function |
|-----|----------|
| Enter | Start/Stop recording |
| Backspace | Discard episode |
| Escape | Quit |
| s | Switch active arm |
| = | Switch active robot |
| l | Toggle render camera |

## Available Tasks

| Task | Description |
|------|-------------|
| `TwoArmLift` | Cooperatively lift a pot |
| `TwoArmPegInHole` | Insert peg into hole |
| `TwoArmHandover` | Pass object between arms |
| `TwoArmTransport` | Transport object together |
| `TwoArmClothFold` | Flexcloth folding with two arms (custom env) |

## Data Format

Data is saved in HDF5 format:

```
episode_0000.hdf5
├── metadata/
│   ├── timestamp
│   ├── success
│   └── episode_length
├── observations/
│   ├── robot0_joint_pos    # (T, 7)
│   ├── robot0_eef_pos      # (T, 3)
│   ├── robot0_eef_quat     # (T, 4)
│   ├── robot0_gripper_qpos # (T, 2)
│   ├── robot1_joint_pos    # (T, 7)
│   ├── robot1_eef_pos      # (T, 3)
│   ├── robot1_eef_quat     # (T, 4)
│   ├── robot1_gripper_qpos # (T, 2)
│   ├── cloth_corners       # (T, 12) optional
│   ├── cloth_center        # (T, 3) optional
├── actions                  # (T, 14) for OSC_POSE
└── rewards                  # (T,)
```

## Deployment

Deploy a trained policy on a bimanual environment:

```bash
python deploy_quad.py --checkpoint checkpoints/best_model.pt --task TwoArmLift --render

# With receding horizon (recompute every 10 steps for error correction)
python deploy_quad.py --checkpoint checkpoints/best_model.pt --two_arm_cloth --render --recompute_freq 10
```

`deploy_quad.py` is a legacy filename from earlier multi-arm experiments.
Use `--input_ref_frame base` if your training data used base-frame deltas.
Use `--two_arm_cloth` to deploy in `TwoArmClothFold`.

### Evaluation

Evaluate a trained policy and get success metrics:

```bash
# Run 20 evaluation episodes
python evaluate.py --checkpoint checkpoints/best_model.pt --task TwoArmClothFold --episodes 20

# With rendering
python evaluate.py --checkpoint checkpoints/best_model.pt --task TwoArmClothFold --episodes 10 --render
```

Results are saved to `eval_results/` as JSON files with success rate, avg reward, and episode statistics.

### Cloth Folding (Flex Cloth)

Run the flexcloth folding task with the scripted policy:

```bash
python collect_scripted.py --two_arm_cloth --policy cloth_fold --episodes 20 --render
```

The cloth folding env uses a grasp-assist (pins nearby cloth vertices to the gripper while closed)
to make folding feasible without teleop.

Train with cloth observations enabled:

```bash
python train_act.py --data_dir data/bimanual --include_object_obs
```

### Mirror Data Augmentation

```bash
python augment_data.py --input_dir data/bimanual --output_dir data/bimanual_mirrored
```

### Mirror Transform Details

For Y-axis symmetry (folding along Y):

| Component | Transformation |
|-----------|----------------|
| Position (x, y, z) | (x, **-y**, z) |
| Quaternion (w, x, y, z) | (w, x, **-y**, **-z**) |
| Velocity dx, dy, dz | (dx, **-dy**, dz) |
| Rotation dax, day, daz | (**-dax**, day, **-daz**) |

### Arm Swapping

With `swap_arms=True` (recommended for symmetric tasks), mirrored demos swap
robot0/robot1 outputs after reflection to preserve left/right consistency.

## Cloth Simulation Notes

This repo includes a MuJoCo flexcloth task (`TwoArmClothFold`) for fast iteration.
If you need higher-fidelity cloth dynamics, consider SoftGym or Isaac.

## Project Structure

```
robosuite_bimanual/
├── collect_robosuite.py    # Data collection with viewer
├── collect_scripted.py     # Scripted data collection
├── dataset.py              # ACT-compatible data loading
├── deploy_quad.py          # Bimanual deployment script (legacy name)
├── evaluate.py             # Policy evaluation with success metrics
├── train_act.py            # ACT training script (state + vision)
├── vision_encoder.py       # ResNet-18 image encoder for vision ACT
├── envs/                   # Two-arm robosuite envs
│   └── two_arm_cloth_fold.py
├── mirror_transform.py     # Mirroring utilities
├── augment_data.py         # Mirror augmentation for datasets
├── scripted_policy.py      # Waypoint policy for scripted demos
└── requirements.txt        # Dependencies
```

## Tips for Good Demonstrations

1. **Consistent start**: Reset to same initial pose
2. **Smooth motions**: Avoid jerky movements
3. **Task completion**: Only save successful episodes
4. **Variation**: Collect demos with slight variations
5. **Mirror-aware**: Think about how actions will look mirrored

## Troubleshooting

### "No display" error
```bash
# Use offscreen rendering
export MUJOCO_GL=egl
# Or
export MUJOCO_GL=osmesa
```

### Robosuite version issues
```bash
# Ensure compatible versions
uv pip install "robosuite>=1.4.0" "mujoco>=2.3.0"
```

### SpaceMouse not detected
```bash
# Install hidapi
uv pip install hidapi

# On Linux, add udev rules
sudo cp 99-spacemouse.rules /etc/udev/rules.d/
```

### evdev build failed (Python.h missing)
On Linux, evdev may build from source and needs Python headers for your interpreter.

```bash
# Ubuntu/Debian (example for Python 3.12)
sudo apt-get install python3.12-dev build-essential
```

Replace `3.12` with your Python version, then retry the install.

## Remote Server / Headless Collection

For collecting data on remote servers without display:

```bash
# Set MuJoCo to use EGL (hardware-accelerated offscreen)
export MUJOCO_GL=egl

# Or use OSmesa (software rendering, no GPU required)
export MUJOCO_GL=osmesa

# Collect scripted demos (no viewer)
python collect_scripted.py --policy cloth_fold --episodes 100

# Collect teleop demos (offscreen rendering only)
python collect_robosuite.py --task TwoArmClothFold --episodes 100 --no_camera_obs
```

### Recommended Settings for Remote

| Setting | Value | Reason |
|---------|-------|--------|
| `--no_camera_obs` | Flag | Faster collection without camera rendering |
| `--cloth_preset` | `fast` | 9×9 vertices, 30 iterations (vs 15×15, 75) |
| `--episodes` | Batch | Run 50-100 per session for efficiency |

### GPU-Accelerated Rendering

If your server has an NVIDIA GPU:

```bash
# Install EGL support
sudo apt-get install nvidia-cuda-toolkit

# Verify GPU rendering
python -c "import mujoco; print(mujoco.viewer.launch_passive())"
```

### Multiprocessing for Speed

Run multiple collectors in parallel (different output directories):

```bash
# Terminal 1
python collect_scripted.py --policy cloth_fold --episodes 50 --save_dir data/bimanual_1

# Terminal 2
python collect_scripted.py --policy cloth_fold --episodes 50 --save_dir data/bimanual_2

# Then merge (manually copy HDF5 files to combined directory)
```

## References

- [Robosuite Documentation](https://robosuite.ai/)
- [ACT Paper](https://arxiv.org/abs/2304.13705)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
