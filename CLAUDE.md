# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A framework for bimanual robot learning with robosuite: collecting demonstrations, training ACT (Action Chunking with Transformers) policies, and deploying them. Primary use case is symmetric bimanual tasks like cloth folding with the custom `TwoArmClothFold` environment.

## Essential Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install robosuite numpy h5py torch mujoco

# View environment
python collect_robosuite.py --demo --task TwoArmLift --robot Panda

# Collect teleop demos
python collect_robosuite.py --task TwoArmClothFold --episodes 50

# Collect scripted demos (cloth folding)
python collect_scripted.py --two_arm_cloth --policy cloth_fold --episodes 50 --render

# Train ACT policy (state-only)
python train_act.py --data_dir data/bimanual --epochs 500

# Train with vision
python train_act.py --data_dir data/bimanual --epochs 500 --use_images --camera_names bimanual_view

# Deploy policy
python deploy_quad.py --checkpoint checkpoints/best_model.pt --two_arm_cloth --render

# Evaluate policy
python evaluate.py --checkpoint checkpoints/best_model.pt --task TwoArmClothFold --episodes 20

# Mirror augmentation
python augment_data.py --input_dir data/bimanual --output_dir data/bimanual_mirrored

# Headless rendering
export MUJOCO_GL=egl  # or osmesa for software rendering
```

No build step, no test suite, no configured linter. If asked to format: `python -m black <files>`. If asked to lint: `python -m ruff check <files>`.

## Architecture

**Pipeline flow:** Data Collection → HDF5 Dataset → ACT Training → Policy Deployment

**Key scripts:**
- `collect_robosuite.py` / `collect_scripted.py` - Demo collection (teleop or waypoint-based)
- `dataset.py` - HDF5 loader with normalization stats
- `train_act.py` - ACT model (transformer encoder-decoder with VAE)
- `vision_encoder.py` - ResNet-18 for image observations
- `deploy_quad.py` - Rollout with optional receding horizon control
- `evaluate.py` - Success metrics to JSON

**Custom environment:** `envs/two_arm_cloth_fold.py` - MuJoCo flexcloth with grasp-assist (pins cloth vertices to gripper while closed)

**HDF5 schema:** Episodes contain `observations/robot{0,1}_{joint_pos,joint_vel,eef_pos,eef_quat,gripper_qpos}`, `actions` (14-dim OSC_POSE), `rewards`, `metadata`

## Key Patterns

**Observation flattening:** 60 dims base (30/arm: 7 joint pos + 7 joint vel + 3 eef pos + 4 eef quat + 2 gripper). Add +15 for cloth obs, +512 per camera for vision.

**Normalization:** Z-score for observations, fixed [-1, 1] bounds for actions (OSC compatible).

**Controller config:**
```python
from robosuite import load_composite_controller_config
config = load_composite_controller_config(controller_type="BASIC", body_part="bimanual", input_controller="OSC_POSE")
```

**Mirror transform (Y-axis):** Position `(x, -y, z)`, Quaternion `(w, x, -y, -z)`, Rotation `(-dax, day, -daz)`. Use `swap_arms=True` for symmetric tasks.

## Common Flags

- `--task` / `--two_arm_cloth`: Select environment
- `--input_ref_frame base|world`: Controller reference frame
- `--include_object_obs`: Add cloth corners/center to observations
- `--recompute_freq N`: Receding horizon control interval
- `--cloth_preset fast|medium|high`: Cloth resolution (9x9, 15x15, 20x20)

## Code Style

- Python 3.9+, PEP 8, 4-space indent, ~100 char lines
- Type hints for public functions, `np.ndarray` / `torch.Tensor` for arrays
- Use `pathlib.Path`, f-strings, argparse with explicit boolean flags
- Observation keys: `robot{idx}_<field>` naming
- Keep changes minimal and focused; clarity over cleverness in robotics math
