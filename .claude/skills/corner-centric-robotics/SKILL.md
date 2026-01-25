---
name: corner-centric-robotics
description: MuJoCo-based robotic manipulation research for policy transfer via corner-centric observation spaces. Use when working on (1) MuJoCo environment development for manipulation tasks, (2) collecting demonstrations with scripted policies, (3) training ACT/imitation learning policies, (4) evaluating policy transfer across symmetric arms, or (5) any task involving this corner-centric manipulation project.
---

# Corner-Centric Robotics Skill

Research project for validating policy transfer via corner-centric observation/action space design.

## Project Context

**Core Hypothesis**: A policy trained on Arm 0 can transfer to Arm 1 (symmetric about Y-axis) through corner-centric observation design.

**Key Insight**: Express observations relative to the active corner, flip x-coordinate for symmetric arms.

```python
# Corner-centric transformation
ee_local = ee_pos - corner_pos
if arm == 1:
    ee_local[0] = -ee_local[0]  # Flip x for symmetry
```

## Project Structure

```
corner-centric-manip/
├── envs/
│   ├── bimanual_env.py    # BimanualCornerEnv + ScriptedPolicy
│   └── scene.xml          # MuJoCo scene (2 arms, 2 cameras)
├── scripts/
│   ├── collect_data.py    # Data collection → HDF5
│   └── test_env.py        # Environment tests
├── data/                  # HDF5 demonstrations
└── models/                # Trained policies
```

## Quick Reference

### Environment API

```python
from envs import BimanualCornerEnv, ScriptedPolicy

env = BimanualCornerEnv()
obs = env.reset(arm_idx=0, randomize=True)
# obs = {"image": (H,W,3), "qpos": (4,), "target": (3,)}

action = np.array([dx, dy, dz, grasp])  # Corner-centric frame
obs, reward, done, info = env.step(action)
```

### Data Format (HDF5)

```
/demo_N/observations/image   (T, 240, 320, 3) uint8
/demo_N/observations/qpos    (T, 4) float32  
/demo_N/actions              (T, 4) float32
```

### Experiment Matrix

| Exp | Train | Test | Expected |
|-----|-------|------|----------|
| Baseline | Arm 0 | Arm 0 | ~85% |
| **Core** | Arm 0 | Arm 1 | ~70%+ |
| Ablation | Arm 0 (global) | Arm 1 | ~30% |

## Detailed References

- **MuJoCo development**: See [references/mujoco.md](references/mujoco.md) for scene design, cameras, constraints
- **Data collection**: See [references/data_collection.md](references/data_collection.md) for HDF5 format, scripted policies
- **ACT training**: See [references/act_training.md](references/act_training.md) for policy training and evaluation

## Common Tasks

| Task | Location |
|------|----------|
| Add camera | `envs/scene.xml` - add `<camera>` with symmetric position |
| Modify observation | `BimanualCornerEnv.get_obs()` in `envs/bimanual_env.py` |
| Change action scale | `action_scale` param in `BimanualCornerEnv.__init__()` |
| Debug symmetry | `python scripts/test_env.py --test symmetry` |
| Collect data | `python scripts/collect_data.py --n_demos 50 --arm 0` |
