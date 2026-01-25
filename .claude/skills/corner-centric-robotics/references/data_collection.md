# Data Collection Reference

## HDF5 Format (ACT-Compatible)

### Structure

```
demos_arm0.hdf5
├── attrs: {num_demos, timestamp}
├── demo_0/
│   ├── observations/
│   │   ├── image     (T, H, W, 3) uint8
│   │   └── qpos      (T, state_dim) float32
│   ├── actions       (T, action_dim) float32
│   └── attrs: {success, length}
├── demo_1/
│   └── ...
```

### Writing HDF5

```python
import h5py

with h5py.File("demos.hdf5", "w") as f:
    f.attrs["num_demos"] = len(demos)
    
    for i, demo in enumerate(demos):
        grp = f.create_group(f"demo_{i}")
        
        obs_grp = grp.create_group("observations")
        obs_grp.create_dataset("image", data=demo["image"],
                               compression="gzip", compression_opts=4)
        obs_grp.create_dataset("qpos", data=demo["qpos"])
        
        grp.create_dataset("actions", data=demo["actions"])
        grp.attrs["success"] = demo["success"]
        grp.attrs["length"] = len(demo["actions"])
```

### Reading HDF5

```python
with h5py.File("demos.hdf5", "r") as f:
    num_demos = f.attrs["num_demos"]
    
    for i in range(num_demos):
        images = f[f"demo_{i}/observations/image"][:]
        qpos = f[f"demo_{i}/observations/qpos"][:]
        actions = f[f"demo_{i}/actions"][:]
```

## Scripted Policy Design

### Phase-Based Structure

```python
class ScriptedPolicy:
    def __init__(self):
        self.phase = "approach"  # approach → descend → grasp → lift → move
    
    def reset(self):
        self.phase = "approach"
    
    def get_action(self, obs, env):
        ee_local = obs["qpos"][:3]
        target_local = obs["target"]
        grasped = obs["qpos"][3] > 0.5
        
        if self.phase == "approach":
            goal = np.array([0, 0, 0.15])  # Above corner
            if close(ee_local, goal):
                self.phase = "descend"
            return move_toward(ee_local, goal, grasp=False)
        
        elif self.phase == "descend":
            goal = np.array([0, 0, 0.02])  # Near corner
            if close(ee_local, goal):
                self.phase = "grasp"
            return move_toward(ee_local, goal, grasp=False)
        
        # ... continue for other phases
```

### Movement Helper

```python
def move_toward(current, goal, grasp, speed=3.0):
    delta = goal - current
    dist = np.linalg.norm(delta)
    
    if dist > 0.001:
        step = min(dist, speed) * (delta / dist)
    else:
        step = np.zeros(3)
    
    return np.array([step[0], step[1], step[2], float(grasp)])
```

## Collection Workflow

### Basic Collection

```python
def collect_demo(env, policy, arm_idx=0):
    obs = env.reset(arm_idx=arm_idx, randomize=True)
    policy.reset()
    
    images, qpos_list, actions = [], [], []
    
    for step in range(max_steps):
        images.append(obs["image"])
        qpos_list.append(obs["qpos"])
        
        action = policy.get_action(obs, env)
        actions.append(action)
        
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    return {
        "image": np.stack(images),
        "qpos": np.stack(qpos_list),
        "actions": np.stack(actions),
        "success": info["success"],
    }
```

### Batch Collection with Retry

```python
def collect_demos(n_demos, arm_idx=0):
    env = BimanualCornerEnv()
    policy = ScriptedPolicy()
    demos = []
    
    for i in range(n_demos):
        demo = collect_demo(env, policy, arm_idx)
        
        if demo["success"]:
            demos.append(demo)
        else:
            # Retry once
            demo = collect_demo(env, policy, arm_idx)
            if demo["success"]:
                demos.append(demo)
    
    return demos
```

## Data Augmentation (Optional)

### Image Augmentation

```python
import torchvision.transforms as T

augment = T.Compose([
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
])
```

### State Noise

```python
qpos_noisy = qpos + np.random.normal(0, 0.005, size=qpos.shape)
```

## Inspection Tools

```python
def inspect_demos(filepath):
    with h5py.File(filepath, "r") as f:
        print(f"Demos: {f.attrs['num_demos']}")
        
        demo = f["demo_0"]
        print(f"Image: {demo['observations/image'].shape}")
        print(f"Qpos: {demo['observations/qpos'].shape}")
        print(f"Actions: {demo['actions'].shape}")
        
        # Length stats
        lengths = [f[f"demo_{i}"].attrs["length"] 
                   for i in range(f.attrs["num_demos"])]
        print(f"Lengths: {min(lengths)}-{max(lengths)}, mean={np.mean(lengths):.1f}")
```
