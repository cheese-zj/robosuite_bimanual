# MuJoCo Development Reference

## Scene Design (MJCF)

### Basic Structure

```xml
<mujoco model="name">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" .../>
    <material name="grid_mat" texture="grid" .../>
  </asset>
  
  <worldbody>
    <!-- Objects, cameras, etc -->
  </worldbody>
  
  <equality>
    <!-- Constraints (welds, connects) -->
  </equality>
</mujoco>
```

### Mocap Bodies (Direct Position Control)

```xml
<body name="ee_0" mocap="true" pos="0.2 0 0.4">
  <geom type="sphere" size="0.02" contype="0" conaffinity="0"/>
</body>
```

Control in Python:
```python
data.mocap_pos[mocap_idx] = new_position
data.mocap_quat[mocap_idx] = new_quaternion
```

### Grasp via Weld Constraints

```xml
<equality>
  <weld name="grasp_0" body1="ee_0" body2="plate" 
        anchor="0.12 0 0" active="false"/>
</equality>
```

Toggle at runtime:
```python
model.eq_active[eq_id] = 1  # Enable grasp
model.eq_active[eq_id] = 0  # Release
```

### Camera Setup

```xml
<!-- Overhead camera -->
<camera name="overhead" pos="0 0 1.0" xyaxes="1 0 0 0 1 0" fovy="60"/>

<!-- Corner camera (symmetric pair) -->
<camera name="cam_0" pos="0.35 0.1 0.5" 
        xyaxes="-0.5 0.866 0 -0.433 -0.25 0.866" fovy="50"/>
<camera name="cam_1" pos="-0.35 0.1 0.5" 
        xyaxes="0.5 0.866 0 0.433 -0.25 0.866" fovy="50"/>
```

**Symmetry Rule**: For Y-axis symmetry, flip x in `pos` and adjust `xyaxes` accordingly.

### Sites (Visual Markers)

```xml
<site name="corner_0" pos="0.12 0 0" size="0.012" rgba="1 0 0 1"/>
<site name="target_0" pos="0.08 0.12 0.3" size="0.015" rgba="1 0 0 0.4"/>
```

Get position: `data.site_xpos[site_id]`

## Python Environment Wrapper

### ID Caching

```python
def _cache_ids(self):
    self.ee_body_ids = [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"ee_{i}")
        for i in range(2)
    ]
    self.corner_site_ids = [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"corner_{i}")
        for i in range(2)
    ]
```

### Rendering

```python
renderer = mujoco.Renderer(model, height=240, width=320)
renderer.update_scene(data, camera="cam_0")
image = renderer.render()  # (H, W, 3) uint8 RGB
```

### Physics Stepping

```python
# Single step
mujoco.mj_step(model, data)

# Sub-stepping for stability (recommended)
for _ in range(10):
    mujoco.mj_step(model, data)
```

### Forward Kinematics Only

```python
mujoco.mj_forward(model, data)  # Update positions without physics
```

## Common Patterns

### Workspace Limits

```python
new_pos = np.clip(new_pos, 
                  [-0.4, -0.3, 0.15],  # Lower bounds
                  [0.4, 0.3, 0.6])     # Upper bounds
```

### Distance-Based Grasp

```python
dist = np.linalg.norm(ee_pos - corner_pos)
if dist < grasp_threshold and grasp_cmd:
    model.eq_active[grasp_eq_id] = 1
```

### Reset Pattern

```python
def reset(self):
    self.data.qpos[:] = self._init_qpos
    self.data.qvel[:] = self._init_qvel
    self.data.mocap_pos[:] = self._init_mocap_pos
    mujoco.mj_forward(self.model, self.data)
```

## Debugging

```python
# Print all body names
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"Body {i}: {name}")

# Visualize with viewer (requires mujoco viewer)
import mujoco.viewer
mujoco.viewer.launch(model, data)
```
