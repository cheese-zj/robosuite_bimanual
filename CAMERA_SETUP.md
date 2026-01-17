# Camera Setup for Cloth Folding Task

## Overview

The camera system has been configured to support:
1. **Bimanual View** - Overhead camera for TwoArmClothFold data collection
2. **Free Camera** - Interactive camera for visualization during testing

## Camera Types

### 1. Bimanual View (`bimanual_view`)
- **Purpose**: Data collection camera for training ACT model
- **Position**: `(0.0, 0.0, 1.4)` in world coordinates
- **Configuration**: Parallel two-arm layout
- **Usage**: Captures observations used for training

### 2. Free Camera (`free_camera`)
- **Purpose**: Interactive camera for visualization only
- **Usage**: Toggle with `l` and use mouse controls
- **Note**: Not saved to datasets


## How It Works

### Data Collection
When you run data collection with `--camera_obs`, the **bimanual_view** camera is included in observations:
```bash
python3 collect_scripted.py --two_arm_cloth --policy cloth_fold --episodes 10 --camera_obs
```

**Saved observations**:
- `bimanual_view_image`: RGB image from overhead of the cloth workspace

**Free camera**:
- Toggle with `l` to enter free camera mode
- Not included in observations

### Visualization
The **render camera** is used for the visualization window:
- Default: `bimanual_view` (overhead of the table)
- Interactive: Mouse controls enabled (free camera mode)
  - **Left-click drag**: Rotate view
  - **Right-click drag**: Pan view
  - **Scroll**: Zoom in/out

## File Locations

- **Camera Definition**: `assets/cameras/two_arm_birdview.xml`
- **Environment Integration**: `envs/two_arm_cloth_fold.py`
- **Default Settings**: `envs/two_arm_cloth_fold.py`

## Testing

### Test camera view:
```bash
python3 collect_scripted.py --two_arm_cloth --render --demo --episodes 1
```

### Collect data with bimanual camera:
```bash
python3 collect_scripted.py --two_arm_cloth --policy cloth_fold --episodes 5 --camera_obs
```

### Verify camera in saved data:
```python
import h5py
with h5py.File('data/bimanual/episode_0000.hdf5', 'r') as f:
    print(f['observations'].keys())  # Should include 'bimanual_view_image'
    img_shape = f['observations/bimanual_view_image'].shape
    print(f"Image shape: {img_shape}")  # Should be (T, H, W, 3)
```

## Customization

### Change camera position:
Edit `assets/cameras/two_arm_birdview.xml`:
```xml
<camera name="bimanual_view"
        mode="fixed"
        pos="0.0 0.0 1.4"  <!-- x, y, z position -->
        quat="0.653 0.271 0.271 0.653"/>  <!-- orientation quaternion -->
```

### Change default cameras:
Edit `envs/two_arm_cloth_fold.py`:
```python
camera_names=["bimanual_view"],  # Cameras for observations
```

### Disable free camera:
Comment out the render override in `envs/two_arm_cloth_fold.py`.

## Notes

- The **render_camera** parameter controls visualization, not data collection
- The **camera_names** parameter controls which cameras are in observations
- Free camera mode may not work on all robosuite versions - fallback to fixed view
- Camera quaternion `(0.653 0.271 0.271 0.653)` ≈ 90° rotation for top-down view
