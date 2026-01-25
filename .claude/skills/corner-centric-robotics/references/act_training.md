# ACT Training Reference

## ACT (Action Chunking with Transformers)

### Key Concepts

- **Action Chunking**: Predicts sequences of actions (chunk_size=100 typical)
- **CVAE**: Conditional VAE for multimodal action distributions
- **Temporal Ensemble**: Averages overlapping action predictions

### Architecture

```
Input:
  - images: (B, N_cams, H, W, 3)
  - qpos: (B, state_dim)

Encoder:
  - ResNet18 per camera → image features
  - Concatenate with qpos
  - Transformer encoder

Decoder:
  - CVAE latent z (training) or z=0 (inference)
  - Transformer decoder
  - Output: (B, chunk_size, action_dim)
```

## Dataset Class

```python
import torch
from torch.utils.data import Dataset
import h5py

class ManipulationDataset(Dataset):
    def __init__(self, hdf5_path, chunk_size=100):
        self.chunk_size = chunk_size
        
        with h5py.File(hdf5_path, "r") as f:
            self.num_demos = f.attrs["num_demos"]
            
            # Load all data into memory
            self.episodes = []
            for i in range(self.num_demos):
                self.episodes.append({
                    "image": f[f"demo_{i}/observations/image"][:],
                    "qpos": f[f"demo_{i}/observations/qpos"][:],
                    "actions": f[f"demo_{i}/actions"][:],
                })
        
        # Build index mapping
        self.indices = []
        for ep_idx, ep in enumerate(self.episodes):
            T = len(ep["actions"])
            for t in range(T):
                self.indices.append((ep_idx, t))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ep_idx, t = self.indices[idx]
        ep = self.episodes[ep_idx]
        
        # Current observation
        image = ep["image"][t]  # (H, W, 3)
        qpos = ep["qpos"][t]    # (state_dim,)
        
        # Action chunk (pad if needed)
        T = len(ep["actions"])
        actions = ep["actions"][t:t+self.chunk_size]
        
        if len(actions) < self.chunk_size:
            pad = np.tile(actions[-1:], (self.chunk_size - len(actions), 1))
            actions = np.concatenate([actions, pad])
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        
        return {
            "image": image,
            "qpos": torch.from_numpy(qpos).float(),
            "actions": torch.from_numpy(actions).float(),
        }
```

## Training Loop

```python
def train_act(dataset, model, epochs=100, lr=1e-4):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in loader:
            optimizer.zero_grad()
            
            # Forward
            pred_actions, kl_loss = model(
                batch["image"], 
                batch["qpos"],
                batch["actions"],  # For CVAE training
            )
            
            # Loss
            recon_loss = F.l1_loss(pred_actions, batch["actions"])
            loss = recon_loss + 0.1 * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")
```

## Evaluation

### Rollout

```python
def evaluate_policy(env, policy, arm_idx, n_trials=20):
    successes = 0
    
    for trial in range(n_trials):
        obs = env.reset(arm_idx=arm_idx, randomize=True)
        action_buffer = None
        buffer_idx = 0
        
        for step in range(300):
            # Get action from policy (with chunking)
            if action_buffer is None or buffer_idx >= len(action_buffer):
                image = torch.from_numpy(obs["image"]).float() / 255.0
                image = image.permute(2, 0, 1).unsqueeze(0)
                qpos = torch.from_numpy(obs["qpos"]).float().unsqueeze(0)
                
                with torch.no_grad():
                    action_buffer = policy.predict(image, qpos)[0]  # (chunk, action_dim)
                buffer_idx = 0
            
            action = action_buffer[buffer_idx].numpy()
            buffer_idx += 1
            
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        if info["success"]:
            successes += 1
    
    return {"success_rate": successes / n_trials}
```

### Temporal Ensemble (Better Performance)

```python
def evaluate_with_ensemble(env, policy, arm_idx, n_trials=20, k=0.01):
    """Use exponential temporal ensemble for smoother actions."""
    
    for trial in range(n_trials):
        obs = env.reset(arm_idx=arm_idx)
        all_actions = {}  # {timestep: [predictions]}
        
        for step in range(300):
            # Predict chunk
            chunk = policy.predict(obs)  # (chunk_size, action_dim)
            
            # Store predictions
            for i, action in enumerate(chunk):
                t = step + i
                if t not in all_actions:
                    all_actions[t] = []
                all_actions[t].append(action)
            
            # Ensemble current timestep
            if step in all_actions:
                weights = np.exp(-k * np.arange(len(all_actions[step])))
                weights /= weights.sum()
                action = np.sum([w * a for w, a in zip(weights, all_actions[step])], axis=0)
            else:
                action = chunk[0]
            
            obs, reward, done, info = env.step(action)
            if done:
                break
```

## Transfer Experiment

```python
# Core experiment: Train on Arm 0, test on both
dataset = ManipulationDataset("data/demos_arm0.hdf5")
model = ACTPolicy(...)
train_act(dataset, model)

# Evaluate
results_arm0 = evaluate_policy(env, model, arm_idx=0, n_trials=20)
results_arm1 = evaluate_policy(env, model, arm_idx=1, n_trials=20)

print(f"Arm 0 (train): {results_arm0['success_rate']:.1%}")
print(f"Arm 1 (transfer): {results_arm1['success_rate']:.1%}")

# Success criteria: Arm 1 >= 70% indicates hypothesis validated
```

## Hyperparameters

| Param | Typical Value |
|-------|--------------|
| chunk_size | 100 |
| hidden_dim | 512 |
| n_heads | 8 |
| n_layers | 4 |
| lr | 1e-4 |
| batch_size | 32 |
| epochs | 100-500 |
| kl_weight | 0.1 |

## Common Issues

### Action Jittering
- Increase chunk_size
- Use temporal ensemble
- Lower action_scale in env

### Poor Transfer
- Check observation symmetry (run symmetry test)
- Verify x-flip is correct
- Ensure cameras are symmetric

### Overfitting
- Collect more demos
- Add data augmentation
- Reduce model size
