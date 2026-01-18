"""
ACT Training Script for Robosuite Bimanual Data

Usage:
    python train_act.py --data_dir data/bimanual --epochs 500
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict

from dataset import create_dataloaders


@dataclass
class ACTConfig:
    """ACT model configuration"""

    obs_dim: int = 60  # 30 per arm × 2 arms (now includes joint velocities)
    action_dim: int = 14  # 7 per arm × 2 arms (OSC_POSE)
    hidden_dim: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    chunk_size: int = 50
    latent_dim: int = 32
    dropout: float = 0.1
    kl_weight: float = 10.0
    kl_warmup_epochs: int = 50  # Epochs over which to anneal KL weight

    # Vision parameters
    use_images: bool = False
    num_cameras: int = 1
    image_size: int = 224  # ResNet input size
    freeze_vision_backbone: bool = False


class ACTModel(nn.Module):
    """
    ACT (Action Chunking with Transformers) model with optional vision support.

    For the full implementation, see: https://github.com/tonyzhaozh/act
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # Vision encoder (optional)
        self.vision_encoder = None
        if config.use_images:
            from vision_encoder import VisionEncoder

            self.vision_encoder = VisionEncoder(
                hidden_dim=config.hidden_dim,
                num_cameras=config.num_cameras,
                pretrained=True,
                freeze_backbone=config.freeze_vision_backbone,
            )

        # Encoder - observation embedding
        # If using images, we combine state + image features
        self.obs_embed = nn.Linear(config.obs_dim, config.hidden_dim)
        if config.use_images:
            # Fusion layer for state + image features
            self.obs_fusion = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(config.hidden_dim),
            )

        self.action_embed = nn.Linear(config.action_dim, config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.n_encoder_layers)

        # VAE
        self.mu_proj = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_proj = nn.Linear(config.hidden_dim, config.latent_dim)

        # Decoder
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)
        self.action_queries = nn.Parameter(
            torch.randn(config.chunk_size, config.hidden_dim) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.n_decoder_layers)

        # Output
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.chunk_size + 1, config.hidden_dim) * 0.02
        )

    def _embed_observation(
        self, obs: torch.Tensor, images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Embed observation, optionally fusing with image features.

        Args:
            obs: (B, obs_dim) state observation
            images: (B, C, H, W) or (B, N, C, H, W) camera images

        Returns:
            (B, hidden_dim) embedded observation
        """
        obs_emb = self.obs_embed(obs)  # (B, H)

        if self.vision_encoder is not None and images is not None:
            img_emb = self.vision_encoder(images)  # (B, H)
            # Fuse state and image features
            combined = torch.cat([obs_emb, img_emb], dim=-1)  # (B, 2H)
            obs_emb = self.obs_fusion(combined)  # (B, H)

        return obs_emb

    def encode(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ):
        """Encode observation (and actions during training) to latent.

        Args:
            obs: (B, obs_dim) state observation
            actions: (B, T, action_dim) action sequence (training only)
            images: (B, C, H, W) or (B, N, C, H, W) camera images (optional)

        Returns:
            mu, logvar: Latent distribution parameters
        """
        B = obs.shape[0]

        obs_emb = self._embed_observation(obs, images).unsqueeze(1)  # (B, 1, H)

        if actions is not None:
            action_emb = self.action_embed(actions)  # (B, T, H)
            seq = torch.cat([obs_emb, action_emb], dim=1)  # (B, T+1, H)
        else:
            seq = obs_emb

        seq = seq + self.pos_embed[:, : seq.size(1)]
        encoded = self.encoder(seq)

        # Use first token for latent
        h = encoded[:, 0]
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)

        return mu, logvar

    def decode(
        self,
        z: torch.Tensor,
        obs: torch.Tensor,
        images: Optional[torch.Tensor] = None,
    ):
        """Decode latent to action sequence.

        Args:
            z: (B, latent_dim) latent vector
            obs: (B, obs_dim) state observation
            images: (B, C, H, W) or (B, N, C, H, W) camera images (optional)

        Returns:
            (B, chunk_size, action_dim) predicted actions
        """
        B = z.shape[0]

        z_emb = self.latent_proj(z).unsqueeze(1)
        obs_emb = self._embed_observation(obs, images).unsqueeze(1)
        memory = torch.cat([z_emb, obs_emb], dim=1)

        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, memory)

        return self.action_head(decoded)

    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        Args:
            obs: (B, obs_dim) state observation
            actions: (B, T, action_dim) action sequence (training only)
            images: (B, C, H, W) or (B, N, C, H, W) camera images (optional)

        Returns:
            pred_actions: (B, chunk_size, action_dim) predicted actions
            mu, logvar: Latent distribution parameters
        """
        mu, logvar = self.encode(obs, actions, images)

        if self.training and actions is not None:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        pred_actions = self.decode(z, obs, images)

        return pred_actions, mu, logvar

    def predict(
        self, obs: torch.Tensor, images: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Inference: predict action chunk.

        Args:
            obs: (obs_dim,) or (B, obs_dim) state observation
            images: (C, H, W), (B, C, H, W) or (B, N, C, H, W) camera images

        Returns:
            (chunk_size, action_dim) predicted actions
        """
        self.eval()
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            if images is not None and images.dim() == 3:
                images = images.unsqueeze(0)
            pred, _, _ = self(obs, images=images)
            return pred[0].cpu().numpy()


def get_kl_weight(epoch: int, warmup_epochs: int = 50, max_weight: float = 10.0) -> float:
    """
    Compute KL weight with linear annealing.

    Starts at 0 and linearly increases to max_weight over warmup_epochs.
    This prevents posterior collapse by letting the model first learn
    good reconstructions before enforcing the latent structure.

    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of epochs to anneal over
        max_weight: Maximum KL weight after warmup

    Returns:
        KL weight for the current epoch
    """
    if epoch < warmup_epochs:
        return max_weight * (epoch / warmup_epochs)
    return max_weight


def compute_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 10.0,
) -> Dict[str, torch.Tensor]:
    """Compute ACT loss"""
    # Reconstruction loss
    recon_loss = F.l1_loss(pred_actions, target_actions)

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    return {
        "total": total_loss,
        "recon": recon_loss,
        "kl": kl_loss,
    }


def train(
    data_dir: str,
    save_dir: str = "checkpoints",
    epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-4,
    chunk_size: int = 50,
    include_object_obs: bool = False,
    use_images: bool = False,
    camera_names: list = None,
    freeze_vision_backbone: bool = False,
    device: str = None,
):
    """Train ACT model.

    Args:
        data_dir: Path to demonstration data directory
        save_dir: Path to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        chunk_size: Action chunk size
        include_object_obs: Include object observations (cloth corners, etc.)
        use_images: Enable vision-based training with camera images
        camera_names: List of camera names to use (default: ["bimanual_view"])
        freeze_vision_backbone: Freeze ResNet backbone weights
        device: Training device (cuda/cpu)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if camera_names is None:
        camera_names = ["bimanual_view"]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device}")
    print(f"Data: {data_dir}")
    print(f"Saving to: {save_path}")
    print(f"Use images: {use_images}")
    if use_images:
        print(f"Cameras: {camera_names}")

    # Create dataloaders
    train_loader, val_loader, stats = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        chunk_size=chunk_size,
        include_object_obs=include_object_obs,
        use_images=use_images,
        camera_names=camera_names,
    )

    # Infer dimensions from data
    sample = next(iter(train_loader))
    obs_dim = sample["obs"].shape[-1]
    action_dim = sample["action"].shape[-1]

    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    # Create model
    config = ACTConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        use_images=use_images,
        num_cameras=len(camera_names) if use_images else 1,
        freeze_vision_backbone=freeze_vision_backbone,
    )
    model = ACTModel(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Compute annealed KL weight
        current_kl_weight = get_kl_weight(
            epoch, config.kl_warmup_epochs, config.kl_weight
        )

        # Train
        model.train()
        train_losses = []
        train_recon_losses = []
        train_kl_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            obs = batch["obs"].to(device)
            actions = batch["action"].to(device)

            # Get images if using vision
            images = None
            if use_images:
                images = batch["images"].to(device)

            optimizer.zero_grad()
            pred, mu, logvar = model(obs, actions, images)
            losses = compute_loss(pred, actions, mu, logvar, current_kl_weight)

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(losses["total"].item())
            train_recon_losses.append(losses["recon"].item())
            train_kl_losses.append(losses["kl"].item())
            pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                actions = batch["action"].to(device)

                # Get images if using vision
                images = None
                if use_images:
                    images = batch["images"].to(device)

                pred, mu, logvar = model(obs, actions, images)
                losses = compute_loss(pred, actions, mu, logvar, current_kl_weight)
                val_losses.append(losses["total"].item())

        train_loss = np.mean(train_losses)
        train_recon = np.mean(train_recon_losses)
        train_kl = np.mean(train_kl_losses)
        val_loss = np.mean(val_losses)

        print(
            f"  Train: {train_loss:.4f} (recon: {train_recon:.4f}, kl: {train_kl:.4f}) "
            f"| Val: {val_loss:.4f} | KL_w: {current_kl_weight:.2f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "stats": stats,
                    "val_loss": val_loss,
                },
                save_path / "best_model.pt",
            )
            print(f"  ✓ Saved best model")

        # Checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                save_path / f"checkpoint_{epoch + 1}.pt",
            )

    # Save final
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "stats": stats,
        },
        save_path / "final_model.pt",
    )

    print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")


def load_policy(
    checkpoint_path: str, device: str = "cpu"
) -> tuple[ACTModel, Optional[Dict]]:
    """Load trained policy"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = ACTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint.get("stats")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ACT model for bimanual manipulation"
    )
    parser.add_argument("--data_dir", type=str, default="data/bimanual")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--include_object_obs", action="store_true")

    # Vision arguments
    parser.add_argument(
        "--use_images",
        action="store_true",
        help="Enable vision-based training with camera images",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["bimanual_view"],
        help="Camera names to use for vision (default: bimanual_view)",
    )
    parser.add_argument(
        "--freeze_vision_backbone",
        action="store_true",
        help="Freeze ResNet backbone weights (faster training, may limit performance)",
    )

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_size=args.chunk_size,
        include_object_obs=args.include_object_obs,
        use_images=args.use_images,
        camera_names=args.camera_names,
        freeze_vision_backbone=args.freeze_vision_backbone,
    )
