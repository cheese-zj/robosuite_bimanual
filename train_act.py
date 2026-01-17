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

    obs_dim: int = 46  # 23 per arm × 2 arms
    action_dim: int = 14  # 7 per arm × 2 arms (OSC_POSE)
    hidden_dim: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    chunk_size: int = 50
    latent_dim: int = 32
    dropout: float = 0.1
    kl_weight: float = 10.0


class ACTModel(nn.Module):
    """
    Simplified ACT (Action Chunking with Transformers) model.

    For the full implementation, see: https://github.com/tonyzhaozh/act
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.obs_embed = nn.Linear(config.obs_dim, config.hidden_dim)
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

    def encode(self, obs: torch.Tensor, actions: Optional[torch.Tensor] = None):
        """Encode observation (and actions during training) to latent"""
        B = obs.shape[0]

        obs_emb = self.obs_embed(obs).unsqueeze(1)  # (B, 1, H)

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

    def decode(self, z: torch.Tensor, obs: torch.Tensor):
        """Decode latent to action sequence"""
        B = z.shape[0]

        z_emb = self.latent_proj(z).unsqueeze(1)
        obs_emb = self.obs_embed(obs).unsqueeze(1)
        memory = torch.cat([z_emb, obs_emb], dim=1)

        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, memory)

        return self.action_head(decoded)

    def forward(self, obs: torch.Tensor, actions: Optional[torch.Tensor] = None):
        """Forward pass"""
        mu, logvar = self.encode(obs, actions)

        if self.training and actions is not None:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        pred_actions = self.decode(z, obs)

        return pred_actions, mu, logvar

    def predict(self, obs: torch.Tensor) -> np.ndarray:
        """Inference: predict action chunk"""
        self.eval()
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            pred, _, _ = self(obs)
            return pred[0].cpu().numpy()


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
    device: str = None,
):
    """Train ACT model"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device}")
    print(f"Data: {data_dir}")
    print(f"Saving to: {save_path}")

    # Create dataloaders
    train_loader, val_loader, stats = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        chunk_size=chunk_size,
        include_object_obs=include_object_obs,
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
        # Train
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            obs = batch["obs"].to(device)
            actions = batch["action"].to(device)

            optimizer.zero_grad()
            pred, mu, logvar = model(obs, actions)
            losses = compute_loss(pred, actions, mu, logvar, config.kl_weight)

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(losses["total"].item())
            pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                actions = batch["action"].to(device)
                pred, mu, logvar = model(obs, actions)
                losses = compute_loss(pred, actions, mu, logvar, config.kl_weight)
                val_losses.append(losses["total"].item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f"  Train: {train_loss:.4f} | Val: {val_loss:.4f}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/bimanual")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--include_object_obs", action="store_true")

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_size=args.chunk_size,
        include_object_obs=args.include_object_obs,
    )
