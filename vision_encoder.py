"""
Vision encoder module for ACT model.

Provides ResNet-18 based image encoding for visual observations.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from typing import List, Optional


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VisionEncoder(nn.Module):
    """
    Vision encoder using ResNet-18 backbone.

    Encodes camera images into feature vectors that can be combined
    with proprioceptive observations for the ACT model.

    Args:
        hidden_dim: Output feature dimension
        num_cameras: Number of camera views to process
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone weights (for faster training)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_cameras: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_cameras = num_cameras

        # Load ResNet-18 backbone
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)

        # Get the feature dimension before the final FC layer
        backbone_out_dim = self.backbone.fc.in_features  # 512 for ResNet-18

        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Project features to hidden_dim
        # If multiple cameras, we concatenate and project
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out_dim * num_cameras, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Store normalization parameters as buffers (move with model)
        self.register_buffer(
            "img_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "img_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
        )

    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images using ImageNet statistics.

        Args:
            images: (B, C, H, W) or (B, N, C, H, W) tensor with values in [0, 1]

        Returns:
            Normalized images
        """
        if images.dim() == 5:
            # (B, N, C, H, W) -> normalize each camera
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            images = (images - self.img_mean) / self.img_std
            images = images.view(B, N, C, H, W)
        else:
            # (B, C, H, W)
            images = (images - self.img_mean) / self.img_std
        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature vectors.

        Args:
            images: Tensor of shape (B, C, H, W) for single camera
                   or (B, N, C, H, W) for N cameras
                   Values should be in [0, 1] range

        Returns:
            features: Tensor of shape (B, hidden_dim)
        """
        # Normalize images
        images = self.normalize_images(images)

        if images.dim() == 5:
            # Multiple cameras: (B, N, C, H, W)
            B, N, C, H, W = images.shape
            assert N == self.num_cameras, f"Expected {self.num_cameras} cameras, got {N}"

            # Process each camera through backbone
            images = images.view(B * N, C, H, W)
            features = self.backbone(images)  # (B*N, 512)
            features = features.view(B, N * features.shape[-1])  # (B, N*512)
        else:
            # Single camera: (B, C, H, W)
            assert self.num_cameras == 1, "Single image but num_cameras > 1"
            features = self.backbone(images)  # (B, 512)

        # Project to hidden_dim
        features = self.feature_proj(features)  # (B, hidden_dim)

        return features


class MultiCameraEncoder(nn.Module):
    """
    Alternative encoder that processes each camera separately with shared weights,
    then fuses features using attention.

    This can be more flexible when the number of cameras varies.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Single shared backbone for all cameras
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)

        backbone_out_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Per-camera feature projection
        self.camera_proj = nn.Linear(backbone_out_dim, hidden_dim)

        # Cross-camera attention for fusion
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Store normalization parameters
        self.register_buffer(
            "img_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "img_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode multiple camera images with attention-based fusion.

        Args:
            images: (B, N, C, H, W) tensor with N cameras

        Returns:
            features: (B, hidden_dim) fused feature vector
        """
        B, N, C, H, W = images.shape

        # Normalize
        images = (images - self.img_mean) / self.img_std

        # Process all cameras through shared backbone
        images = images.view(B * N, C, H, W)
        features = self.backbone(images)  # (B*N, 512)
        features = self.camera_proj(features)  # (B*N, hidden_dim)
        features = features.view(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        # Self-attention fusion across cameras
        fused, _ = self.fusion_attn(features, features, features)
        fused = self.fusion_norm(fused + features)  # Residual connection

        # Mean pool across cameras
        fused = fused.mean(dim=1)  # (B, hidden_dim)

        return fused


def create_vision_encoder(
    hidden_dim: int = 256,
    num_cameras: int = 1,
    encoder_type: str = "simple",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory function to create vision encoder.

    Args:
        hidden_dim: Output feature dimension
        num_cameras: Number of camera views
        encoder_type: "simple" for VisionEncoder, "attention" for MultiCameraEncoder
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone weights

    Returns:
        Vision encoder module
    """
    if encoder_type == "simple":
        return VisionEncoder(
            hidden_dim=hidden_dim,
            num_cameras=num_cameras,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    elif encoder_type == "attention":
        return MultiCameraEncoder(
            hidden_dim=hidden_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == "__main__":
    # Test the vision encoder
    print("Testing VisionEncoder...")

    # Single camera
    encoder = VisionEncoder(hidden_dim=256, num_cameras=1)
    x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    out = encoder(x)
    print(f"Single camera: input {x.shape} -> output {out.shape}")

    # Multiple cameras
    encoder = VisionEncoder(hidden_dim=256, num_cameras=2)
    x = torch.randn(4, 2, 3, 224, 224)  # Batch of 4, 2 cameras each
    out = encoder(x)
    print(f"Multi camera: input {x.shape} -> output {out.shape}")

    # Attention-based encoder
    print("\nTesting MultiCameraEncoder...")
    encoder = MultiCameraEncoder(hidden_dim=256)
    x = torch.randn(4, 3, 3, 224, 224)  # 3 cameras
    out = encoder(x)
    print(f"Attention encoder: input {x.shape} -> output {out.shape}")

    print("\nAll tests passed!")
