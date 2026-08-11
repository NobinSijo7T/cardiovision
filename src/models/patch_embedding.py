"""
CARDIOVISION - ViT Patch Embedding
Splits 2D scalograms into patches and embeds them.
"""

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger("cardiovision.models.patch_embedding")


class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer for Vision Transformer.

    Divides an image into non-overlapping patches and projects each patch
    into a fixed-dimensional embedding space using a 2D convolution.
    Adds a learnable [CLS] token and positional embeddings.
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        use_cls_token: bool = True,
        use_positional_embedding: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.use_positional_embedding = use_positional_embedding

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2

        # 2D Convolution to extract patches and project them to embed_dim
        # kernel_size=patch_size and stride=patch_size ensures non-overlapping patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # [CLS] token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.seq_len = self.num_patches + 1
        else:
            self.seq_len = self.num_patches

        # Positional Embedding
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(torch.randn(1, self.seq_len, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Embedded token sequence of shape (B, seq_len, embed_dim).
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, \
            f"Input image size ({H}x{W}) doesn't match expected ({self.image_size}x{self.image_size})"

        # x shape: (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.projection(x)

        # Flatten spatial dimensions: (B, embed_dim, num_patches)
        x = x.flatten(2)

        # Transpose to sequence format: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        # Prepend [CLS] token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embeddings
        if self.use_positional_embedding:
            x = x + self.positional_embedding

        return x
