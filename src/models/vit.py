"""
CARDIOVISION - Vision Transformer
Custom Vision Transformer trained from scratch (no pre-trained weights).
"""

import torch
import torch.nn as nn

from src.models.patch_embedding import PatchEmbedding
from src.models.transformer_encoder import TransformerEncoder
from src.utils.logger import get_logger

logger = get_logger("cardiovision.models.vit")


class CardioViT(nn.Module):
    """
    Vision Transformer for Cardiovascular Disease Classification from ECG Scalograms.

    Architecture:
    Input Image -> Patch Embedding -> [CLS] Token + Positional Embedding
    -> Transformer Encoder Stack -> LayerNorm -> Linear Classifier Head
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 5,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_cls_token: bool = True,
        use_positional_embedding: bool = True
    ):
        super().__init__()
        self.use_cls_token = use_cls_token

        logger.info(f"Initializing Custom CardioViT: {num_layers} layers, {num_heads} heads, "
                    f"embed_dim={embed_dim}, num_classes={num_classes}")

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_cls_token=use_cls_token,
            use_positional_embedding=use_positional_embedding
        )

        # Transformer Encoder Stack
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout
        )

        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights appropriately for training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features up to the final layer norm.
        Useful for Grad-CAM explainability.
        """
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor (B, C, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        x = self.forward_features(x)

        if self.use_cls_token:
            # Use only the [CLS] token for classification
            x = x[:, 0]
        else:
            # Global average pooling over all patches
            x = x.mean(dim=1)

        logits = self.head(x)
        return logits
