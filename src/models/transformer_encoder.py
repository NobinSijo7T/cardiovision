"""
CARDIOVISION - Transformer Encoder
Multi-head self-attention and MLP blocks for ViT.
"""

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger("cardiovision.models.transformer_encoder")


class MLP(nn.Module):
    """Multi-Layer Perceptron used in Transformer block."""
    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop_prob: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer Encoder block.
    LayerNorm -> MHSA -> Dropout -> Add -> LayerNorm -> MLP -> Add
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()

        # Multi-Head Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        # Feed-Forward Network
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_dim,
            out_features=embed_dim,
            drop_prob=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, seq_len, embed_dim).

        Returns:
            Output tensor of shape (B, seq_len, embed_dim).
        """
        # MHSA path with residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out

        # MLP path with residual
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer blocks.
    """
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, attention_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all blocks.

        Args:
            x: Input tensor of shape (B, seq_len, embed_dim).

        Returns:
            Output tensor of shape (B, seq_len, embed_dim).
        """
        for block in self.blocks:
            x = block(x)
        return x
