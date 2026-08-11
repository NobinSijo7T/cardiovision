"""
CARDIOVISION - Model Tests
"""

import torch
import pytest

from src.models.vit import CardioViT

def test_vit_forward():
    model = CardioViT(
        image_size=64, # small for test
        patch_size=16,
        in_channels=3,
        num_classes=5,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128
    )
    
    # Batch size 2, 3 channels, 64x64
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    
    assert out.shape == (2, 5) # (batch_size, num_classes)
    
def test_vit_no_cls_token():
    model = CardioViT(
        image_size=64,
        patch_size=16,
        use_cls_token=False, # Test global average pooling path
        num_classes=5,
        embed_dim=64,
        num_layers=2
    )
    
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    
    assert out.shape == (2, 5)
