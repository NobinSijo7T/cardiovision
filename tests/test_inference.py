"""
CARDIOVISION - Inference Tests
"""

import numpy as np
import torch
import pytest

from src.models.vit import CardioViT
from src.inference.predict import CardiovisionPredictor
from src.utils.config import Config

def test_predict_pipeline():
    cfg = Config()
    # Small model for fast test
    model = CardioViT(
        image_size=224,
        patch_size=16,
        num_classes=5,
        embed_dim=64,
        num_layers=1,
        num_heads=2
    )
    
    predictor = CardiovisionPredictor(model, cfg, device="cpu")
    
    # 10s at 100Hz = 1000 samples
    signal = np.random.randn(1000, 12)
    
    # Override scales for faster test
    predictor.cfg.cwt.scales_end = 16
    
    result = predictor.predict_from_signal(signal, sampling_rate=100, generate_explanation=True)
    
    assert result["status"] == "success"
    assert "filtered_signal" in result
    assert "predicted_class_name" in result
    assert "confidence" in result
    assert "gradcam_heatmap" in result
    assert result["gradcam_heatmap"].shape == (224, 224)
