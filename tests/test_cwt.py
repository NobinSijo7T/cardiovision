"""
CARDIOVISION - CWT Tests
"""

import numpy as np
import pytest

from src.preprocessing.cwt_transform import generate_composite_scalogram

def test_generate_composite_scalogram():
    # 10 seconds at 100Hz = 1000 samples, 12 leads
    signal = np.random.randn(1000, 12)
    
    img = generate_composite_scalogram(
        signal,
        sampling_rate=100,
        wavelet='morl',
        scales_start=1,
        scales_end=32, # small for fast test
        image_size=(224, 224),
        layout=(3, 4)
    )
    
    # Output should be PyTorch format (3, H, W) and normalized
    assert img.shape == (3, 224, 224)
    assert np.min(img) >= 0.0
    assert np.max(img) <= 1.0
