"""
CARDIOVISION - Preprocessing Tests
"""

import numpy as np
import pytest

from src.preprocessing.signal_quality import check_signal_quality
from src.preprocessing.butterworth import apply_butterworth_filter
from src.preprocessing.normalization import min_max_normalize

def test_signal_quality_nan():
    signal = np.zeros((100, 12))
    signal[0, 0] = np.nan
    is_valid, issues = check_signal_quality(signal, max_nan_ratio=0.0)
    assert not is_valid
    assert any("NaN" in iss for iss in issues)

def test_signal_quality_flat():
    signal = np.ones((100, 12)) # Flat signal
    is_valid, issues = check_signal_quality(signal, max_flat_ratio=0.5)
    assert not is_valid
    assert any("flat" in iss for iss in issues)

def test_normalization():
    signal = np.random.randn(100, 12) * 5 # values between approx -15 and 15
    norm_signal = min_max_normalize(signal, range_min=0.0, range_max=1.0)
    
    assert np.all(norm_signal >= 0.0)
    assert np.all(norm_signal <= 1.0)
    
    # Check per-lead scaling
    for i in range(12):
        assert np.isclose(np.min(norm_signal[:, i]), 0.0)
        assert np.isclose(np.max(norm_signal[:, i]), 1.0)
