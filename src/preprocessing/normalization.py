"""
CARDIOVISION - Min-Max Normalization
Normalizes ECG signals to a specified range.
"""

from typing import Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("cardiovision.preprocessing.normalization")


def min_max_normalize(
    signal: np.ndarray,
    range_min: float = 0.0,
    range_max: float = 1.0,
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
    per_lead: bool = True,
) -> np.ndarray:
    """
    Apply Min-Max normalization to ECG signal.

    Formula: X_norm = (X - X_min) / (X_max - X_min) * (range_max - range_min) + range_min

    Args:
        signal: ECG signal array of shape (num_samples, num_leads).
        range_min: Target minimum value.
        range_max: Target maximum value.
        global_min: Optional pre-computed minimum (for consistent normalization).
        global_max: Optional pre-computed maximum (for consistent normalization).
        per_lead: If True, normalize each lead independently.

    Returns:
        Normalized signal array with same shape.
    """
    normalized = np.zeros_like(signal, dtype=np.float64)
    target_range = range_max - range_min

    if per_lead:
        for lead_idx in range(signal.shape[1]):
            lead = signal[:, lead_idx].astype(np.float64)

            if global_min is not None and global_max is not None:
                s_min, s_max = global_min, global_max
            else:
                # Filter out NaN for min/max computation
                valid = lead[~np.isnan(lead)]
                if len(valid) == 0:
                    normalized[:, lead_idx] = 0.0
                    continue
                s_min, s_max = np.min(valid), np.max(valid)

            denom = s_max - s_min
            if denom < 1e-10:
                # Constant signal, map to midpoint
                normalized[:, lead_idx] = (range_min + range_max) / 2.0
            else:
                normalized[:, lead_idx] = (lead - s_min) / denom * target_range + range_min
    else:
        # Global normalization across all leads
        valid = signal[~np.isnan(signal)]
        if len(valid) == 0:
            return np.full_like(signal, (range_min + range_max) / 2.0)

        if global_min is not None and global_max is not None:
            s_min, s_max = global_min, global_max
        else:
            s_min, s_max = np.min(valid), np.max(valid)

        denom = s_max - s_min
        if denom < 1e-10:
            return np.full_like(signal, (range_min + range_max) / 2.0, dtype=np.float64)
        normalized = (signal.astype(np.float64) - s_min) / denom * target_range + range_min

    return normalized


def compute_normalization_stats(
    signals: list,
) -> Tuple[float, float]:
    """
    Compute global min/max from a collection of signals (training set only).

    Args:
        signals: List of signal arrays, each of shape (num_samples, num_leads).

    Returns:
        Tuple of (global_min, global_max).
    """
    all_min = float('inf')
    all_max = float('-inf')

    for signal in signals:
        valid = signal[~np.isnan(signal) & ~np.isinf(signal)]
        if len(valid) > 0:
            all_min = min(all_min, np.min(valid))
            all_max = max(all_max, np.max(valid))

    logger.info(f"Global normalization stats: min={all_min:.4f}, max={all_max:.4f}")
    return all_min, all_max
