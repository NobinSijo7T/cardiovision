"""
CARDIOVISION - Signal Quality Checker
Detects missing, corrupted, NaN, infinite, or unusable ECG recordings.
"""

from typing import Tuple, List

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("cardiovision.preprocessing.quality")


def check_signal_quality(
    signal: np.ndarray,
    max_nan_ratio: float = 0.01,
    max_flat_ratio: float = 0.50,
    amplitude_min_mv: float = -10.0,
    amplitude_max_mv: float = 10.0,
) -> Tuple[bool, List[str]]:
    """
    Check ECG signal quality across all leads.

    Args:
        signal: ECG signal array of shape (num_samples, num_leads).
        max_nan_ratio: Maximum allowed ratio of NaN values per lead.
        max_flat_ratio: Maximum allowed ratio of identical consecutive values.
        amplitude_min_mv: Minimum acceptable amplitude in mV.
        amplitude_max_mv: Maximum acceptable amplitude in mV.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues = []

    if signal is None:
        return False, ["Signal is None"]

    # Check dimensions
    if signal.ndim != 2:
        issues.append(f"Expected 2D signal, got {signal.ndim}D")
        return False, issues

    num_samples, num_leads = signal.shape

    if num_leads == 0:
        issues.append("No leads found")
        return False, issues

    if num_samples < 10:
        issues.append(f"Signal too short: {num_samples} samples")
        return False, issues

    # Check for NaN values
    nan_count = np.isnan(signal).sum()
    nan_ratio = nan_count / signal.size
    if nan_ratio > max_nan_ratio:
        issues.append(f"NaN ratio {nan_ratio:.4f} exceeds threshold {max_nan_ratio}")

    # Check for infinite values
    inf_count = np.isinf(signal).sum()
    if inf_count > 0:
        issues.append(f"Found {inf_count} infinite values")

    # Check for missing leads (all-zero leads)
    for lead_idx in range(num_leads):
        lead_signal = signal[:, lead_idx]

        # All zeros
        if np.all(lead_signal == 0):
            issues.append(f"Lead {lead_idx} is all zeros")
            continue

        # Flat-line detection (consecutive identical values)
        if len(lead_signal) > 1:
            diffs = np.diff(lead_signal)
            flat_ratio = np.sum(diffs == 0) / len(diffs)
            if flat_ratio > max_flat_ratio:
                issues.append(f"Lead {lead_idx} flat ratio {flat_ratio:.2f} exceeds {max_flat_ratio}")

    # Check amplitude range (using non-NaN values)
    valid_signal = signal[~np.isnan(signal) & ~np.isinf(signal)]
    if len(valid_signal) > 0:
        sig_min = np.min(valid_signal)
        sig_max = np.max(valid_signal)
        if sig_min < amplitude_min_mv or sig_max > amplitude_max_mv:
            issues.append(
                f"Amplitude out of range: [{sig_min:.2f}, {sig_max:.2f}] "
                f"vs [{amplitude_min_mv}, {amplitude_max_mv}]"
            )

    is_valid = len(issues) == 0
    return is_valid, issues


def repair_signal(signal: np.ndarray) -> np.ndarray:
    """
    Attempt to repair minor signal issues (NaN interpolation).

    Args:
        signal: ECG signal array of shape (num_samples, num_leads).

    Returns:
        Repaired signal array.
    """
    repaired = signal.copy()

    for lead_idx in range(repaired.shape[1]):
        lead = repaired[:, lead_idx]

        # Replace NaN with linear interpolation
        nan_mask = np.isnan(lead)
        if nan_mask.any() and not nan_mask.all():
            valid_indices = np.where(~nan_mask)[0]
            nan_indices = np.where(nan_mask)[0]
            lead[nan_indices] = np.interp(nan_indices, valid_indices, lead[valid_indices])

        # Replace Inf with neighboring values
        inf_mask = np.isinf(lead)
        if inf_mask.any():
            valid_indices = np.where(~inf_mask)[0]
            if len(valid_indices) > 0:
                inf_indices = np.where(inf_mask)[0]
                lead[inf_indices] = np.interp(inf_indices, valid_indices, lead[valid_indices])

        repaired[:, lead_idx] = lead

    return repaired
