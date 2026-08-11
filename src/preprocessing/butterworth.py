"""
CARDIOVISION - Butterworth Bandpass Filter
Removes baseline wander and high-frequency noise from ECG signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from src.utils.logger import get_logger

logger = get_logger("cardiovision.preprocessing.butterworth")


def design_butterworth_filter(
    sampling_rate: int,
    low_cutoff_hz: float = 0.5,
    high_cutoff_hz: float = 40.0,
    filter_order: int = 4,
) -> tuple:
    """
    Design a Butterworth bandpass filter.

    Args:
        sampling_rate: Signal sampling rate in Hz.
        low_cutoff_hz: Low cutoff frequency in Hz.
        high_cutoff_hz: High cutoff frequency in Hz.
        filter_order: Filter order.

    Returns:
        Tuple of (b, a) filter coefficients.
    """
    nyquist = sampling_rate / 2.0

    # Validate frequencies
    if low_cutoff_hz >= nyquist:
        raise ValueError(
            f"Low cutoff ({low_cutoff_hz} Hz) must be < Nyquist ({nyquist} Hz)"
        )
    if high_cutoff_hz >= nyquist:
        logger.warning(
            f"High cutoff ({high_cutoff_hz} Hz) >= Nyquist ({nyquist} Hz), "
            f"clamping to {nyquist * 0.95:.1f} Hz"
        )
        high_cutoff_hz = nyquist * 0.95

    low_normalized = low_cutoff_hz / nyquist
    high_normalized = high_cutoff_hz / nyquist

    b, a = butter(filter_order, [low_normalized, high_normalized], btype='bandpass')
    return b, a


def apply_butterworth_filter(
    signal: np.ndarray,
    sampling_rate: int,
    low_cutoff_hz: float = 0.5,
    high_cutoff_hz: float = 40.0,
    filter_order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter to all ECG leads.

    Uses scipy.signal.filtfilt for zero-phase filtering (no phase distortion).

    Args:
        signal: ECG signal array of shape (num_samples, num_leads).
        sampling_rate: Sampling rate in Hz.
        low_cutoff_hz: Low cutoff frequency in Hz.
        high_cutoff_hz: High cutoff frequency in Hz.
        filter_order: Filter order.

    Returns:
        Filtered signal array with same shape.
    """
    b, a = design_butterworth_filter(
        sampling_rate, low_cutoff_hz, high_cutoff_hz, filter_order
    )

    filtered = np.zeros_like(signal)

    for lead_idx in range(signal.shape[1]):
        lead_signal = signal[:, lead_idx]

        # Skip leads with NaN/Inf (should have been caught by quality check)
        if np.any(np.isnan(lead_signal)) or np.any(np.isinf(lead_signal)):
            filtered[:, lead_idx] = lead_signal
            continue

        # Apply zero-phase filtering
        try:
            # padlen must be less than signal length
            padlen = min(3 * max(len(b), len(a)), len(lead_signal) - 1)
            filtered[:, lead_idx] = filtfilt(b, a, lead_signal, padlen=padlen)
        except ValueError as e:
            logger.warning(f"Filter failed for lead {lead_idx}: {e}. Using original.")
            filtered[:, lead_idx] = lead_signal

    return filtered
