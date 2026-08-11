"""
CARDIOVISION - Pan-Tompkins R-Peak Detection Algorithm
Full implementation of the Pan-Tompkins QRS detection algorithm.

Stages:
    1. Bandpass filtering (5-15 Hz)
    2. Differentiation
    3. Squaring
    4. Moving-window integration
    5. Adaptive thresholding
    6. R-peak localization
"""

from typing import Tuple, List, Optional

import numpy as np
from scipy.signal import butter, filtfilt

from src.utils.logger import get_logger

logger = get_logger("cardiovision.preprocessing.pan_tompkins")


def _bandpass_filter(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Pan-Tompkins bandpass filter (5-15 Hz).
    Isolates QRS complex frequency content.
    """
    nyquist = fs / 2.0
    low = 5.0 / nyquist
    high = min(15.0 / nyquist, 0.99)
    b, a = butter(2, [low, high], btype='bandpass')
    padlen = min(3 * max(len(b), len(a)), len(signal) - 1)
    return filtfilt(b, a, signal, padlen=padlen)


def _differentiate(signal: np.ndarray) -> np.ndarray:
    """
    Five-point derivative as specified by Pan-Tompkins.
    Approximates: y[n] = (1/8T)(-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])
    """
    diff = np.zeros_like(signal)
    for i in range(2, len(signal) - 2):
        diff[i] = (-signal[i - 2] - 2 * signal[i - 1] +
                   2 * signal[i + 1] + signal[i + 2]) / 8.0
    return diff


def _squaring(signal: np.ndarray) -> np.ndarray:
    """Point-wise squaring to emphasize large QRS slopes."""
    return signal ** 2


def _moving_window_integration(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Moving-window integration.

    Args:
        signal: Squared signal.
        window_size: Integration window size in samples.
    """
    integrated = np.zeros_like(signal)
    cumsum = np.cumsum(signal)
    cumsum = np.insert(cumsum, 0, 0)

    for i in range(len(signal)):
        start = max(0, i - window_size + 1)
        integrated[i] = (cumsum[i + 1] - cumsum[start]) / window_size

    return integrated


def _adaptive_threshold(
    integrated: np.ndarray,
    filtered: np.ndarray,
    fs: int,
) -> List[int]:
    """
    Adaptive thresholding with search-back for R-peak localization.

    Uses dual thresholds on both the integrated and bandpass-filtered signals.

    Args:
        integrated: Moving-window integrated signal.
        filtered: Bandpass filtered signal.
        fs: Sampling frequency.

    Returns:
        List of detected R-peak indices.
    """
    # Initialize thresholds
    # Signal peaks
    spki = np.max(integrated[:2 * fs]) * 0.25  # signal peak running estimate
    npki = np.mean(integrated[:2 * fs]) * 0.5  # noise peak running estimate
    threshold_i1 = npki + 0.25 * (spki - npki)
    threshold_i2 = 0.5 * threshold_i1

    # Filtered signal peaks
    spkf = np.max(filtered[:2 * fs]) * 0.25
    npkf = np.mean(filtered[:2 * fs]) * 0.5
    threshold_f1 = npkf + 0.25 * (spkf - npkf)
    threshold_f2 = 0.5 * threshold_f1

    # Minimum distance between R-peaks (refractory period: 200ms)
    refractory_samples = int(0.2 * fs)

    # RR interval tracking
    rr_intervals: List[int] = []
    rr_average = int(0.8 * fs)  # Initial RR estimate (75 bpm)
    rr_missed_limit = int(1.66 * rr_average)

    # Find all peaks in integrated signal
    peaks = _find_local_peaks(integrated, min_distance=refractory_samples)

    r_peaks: List[int] = []

    for peak_idx in peaks:
        peak_val_i = integrated[peak_idx]
        peak_val_f = abs(filtered[peak_idx])

        is_qrs = False

        # Check if peak exceeds thresholds
        if peak_val_i > threshold_i1 and peak_val_f > threshold_f1:
            is_qrs = True

            # Check RR interval
            if len(r_peaks) > 0:
                rr = peak_idx - r_peaks[-1]
                if rr < refractory_samples:
                    is_qrs = False
                else:
                    rr_intervals.append(rr)
                    if len(rr_intervals) > 8:
                        rr_intervals = rr_intervals[-8:]
                    rr_average = int(np.mean(rr_intervals))
                    rr_missed_limit = int(1.66 * rr_average)

        if is_qrs:
            # Refine peak location in original filtered signal
            # Search in a window around the detected peak
            search_start = max(0, peak_idx - int(0.075 * fs))
            search_end = min(len(filtered), peak_idx + int(0.075 * fs))
            window = abs(filtered[search_start:search_end])
            if len(window) > 0:
                refined_idx = search_start + np.argmax(window)
                r_peaks.append(refined_idx)
            else:
                r_peaks.append(peak_idx)

            # Update signal peak estimates
            spki = 0.125 * peak_val_i + 0.875 * spki
            spkf = 0.125 * peak_val_f + 0.875 * spkf
        else:
            # Update noise peak estimates
            npki = 0.125 * peak_val_i + 0.875 * npki
            npkf = 0.125 * peak_val_f + 0.875 * npkf

        # Update thresholds
        threshold_i1 = npki + 0.25 * (spki - npki)
        threshold_i2 = 0.5 * threshold_i1
        threshold_f1 = npkf + 0.25 * (spkf - npkf)
        threshold_f2 = 0.5 * threshold_f1

    # Search-back for missed peaks
    if len(r_peaks) > 1:
        r_peaks = _searchback(
            r_peaks, integrated, filtered, fs,
            threshold_i2, threshold_f2, rr_average
        )

    return sorted(set(r_peaks))


def _find_local_peaks(signal: np.ndarray, min_distance: int = 10) -> List[int]:
    """Find local maxima in signal with minimum distance constraint."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            if len(peaks) == 0 or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
            elif signal[i] > signal[peaks[-1]]:
                peaks[-1] = i
    return peaks


def _searchback(
    r_peaks: List[int],
    integrated: np.ndarray,
    filtered: np.ndarray,
    fs: int,
    threshold_i2: float,
    threshold_f2: float,
    rr_average: int,
) -> List[int]:
    """
    Search back for missed R-peaks when RR interval is too long.
    Uses lower thresholds (threshold_i2, threshold_f2).
    """
    augmented_peaks = list(r_peaks)
    rr_missed_limit = int(1.66 * rr_average)

    i = 1
    while i < len(augmented_peaks):
        rr = augmented_peaks[i] - augmented_peaks[i - 1]
        if rr > rr_missed_limit:
            # Search for peaks between the two detected peaks using lower threshold
            search_start = augmented_peaks[i - 1] + int(0.2 * fs)
            search_end = augmented_peaks[i] - int(0.2 * fs)

            if search_start < search_end:
                segment = integrated[search_start:search_end]
                candidates = _find_local_peaks(segment, min_distance=int(0.2 * fs))

                for c in candidates:
                    idx = search_start + c
                    if integrated[idx] > threshold_i2 and abs(filtered[idx]) > threshold_f2:
                        augmented_peaks.insert(i, idx)
                        break

        i += 1

    return augmented_peaks


def detect_r_peaks(
    signal: np.ndarray,
    fs: int,
    primary_lead: int = 1,
    fallback_leads: Optional[List[int]] = None,
    integration_window_ms: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect R-peaks using the Pan-Tompkins algorithm.

    Args:
        signal: ECG signal array of shape (num_samples, num_leads).
        fs: Sampling frequency in Hz.
        primary_lead: Primary lead index for R-peak detection (default: Lead II = 1).
        fallback_leads: List of fallback lead indices if primary fails.
        integration_window_ms: Integration window duration in milliseconds.

    Returns:
        Tuple of:
            - r_peak_indices: Array of R-peak sample indices.
            - rr_intervals: Array of RR intervals in samples.
    """
    if fallback_leads is None:
        fallback_leads = [0, 5, 6]  # I, AVF, V1

    # Determine integration window size in samples
    window_size = int(integration_window_ms * fs / 1000)
    window_size = max(window_size, 3)  # minimum 3 samples

    # Try primary lead first, then fallbacks
    leads_to_try = [primary_lead] + [l for l in fallback_leads if l != primary_lead]

    for lead_idx in leads_to_try:
        if lead_idx >= signal.shape[1]:
            continue

        lead_signal = signal[:, lead_idx].astype(np.float64)

        # Skip if lead is all zeros or has issues
        if np.all(lead_signal == 0) or np.any(np.isnan(lead_signal)):
            continue

        try:
            # Stage 1: Bandpass filtering (5-15 Hz)
            bp_filtered = _bandpass_filter(lead_signal, fs)

            # Stage 2: Differentiation
            differentiated = _differentiate(bp_filtered)

            # Stage 3: Squaring
            squared = _squaring(differentiated)

            # Stage 4: Moving-window integration
            integrated = _moving_window_integration(squared, window_size)

            # Stage 5 & 6: Adaptive thresholding & R-peak localization
            r_peaks = _adaptive_threshold(integrated, bp_filtered, fs)

            if len(r_peaks) >= 2:
                r_peak_indices = np.array(r_peaks)
                rr_intervals = np.diff(r_peak_indices)
                logger.debug(
                    f"Lead {lead_idx}: detected {len(r_peaks)} R-peaks, "
                    f"mean HR: {60 * fs / np.mean(rr_intervals):.0f} bpm"
                )
                return r_peak_indices, rr_intervals

        except Exception as e:
            logger.warning(f"Pan-Tompkins failed on lead {lead_idx}: {e}")
            continue

    # If all leads fail, return empty arrays
    logger.warning("R-peak detection failed on all leads")
    return np.array([]), np.array([])


def compute_heart_rate(rr_intervals: np.ndarray, fs: int) -> float:
    """
    Compute average heart rate from RR intervals.

    Args:
        rr_intervals: Array of RR intervals in samples.
        fs: Sampling frequency.

    Returns:
        Average heart rate in beats per minute.
    """
    if len(rr_intervals) == 0:
        return 0.0
    mean_rr_seconds = np.mean(rr_intervals) / fs
    return 60.0 / mean_rr_seconds
