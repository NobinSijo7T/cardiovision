"""
CARDIOVISION - Continuous Wavelet Transform
Converts 1D ECG signals into 2D time-frequency scalogram representations.
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pywt
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.utils.logger import get_logger

logger = get_logger("cardiovision.preprocessing.cwt")

# Standard 12-lead ECG names
LEAD_NAMES = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def compute_cwt(
    signal_1d: np.ndarray,
    sampling_rate: int,
    wavelet: str = "morl",
    scales_start: int = 1,
    scales_end: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Continuous Wavelet Transform of a 1D signal.

    Args:
        signal_1d: 1D signal array.
        sampling_rate: Signal sampling rate in Hz.
        wavelet: Wavelet name (default: Morlet 'morl').
        scales_start: Starting scale value.
        scales_end: Ending scale value.

    Returns:
        Tuple of (coefficients, frequencies).
        coefficients: 2D array of shape (num_scales, num_samples).
        frequencies: 1D array of pseudo-frequencies for each scale.
    """
    scales = np.arange(scales_start, scales_end)
    coefficients, frequencies = pywt.cwt(signal_1d, scales, wavelet, 1.0 / sampling_rate)
    return np.abs(coefficients), frequencies


def scalogram_to_image(
    coefficients: np.ndarray,
    image_size: Tuple[int, int] = (224, 224),
    colormap: str = "jet",
) -> np.ndarray:
    """
    Convert CWT coefficients to a colored scalogram image.

    Args:
        coefficients: 2D array of CWT coefficients (scales × time).
        image_size: Output image size (H, W).
        colormap: Matplotlib colormap name.

    Returns:
        RGB image array of shape (H, W, 3) with values in [0, 255].
    """
    # Normalize coefficients to [0, 1]
    c_min, c_max = coefficients.min(), coefficients.max()
    if c_max - c_min > 1e-10:
        normalized = (coefficients - c_min) / (c_max - c_min)
    else:
        normalized = np.zeros_like(coefficients)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(normalized)[:, :, :3]  # Drop alpha channel

    # Resize to target size
    img = Image.fromarray((colored * 255).astype(np.uint8))
    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)

    return np.array(img)


def generate_lead_scalogram(
    signal: np.ndarray,
    lead_idx: int,
    sampling_rate: int,
    wavelet: str = "morl",
    scales_start: int = 1,
    scales_end: int = 128,
    image_size: Tuple[int, int] = (224, 224),
    colormap: str = "jet",
) -> np.ndarray:
    """
    Generate a scalogram image for a single ECG lead.

    Args:
        signal: ECG signal array of shape (num_samples, num_leads).
        lead_idx: Index of the lead to process.
        sampling_rate: Sampling rate in Hz.
        wavelet: CWT wavelet name.
        scales_start: Starting scale.
        scales_end: Ending scale.
        image_size: Output image size.
        colormap: Colormap for visualization.

    Returns:
        RGB image array of shape (H, W, 3).
    """
    lead_signal = signal[:, lead_idx].astype(np.float64)
    coefficients, _ = compute_cwt(lead_signal, sampling_rate, wavelet, scales_start, scales_end)
    return scalogram_to_image(coefficients, image_size, colormap)


def generate_composite_scalogram(
    signal: np.ndarray,
    sampling_rate: int,
    wavelet: str = "morl",
    scales_start: int = 1,
    scales_end: int = 128,
    image_size: Tuple[int, int] = (224, 224),
    colormap: str = "jet",
    layout: Tuple[int, int] = (3, 4),
) -> np.ndarray:
    """
    Generate a composite scalogram image from all 12 ECG leads.

    Arranges 12 lead scalograms in a grid layout and resizes to target size.

    Args:
        signal: ECG signal array of shape (num_samples, num_leads).
        sampling_rate: Sampling rate in Hz.
        wavelet: CWT wavelet name.
        scales_start: Starting scale.
        scales_end: Ending scale.
        image_size: Final output image size (H, W).
        colormap: Colormap for visualization.
        layout: Grid layout (rows, cols) for arranging leads.

    Returns:
        RGB image array of shape (3, H, W) normalized to [0, 1] for PyTorch.
    """
    rows, cols = layout
    num_leads = min(signal.shape[1], rows * cols)

    # Calculate per-lead tile size
    tile_h = image_size[0] // rows
    tile_w = image_size[1] // cols

    # Generate scalograms for each lead
    tiles = []
    for lead_idx in range(num_leads):
        lead_signal = signal[:, lead_idx].astype(np.float64)
        coefficients, _ = compute_cwt(lead_signal, sampling_rate, wavelet, scales_start, scales_end)
        tile = scalogram_to_image(coefficients, (tile_h, tile_w), colormap)
        tiles.append(tile)

    # Pad if fewer leads than grid slots
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    # Arrange in grid
    grid_rows = []
    for r in range(rows):
        row_tiles = tiles[r * cols: (r + 1) * cols]
        grid_rows.append(np.concatenate(row_tiles, axis=1))
    composite = np.concatenate(grid_rows, axis=0)

    # Resize to exact target size
    img = Image.fromarray(composite.astype(np.uint8))
    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    result = np.array(img)

    # Convert to (3, H, W) float32 normalized to [0, 1]
    result = result.transpose(2, 0, 1).astype(np.float32) / 255.0

    return result


def save_scalogram(
    image: np.ndarray,
    output_path: str,
    is_chw: bool = True,
) -> None:
    """
    Save a scalogram image to disk.

    Args:
        image: Image array. If is_chw=True, shape is (3, H, W); else (H, W, 3).
        output_path: Path to save the image.
        is_chw: Whether image is in CHW format (PyTorch convention).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_chw:
        img_array = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        if image.max() <= 1.0:
            img_array = (image * 255).astype(np.uint8)
        else:
            img_array = image.astype(np.uint8)

    img = Image.fromarray(img_array)
    img.save(str(output_path))
