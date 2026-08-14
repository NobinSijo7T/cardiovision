"""
CARDIOVISION - Continuous Wavelet Transform
Converts 1D ECG signals into 2D time-frequency scalogram representations.
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')

from src.utils.logger import get_logger
from src.data.validation import sha256_file

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
    magnitude = np.abs(coefficients).astype(np.float64)
    energy = np.sqrt(np.mean(np.square(magnitude)))
    if energy > 1e-12:
        magnitude = magnitude / energy
    return magnitude, frequencies


def compute_cwt_torch(
    signal_1d: np.ndarray,
    sampling_rate: int,
    wavelet: str = "morl",
    scales_start: int = 1,
    scales_end: int = 128,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute a deterministic Morlet CWT approximation with PyTorch.

    This is used only for accelerated preprocessing. The CPU PyWavelets path
    remains available as the reference implementation.
    """
    if wavelet != "morl":
        raise ValueError(f"Torch CWT currently supports wavelet='morl', got {wavelet!r}")
    if scales_end <= scales_start:
        raise ValueError("scales_end must be greater than scales_start")

    sig = torch.as_tensor(signal_1d, dtype=torch.float32, device=device).view(1, 1, -1)
    scales = torch.arange(scales_start, scales_end, dtype=torch.float32, device=device)
    max_scale = float(scales[-1].item())
    half_width = min(int(sig.shape[-1] // 2), max(16, int(np.ceil(8 * max_scale))))
    t = torch.arange(-half_width, half_width + 1, dtype=torch.float32, device=device)
    x = t.unsqueeze(0) / scales.unsqueeze(1)
    kernels = torch.cos(5.0 * x) * torch.exp(-0.5 * x * x) / torch.sqrt(scales).unsqueeze(1)
    kernels = kernels - kernels.mean(dim=1, keepdim=True)
    kernels = kernels / (torch.linalg.vector_norm(kernels, dim=1, keepdim=True) + 1e-12)
    kernels = kernels.flip(dims=[1]).unsqueeze(1)

    coeff = F.conv1d(sig, kernels, padding=half_width).abs().squeeze(0)
    coeff = coeff[:, : signal_1d.shape[0]]
    energy = torch.sqrt(torch.mean(coeff.square()))
    if float(energy.detach().cpu()) > 1e-12:
        coeff = coeff / energy
    return coeff.detach().cpu().numpy()


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
    coefficients = np.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
    c_min, c_max = coefficients.min(), coefficients.max()
    if c_max - c_min > 1e-10:
        normalized = (coefficients - c_min) / (c_max - c_min)
    else:
        normalized = np.zeros_like(coefficients)

    # Apply colormap
    cmap = matplotlib.colormaps[colormap]
    colored = cmap(normalized)[:, :, :3]  # Drop alpha channel

    # Resize to target size
    img = Image.fromarray((colored * 255).astype(np.uint8))
    img = img.resize((image_size[1], image_size[0]), Image.Resampling.LANCZOS)

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
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D ECG signal, got shape {signal.shape}")
    if signal.shape[1] < 1:
        raise ValueError("ECG signal has no leads")
    if scales_end <= scales_start:
        raise ValueError("scales_end must be greater than scales_start")

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
    img = img.resize((image_size[1], image_size[0]), Image.Resampling.LANCZOS)
    result = np.array(img)

    # Convert to (3, H, W) float32 normalized to [0, 1]
    result = result.transpose(2, 0, 1).astype(np.float32) / 255.0

    return result


def generate_composite_scalogram_torch(
    signal: np.ndarray,
    sampling_rate: int,
    wavelet: str = "morl",
    scales_start: int = 1,
    scales_end: int = 128,
    image_size: Tuple[int, int] = (224, 224),
    colormap: str = "jet",
    layout: Tuple[int, int] = (3, 4),
    device: str = "cuda",
) -> np.ndarray:
    """Generate a composite scalogram using Torch/CUDA for CWT coefficients."""
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D ECG signal, got shape {signal.shape}")
    if signal.shape[1] < 1:
        raise ValueError("ECG signal has no leads")

    rows, cols = layout
    num_leads = min(signal.shape[1], rows * cols)
    tile_h = image_size[0] // rows
    tile_w = image_size[1] // cols

    tiles = []
    for lead_idx in range(num_leads):
        lead_signal = signal[:, lead_idx].astype(np.float32)
        coefficients = compute_cwt_torch(
            lead_signal,
            sampling_rate,
            wavelet=wavelet,
            scales_start=scales_start,
            scales_end=scales_end,
            device=device,
        )
        tiles.append(scalogram_to_image(coefficients, (tile_h, tile_w), colormap))

    while len(tiles) < rows * cols:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        grid_rows.append(np.concatenate(tiles[r * cols: (r + 1) * cols], axis=1))
    composite = np.concatenate(grid_rows, axis=0)

    img = Image.fromarray(composite.astype(np.uint8))
    img = img.resize((image_size[1], image_size[0]), Image.Resampling.LANCZOS)
    result = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    return result


def scalogram_quality_issues(
    image: np.ndarray,
    min_dynamic_range: float = 0.03,
    min_nonzero_ratio: float = 0.001,
) -> list:
    """
    Return quality issues for an in-memory scalogram image.

    Args:
        image: CHW or HWC RGB scalogram normalized to [0, 1].
        min_dynamic_range: Minimum max-min intensity range.
        min_nonzero_ratio: Minimum fraction of non-zero pixels.
    """
    arr = image
    if arr.ndim != 3:
        return [f"invalid_ndim={arr.ndim}"]
    if arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    issues = []
    if arr.shape[2] != 3:
        issues.append(f"invalid_channels={arr.shape[2]}")
    if float(arr.max() - arr.min()) < min_dynamic_range:
        issues.append("low_dynamic_range")
    if float(np.count_nonzero(arr) / arr.size) < min_nonzero_ratio:
        issues.append("nearly_empty")
    if not np.isfinite(arr).all():
        issues.append("non_finite_pixels")
    return issues


def save_scalogram(
    image: np.ndarray,
    output_path: str,
    is_chw: bool = True,
) -> str:
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

    img = Image.fromarray(img_array, mode="RGB")
    img.save(str(output_path), format="PNG", optimize=False, compress_level=6)
    return sha256_file(output_path)
