"""
CARDIOVISION - Dataset Loader
PyTorch Dataset for lazy-loading PTB-XL ECG records and CWT scalograms.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import wfdb

from src.utils.logger import get_logger
from src.preprocessing.signal_quality import check_signal_quality
from src.preprocessing.butterworth import apply_butterworth_filter
from src.preprocessing.pan_tompkins import detect_r_peaks
from src.preprocessing.normalization import min_max_normalize
from src.preprocessing.cwt_transform import generate_composite_scalogram

logger = get_logger("cardiovision.data.dataset")


class ECGScalogramDataset(Dataset):
    """
    PyTorch Dataset for ECG CWT scalogram images.

    Supports two modes:
    1. Pre-generated: Load saved scalogram images from disk
    2. On-the-fly: Generate scalograms from raw ECG signals during loading
    """

    def __init__(
        self,
        records_df: pd.DataFrame,
        dataset_root: str,
        cwt_image_dir: Optional[str] = None,
        sampling_rate: int = 100,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[object] = None,
        augment: bool = False,
        augmentation_config: Optional[Dict] = None,
        on_the_fly: bool = False,
        cwt_config: Optional[Dict] = None,
        filter_config: Optional[Dict] = None,
    ):
        """
        Args:
            records_df: DataFrame with ecg_id index, 'label', 'filename_lr'/'filename_hr' columns.
            dataset_root: Root path to PTB-XL dataset.
            cwt_image_dir: Directory containing pre-generated scalogram images.
            sampling_rate: ECG sampling rate (100 or 500).
            image_size: Target image size (H, W).
            transform: Optional torchvision transforms.
            augment: Whether to apply data augmentation.
            augmentation_config: Augmentation parameters.
            on_the_fly: If True, generate scalograms on-the-fly from raw ECG.
            cwt_config: CWT transform parameters.
            filter_config: Butterworth filter parameters.
        """
        self.records_df = records_df.reset_index()
        self.dataset_root = Path(dataset_root)
        self.cwt_image_dir = Path(cwt_image_dir) if cwt_image_dir else None
        self.sampling_rate = sampling_rate
        self.image_size = image_size
        self.transform = transform
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        self.on_the_fly = on_the_fly
        self.cwt_config = cwt_config or {}
        self.filter_config = filter_config or {}

        # Use filename based on sampling rate
        self.filename_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'

        logger.info(
            f"Dataset initialized: {len(self.records_df)} samples, "
            f"on_the_fly={on_the_fly}, augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.records_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load a single sample.

        Returns:
            Tuple of (image_tensor [3, H, W], label_int).
        """
        row = self.records_df.iloc[idx]
        label = int(row['label'])
        ecg_id = row['ecg_id'] if 'ecg_id' in row else row.name

        if self.on_the_fly:
            image = self._generate_scalogram(row)
        else:
            image = self._load_pregenerated(ecg_id)

        if image is None:
            # Fallback: return a zero tensor (should be rare after quality filtering)
            image = torch.zeros(3, self.image_size[0], self.image_size[1])
        else:
            # Apply augmentation to signal-level data if needed
            if self.augment:
                image = self._apply_augmentation(image)

            if self.transform:
                image = self.transform(image)
            elif not isinstance(image, torch.Tensor):
                # Convert numpy/PIL to tensor
                if isinstance(image, np.ndarray):
                    if image.ndim == 2:
                        image = np.stack([image] * 3, axis=0)
                    elif image.ndim == 3 and image.shape[2] == 3:
                        image = image.transpose(2, 0, 1)
                    image = torch.from_numpy(image).float()
                elif isinstance(image, Image.Image):
                    image = np.array(image.convert('RGB')).transpose(2, 0, 1)
                    image = torch.from_numpy(image).float() / 255.0

        return image, label

    def _load_pregenerated(self, ecg_id: int) -> Optional[np.ndarray]:
        """Load a pre-generated CWT scalogram image."""
        if self.cwt_image_dir is None:
            raise ValueError("cwt_image_dir must be set when on_the_fly=False")

        img_path = self.cwt_image_dir / f"{ecg_id}.png"
        if not img_path.exists():
            logger.warning(f"Scalogram not found for ecg_id={ecg_id}: {img_path}")
            return None

        img = Image.open(img_path).convert('RGB').resize(
            (self.image_size[1], self.image_size[0]), Image.BILINEAR
        )
        return np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0

    def _generate_scalogram(self, row: pd.Series) -> Optional[np.ndarray]:
        """Generate a CWT scalogram on-the-fly from raw ECG."""
        try:
            filename = row[self.filename_col]
            record_path = str(self.dataset_root / filename)
            signal, meta = wfdb.rdsamp(record_path)

            # Signal quality check
            is_valid, _ = check_signal_quality(signal)
            if not is_valid:
                return None

            # Butterworth filtering
            low = self.filter_config.get('low_cutoff_hz', 0.5)
            high = self.filter_config.get('high_cutoff_hz', 40.0)
            order = self.filter_config.get('filter_order', 4)
            signal = apply_butterworth_filter(signal, self.sampling_rate, low, high, order)

            # Min-Max normalization
            signal = min_max_normalize(signal)

            # Generate composite scalogram
            wavelet = self.cwt_config.get('wavelet', 'morl')
            scales_start = self.cwt_config.get('scales_start', 1)
            scales_end = self.cwt_config.get('scales_end', 128)
            img = generate_composite_scalogram(
                signal, self.sampling_rate, wavelet,
                scales_start, scales_end, self.image_size
            )

            return img

        except Exception as e:
            logger.warning(f"Error generating scalogram for record: {e}")
            return None

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to an image."""
        if not isinstance(image, np.ndarray):
            return image

        # Small amplitude scaling
        scale_range = self.augmentation_config.get('amplitude_scale_range', [0.95, 1.05])
        scale = np.random.uniform(scale_range[0], scale_range[1])
        image = image * scale

        # Low-level Gaussian noise
        noise_std = self.augmentation_config.get('gaussian_noise_std', 0.005)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
            image = image + noise

        # Clip to valid range
        image = np.clip(image, 0.0, 1.0)

        return image


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_root: str,
    cwt_image_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for train, val, and test splits.

    Args:
        train_df: Training records DataFrame.
        val_df: Validation records DataFrame.
        test_df: Test records DataFrame.
        dataset_root: PTB-XL dataset root path.
        cwt_image_dir: Directory with pre-generated scalograms.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        image_size: Image dimensions.
        augmentation_config: Augmentation parameters.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = ECGScalogramDataset(
        records_df=train_df,
        dataset_root=dataset_root,
        cwt_image_dir=cwt_image_dir,
        image_size=image_size,
        augment=True,
        augmentation_config=augmentation_config,
    )
    val_dataset = ECGScalogramDataset(
        records_df=val_df,
        dataset_root=dataset_root,
        cwt_image_dir=cwt_image_dir,
        image_size=image_size,
        augment=False,
    )
    test_dataset = ECGScalogramDataset(
        records_df=test_df,
        dataset_root=dataset_root,
        cwt_image_dir=cwt_image_dir,
        image_size=image_size,
        augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
