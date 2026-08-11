"""
CARDIOVISION - Dataset Splitter
Patient-level stratified train/validation/test split.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.utils.logger import get_logger

logger = get_logger("cardiovision.data.splitter")


def patient_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset at the patient level to prevent data leakage.

    No ECG recordings from the same patient will appear in different splits.

    Args:
        df: Labeled DataFrame with 'patient_id' and 'label' columns.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    logger.info(f"Splitting dataset: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logger.info(f"Total records: {len(df)}, Unique patients: {df['patient_id'].nunique()}")

    # Step 1: Split into train+val vs test
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=random_seed
    )
    train_val_idx, test_idx = next(gss_test.split(df, df['label'], groups=df['patient_id']))
    df_train_val = df.iloc[train_val_idx]
    df_test = df.iloc[test_idx]

    # Step 2: Split train+val into train vs val
    val_fraction_of_train_val = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction_of_train_val,
        random_state=random_seed
    )
    train_idx, val_idx = next(gss_val.split(
        df_train_val, df_train_val['label'], groups=df_train_val['patient_id']
    ))
    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]

    # Verify no patient leakage
    train_patients = set(df_train['patient_id'].unique())
    val_patients = set(df_val['patient_id'].unique())
    test_patients = set(df_test['patient_id'].unique())

    assert len(train_patients & val_patients) == 0, "Patient leakage between train and val!"
    assert len(train_patients & test_patients) == 0, "Patient leakage between train and test!"
    assert len(val_patients & test_patients) == 0, "Patient leakage between val and test!"

    logger.info(f"Train: {len(df_train)} records, {len(train_patients)} patients")
    logger.info(f"Val:   {len(df_val)} records, {len(val_patients)} patients")
    logger.info(f"Test:  {len(df_test)} records, {len(test_patients)} patients")

    # Log class distribution per split
    for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        dist = split_df['label_name'].value_counts().to_dict()
        logger.info(f"  {split_name} class distribution: {dist}")

    return df_train, df_val, df_test


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: str = "data/splits"
) -> None:
    """
    Save split information (ecg_ids, patient_ids) to JSON files.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.
        df_test: Test DataFrame.
        output_dir: Directory to save split files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        split_info = {
            "ecg_ids": split_df.index.tolist(),
            "patient_ids": split_df['patient_id'].unique().tolist(),
            "num_records": len(split_df),
            "num_patients": split_df['patient_id'].nunique(),
            "class_distribution": split_df['label_name'].value_counts().to_dict(),
        }
        filepath = output_path / f"{split_name}_split.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, default=str)
        logger.info(f"Saved {split_name} split info to {filepath}")

    # Also save as CSV for easy loading
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        csv_path = output_path / f"{split_name}_records.csv"
        split_df[['patient_id', 'label', 'label_name', 'filename_lr', 'filename_hr']].to_csv(csv_path)
        logger.info(f"Saved {split_name} records CSV to {csv_path}")


def load_split_ids(splits_dir: str = "data/splits") -> Dict[str, list]:
    """
    Load previously saved split IDs.

    Args:
        splits_dir: Directory containing split JSON files.

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing ecg_id lists.
    """
    splits_path = Path(splits_dir)
    result = {}
    for split_name in ["train", "val", "test"]:
        filepath = splits_path / f"{split_name}_split.json"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                info = json.load(f)
            result[split_name] = info['ecg_ids']
        else:
            logger.warning(f"Split file not found: {filepath}")
    return result
