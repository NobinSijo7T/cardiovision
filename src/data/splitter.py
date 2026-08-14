"""
CARDIOVISION - Dataset Splitter
Patient-level stratified train/validation/test split.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("cardiovision.data.splitter")


def _dominant_patient_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ECG records to one deterministic stratification label per patient."""
    rows = []
    for patient_id, group in df.groupby("patient_id", sort=True):
        counts = group["label"].value_counts()
        dominant_label = int(counts.sort_index().idxmax())
        rows.append({
            "patient_id": patient_id,
            "dominant_label": dominant_label,
            "num_records": len(group),
        })
    return pd.DataFrame(rows)


def _assign_patients_for_ratio(
    patient_df: pd.DataFrame,
    ratio: float,
    rng: np.random.Generator,
) -> set:
    """Select patients class-by-class to approximate a stratified record ratio."""
    selected = set()
    for _, group in patient_df.groupby("dominant_label", sort=True):
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        target_records = max(1, int(round(shuffled["num_records"].sum() * ratio)))
        running = 0
        for row in shuffled.itertuples(index=False):
            if running >= target_records and len(selected) > 0:
                break
            selected.add(row.patient_id)
            running += int(row.num_records)
    return selected


def _assert_disjoint_patient_sets(splits: Iterable[Tuple[str, pd.DataFrame]]) -> None:
    """Raise if any patient appears in more than one split."""
    seen = {}
    for split_name, split_df in splits:
        for patient_id in split_df["patient_id"].unique():
            if patient_id in seen:
                raise AssertionError(f"Patient leakage: patient {patient_id} in {seen[patient_id]} and {split_name}")
            seen[patient_id] = split_name


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

    rng = np.random.default_rng(random_seed)
    patient_df = _dominant_patient_labels(df)

    test_patients = _assign_patients_for_ratio(patient_df, test_ratio, rng)
    remaining_patients = patient_df[~patient_df["patient_id"].isin(test_patients)]
    val_ratio_remaining = val_ratio / (train_ratio + val_ratio)
    val_patients = _assign_patients_for_ratio(remaining_patients, val_ratio_remaining, rng)
    train_patients = set(remaining_patients["patient_id"]) - val_patients

    df_train = df[df["patient_id"].isin(train_patients)].copy()
    df_val = df[df["patient_id"].isin(val_patients)].copy()
    df_test = df[df["patient_id"].isin(test_patients)].copy()

    # Verify no patient leakage
    train_patients = set(df_train["patient_id"].unique())
    val_patients = set(df_val["patient_id"].unique())
    test_patients = set(df_test["patient_id"].unique())
    _assert_disjoint_patient_sets([("train", df_train), ("val", df_val), ("test", df_test)])
    assert len(df_train) + len(df_val) + len(df_test) == len(df), "Split record count mismatch"

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
            "label_distribution": split_df['label'].value_counts().sort_index().to_dict(),
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

    stats = {
        split_name: {
            "records": int(len(split_df)),
            "patients": int(split_df["patient_id"].nunique()),
            "class_distribution": {
                str(k): int(v)
                for k, v in split_df["label_name"].value_counts().sort_index().items()
            },
        }
        for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]
    }
    with open(output_path / "split_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)


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
