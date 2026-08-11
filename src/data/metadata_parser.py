"""
CARDIOVISION - Metadata Parser
Loads and processes PTB-XL metadata files (ptbxl_database.csv, scp_statements.csv).
"""

import ast
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("cardiovision.data.metadata")


def load_ptbxl_database(dataset_root: str, metadata_file: str = "ptbxl_database.csv") -> pd.DataFrame:
    """
    Load the PTB-XL database CSV and parse the scp_codes column.

    Args:
        dataset_root: Path to the PTB-XL dataset root directory.
        metadata_file: Name of the metadata CSV file.

    Returns:
        DataFrame with parsed scp_codes (as Python dicts).
    """
    csv_path = Path(dataset_root) / metadata_file
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    logger.info(f"Loading PTB-XL database from {csv_path}")
    df = pd.read_csv(csv_path, index_col='ecg_id')

    # Parse scp_codes from string representation to dict
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))

    logger.info(f"Loaded {len(df)} ECG records")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Unique patients: {df['patient_id'].nunique()}")

    return df


def load_scp_statements(dataset_root: str, scp_file: str = "scp_statements.csv") -> pd.DataFrame:
    """
    Load the SCP statements file with diagnostic code descriptions.

    Args:
        dataset_root: Path to the PTB-XL dataset root directory.
        scp_file: Name of the SCP statements CSV file.

    Returns:
        DataFrame indexed by SCP code with diagnostic metadata.
    """
    csv_path = Path(dataset_root) / scp_file
    if not csv_path.exists():
        raise FileNotFoundError(f"SCP statements file not found: {csv_path}")

    logger.info(f"Loading SCP statements from {csv_path}")
    scp_df = pd.read_csv(csv_path, index_col=0)

    logger.info(f"Total SCP codes: {len(scp_df)}")
    logger.info(f"Diagnostic codes: {scp_df['diagnostic'].sum():.0f}")
    logger.info(f"Form codes: {scp_df['form'].sum():.0f}")
    logger.info(f"Rhythm codes: {scp_df['rhythm'].sum():.0f}")
    logger.info(f"Diagnostic superclasses: {scp_df['diagnostic_class'].dropna().unique().tolist()}")

    return scp_df


def get_diagnostic_codes(scp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter SCP statements to diagnostic-only codes.

    Args:
        scp_df: Full SCP statements DataFrame.

    Returns:
        DataFrame containing only rows where diagnostic == 1.
    """
    diag_df = scp_df[scp_df['diagnostic'] == 1.0].copy()
    logger.info(f"Filtered to {len(diag_df)} diagnostic SCP codes")
    return diag_df


def get_rhythm_codes(scp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter SCP statements to rhythm-only codes.

    Args:
        scp_df: Full SCP statements DataFrame.

    Returns:
        DataFrame containing only rows where rhythm == 1.
    """
    rhythm_df = scp_df[scp_df['rhythm'] == 1.0].copy()
    logger.info(f"Filtered to {len(rhythm_df)} rhythm SCP codes")
    return rhythm_df


def load_all_metadata(dataset_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all metadata and return structured components.

    Args:
        dataset_root: Path to the PTB-XL dataset root directory.

    Returns:
        Tuple of (database_df, scp_df, diagnostic_df, rhythm_df)
    """
    db_df = load_ptbxl_database(dataset_root)
    scp_df = load_scp_statements(dataset_root)
    diag_df = get_diagnostic_codes(scp_df)
    rhythm_df = get_rhythm_codes(scp_df)

    return db_df, scp_df, diag_df, rhythm_df
