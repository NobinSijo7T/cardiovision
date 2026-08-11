"""
CARDIOVISION - Label Mapper
Maps PTB-XL SCP diagnostic codes to 5 target cardiovascular disease classes.

Target Classes:
    0: Normal
    1: Myocardial Infarction
    2: Arrhythmia
    3: Left Ventricular Hypertrophy
    4: ST/T Wave Abnormalities

Multi-label Resolution:
    Priority-based (MI > Arrhythmia > LVH > STTC > Normal).
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("cardiovision.data.label_mapper")

# ==============================================================================
# SCP Code → Target Class Mapping
# ==============================================================================

# Diagnostic SCP codes (diagnostic == 1 in scp_statements.csv)
SCP_TO_CLASS: Dict[str, str] = {
    # --- Normal ---
    "NORM": "Normal",

    # --- Myocardial Infarction (MI superclass) ---
    "IMI": "Myocardial Infarction",
    "ASMI": "Myocardial Infarction",
    "AMI": "Myocardial Infarction",
    "ALMI": "Myocardial Infarction",
    "LMI": "Myocardial Infarction",
    "ILMI": "Myocardial Infarction",
    "IPMI": "Myocardial Infarction",
    "IPLMI": "Myocardial Infarction",
    "PMI": "Myocardial Infarction",
    "INJAS": "Myocardial Infarction",
    "INJAL": "Myocardial Infarction",
    "INJIN": "Myocardial Infarction",
    "INJLA": "Myocardial Infarction",
    "INJIL": "Myocardial Infarction",

    # --- Arrhythmia (CD superclass - conduction disturbances) ---
    "LAFB": "Arrhythmia",
    "IRBBB": "Arrhythmia",
    "1AVB": "Arrhythmia",
    "IVCD": "Arrhythmia",
    "CRBBB": "Arrhythmia",
    "CLBBB": "Arrhythmia",
    "LPFB": "Arrhythmia",
    "WPW": "Arrhythmia",
    "ILBBB": "Arrhythmia",
    "3AVB": "Arrhythmia",
    "2AVB": "Arrhythmia",

    # --- Left Ventricular Hypertrophy (HYP superclass) ---
    "LVH": "Left Ventricular Hypertrophy",
    "LAO/LAE": "Left Ventricular Hypertrophy",
    "RVH": "Left Ventricular Hypertrophy",
    "RAO/RAE": "Left Ventricular Hypertrophy",
    "SEHYP": "Left Ventricular Hypertrophy",

    # --- ST/T Wave Abnormalities (STTC superclass) ---
    "NDT": "ST/T Wave Abnormalities",
    "NST_": "ST/T Wave Abnormalities",
    "DIG": "ST/T Wave Abnormalities",
    "LNGQT": "ST/T Wave Abnormalities",
    "ISC_": "ST/T Wave Abnormalities",
    "ISCAL": "ST/T Wave Abnormalities",
    "ISCIN": "ST/T Wave Abnormalities",
    "ISCIL": "ST/T Wave Abnormalities",
    "ISCAS": "ST/T Wave Abnormalities",
    "ISCLA": "ST/T Wave Abnormalities",
    "ISCAN": "ST/T Wave Abnormalities",
    "ANEUR": "ST/T Wave Abnormalities",
    "EL": "ST/T Wave Abnormalities",
}

# Rhythm codes mapped to Arrhythmia (rhythm == 1, not diagnostic)
RHYTHM_TO_CLASS: Dict[str, str] = {
    "AFIB": "Arrhythmia",
    "STACH": "Arrhythmia",
    "SARRH": "Arrhythmia",
    "SBRAD": "Arrhythmia",
    "SVARR": "Arrhythmia",
    "AFLT": "Arrhythmia",
    "SVTAC": "Arrhythmia",
    "PSVT": "Arrhythmia",
    "BIGU": "Arrhythmia",
    "TRIGU": "Arrhythmia",
    "PACE": "Arrhythmia",
}

# Class name to integer label
CLASS_TO_IDX: Dict[str, int] = {
    "Normal": 0,
    "Myocardial Infarction": 1,
    "Arrhythmia": 2,
    "Left Ventricular Hypertrophy": 3,
    "ST/T Wave Abnormalities": 4,
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

# Priority order for multi-label resolution (highest priority first)
PRIORITY_ORDER: List[str] = [
    "Myocardial Infarction",
    "Arrhythmia",
    "Left Ventricular Hypertrophy",
    "ST/T Wave Abnormalities",
    "Normal",
]


def map_scp_to_classes(scp_codes: Dict[str, float]) -> List[str]:
    """
    Map a record's SCP codes to target class names.

    Args:
        scp_codes: Dictionary of {scp_code: likelihood} from a PTB-XL record.

    Returns:
        List of unique target class names this record maps to.
    """
    classes = set()
    for code, likelihood in scp_codes.items():
        # Only consider codes with non-zero likelihood
        if code in SCP_TO_CLASS:
            classes.add(SCP_TO_CLASS[code])
        elif code in RHYTHM_TO_CLASS:
            classes.add(RHYTHM_TO_CLASS[code])
    return list(classes)


def resolve_single_label(classes: List[str]) -> Optional[str]:
    """
    Resolve multiple target classes to a single label using priority order.

    Args:
        classes: List of target class names for a record.

    Returns:
        Single target class name, or None if no classes present.
    """
    if not classes:
        return None

    for priority_class in PRIORITY_ORDER:
        if priority_class in classes:
            return priority_class

    return None


def assign_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Assign 5-class labels to the entire PTB-XL dataset.

    Args:
        df: PTB-XL database DataFrame with parsed scp_codes.

    Returns:
        Tuple of:
            - DataFrame with added 'target_classes', 'label_name', 'label' columns
            - Statistics dictionary with mapping report
    """
    logger.info("Mapping SCP codes to 5 target classes...")

    # Map each record's SCP codes to target classes
    df['target_classes'] = df['scp_codes'].apply(map_scp_to_classes)
    df['num_target_classes'] = df['target_classes'].apply(len)

    # Resolve multi-label to single label
    df['label_name'] = df['target_classes'].apply(resolve_single_label)

    # Count records per mapping category
    total_records = len(df)
    no_mapping = df['label_name'].isna().sum()
    single_class = (df['num_target_classes'] == 1).sum()
    multi_class = (df['num_target_classes'] > 1).sum()

    logger.info(f"Total records: {total_records}")
    logger.info(f"Records with single class: {single_class}")
    logger.info(f"Records with multiple classes (resolved by priority): {multi_class}")
    logger.info(f"Records with no diagnostic mapping (excluded): {no_mapping}")

    # Exclude records with no mapping
    df_labeled = df.dropna(subset=['label_name']).copy()
    df_labeled['label'] = df_labeled['label_name'].map(CLASS_TO_IDX)

    # Per-class statistics
    class_counts = df_labeled['label_name'].value_counts().to_dict()
    class_percentages = {
        k: round(v / len(df_labeled) * 100, 2)
        for k, v in class_counts.items()
    }

    logger.info("Class distribution:")
    for cls_name in PRIORITY_ORDER:
        count = class_counts.get(cls_name, 0)
        pct = class_percentages.get(cls_name, 0)
        logger.info(f"  {cls_name}: {count} ({pct}%)")

    # Build statistics report
    stats = {
        "total_records": total_records,
        "usable_records": len(df_labeled),
        "excluded_records": no_mapping,
        "single_class_records": int(single_class),
        "multi_class_records": int(multi_class),
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "class_to_idx": CLASS_TO_IDX,
        "scp_mapping": {
            "diagnostic_codes_mapped": len(SCP_TO_CLASS),
            "rhythm_codes_mapped": len(RHYTHM_TO_CLASS),
        },
        "priority_order": PRIORITY_ORDER,
    }

    return df_labeled, stats


def get_class_weights(df: pd.DataFrame) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights for loss function.

    Args:
        df: Labeled DataFrame with 'label' column.

    Returns:
        Dictionary of {class_idx: weight}.
    """
    label_counts = Counter(df['label'].values)
    total = sum(label_counts.values())
    n_classes = len(CLASS_TO_IDX)
    weights = {}
    for idx in range(n_classes):
        count = label_counts.get(idx, 1)
        weights[idx] = total / (n_classes * count)
    return weights
