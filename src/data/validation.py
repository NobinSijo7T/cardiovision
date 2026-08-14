"""
Dataset validation and preprocessing reports for PTB-XL.

The checks in this module run before model training so every downstream sample has
an auditable metadata row, a valid label, a readable ECG record, and exactly one
lossless scalogram image.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.data.label_mapper import CLASS_TO_IDX, get_class_weights


@dataclass(frozen=True)
class ImageIntegrity:
    """Integrity information for a scalogram PNG."""

    exists: bool
    valid: bool
    checksum: Optional[str]
    width: Optional[int]
    height: Optional[int]
    mode: Optional[str]
    dynamic_range: Optional[int]
    nearly_empty: bool
    reason: str = ""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_files_for_row(dataset_root: Path, row: pd.Series) -> List[Path]:
    """Return expected WFDB files for a PTB-XL metadata row."""
    filenames: List[Path] = []
    for col in ("filename_lr", "filename_hr"):
        if col in row and pd.notna(row[col]):
            base = dataset_root / str(row[col])
            filenames.extend([base.with_suffix(".hea"), base.with_suffix(".dat")])
    return filenames


def validate_metadata(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate required metadata columns, duplicates, patients, and labels."""
    required = {"patient_id", "scp_codes", "filename_lr", "filename_hr"}
    if "label" in df.columns:
        required.add("label")
    if "label_name" in df.columns:
        required.add("label_name")

    rows: List[Dict[str, Any]] = []
    missing_columns = sorted(required - set(df.columns))
    duplicate_ecg_ids = df.index[df.index.duplicated()].unique().tolist()

    for ecg_id, row in df.iterrows():
        issues: List[str] = []
        if missing_columns:
            issues.append(f"missing_columns={missing_columns}")
        if pd.isna(row.get("patient_id")):
            issues.append("missing_patient_id")
        if "label" in df.columns:
            label = row.get("label")
            if pd.isna(label) or int(label) not in CLASS_TO_IDX.values():
                issues.append(f"invalid_label={label}")
        if "label_name" in df.columns:
            label_name = row.get("label_name")
            if pd.isna(label_name) or label_name not in CLASS_TO_IDX:
                issues.append(f"invalid_label_name={label_name}")
        if pd.isna(row.get("scp_codes")):
            issues.append("missing_scp_codes")
        if ecg_id in duplicate_ecg_ids:
            issues.append("duplicate_ecg_id")
        rows.append({"ecg_id": ecg_id, "valid_metadata": len(issues) == 0, "issues": ";".join(issues)})

    report = pd.DataFrame(rows)
    summary = {
        "total_records": int(len(df)),
        "valid_metadata_records": int(report["valid_metadata"].sum()),
        "invalid_metadata_records": int((~report["valid_metadata"]).sum()),
        "duplicate_ecg_ids": [int(x) if isinstance(x, (np.integer, int)) else x for x in duplicate_ecg_ids],
        "missing_columns": missing_columns,
    }
    return report, summary


def validate_record_files(df: pd.DataFrame, dataset_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Verify expected PTB-XL WFDB files exist for every record."""
    root = Path(dataset_root)
    rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    for ecg_id, row in df.iterrows():
        expected = record_files_for_row(root, row)
        missing = [str(path) for path in expected if not path.exists()]
        rows.append({
            "ecg_id": ecg_id,
            "expected_files": len(expected),
            "missing_files": len(missing),
            "files_exist": len(missing) == 0,
        })
        for path in missing:
            missing_rows.append({"ecg_id": ecg_id, "missing_file": path})

    report = pd.DataFrame(rows)
    missing_df = pd.DataFrame(missing_rows, columns=["ecg_id", "missing_file"])
    summary = {
        "records_checked": int(len(df)),
        "records_with_all_files": int(report["files_exist"].sum()) if not report.empty else 0,
        "records_with_missing_files": int((~report["files_exist"]).sum()) if not report.empty else 0,
        "missing_file_count": int(len(missing_df)),
    }
    return report, missing_df, summary


def validate_scalogram_image(
    path: Path,
    expected_size: Tuple[int, int],
    min_dynamic_range: int = 8,
    min_nonzero_ratio: float = 0.001,
) -> ImageIntegrity:
    """Validate a saved PNG without trusting cache metadata."""
    if not path.exists():
        return ImageIntegrity(False, False, None, None, None, None, None, True, "missing")

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb)
            dynamic_range = int(arr.max()) - int(arr.min())
            nonzero_ratio = float(np.count_nonzero(arr) / arr.size)
            expected_h, expected_w = expected_size
            size_ok = rgb.size == (expected_w, expected_h)
            is_png = path.suffix.lower() == ".png"
            nearly_empty = dynamic_range < min_dynamic_range or nonzero_ratio < min_nonzero_ratio
            valid = is_png and size_ok and not nearly_empty
            reasons = []
            if not is_png:
                reasons.append("not_png")
            if not size_ok:
                reasons.append(f"bad_size={rgb.size}")
            if nearly_empty:
                reasons.append(f"nearly_empty_range={dynamic_range}_nonzero={nonzero_ratio:.6f}")
            return ImageIntegrity(
                True, valid, sha256_file(path), rgb.size[0], rgb.size[1],
                rgb.mode, dynamic_range, nearly_empty, ";".join(reasons),
            )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return ImageIntegrity(True, False, None, None, None, None, None, True, f"corrupt_image={exc}")


def build_class_distribution_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Return class frequencies, recommended weights, and imbalance warnings."""
    counts = df["label_name"].value_counts().reindex(CLASS_TO_IDX.keys(), fill_value=0)
    weights = get_class_weights(df) if len(df) else {}
    warnings = [
        f"{label_name} has extremely few samples ({count})"
        for label_name, count in counts.items()
        if count > 0 and count < 10
    ]
    return {
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_percentages": {
            str(k): float(v / len(df) * 100.0) if len(df) else 0.0
            for k, v in counts.items()
        },
        "recommended_class_weights": {int(k): float(v) for k, v in weights.items()},
        "weighted_random_sampler_ready": True,
        "focal_loss_alpha": {int(k): float(v) for k, v in weights.items()},
        "warnings": warnings,
    }


def write_class_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Write a deterministic class distribution bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = df["label_name"].value_counts().reindex(CLASS_TO_IDX.keys(), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="#2f6f73")
    ax.set_ylabel("Records")
    ax.set_xlabel("Class")
    ax.set_title("PTB-XL 5-Class Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_patient_distribution(df: pd.DataFrame, output_path: Path, split_name: Optional[str] = None) -> None:
    """Write per-patient sample counts and dominant labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = (
        df.groupby("patient_id")
        .agg(
            num_records=("label", "size"),
            labels=("label_name", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    if split_name is not None:
        grouped.insert(0, "split", split_name)
    grouped.to_csv(output_path, index=False)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
