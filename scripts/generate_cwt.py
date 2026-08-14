"""
CARDIOVISION - Generate CWT Scalograms Script

Batch generates one verified PNG scalogram for every split ECG record. Missing or
invalid images are never skipped silently: each failure is written with a reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Optional

from _bootstrap import ensure_project_venv

ensure_project_venv()

import pandas as pd
import torch
from tqdm import tqdm
import wfdb

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.validation import validate_scalogram_image, write_json
from src.preprocessing.butterworth import apply_butterworth_filter
from src.preprocessing.cwt_transform import (
    generate_composite_scalogram,
    generate_composite_scalogram_torch,
    save_scalogram,
    scalogram_quality_issues,
)
from src.preprocessing.normalization import min_max_normalize
from src.preprocessing.signal_quality import check_signal_quality, clean_ecg_signal
from src.utils.config import load_config
from src.utils.logger import setup_logger


@dataclass
class ProcessResult:
    """One-row outcome for preprocessing an ECG record."""

    ecg_id: int
    filename: str
    status: str
    output_path: str
    checksum: Optional[str] = None
    reason: str = ""


def _record_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _fatal_signal_quality_issues(issues: List[str], num_leads: int) -> List[str]:
    """Return only quality issues that should block scalogram generation."""
    nonfatal_prefixes = ("Amplitude out of range",)
    bad_lead_issues = [
        issue for issue in issues
        if "is all zeros" in issue or "flat ratio" in issue
    ]
    fatal = [
        issue for issue in issues
        if not issue.startswith(nonfatal_prefixes)
        and "is all zeros" not in issue
        and "flat ratio" not in issue
    ]
    if len(bad_lead_issues) >= max(1, num_leads // 2):
        fatal.extend(bad_lead_issues)
    return fatal


def _load_splits(splits_dir: Path) -> pd.DataFrame:
    paths = {
        "train": splits_dir / "train_records.csv",
        "val": splits_dir / "val_records.csv",
        "test": splits_dir / "test_records.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Split records not found: {missing}. Run prepare_dataset.py first.")

    frames = []
    for split_name, path in paths.items():
        df = pd.read_csv(path, index_col=0)
        df["split"] = split_name
        frames.append(df)
    combined = pd.concat(frames, axis=0)
    if combined.index.duplicated().any():
        dupes = combined.index[combined.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate ECG IDs in splits: {dupes[:10]}")
    return combined


def _filter_failed_records(df_all: pd.DataFrame, failed_csv: Path) -> pd.DataFrame:
    """Restrict processing to ECG IDs from the previous failure report."""
    if not failed_csv.exists():
        raise FileNotFoundError(f"Failed ECG report not found: {failed_csv}")
    failed_df = pd.read_csv(failed_csv)
    if failed_df.empty:
        return df_all.iloc[0:0].copy()
    failed_ids = failed_df["ecg_id"].astype(df_all.index.dtype, copy=False).tolist()
    missing_ids = sorted(set(failed_ids) - set(df_all.index))
    if missing_ids:
        raise ValueError(f"Failed ECG IDs are not present in split CSVs: {missing_ids[:10]}")
    return df_all.loc[failed_ids].copy()


def _resolve_cwt_device(device: str) -> Optional[str]:
    """Resolve requested CWT device to a concrete Torch device or CPU path."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
        return "cuda"
    return None


def process_record(
    row: pd.Series,
    cfg: Any,
    dataset_root: Path,
    cwt_dir: Path,
    force: bool = False,
    cwt_device: Optional[str] = None,
) -> ProcessResult:
    """Generate or validate the scalogram for one ECG record."""
    ecg_id = _record_id(row.name)
    filename_col = "filename_lr" if cfg.dataset.sampling_rate == 100 else "filename_hr"
    filename = str(row[filename_col])
    out_path = cwt_dir / f"{ecg_id}.{cfg.cwt.save_format}"
    expected_size = tuple(cfg.cwt.image_size)

    if not force:
        integrity = validate_scalogram_image(out_path, expected_size)
        if integrity.valid:
            return ProcessResult(ecg_id, filename, "cached", str(out_path), integrity.checksum)

    try:
        record_path = dataset_root / filename
        for suffix in (".hea", ".dat"):
            if not record_path.with_suffix(suffix).exists():
                return ProcessResult(ecg_id, filename, "failed", str(out_path), reason=f"missing_wfdb_{suffix}")

        signal, _ = wfdb.rdsamp(str(record_path))
        if signal.shape[1] != 12:
            return ProcessResult(ecg_id, filename, "failed", str(out_path), reason=f"expected_12_leads_got_{signal.shape[1]}")

        cleaned = clean_ecg_signal(
            signal,
            cfg.dataset.sampling_rate,
            amplitude_min_mv=cfg.preprocessing.signal_quality.amplitude_min_mv,
            amplitude_max_mv=cfg.preprocessing.signal_quality.amplitude_max_mv,
        )
        is_valid, issues = check_signal_quality(
            cleaned,
            max_nan_ratio=cfg.preprocessing.signal_quality.max_nan_ratio,
            max_flat_ratio=cfg.preprocessing.signal_quality.max_flat_ratio,
            amplitude_min_mv=cfg.preprocessing.signal_quality.amplitude_min_mv,
            amplitude_max_mv=cfg.preprocessing.signal_quality.amplitude_max_mv,
        )
        fatal_issues = _fatal_signal_quality_issues(issues, cleaned.shape[1])
        if fatal_issues:
            return ProcessResult(ecg_id, filename, "failed", str(out_path), reason=";".join(fatal_issues))

        filtered = apply_butterworth_filter(
            cleaned,
            cfg.dataset.sampling_rate,
            cfg.preprocessing.butterworth.low_cutoff_hz,
            cfg.preprocessing.butterworth.high_cutoff_hz,
            cfg.preprocessing.butterworth.filter_order,
        )
        normalized = min_max_normalize(
            filtered,
            cfg.preprocessing.normalization.range_min,
            cfg.preprocessing.normalization.range_max,
        )
        if cwt_device is not None:
            image = generate_composite_scalogram_torch(
                normalized,
                cfg.dataset.sampling_rate,
                wavelet=cfg.cwt.wavelet,
                scales_start=cfg.cwt.scales_start,
                scales_end=cfg.cwt.scales_end,
                image_size=expected_size,
                colormap=cfg.cwt.colormap,
                layout=tuple(cfg.cwt.composite_layout),
                device=cwt_device,
            )
        else:
            image = generate_composite_scalogram(
                normalized,
                cfg.dataset.sampling_rate,
                wavelet=cfg.cwt.wavelet,
                scales_start=cfg.cwt.scales_start,
                scales_end=cfg.cwt.scales_end,
                image_size=expected_size,
                colormap=cfg.cwt.colormap,
                layout=tuple(cfg.cwt.composite_layout),
            )
        quality_issues = scalogram_quality_issues(image)
        if quality_issues:
            return ProcessResult(ecg_id, filename, "failed", str(out_path), reason=";".join(quality_issues))

        checksum = save_scalogram(image, str(out_path), is_chw=True)
        integrity = validate_scalogram_image(out_path, expected_size)
        if not integrity.valid:
            return ProcessResult(ecg_id, filename, "failed", str(out_path), checksum, integrity.reason)

        return ProcessResult(ecg_id, filename, "generated", str(out_path), checksum, reason=";".join(issues))
    except Exception as exc:
        return ProcessResult(ecg_id, filename, "failed", str(out_path), reason=repr(exc))


def _write_reports(
    results: List[ProcessResult],
    expected_count: int,
    reports_dir: Path,
    cwt_dir: Path,
    selected_count: Optional[int] = None,
) -> None:
    report_df = pd.DataFrame([asdict(result) for result in results])
    failed_df = report_df[report_df["status"] == "failed"].copy()

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(reports_dir / "preprocessing_report.csv", index=False)
    failed_df[["ecg_id", "filename", "reason"]].to_csv(reports_dir / "preprocessing_failed_ecgs.csv", index=False)
    failed_df[["ecg_id", "filename", "reason"]].to_csv(cwt_dir.parent / "preprocessing_failed_ecgs.csv", index=False)

    pngs = list(cwt_dir.glob("*.png"))
    duplicate_outputs = int(report_df["output_path"].duplicated().sum())
    summary = {
        "expected_ecg_samples": int(expected_count),
        "processed_this_run": int(selected_count if selected_count is not None else len(report_df)),
        "result_rows": int(len(report_df)),
        "generated": int((report_df["status"] == "generated").sum()),
        "cached": int((report_df["status"] == "cached").sum()),
        "failed": int((report_df["status"] == "failed").sum()),
        "png_files_in_output_dir": int(len(pngs)),
        "all_records_have_verified_png": bool(len(failed_df) == 0 and len(pngs) >= expected_count),
        "duplicate_ecg_ids": int(report_df["ecg_id"].duplicated().sum()),
        "duplicate_output_paths": duplicate_outputs,
    }
    write_json(reports_dir / "preprocessing_summary.json", summary)

    with (reports_dir / "preprocessing_logs.txt").open("w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, sort_keys=True))
        f.write("\n\nFailures:\n")
        if failed_df.empty:
            f.write("none\n")
        else:
            for row in failed_df.itertuples(index=False):
                f.write(f"{row.ecg_id}: {row.reason}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate verified CWT scalograms")
    parser.add_argument("--subset", type=int, default=None, help="Process only the first N records")
    parser.add_argument("--force", action="store_true", help="Regenerate even when cache integrity is valid")
    parser.add_argument("--failed-only", action="store_true", help="Regenerate only ECG IDs from the previous failure CSV")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="CWT compute device")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_logger("generate_cwt", log_file="generate_cwt.log")
    logger.info("Starting verified CWT generation")
    cwt_device = _resolve_cwt_device(args.device)
    logger.info(f"CWT device: {cwt_device or 'cpu'}")

    df_all = _load_splits(Path(cfg.output.splits_dir))
    expected_count = len(df_all)
    if args.subset is not None:
        df_all = df_all.head(args.subset)
        logger.info(f"Processing subset of {len(df_all)} records")

    cwt_dir = Path(cfg.cwt.output_dir)
    cwt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(cfg.output.metrics_dir).parent / "preprocessing"
    if args.failed_only:
        df_all = _filter_failed_records(df_all, reports_dir / "preprocessing_failed_ecgs.csv")
        logger.info(f"Processing failed-only subset of {len(df_all)} records")

    results: List[ProcessResult] = []
    for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Generating CWTs"):
        result = process_record(row, cfg, Path(cfg.dataset.root_dir), cwt_dir, force=args.force, cwt_device=cwt_device)
        results.append(result)
        if result.status == "failed":
            logger.warning(f"Failed ECG {result.ecg_id}: {result.reason}")

    _write_reports(results, expected_count, reports_dir, cwt_dir, selected_count=len(df_all))

    failures = sum(1 for result in results if result.status == "failed")
    logger.info(f"Finished CWT generation: {len(results) - failures}/{len(results)} verified.")
    if failures:
        logger.error(f"{failures} ECGs failed preprocessing. See {reports_dir / 'preprocessing_failed_ecgs.csv'}")
        sys.exit(2)


if __name__ == "__main__":
    main()
