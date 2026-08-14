"""
CARDIOVISION - Dataset Preparation Script
Orchestrates metadata parsing, label mapping, and patient-level splitting.
"""

import sys
from pathlib import Path
import json

from _bootstrap import ensure_project_venv

ensure_project_venv()

# Add project root to path so we can import from src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.metadata_parser import load_all_metadata
from src.data.label_mapper import assign_labels
from src.data.splitter import patient_level_split, save_splits
from src.data.validation import (
    build_class_distribution_report,
    validate_metadata,
    validate_record_files,
    write_class_distribution_plot,
    write_json,
    write_patient_distribution,
)

def main():
    # Load configuration
    cfg = load_config()
    
    # Set seed for reproducibility
    set_seed(cfg.reproducibility.seed, cfg.reproducibility.deterministic)
    
    # Setup logger
    logger = setup_logger("prepare_dataset", log_file="prepare_dataset.log")
    logger.info("Starting Dataset Preparation")
    
    # 1. Parse Metadata
    dataset_root = cfg.dataset.root_dir
    db_df, scp_df, diag_df, rhythm_df = load_all_metadata(dataset_root)
    
    # 2. Map Labels
    df_labeled, mapping_stats = assign_labels(db_df)

    metadata_report, metadata_summary = validate_metadata(df_labeled)
    file_report, missing_files, file_summary = validate_record_files(df_labeled, dataset_root)

    # Save mapping report
    report_path = Path(cfg.output.dataset_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    class_report = build_class_distribution_report(df_labeled)
    dataset_report = {
        "label_mapping": mapping_stats,
        "metadata_validation": metadata_summary,
        "file_validation": file_summary,
        "class_distribution": class_report,
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_report, f, indent=2, default=str)
    logger.info(f"Saved dataset report to {report_path}")

    reports_dir = Path(cfg.output.metrics_dir).parent / "preprocessing"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata_report.to_csv(reports_dir / "preprocessing_report.csv", index=False)
    missing_files.to_csv(reports_dir / "missing_files.csv", index=False)
    write_patient_distribution(df_labeled, reports_dir / "patient_distribution.csv")
    write_class_distribution_plot(df_labeled, Path(cfg.output.figures_dir) / "class_distribution.png")
    
    # 3. Patient-Level Splitting
    df_train, df_val, df_test = patient_level_split(
        df_labeled,
        train_ratio=cfg.splitting.train_ratio,
        val_ratio=cfg.splitting.val_ratio,
        test_ratio=cfg.splitting.test_ratio,
        random_seed=cfg.splitting.random_seed
    )
    
    # Save splits
    save_splits(df_train, df_val, df_test, output_dir=cfg.output.splits_dir)
    split_summary = {
        "train": {"records": len(df_train), "patients": df_train["patient_id"].nunique()},
        "val": {"records": len(df_val), "patients": df_val["patient_id"].nunique()},
        "test": {"records": len(df_test), "patients": df_test["patient_id"].nunique()},
        "patient_leakage": False,
    }
    write_json(reports_dir / "preprocessing_summary.json", {**dataset_report, "splits": split_summary})
    logger.info("Dataset Preparation Complete")

if __name__ == "__main__":
    main()
