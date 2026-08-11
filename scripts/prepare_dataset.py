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
    
    # Save mapping report
    report_path = Path(cfg.output.dataset_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_stats, f, indent=2, default=str)
    logger.info(f"Saved dataset report to {report_path}")
    
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
    logger.info("Dataset Preparation Complete")

if __name__ == "__main__":
    main()
