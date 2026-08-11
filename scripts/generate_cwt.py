"""
CARDIOVISION - Generate CWT Scalograms Script
Batch generates CWT images from raw ECG signals.
"""

import sys
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

from _bootstrap import ensure_project_venv

ensure_project_venv()

import pandas as pd
from tqdm import tqdm
import wfdb
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.preprocessing.signal_quality import check_signal_quality
from src.preprocessing.butterworth import apply_butterworth_filter
from src.preprocessing.normalization import min_max_normalize
from src.preprocessing.cwt_transform import generate_composite_scalogram, save_scalogram

def process_record(ecg_id, filename, cfg, dataset_root, cwt_dir, logger):
    """Process a single ECG record and save its CWT scalogram."""
    try:
        record_path = str(Path(dataset_root) / filename)
        signal, _ = wfdb.rdsamp(record_path)
        
        # 1. Quality Check
        is_valid, _ = check_signal_quality(
            signal,
            max_nan_ratio=cfg.preprocessing.signal_quality.max_nan_ratio,
            max_flat_ratio=cfg.preprocessing.signal_quality.max_flat_ratio,
            amplitude_min_mv=cfg.preprocessing.signal_quality.amplitude_min_mv,
            amplitude_max_mv=cfg.preprocessing.signal_quality.amplitude_max_mv
        )
        if not is_valid:
            return False
            
        # 2. Filter
        signal = apply_butterworth_filter(
            signal,
            cfg.dataset.sampling_rate,
            cfg.preprocessing.butterworth.low_cutoff_hz,
            cfg.preprocessing.butterworth.high_cutoff_hz,
            cfg.preprocessing.butterworth.filter_order
        )
        
        # 3. Normalize
        signal = min_max_normalize(
            signal,
            cfg.preprocessing.normalization.range_min,
            cfg.preprocessing.normalization.range_max
        )
        
        # 4. Generate CWT
        img = generate_composite_scalogram(
            signal,
            cfg.dataset.sampling_rate,
            wavelet=cfg.cwt.wavelet,
            scales_start=cfg.cwt.scales_start,
            scales_end=cfg.cwt.scales_end,
            image_size=tuple(cfg.cwt.image_size),
            colormap=cfg.cwt.colormap,
            layout=tuple(cfg.cwt.composite_layout)
        )
        
        # 5. Save
        # The generate_composite_scalogram returns CHW float [0,1] format
        out_path = Path(cwt_dir) / f"{ecg_id}.{cfg.cwt.save_format}"
        save_scalogram(img, str(out_path), is_chw=True)
        return True
        
    except Exception as e:
        logger.warning(f"Error processing {ecg_id}: {e}")
        return False

def _process_wrapper(args, cfg, dataset_root, cwt_dir, logger):
    ecg_id, filename = args
    return process_record(ecg_id, filename, cfg, dataset_root, cwt_dir, logger)

def main():
    parser = argparse.ArgumentParser(description="Generate CWT Scalograms")
    parser.add_argument("--subset", type=int, default=None, help="Process only a subset of records")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_logger("generate_cwt", log_file="generate_cwt.log")
    
    logger.info("Starting CWT generation")
    
    # Load dataset records
    train_csv = Path(cfg.output.splits_dir) / "train_records.csv"
    val_csv = Path(cfg.output.splits_dir) / "val_records.csv"
    test_csv = Path(cfg.output.splits_dir) / "test_records.csv"
    
    if not train_csv.exists():
        logger.error("Split records not found. Run prepare_dataset.py first.")
        sys.exit(1)
        
    df_train = pd.read_csv(train_csv, index_col=0)
    df_val = pd.read_csv(val_csv, index_col=0)
    df_test = pd.read_csv(test_csv, index_col=0)
    df_all = pd.concat([df_train, df_val, df_test])
    
    logger.info(f"Loaded {len(df_all)} records from splits.")
    
    if args.subset:
        df_all = df_all.head(args.subset)
        logger.info(f"Processing subset of {args.subset} records.")
        
    filename_col = 'filename_lr' if cfg.dataset.sampling_rate == 100 else 'filename_hr'
    tasks = [(idx, row[filename_col]) for idx, row in df_all.iterrows()]
    
    cwt_dir = Path(cfg.cwt.output_dir)
    cwt_dir.mkdir(parents=True, exist_ok=True)
    
    num_workers = args.workers if args.workers else max(1, cpu_count() - 1)
    logger.info(f"Using {num_workers} parallel workers")
    
    process_func = partial(
        _process_wrapper, 
        cfg=cfg, 
        dataset_root=cfg.dataset.root_dir, 
        cwt_dir=str(cwt_dir),
        logger=logger
    )
    
    success_count = 0
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_func, tasks), 
            total=len(tasks),
            desc="Generating CWTs"
        ))
        success_count = sum(1 for r in results if r)
        
    logger.info(f"Finished generating {success_count}/{len(tasks)} CWT scalograms.")

if __name__ == "__main__":
    main()
