"""
CARDIOVISION - Evaluate Model Script
Evaluates a trained model on the held-out test set.
"""

import sys
from pathlib import Path
import json

from _bootstrap import ensure_project_venv

ensure_project_venv()

import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config import load_config, get_device
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.dataset_loader import create_data_loaders
from src.models.vit import CardioViT
from src.training.losses import build_criterion
from src.training.validate import validate_epoch

def main():
    cfg = load_config()
    set_seed(cfg.reproducibility.seed, cfg.reproducibility.deterministic)
    logger = setup_logger("evaluate_model", log_file="evaluate_model.log")
    
    device_name = get_device(cfg)
    device = torch.device(device_name)
    
    # Load splits
    try:
        df_train = pd.read_csv(Path(cfg.output.splits_dir) / "train_records.csv", index_col=0)
        df_val = pd.read_csv(Path(cfg.output.splits_dir) / "val_records.csv", index_col=0)
        df_test = pd.read_csv(Path(cfg.output.splits_dir) / "test_records.csv", index_col=0)
    except FileNotFoundError:
        logger.error("Split CSVs not found. Run prepare_dataset.py first.")
        sys.exit(1)
        
    _, _, test_loader = create_data_loaders(
        df_train, df_val, df_test,
        dataset_root=cfg.dataset.root_dir,
        cwt_image_dir=cfg.cwt.output_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        image_size=tuple(cfg.model.input_size),
        augmentation_config=None
    )
    
    criterion = build_criterion(
        loss_name=cfg.training.loss.name,
        device=device_name
    )
    
    # Build Model
    model = CardioViT(
        image_size=cfg.model.input_size[0],
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.input_channels,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embedding_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_dim=cfg.model.mlp_dim,
        use_cls_token=cfg.model.use_cls_token,
        use_positional_embedding=cfg.model.use_positional_embedding
    ).to(device)
    
    # Load checkpoint
    best_model_path = Path(cfg.output.checkpoints_dir) / "best_model.pth"
    if not best_model_path.exists():
        logger.error(f"Checkpoint not found at {best_model_path}")
        sys.exit(1)
        
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate
    logger.info("Evaluating on Test Set...")
    test_loss, test_acc, test_metrics = validate_epoch(
        model, test_loader, criterion, device, 
        class_names=cfg.labels.class_names, epoch_idx=checkpoint['epoch'], prefix="Test"
    )
    
    # Save test metrics
    metrics_path = Path(cfg.output.metrics_dir) / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2, default=str)
    
    logger.info(f"Evaluation complete. Metrics saved to {metrics_path}")
    logger.info(f"Test Accuracy: {test_acc:.4f}, Test Macro F1: {test_metrics.get('macro_f1', 0):.4f}")

if __name__ == "__main__":
    main()
