"""
CARDIOVISION - Train Model Script
Orchestrates the training pipeline for the ViT model.
"""

import sys
from pathlib import Path
import json

from _bootstrap import ensure_project_venv

ensure_project_venv()

import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config import load_config, get_device
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.dataset_loader import create_data_loaders
from src.models.vit import CardioViT
from src.training.losses import build_criterion
from src.training.train import train_epoch
from src.training.validate import validate_epoch


def is_improvement(current_value, best_value, monitor, min_delta):
    if monitor == "val_loss":
        return current_value < best_value - min_delta
    return current_value > best_value + min_delta


def main():
    cfg = load_config()
    set_seed(cfg.reproducibility.seed, cfg.reproducibility.deterministic)
    logger = setup_logger("train_model", log_file="train_model.log")
    
    device_name = get_device(cfg)
    device = torch.device(device_name)
    logger.info(f"Training on device: {device}")
    
    # Load data splits
    try:
        df_train = pd.read_csv(Path(cfg.output.splits_dir) / "train_records.csv", index_col=0)
        df_val = pd.read_csv(Path(cfg.output.splits_dir) / "val_records.csv", index_col=0)
        df_test = pd.read_csv(Path(cfg.output.splits_dir) / "test_records.csv", index_col=0)
    except FileNotFoundError:
        logger.error("Split CSVs not found. Run prepare_dataset.py first.")
        sys.exit(1)
        
    # Build DataLoaders
    train_loader, val_loader, _ = create_data_loaders(
        df_train, df_val, df_test,
        dataset_root=cfg.dataset.root_dir,
        cwt_image_dir=cfg.cwt.output_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        image_size=tuple(cfg.model.input_size),
        augmentation_config=cfg.training.augmentation.__dict__,
        use_weighted_sampler=cfg.training.use_weighted_sampler,
    )
    
    # Class weights
    class_weights = None
    if cfg.training.loss.use_class_weights:
        from src.data.label_mapper import get_class_weights
        class_weights = get_class_weights(df_train)
        
    criterion = build_criterion(
        loss_name=cfg.training.loss.name,
        class_weights=class_weights,
        label_smoothing=cfg.training.loss.label_smoothing,
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
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        use_cls_token=cfg.model.use_cls_token,
        use_positional_embedding=cfg.model.use_positional_embedding
    ).to(device)
    
    # Optimizer & Scheduler
    if cfg.training.optimizer.name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=cfg.training.optimizer.learning_rate,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=tuple(cfg.training.optimizer.betas)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.optimizer.learning_rate)
        
    if cfg.training.scheduler.name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cfg.training.scheduler.T_max,
            eta_min=cfg.training.scheduler.eta_min
        )
    else:
        scheduler = None
        
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if (cfg.training.mixed_precision and device.type == 'cuda') else None
    
    # Training Loop
    monitor = cfg.training.early_stopping.monitor
    best_score = float('inf') if monitor == "val_loss" else float('-inf')
    epochs_no_improve = 0
    checkpoint_dir = Path(cfg.output.checkpoints_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    history = []
    
    logger.info("Starting training loop...")
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, cfg.training.gradient_clip_norm, epoch
        )
        
        val_loss, val_acc, val_metrics = validate_epoch(
            model, val_loader, criterion, device, 
            class_names=cfg.labels.class_names, epoch_idx=epoch, prefix="Val"
        )
        
        if scheduler is not None:
            scheduler.step()
            
        current_metrics = {
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'val_macro_f1': val_metrics.get('macro_f1', 0)
        }
        history.append(current_metrics)
        if monitor not in current_metrics:
            logger.error(f"Unknown early stopping monitor: {monitor}")
            sys.exit(1)
        current_score = current_metrics[monitor]
        
        # Checkpointing
        if is_improvement(current_score, best_score, monitor, cfg.training.early_stopping.min_delta):
            best_score = current_score
            epochs_no_improve = 0
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_macro_f1': val_metrics.get('macro_f1', 0),
                'monitor': monitor,
                'best_score': best_score,
            }, best_model_path)
            logger.info(
                f"Saved new best model to {best_model_path} "
                f"({monitor}={current_score:.4f})"
            )
        else:
            epochs_no_improve += 1
            if cfg.training.early_stopping.enabled and epochs_no_improve >= cfg.training.early_stopping.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = Path(cfg.output.training_log)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path, index=False)
    logger.info(f"Training completed. History saved to {history_path}")

if __name__ == "__main__":
    main()
