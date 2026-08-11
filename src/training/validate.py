"""
CARDIOVISION - Validation Loop
Validation/Evaluation loop without gradients.
"""

from typing import Dict, List, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from src.utils.logger import get_logger
from src.training.metrics import compute_metrics

logger = get_logger("cardiovision.training.validate")

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
    epoch_idx: int = 0,
    prefix: str = "Val"
) -> Tuple[float, float, Dict]:
    """
    Evaluate the model on a validation or test set.

    Args:
        model: The neural network model.
        dataloader: Validation/Test dataloader.
        criterion: Loss function.
        device: Device to run on.
        class_names: List of class names for metrics.
        epoch_idx: Current epoch index (for logging).
        prefix: Prefix for logging (e.g., "Val" or "Test").

    Returns:
        Tuple of (average_loss, average_accuracy, detailed_metrics_dict).
    """
    model.eval()
    
    total_loss = 0.0
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    start_time = time.time()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [{prefix}]", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Probabilities and predictions
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        total_loss += loss.item() * inputs.size(0)
        
        all_targets.append(targets.cpu().numpy())
        all_preds.append(predicted.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        
        # Update progress bar
        current_loss = total_loss / (len(all_targets) * dataloader.batch_size) # approximate
        pbar.set_postfix({'loss': f"{current_loss:.4f}"})
        
    epoch_time = time.time() - start_time
    
    # Concatenate all batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    
    total_samples = len(y_true)
    avg_loss = total_loss / total_samples
    
    # Compute detailed metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    avg_acc = metrics["accuracy"]
    
    logger.info(f"{prefix} Epoch {epoch_idx} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
                f"Macro F1: {metrics.get('macro_f1', 0):.4f} | Time: {epoch_time:.2f}s")
                
    return avg_loss, avg_acc, metrics
