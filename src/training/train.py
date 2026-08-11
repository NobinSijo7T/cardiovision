"""
CARDIOVISION - Training Loop
Main training epoch loop.
"""

from typing import Dict, Optional, Tuple
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger("cardiovision.training.train")

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    clip_grad_norm: float = 1.0,
    epoch_idx: int = 0
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: The neural network model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on (cpu/cuda).
        scaler: GradScaler for mixed precision (if using).
        clip_grad_norm: Max gradient norm for clipping.
        epoch_idx: Current epoch index (for logging).

    Returns:
        Tuple of (average_loss, average_accuracy).
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [Train]", leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
            optimizer.step()
            
        # Statistics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss/total:.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
        
    epoch_time = time.time() - start_time
    avg_loss = total_loss / total
    avg_acc = correct / total
    
    logger.info(f"Train Epoch {epoch_idx} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Time: {epoch_time:.2f}s")
    
    return avg_loss, avg_acc
