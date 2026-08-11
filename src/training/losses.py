"""
CARDIOVISION - Loss Functions
Defines loss criteria for training.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger("cardiovision.training.losses")

def build_criterion(
    loss_name: str = "CrossEntropyLoss",
    class_weights: Optional[Dict[int, float]] = None,
    label_smoothing: float = 0.0,
    device: str = "cpu"
) -> nn.Module:
    """
    Build the loss function criterion.

    Args:
        loss_name: Name of the loss function.
        class_weights: Dictionary mapping class index to weight.
        label_smoothing: Amount of label smoothing [0.0, 1.0].
        device: Device to place the weights tensor on.

    Returns:
        Configured nn.Module loss function.
    """
    weights_tensor = None
    if class_weights is not None:
        num_classes = len(class_weights)
        weights_tensor = torch.zeros(num_classes, dtype=torch.float)
        for idx, weight in class_weights.items():
            weights_tensor[idx] = weight
        weights_tensor = weights_tensor.to(device)
        logger.info(f"Using class weights: {weights_tensor.tolist()}")

    if loss_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(
            weight=weights_tensor,
            label_smoothing=label_smoothing
        )
        logger.info(f"Built CrossEntropyLoss with smoothing={label_smoothing}")
        return criterion
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
