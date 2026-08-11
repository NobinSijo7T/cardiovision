"""
CARDIOVISION - ViT Grad-CAM
Provides visual explanations for Vision Transformer predictions.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from src.utils.logger import get_logger

logger = get_logger("cardiovision.explainability.gradcam")

class ViTGradCAM:
    """
    Grad-CAM implementation specifically designed for Vision Transformers.
    """
    def __init__(self, model: nn.Module, target_layer_idx: int = -1):
        self.model = model
        self.model.eval()
        
        # Get target layer (default: last transformer block's norm1 before attention)
        # Note: In ViT, gradients with respect to attention weights or outputs are used.
        # We hook into the output of the Add operation after Attention, or the Norm before it.
        # Here we hook the final norm layer output which contains the final token representations.
        
        # In CardioViT, the transformer blocks are in model.encoder.blocks
        blocks = self.model.encoder.blocks
        if target_layer_idx < 0:
            target_layer_idx = len(blocks) + target_layer_idx
            
        self.target_layer = blocks[target_layer_idx].norm1
        
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]
        
    def __call__(self, x: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            x: Input image tensor (1, C, H, W).
            target_class: Class index to explain. If None, explains the predicted class.
            
        Returns:
            Heatmap array of shape (H, W) with values in [0, 1].
        """
        b, c, h, w = x.shape
        assert b == 1, "Grad-CAM only supports batch size 1"
        
        # Forward pass
        self.model.zero_grad()
        logits = self.model(x)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        # Backward pass for target class
        target = logits[0, target_class]
        target.backward()
        
        # Get activations and gradients
        # Shape: (1, seq_len, embed_dim)
        activations = self.activations[0] # (seq_len, embed_dim)
        gradients = self.gradients[0] # (seq_len, embed_dim)
        
        # Remove CLS token if present
        if self.model.use_cls_token:
            activations = activations[1:]
            gradients = gradients[1:]
            
        # Calculate weights: Global Average Pooling of gradients across spatial dimensions
        # ViT doesn't have spatial dimensions like CNN, but each token corresponds to a patch.
        # weight shape: (embed_dim,)
        weights = torch.mean(gradients, dim=0)
        
        # Calculate CAM: weighted combination of activations
        # CAM shape: (seq_len,)
        cam = torch.matmul(activations, weights)
        
        # Apply ReLU (we only care about positive influences)
        cam = F.relu(cam)
        
        # Reshape to spatial grid
        num_patches = cam.shape[0]
        grid_size = int(np.sqrt(num_patches))
        cam = cam.reshape(grid_size, grid_size)
        
        # Normalize to [0, 1]
        cam = cam - torch.min(cam)
        cam_max = torch.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
            
        # Upsample to original image size
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (w, h))
        
        return cam

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay a heatmap on an image.
    
    Args:
        image: Original image array (H, W, 3) in [0, 1].
        heatmap: Heatmap array (H, W) in [0, 1].
        alpha: Blending factor.
        colormap: OpenCV colormap.
        
    Returns:
        Overlay image array (H, W, 3) in [0, 255].
    """
    # Convert image to [0, 255] uint8
    if image.max() <= 1.0:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)
        
    # Convert heatmap to [0, 255] uint8 and apply colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Ensure image is RGB (OpenCV colormap returns BGR)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay
