"""
CARDIOVISION - CWT Plotting
Visualizations for CWT scalograms.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_scalogram(scalogram: np.ndarray, title: str = "CWT Scalogram", figsize=(8, 6)):
    """
    Plot a single scalogram image.
    
    Args:
        scalogram: Scalogram image array. Can be (H, W, 3) or (3, H, W).
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if CHW format and convert to HWC
    if scalogram.shape[0] == 3 and len(scalogram.shape) == 3:
        display_img = scalogram.transpose(1, 2, 0)
    else:
        display_img = scalogram
        
    ax.imshow(display_img)
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_overlay(image: np.ndarray, title: str = "Grad-CAM Explanation", figsize=(8, 6)):
    """Plot an overlay image."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if BGR from OpenCV and convert to RGB
    if image.shape[-1] == 3:
        # Assuming RGB if passed through Streamlit/PIL, but checking just in case
        pass
        
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    return fig
