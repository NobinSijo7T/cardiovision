"""
CARDIOVISION - Results Plotting
Visualizations for training curves and evaluation metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_training_history(csv_path: str, figsize=(12, 5)):
    """Plot training and validation loss and accuracy from history CSV."""
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(df['epoch'], df['train_acc'], label='Train Acc', marker='o')
    ax2.plot(df['epoch'], df['val_acc'], label='Val Acc', marker='s')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm: list, class_names: list, title: str = "Confusion Matrix", figsize=(8, 6)):
    """Plot confusion matrix heatmap."""
    cm_array = np.array(cm)
    
    # Normalize confusion matrix
    cm_norm = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_norm, annot=cm_array, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
                
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig
