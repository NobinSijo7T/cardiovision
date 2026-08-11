"""
CARDIOVISION - Metrics
Evaluation metrics for training and testing.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels (N,).
        y_pred: Predicted labels (N,).
        y_prob: Prediction probabilities (N, num_classes).
        class_names: List of class names for reporting.

    Returns:
        Dictionary of computed metrics.
    """
    metrics = {}
    
    # Basic Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Store per-class metrics
    metrics["per_class"] = {}
    for i, cls_name in enumerate(class_names):
        if i < len(precision):
            metrics["per_class"][cls_name] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i]
            }
            
    # Macro averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["macro_precision"] = macro_p
    metrics["macro_recall"] = macro_r
    metrics["macro_f1"] = macro_f1
    
    # Weighted averages
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["weighted_precision"] = weighted_p
    metrics["weighted_recall"] = weighted_r
    metrics["weighted_f1"] = weighted_f1
    
    # ROC AUC (One-vs-Rest)
    try:
        # Check if all classes are present in y_true, otherwise roc_auc_score fails
        if len(np.unique(y_true)) == len(class_names):
            metrics["roc_auc_macro"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
            metrics["roc_auc_weighted"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )
        else:
            metrics["roc_auc_macro"] = float('nan')
            metrics["roc_auc_weighted"] = float('nan')
    except ValueError:
        metrics["roc_auc_macro"] = float('nan')
        metrics["roc_auc_weighted"] = float('nan')
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    metrics["confusion_matrix"] = cm.tolist()
    
    # Full classification report string
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    
    return metrics
