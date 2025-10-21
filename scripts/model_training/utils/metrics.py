"""
Metrics calculation utilities
"""

import logging
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

from ..config.optimized_config import OptimizedModelConfig as ModelConfig


class MetricsCalculator:
    """Metrics calculator"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize metrics calculator
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Metrics dictionary
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        
        # ROC-AUC and PR-AUC (if probabilities provided)
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix related metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
        
        return metrics
    
    def calculate_custom_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate custom metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Custom metrics dictionary
        """
        metrics = {}
        
        # Calculate basic metrics
        basic_metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        metrics.update(basic_metrics)
        
        # Calculate additional medical-related metrics
        if y_pred_proba is not None:
            # Performance at different thresholds
            thresholds = [0.3, 0.5, 0.7]
            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                metrics[f'precision_at_{threshold}'] = precision_score(y_true, y_pred_thresh)
                metrics[f'recall_at_{threshold}'] = recall_score(y_true, y_pred_thresh)
                metrics[f'f1_at_{threshold}'] = f1_score(y_true, y_pred_thresh)
        
        return metrics
    
    def get_classification_report_dict(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Get classification report dictionary
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report dictionary
        """
        return classification_report(y_true, y_pred, output_dict=True)
    
    def calculate_class_balance_metrics(self, y_true: np.ndarray) -> Dict[str, float]:
        """
        Calculate class balance metrics
        
        Args:
            y_true: True labels
            
        Returns:
            Class balance metrics
        """
        unique, counts = np.unique(y_true, return_counts=True)
        
        metrics = {}
        metrics['class_counts'] = dict(zip(unique, counts))
        metrics['class_ratios'] = dict(zip(unique, counts / len(y_true)))
        
        if len(counts) == 2:
            metrics['imbalance_ratio'] = counts.max() / counts.min()
            metrics['minority_class_ratio'] = counts.min() / len(y_true)
        
        return metrics
    
    def get_metric_summary(self, metrics: Dict[str, float]) -> str:
        """
        Get metrics summary string
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        if 'accuracy' in metrics:
            summary_parts.append(f"Accuracy: {metrics['accuracy']:.2f}")
        if 'precision' in metrics:
            summary_parts.append(f"Precision: {metrics['precision']:.2f}")
        if 'recall' in metrics:
            summary_parts.append(f"Recall: {metrics['recall']:.2f}")
        if 'f1' in metrics:
            summary_parts.append(f"F1-Score: {metrics['f1']:.2f}")
        if 'roc_auc' in metrics:
            summary_parts.append(f"ROC-AUC: {metrics['roc_auc']:.2f}")
        if 'pr_auc' in metrics:
            summary_parts.append(f"PR-AUC: {metrics['pr_auc']:.2f}")
        
        return " | ".join(summary_parts)
