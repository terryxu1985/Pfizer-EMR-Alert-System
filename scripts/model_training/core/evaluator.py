"""
Model evaluator for comprehensive model assessment
"""

import logging
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)

from ..config.optimized_config import OptimizedModelConfig as ModelConfig


class ModelEvaluator:
    """Model evaluator"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize evaluator
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        
        # Create output directories
        os.makedirs(self.config.path_config.model_evaluation_dir, exist_ok=True)
        os.makedirs(self.config.path_config.visualizations_dir, exist_ok=True)
    
    def evaluate_single_model(self, trainer: 'BaseTrainer', X_test: pd.DataFrame, 
                            y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate single model
        
        Args:
            trainer: Trainer instance
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation results dictionary
        """
        model_name = trainer._get_model_name()
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Get evaluation metrics
        metrics = trainer.evaluate(X_test, y_test)
        
        # Get feature importance
        feature_importance = trainer.get_feature_importance()
        
        # Get classification report
        y_pred = metrics.get('y_pred')
        if y_pred is not None:
            classification_rep = classification_report(
                y_test, y_pred, output_dict=True
            )
        else:
            classification_rep = {}
        
        # Get confusion matrix
        if y_pred is not None:
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            confusion_metrics = {
                'confusion_matrix': cm.tolist(),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': specificity,
                'false_positive_rate': fpr,
                'false_negative_rate': fnr
            }
        else:
            confusion_metrics = {}
        
        # Merge all results
        result = {
            'model_name': model_name,
            'model_type': trainer.config.model_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'classification_report': classification_rep,
            'confusion_metrics': confusion_metrics
        }
        
        # Store results
        self.results[model_name] = result
        
        return result
    
    def compare_models(self, model_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models using PR-AUC, Recall, Precision, F1 priority order
        
        Args:
            model_results: List of model results
            
        Returns:
            Comparison results DataFrame
        """
        comparison_data = []
        
        for result in model_results:
            metrics = result['metrics']
            model_name = result['model_name']
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1', 0),
                'ROC-AUC': metrics.get('roc_auc', 0),
                'PR-AUC': metrics.get('pr_auc', 0),
                'CV ROC-AUC Mean': metrics.get('cv_mean', 0),
                'CV ROC-AUC Std': metrics.get('cv_std', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate comprehensive score using weighted metrics
        # Medical scenario weights: Precision > PR-AUC > ROC-AUC > F1-Score > Recall > Accuracy
        weights = {
            'PR-AUC': 0.25,      # Important for imbalanced data
            'Recall': 0.15,      # Important but less than precision in medical scenarios
            'Precision': 0.30,   # Most important - avoid false positives
            'F1-Score': 0.15,    # Balanced measure
            'ROC-AUC': 0.10,     # Overall discrimination ability
            'Accuracy': 0.05     # Least important for imbalanced medical data
        }
        
        # Calculate weighted composite score
        comparison_df['Composite_Score'] = (
            comparison_df['PR-AUC'] * weights['PR-AUC'] +
            comparison_df['Recall'] * weights['Recall'] +
            comparison_df['Precision'] * weights['Precision'] +
            comparison_df['F1-Score'] * weights['F1-Score'] +
            comparison_df['ROC-AUC'] * weights['ROC-AUC'] +
            comparison_df['Accuracy'] * weights['Accuracy']
        )
        
        # Sort by composite score (descending)
        comparison_df = comparison_df.sort_values('Composite_Score', ascending=False)
        
        # Round numeric metrics to two decimals for consistency in saved CSV
        numeric_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'CV ROC-AUC Mean', 'CV ROC-AUC Std', 'Composite_Score']
        for col in numeric_columns:
            if col in comparison_df.columns:
                comparison_df[col] = comparison_df[col].astype(float).round(2)

        # Save comparison results
        output_path = os.path.join(
            self.config.path_config.model_evaluation_dir, 
            'model_comparison_results.csv'
        )
        comparison_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Model comparison results saved to: {output_path}")
        self.logger.info(f"Models ranked by: PR-AUC > Recall > Precision > F1-Score")
        
        return comparison_df
    
    def create_visualizations(self, model_results: List[Dict[str, Any]]):
        """
        Create visualization charts - only keep Model Performance Metrics
        
        Args:
            model_results: List of model results
        """
        self.logger.info("Creating model performance comparison charts...")
        
        # Set plotting style
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Model Performance Metrics
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
        x = np.arange(len(metrics_names))
        width = 0.15
        
        # Calculate appropriate width
        num_models = len(model_results)
        if num_models > 6:
            width = 0.12  # Reduce width if too many models
        elif num_models > 4:
            width = 0.13
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, result in enumerate(model_results):
            metrics = result['metrics']
            values = [
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0),
                metrics.get('roc_auc', 0),
                metrics.get('pr_auc', 0)
            ]
            
            # Use cycling colors
            color = colors[i % len(colors)]
            bars = ax.bar(x + i * width, values, width, label=result['model_name'], color=color, alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels(metrics_names, fontsize=11)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)  # Set y-axis range with space for value labels
        
        # Set y-axis ticks
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(
            self.config.path_config.visualizations_dir, 
            'model_comparison_plots.png'
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close chart to release memory
        
        self.logger.info(f"Model performance comparison chart saved to: {output_path}")
        
        # Generate separate feature importance analysis chart
        self._generate_feature_importance_plot(model_results)
    
    def _generate_feature_importance_plot(self, model_results: List[Dict[str, Any]]):
        """Generate separate feature importance analysis chart"""
        tree_models = [r for r in model_results if r['feature_importance'] is not None]
        
        if not tree_models:
            self.logger.warning("No models support feature importance, skipping feature importance analysis chart generation")
            return
        
        # Select best tree model
        best_tree_model = max(tree_models, key=lambda x: x['metrics'].get('roc_auc', 0))
        importance = best_tree_model['feature_importance']
        
        # Take top 20 most important features
        top_features = dict(list(importance.items())[:20])
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        # Create chart
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'{best_tree_model["model_name"]} - Top 20 Feature Importance Analysis')
        plt.gca().invert_yaxis()  # Most important features at top
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{imp:.2f}', ha='left', va='center', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(
            self.config.path_config.visualizations_dir, 
            'feature_importance_analysis.png'
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close chart to release memory
        
        self.logger.info(f"Feature importance analysis chart saved to: {output_path}")
    
    def generate_detailed_report(self, model_results: List[Dict[str, Any]]) -> str:
        """
        Generate detailed evaluation report
        
        Args:
            model_results: List of model results
            
        Returns:
            Report content
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Pfizer EMR Alert System - Model Evaluation Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Best model
        best_model = max(model_results, key=lambda x: x['metrics'].get('roc_auc', 0))
        report_lines.append(f"Best Model: {best_model['model_name']}")
        report_lines.append(f"ROC-AUC: {best_model['metrics'].get('roc_auc', 0):.2f}")
        report_lines.append(f"F1-Score: {best_model['metrics'].get('f1', 0):.2f}")
        report_lines.append("")
        
        # Detailed results for all models
        for result in model_results:
            model_name = result['model_name']
            metrics = result['metrics']
            
            report_lines.append(f"Model: {model_name}")
            report_lines.append("-" * 40)
            report_lines.append(f"Accuracy: {metrics.get('accuracy', 0):.2f}")
            report_lines.append(f"Precision: {metrics.get('precision', 0):.2f}")
            report_lines.append(f"Recall: {metrics.get('recall', 0):.2f}")
            report_lines.append(f"F1-Score: {metrics.get('f1', 0):.2f}")
            report_lines.append(f"ROC-AUC: {metrics.get('roc_auc', 0):.2f}")
            report_lines.append(f"PR-AUC: {metrics.get('pr_auc', 0):.2f}")
            report_lines.append(f"Cross-validation ROC-AUC: {metrics.get('cv_mean', 0):.2f} (+/- {metrics.get('cv_std', 0) * 2:.2f})")
            report_lines.append("")
        
        # Feature importance analysis
        tree_models = [r for r in model_results if r['feature_importance'] is not None]
        if tree_models:
            report_lines.append("Feature Importance Analysis (based on best tree model):")
            report_lines.append("-" * 40)
            best_tree_model = max(tree_models, key=lambda x: x['metrics'].get('roc_auc', 0))
            importance = best_tree_model['feature_importance']
            
            for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                report_lines.append(f"{i+1:2d}. {feature}: {imp:.2f}")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        output_path = os.path.join(
            self.config.path_config.model_evaluation_dir, 
            'model_evaluation_report.txt'
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Detailed evaluation report saved to: {output_path}")
        
        return report_content
    
    def save_results(self, model_results: List[Dict[str, Any]]):
        """
        Save all results
        
        Args:
            model_results: List of model results
        """
        # Create detailed results DataFrame
        detailed_results = []
        
        for result in model_results:
            model_name = result['model_name']
            metrics = result['metrics']
            confusion_metrics = result.get('confusion_metrics', {})
            
            detailed_results.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1', 0),
                'ROC-AUC': metrics.get('roc_auc', 0),
                'PR-AUC': metrics.get('pr_auc', 0),
                'CV ROC-AUC Mean': metrics.get('cv_mean', 0),
                'CV ROC-AUC Std': metrics.get('cv_std', 0),
                'Specificity': confusion_metrics.get('specificity', 0),
                'False Positive Rate': confusion_metrics.get('false_positive_rate', 0),
                'False Negative Rate': confusion_metrics.get('false_negative_rate', 0),
                'True Positives': confusion_metrics.get('true_positives', 0),
                'True Negatives': confusion_metrics.get('true_negatives', 0),
                'False Positives': confusion_metrics.get('false_positives', 0),
                'False Negatives': confusion_metrics.get('false_negatives', 0)
            })
        
        # Save detailed results (round numeric metrics to two decimals, keep counts as integers)
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df = detailed_df.sort_values('ROC-AUC', ascending=False)
        float_columns = [
            'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC',
            'CV ROC-AUC Mean', 'CV ROC-AUC Std', 'Specificity',
            'False Positive Rate', 'False Negative Rate'
        ]
        for col in float_columns:
            if col in detailed_df.columns:
                detailed_df[col] = detailed_df[col].astype(float).round(2)
        
        output_path = os.path.join(
            self.config.path_config.model_evaluation_dir, 
            'model_evaluation_results.csv'
        )
        detailed_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Detailed results saved to: {output_path}")
        
        return detailed_df
