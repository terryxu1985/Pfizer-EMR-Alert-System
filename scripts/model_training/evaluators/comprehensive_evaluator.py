"""
Comprehensive Model Evaluation Script for Pfizer EMR Alert System

This script provides a complete model evaluation pipeline that:
1. Trains and evaluates multiple machine learning models
2. Generates detailed performance comparisons
3. Creates comprehensive visualizations
4. Analyzes feature importance
5. Saves all results to reports/model_evaluation directory

All outputs, comments, and reports are in English.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.model_training.config.optimized_config import OptimizedModelConfig as ModelConfig
from scripts.model_training.pipelines.evaluation_pipeline import EvaluationPipeline
from scripts.model_training.core.evaluator import ModelEvaluator
from scripts.model_training.utils.data_loader import DataLoader
from scripts.model_training.utils.preprocessor import DataPreprocessor
from scripts.model_training.core.model_factory import ModelFactory

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure matplotlib for better visualizations
plt.style.use('default')
sns.set_palette("husl")


class ComprehensiveModelEvaluator:
    """
    Comprehensive model evaluator that provides detailed analysis and visualizations
    """
    
    def __init__(self, config: ModelConfig = None):
        """
        Initialize comprehensive model evaluator
        
        Args:
            config: Model configuration object. If None, uses default configuration.
        """
        self.config = config or ModelConfig()
        
        # Resolve paths to absolute paths
        project_root = Path(__file__).parent.parent.parent
        self.config = self.config.resolve_paths(project_root)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.evaluator = ModelEvaluator(self.config)
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_results = []
        
        # Create output directories
        self._create_output_directories()
        
        self.logger.info("Comprehensive Model Evaluator initialized successfully")
        self.logger.info(f"Output directory: {self.config.path_config.model_evaluation_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs(self.config.path_config.logs_dir, exist_ok=True)
        
        # Configure logging
        log_file = os.path.join(
            self.config.path_config.logs_dir, 
            f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(self.__class__.__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
        
        return logger
    
    def _create_output_directories(self):
        """Create necessary output directories"""
        directories = [
            self.config.path_config.model_evaluation_dir,
            self.config.path_config.visualizations_dir,
            os.path.join(self.config.path_config.model_evaluation_dir, "detailed_analysis"),
            os.path.join(self.config.path_config.model_evaluation_dir, "feature_importance"),
            os.path.join(self.config.path_config.model_evaluation_dir, "performance_metrics")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def load_and_prepare_data(self, dataset_path: Optional[str] = None) -> bool:
        """
        Load and prepare data for model evaluation
        
        Args:
            dataset_path: Path to dataset. If None, uses default path from config.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STEP 1: Loading and Preparing Data")
            self.logger.info("=" * 80)
            
            # Load dataset
            self.logger.info("Loading dataset...")
            df = self.data_loader.load_dataset(dataset_path)
            
            if df is None or df.empty:
                self.logger.error("Failed to load dataset or dataset is empty")
                return False
            
            self.logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Validate dataset
            self.logger.info("Validating dataset...")
            if not self.data_loader.validate_dataset():
                self.logger.error("Dataset validation failed")
                return False
            
            self.logger.info("Dataset validation passed")
            
            # Get clean dataset
            self.logger.info("Preparing clean dataset...")
            X, y = self.data_loader.get_clean_dataset()
            
            self.logger.info(f"Clean dataset prepared. Features: {X.shape[1]}, Samples: {X.shape[0]}")
            self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            # Split data
            self.logger.info("Splitting data into train/test sets...")
            self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.split_data(X, y)
            
            self.logger.info(f"Data split completed:")
            self.logger.info(f"  Training set: {self.X_train.shape}")
            self.logger.info(f"  Test set: {self.X_test.shape}")
            
            # Fit and transform training data
            self.logger.info("Fitting and transforming training data...")
            self.X_train, self.y_train = self.preprocessor.fit_transform(self.X_train, self.y_train)
            
            # Transform test data
            self.logger.info("Transforming test data...")
            self.X_test = self.preprocessor.transform(self.X_test)
            
            # Handle class imbalance
            if self.config.use_smote:
                self.logger.info("Handling class imbalance with SMOTE...")
                self.X_train, self.y_train = self.preprocessor.handle_class_imbalance(
                    self.X_train, self.y_train
                )
                self.logger.info(f"After SMOTE - Training set: {self.X_train.shape}")
            
            self.logger.info("Data preparation completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def train_and_evaluate_models(self, model_types: List[str] = None) -> bool:
        """
        Train and evaluate multiple models
        
        Args:
            model_types: List of model types to evaluate. If None, uses default models.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STEP 2: Training and Evaluating Models")
            self.logger.info("=" * 80)
            
            if model_types is None:
                model_types = [
                    'xgboost', 'random_forest', 'logistic_regression',
                    'gradient_boosting', 'svm', 'naive_bayes'
                ]
            
            self.logger.info(f"Models to evaluate: {model_types}")
            
            self.model_results = []
            successful_models = 0
            
            for i, model_type in enumerate(model_types, 1):
                self.logger.info(f"\n--- Training Model {i}/{len(model_types)}: {model_type.upper()} ---")
                
                try:
                    # Create trainer
                    trainer = ModelFactory.create_trainer(self.config, model_type)
                    trainer.feature_columns = self.preprocessor.feature_columns
                    
                    # Train model
                    self.logger.info(f"Training {model_type} model...")
                    trainer.train(self.X_train, self.y_train)
                    
                    # Evaluate model
                    self.logger.info(f"Evaluating {model_type} model...")
                    result = self.evaluator.evaluate_single_model(trainer, self.X_test, self.y_test)
                    
                    self.model_results.append(result)
                    successful_models += 1
                    
                    # Log model performance
                    metrics = result['metrics']
                    self.logger.info(f"✅ {model_type.upper()} - ROC-AUC: {metrics.get('roc_auc', 0):.2f}, "
                                   f"F1-Score: {metrics.get('f1', 0):.2f}, "
                                   f"Accuracy: {metrics.get('accuracy', 0):.2f}")
                    
                except Exception as e:
                    self.logger.error(f"❌ {model_type.upper()} training failed: {e}")
                    continue
            
            self.logger.info(f"\nModel training completed!")
            self.logger.info(f"Successfully trained: {successful_models}/{len(model_types)} models")
            
            if successful_models == 0:
                self.logger.error("No models were successfully trained!")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis including comparisons, visualizations, and reports
        
        Returns:
            Dictionary containing all analysis results
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STEP 3: Generating Comprehensive Analysis")
            self.logger.info("=" * 80)
            
            if not self.model_results:
                raise ValueError("No model results available. Please train models first.")
            
            analysis_results = {}
            
            # 1. Model Performance Comparison
            self.logger.info("Generating model performance comparison...")
            comparison_df = self.evaluator.compare_models(self.model_results)
            analysis_results['comparison_df'] = comparison_df
            
            # 2. Detailed Results
            self.logger.info("Saving detailed results...")
            detailed_df = self.evaluator.save_results(self.model_results)
            analysis_results['detailed_df'] = detailed_df
            
            # 3. Enhanced Visualizations
            self.logger.info("Creating enhanced visualizations...")
            self._create_enhanced_visualizations()
            
            # 4. Feature Importance Analysis
            self.logger.info("Analyzing feature importance...")
            feature_analysis = self._analyze_feature_importance()
            analysis_results['feature_analysis'] = feature_analysis
            
            # 5. Performance Metrics Analysis
            self.logger.info("Analyzing performance metrics...")
            performance_analysis = self._analyze_performance_metrics()
            analysis_results['performance_analysis'] = performance_analysis
            
            # 6. Generate Comprehensive Report
            self.logger.info("Generating comprehensive report...")
            report_content = self._generate_comprehensive_report(analysis_results)
            analysis_results['report_content'] = report_content
            
            # 7. Save Analysis Summary
            self.logger.info("Saving analysis summary...")
            self._save_analysis_summary(analysis_results)
            
            self.logger.info("Comprehensive analysis completed successfully!")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def _create_enhanced_visualizations(self):
        """Create enhanced visualization charts"""
        try:
            # 1. Model Performance Comparison Chart
            self._create_performance_comparison_chart()
            
            # 2. ROC Curves Comparison
            self._create_roc_curves_comparison()
            
            # 3. Feature Importance Analysis Chart
            self._create_feature_importance_chart()
            
            # 4. Confusion Matrix Heatmaps
            self._create_confusion_matrix_heatmaps()
            
            # 5. Performance Metrics Heatmap
            self._create_performance_heatmap()
            
            # 6. Model Ranking Chart
            self._create_model_ranking_chart()
            
            self.logger.info("Enhanced visualizations created successfully")
            
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
    
    def _create_performance_comparison_chart(self):
        """Create comprehensive performance comparison chart"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
            
            # Prepare data
            model_names = [result['model_name'] for result in self.model_results]
            metrics_data = {
                'Accuracy': [result['metrics'].get('accuracy', 0) for result in self.model_results],
                'Precision': [result['metrics'].get('precision', 0) for result in self.model_results],
                'Recall': [result['metrics'].get('recall', 0) for result in self.model_results],
                'F1-Score': [result['metrics'].get('f1', 0) for result in self.model_results],
                'ROC-AUC': [result['metrics'].get('roc_auc', 0) for result in self.model_results],
                'PR-AUC': [result['metrics'].get('pr_auc', 0) for result in self.model_results]
            }
            
            # 1. Bar chart for main metrics
            ax1 = axes[0, 0]
            x = np.arange(len(model_names))
            width = 0.15
            
            for i, (metric, values) in enumerate(metrics_data.items()):
                ax1.bar(x + i * width, values, width, label=metric, alpha=0.8)
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Score')
            ax1.set_title('Performance Metrics Comparison')
            ax1.set_xticks(x + width * 2)
            ax1.set_xticklabels(model_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)
            
            # 2. ROC-AUC comparison
            ax2 = axes[0, 1]
            roc_auc_values = metrics_data['ROC-AUC']
            bars = ax2.bar(model_names, roc_auc_values, color='skyblue', alpha=0.7)
            ax2.set_ylabel('ROC-AUC Score')
            ax2.set_title('ROC-AUC Comparison')
            ax2.set_ylim(0, 1.0)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, roc_auc_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # 3. F1-Score vs Precision scatter
            ax3 = axes[1, 0]
            precision_values = metrics_data['Precision']
            f1_values = metrics_data['F1-Score']
            
            scatter = ax3.scatter(precision_values, f1_values, s=100, alpha=0.7)
            ax3.set_xlabel('Precision')
            ax3.set_ylabel('F1-Score')
            ax3.set_title('Precision vs F1-Score')
            ax3.grid(True, alpha=0.3)
            
            # Add model labels
            for i, model_name in enumerate(model_names):
                ax3.annotate(model_name, (precision_values[i], f1_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # 4. Performance radar chart (simplified)
            ax4 = axes[1, 1]
            metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
            
            # Find best model
            best_model_idx = np.argmax([result['metrics'].get('pr_auc', 0) for result in self.model_results])
            best_model_values = [metrics_data[metric][best_model_idx] for metric in metrics_for_radar]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
            best_model_values += best_model_values[:1]  # Complete the circle
            angles += angles[:1]
            
            ax4.plot(angles, best_model_values, 'o-', linewidth=2, label=f'Best Model ({model_names[best_model_idx]})')
            ax4.fill(angles, best_model_values, alpha=0.25)
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics_for_radar)
            ax4.set_ylim(0, 1)
            ax4.set_title('Best Model Performance Profile')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Save chart
            output_path = os.path.join(
                self.config.path_config.visualizations_dir,
                'comprehensive_performance_comparison.png'
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance comparison chart saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Performance comparison chart creation failed: {e}")
    
    def _create_roc_curves_comparison(self):
        """Create ROC curves comparison chart"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot ROC curves for models that support probability prediction
            for result in self.model_results:
                model_name = result['model_name']
                metrics = result['metrics']
                
                # Check if model has probability predictions
                if 'y_pred_proba' in metrics and metrics['y_pred_proba'] is not None:
                    from sklearn.metrics import roc_curve, auc
                    
                    fpr, tpr, _ = roc_curve(self.y_test, metrics['y_pred_proba'])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, linewidth=2, 
                            label=f'{model_name} (AUC = {roc_auc:.2f})')
            
            # Plot diagonal line (random classifier)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Save chart
            output_path = os.path.join(
                self.config.path_config.visualizations_dir,
                'roc_curves_comparison.png'
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"ROC curves comparison chart saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"ROC curves comparison chart creation failed: {e}")
    
    def _create_feature_importance_chart(self):
        """Create comprehensive feature importance analysis chart"""
        try:
            # Find models that support feature importance
            tree_models = [r for r in self.model_results if r['feature_importance'] is not None]
            
            if not tree_models:
                self.logger.warning("No models support feature importance analysis")
                return
            
            # Create subplot for each tree model
            num_models = len(tree_models)
            fig, axes = plt.subplots(2, (num_models + 1) // 2, figsize=(16, 10))
            if num_models == 1:
                axes = [axes]
            elif num_models <= 2:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            fig.suptitle('Feature Importance Analysis by Model', fontsize=16, fontweight='bold')
            
            for i, result in enumerate(tree_models):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                model_name = result['model_name']
                importance = result['feature_importance']
                
                # Get top 15 features
                top_features = dict(list(importance.items())[:15])
                features = list(top_features.keys())
                importances = list(top_features.values())
                
                # Create horizontal bar chart
                bars = ax.barh(range(len(features)), importances)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'{model_name} - Top 15 Features')
                ax.invert_yaxis()  # Most important features at top
                
                # Add value labels
                for j, (bar, imp) in enumerate(zip(bars, importances)):
                    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                           f'{imp:.2f}', ha='left', va='center', fontsize=8)
                
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(tree_models), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save chart
            output_path = os.path.join(
                self.config.path_config.visualizations_dir,
                'comprehensive_feature_importance.png'
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Feature importance analysis chart saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Feature importance chart creation failed: {e}")
    
    def _create_confusion_matrix_heatmaps(self):
        """Create confusion matrix heatmaps for all models"""
        try:
            num_models = len(self.model_results)
            fig, axes = plt.subplots(2, (num_models + 1) // 2, figsize=(16, 10))
            if num_models == 1:
                axes = [axes]
            elif num_models <= 2:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
            
            for i, result in enumerate(self.model_results):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                model_name = result['model_name']
                confusion_metrics = result.get('confusion_metrics', {})
                
                if 'confusion_matrix' in confusion_metrics:
                    cm = np.array(confusion_metrics['confusion_matrix'])
                    
                    # Create heatmap
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                              xticklabels=['Predicted Negative', 'Predicted Positive'],
                              yticklabels=['Actual Negative', 'Actual Positive'])
                    
                    ax.set_title(f'{model_name} Confusion Matrix')
                    ax.set_xlabel('Predicted Label')
                    ax.set_ylabel('Actual Label')
            
            # Hide unused subplots
            for i in range(len(self.model_results), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save chart
            output_path = os.path.join(
                self.config.path_config.visualizations_dir,
                'confusion_matrices_comparison.png'
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confusion matrices comparison chart saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Confusion matrices chart creation failed: {e}")
    
    def _create_performance_heatmap(self):
        """Create performance metrics heatmap"""
        try:
            # Prepare data for heatmap
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
            model_names = [result['model_name'] for result in self.model_results]
            
            # Create metrics matrix
            metrics_matrix = []
            for result in self.model_results:
                metrics = result['metrics']
                row = [
                    metrics.get('accuracy', 0),
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('f1', 0),
                    metrics.get('roc_auc', 0),
                    metrics.get('pr_auc', 0)
                ]
                metrics_matrix.append(row)
            
            metrics_matrix = np.array(metrics_matrix)
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics_matrix, 
                       xticklabels=metrics_names,
                       yticklabels=model_names,
                       annot=True, 
                       fmt='.2f',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Performance Score'})
            
            plt.title('Model Performance Metrics Heatmap')
            plt.xlabel('Metrics')
            plt.ylabel('Models')
            plt.tight_layout()
            
            # Save chart
            output_path = os.path.join(
                self.config.path_config.visualizations_dir,
                'performance_metrics_heatmap.png'
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance metrics heatmap saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Performance heatmap creation failed: {e}")
    
    def _create_model_ranking_chart(self):
        """Create model ranking chart based on different metrics"""
        try:
            # Prepare ranking data
            metrics_for_ranking = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
            model_names = [result['model_name'] for result in self.model_results]
            
            # Calculate rankings for each metric
            rankings = {}
            for metric in metrics_for_ranking:
                values = [result['metrics'].get(metric.lower().replace('-', '_'), 0) 
                         for result in self.model_results]
                # Rank in descending order (higher is better)
                rankings[metric] = [sorted(values, reverse=True).index(v) + 1 for v in values]
            
            # Create ranking chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(model_names))
            width = 0.15
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, metric in enumerate(metrics_for_ranking):
                ax.bar(x + i * width, rankings[metric], width, 
                      label=metric, color=colors[i], alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Ranking (1 = Best)')
            ax.set_title('Model Rankings by Different Metrics')
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()  # Lower rank number at top
            
            plt.tight_layout()
            
            # Save chart
            output_path = os.path.join(
                self.config.path_config.visualizations_dir,
                'model_rankings.png'
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Model rankings chart saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Model ranking chart creation failed: {e}")
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance across all models"""
        try:
            tree_models = [r for r in self.model_results if r['feature_importance'] is not None]
            
            if not tree_models:
                return {'message': 'No models support feature importance analysis'}
            
            # Collect all features and their importance scores
            all_features = set()
            feature_importance_data = {}
            
            for result in tree_models:
                model_name = result['model_name']
                importance = result['feature_importance']
                
                feature_importance_data[model_name] = importance
                all_features.update(importance.keys())
            
            # Calculate average importance across models
            avg_importance = {}
            for feature in all_features:
                scores = []
                for model_name, importance in feature_importance_data.items():
                    if feature in importance:
                        scores.append(importance[feature])
                
                if scores:
                    avg_importance[feature] = np.mean(scores)
            
            # Sort by average importance
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Create detailed analysis
            analysis = {
                'total_features': len(all_features),
                'models_with_importance': len(tree_models),
                'top_10_features': sorted_features[:10],
                'bottom_10_features': sorted_features[-10:],
                'feature_importance_data': feature_importance_data,
                'average_importance': avg_importance
            }
            
            # Save detailed feature importance analysis (round numeric columns to two decimals)
            feature_analysis_path = os.path.join(
                self.config.path_config.model_evaluation_dir,
                "feature_importance",
                "detailed_feature_importance_analysis.csv"
            )
            
            # Create DataFrame for detailed analysis
            feature_df_data = []
            for feature, avg_imp in sorted_features:
                row = {'Feature': feature, 'Average_Importance': avg_imp}
                
                # Add individual model importance scores
                for model_name, importance in feature_importance_data.items():
                    row[f'{model_name}_Importance'] = importance.get(feature, 0)
                
                feature_df_data.append(row)
            
            feature_df = pd.DataFrame(feature_df_data)
            for col in feature_df.columns:
                if col != 'Feature':
                    feature_df[col] = feature_df[col].astype(float).round(2)
            feature_df.to_csv(feature_analysis_path, index=False)
            
            self.logger.info(f"Detailed feature importance analysis saved to: {feature_analysis_path}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics in detail"""
        try:
            # Extract metrics for all models
            metrics_data = []
            for result in self.model_results:
                model_name = result['model_name']
                metrics = result['metrics']
                confusion_metrics = result.get('confusion_metrics', {})
                
                model_metrics = {
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1', 0),
                    'ROC-AUC': metrics.get('roc_auc', 0),
                    'PR-AUC': metrics.get('pr_auc', 0),
                    'Specificity': confusion_metrics.get('specificity', 0),
                    'False_Positive_Rate': confusion_metrics.get('false_positive_rate', 0),
                    'False_Negative_Rate': confusion_metrics.get('false_negative_rate', 0)
                }
                
                metrics_data.append(model_metrics)
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Calculate statistics
            analysis = {
                'metrics_summary': metrics_df.describe(),
                'best_model_by_metric': {},
                'performance_statistics': {}
            }
            
            # Find best model for each metric
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']:
                if metric in metrics_df.columns:
                    best_idx = metrics_df[metric].idxmax()
                    best_model = metrics_df.loc[best_idx, 'Model']
                    best_score = metrics_df.loc[best_idx, metric]
                    analysis['best_model_by_metric'][metric] = {
                        'model': best_model,
                        'score': best_score
                    }
            
            # Calculate performance statistics
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']:
                if metric in metrics_df.columns:
                    values = metrics_df[metric]
                    analysis['performance_statistics'][metric] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'range': values.max() - values.min()
                    }
            
            # Save detailed performance analysis (round numeric columns to two decimals)
            performance_path = os.path.join(
                self.config.path_config.model_evaluation_dir,
                "performance_metrics",
                "detailed_performance_analysis.csv"
            )
            
            for col in metrics_df.columns:
                if col != 'Model':
                    metrics_df[col] = metrics_df[col].astype(float).round(2)
            metrics_df.to_csv(performance_path, index=False)
            
            self.logger.info(f"Detailed performance analysis saved to: {performance_path}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Performance metrics analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report"""
        try:
            report_lines = []
            
            # Header
            report_lines.append("=" * 100)
            report_lines.append("PFIZER EMR ALERT SYSTEM - COMPREHENSIVE MODEL EVALUATION REPORT")
            report_lines.append("=" * 100)
            report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Executive Summary
            report_lines.append("EXECUTIVE SUMMARY")
            report_lines.append("-" * 50)
            
            if 'comparison_df' in analysis_results and not analysis_results['comparison_df'].empty:
                best_model = analysis_results['comparison_df'].iloc[0]
                report_lines.append(f"Best Performing Model: {best_model['Model']}")
                report_lines.append(f"Composite Score: {best_model['Composite_Score']:.2f}")
                report_lines.append(f"PR-AUC: {best_model['PR-AUC']:.2f}")
                report_lines.append(f"Precision: {best_model['Precision']:.2f}")
                report_lines.append(f"Recall: {best_model['Recall']:.2f}")
                report_lines.append(f"F1-Score: {best_model['F1-Score']:.2f}")
                report_lines.append(f"ROC-AUC: {best_model['ROC-AUC']:.2f}")
                report_lines.append(f"Accuracy: {best_model['Accuracy']:.2f}")
                report_lines.append("")
            
            # Model Performance Comparison
            report_lines.append("MODEL PERFORMANCE COMPARISON")
            report_lines.append("-" * 50)
            
            if 'comparison_df' in analysis_results:
                comparison_df = analysis_results['comparison_df']
                report_lines.append("Ranked by Composite Score (Weighted: Precision 30%, PR-AUC 25%, F1-Score 15%, Recall 15%, ROC-AUC 10%, Accuracy 5%):")
                report_lines.append("")
                
                for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
                    report_lines.append(f"{i}. {row['Model']}")
                    report_lines.append(f"   Composite Score: {row['Composite_Score']:.2f}")
                    report_lines.append(f"   PR-AUC: {row['PR-AUC']:.2f}")
                    report_lines.append(f"   Precision: {row['Precision']:.2f}")
                    report_lines.append(f"   Recall: {row['Recall']:.2f}")
                    report_lines.append(f"   F1-Score: {row['F1-Score']:.2f}")
                    report_lines.append(f"   ROC-AUC: {row['ROC-AUC']:.2f}")
                    report_lines.append(f"   Accuracy: {row['Accuracy']:.2f}")
                    report_lines.append("")
            
            # Feature Importance Analysis
            if 'feature_analysis' in analysis_results:
                feature_analysis = analysis_results['feature_analysis']
                
                if 'top_10_features' in feature_analysis:
                    report_lines.append("FEATURE IMPORTANCE ANALYSIS")
                    report_lines.append("-" * 50)
                    report_lines.append("Top 10 Most Important Features:")
                    report_lines.append("")
                    
                    for i, (feature, importance) in enumerate(feature_analysis['top_10_features'], 1):
                        report_lines.append(f"{i:2d}. {feature}: {importance:.2f}")
                    
                    report_lines.append("")
            
            # Performance Statistics
            if 'performance_analysis' in analysis_results:
                performance_analysis = analysis_results['performance_analysis']
                
                if 'performance_statistics' in performance_analysis:
                    report_lines.append("PERFORMANCE STATISTICS")
                    report_lines.append("-" * 50)
                    
                    for metric, stats in performance_analysis['performance_statistics'].items():
                        report_lines.append(f"{metric}:")
                        report_lines.append(f"  Mean: {stats['mean']:.2f}")
                        report_lines.append(f"  Std:  {stats['std']:.2f}")
                        report_lines.append(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
                        report_lines.append("")
            
            # Best Model by Metric
            if 'performance_analysis' in analysis_results:
                performance_analysis = analysis_results['performance_analysis']
                
                if 'best_model_by_metric' in performance_analysis:
                    report_lines.append("BEST MODEL BY METRIC")
                    report_lines.append("-" * 50)
                    
                    for metric, info in performance_analysis['best_model_by_metric'].items():
                        report_lines.append(f"{metric}: {info['model']} ({info['score']:.2f})")
                    
                    report_lines.append("")
            
            # Configuration Summary
            report_lines.append("CONFIGURATION SUMMARY")
            report_lines.append("-" * 50)
            report_lines.append(f"Test Size: {self.config.test_size}")
            report_lines.append(f"Cross-Validation Folds: {self.config.cv_folds}")
            report_lines.append(f"Use SMOTE: {self.config.use_smote}")
            report_lines.append(f"Primary Metric: {self.config.primary_metric}")
            report_lines.append(f"Total Features Used: {len(self.config.feature_config.production_features)}")
            report_lines.append("")
            
            # Recommendations
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 50)
            
            if 'comparison_df' in analysis_results and not analysis_results['comparison_df'].empty:
                best_model = analysis_results['comparison_df'].iloc[0]
                report_lines.append(f"1. Deploy {best_model['Model']} as the primary model for production use.")
                report_lines.append(f"2. Monitor performance with PR-AUC threshold of {best_model['PR-AUC']:.2f}.")
                report_lines.append(f"3. Focus on Recall ({best_model['Recall']:.2f}) and Precision ({best_model['Precision']:.2f}) for clinical decision support.")
                
                if best_model['PR-AUC'] < 0.7:
                    report_lines.append("4. Consider additional feature engineering or data collection to improve performance.")
                else:
                    report_lines.append("4. Model performance is satisfactory for production deployment.")
                
                report_lines.append("5. Implement continuous monitoring and retraining procedures.")
                report_lines.append("6. Consider ensemble methods combining top-performing models.")
            
            report_lines.append("")
            report_lines.append("=" * 100)
            report_lines.append("END OF REPORT")
            report_lines.append("=" * 100)
            
            report_content = "\n".join(report_lines)
            
            # Save report (already uses .2f formatting in strings above)
            report_path = os.path.join(
                self.config.path_config.model_evaluation_dir,
                "comprehensive_model_evaluation_report.txt"
            )
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"Comprehensive evaluation report saved to: {report_path}")
            
            return report_content
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"
    
    def _save_analysis_summary(self, analysis_results: Dict[str, Any]):
        """Save analysis summary in JSON format"""
        try:
            import json
            
            # Prepare summary data
            summary = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'total_models_evaluated': len(self.model_results),
                'best_model': None,
                'performance_summary': {},
                'feature_analysis_summary': {},
                'configuration': {
                    'test_size': self.config.test_size,
                    'cv_folds': self.config.cv_folds,
                    'use_smote': self.config.use_smote,
                    'primary_metric': self.config.primary_metric
                }
            }
            
            # Add best model information
            if 'comparison_df' in analysis_results and not analysis_results['comparison_df'].empty:
                best_model = analysis_results['comparison_df'].iloc[0]
                summary['best_model'] = {
                    'name': best_model['Model'],
                    'roc_auc': float(best_model['ROC-AUC']),
                    'f1_score': float(best_model['F1-Score']),
                    'accuracy': float(best_model['Accuracy'])
                }
            
            # Add performance summary
            if 'performance_analysis' in analysis_results:
                perf_analysis = analysis_results['performance_analysis']
                if 'performance_statistics' in perf_analysis:
                    summary['performance_summary'] = {
                        metric: {
                            'mean': float(stats['mean']),
                            'std': float(stats['std']),
                            'range': [float(stats['min']), float(stats['max'])]
                        }
                        for metric, stats in perf_analysis['performance_statistics'].items()
                    }
            
            # Add feature analysis summary
            if 'feature_analysis' in analysis_results:
                feature_analysis = analysis_results['feature_analysis']
                if 'top_10_features' in feature_analysis:
                    summary['feature_analysis_summary'] = {
                        'top_features': [
                            {'feature': feature, 'importance': float(importance)}
                            for feature, importance in feature_analysis['top_10_features']
                        ],
                        'total_features': feature_analysis.get('total_features', 0)
                    }
            
            # Save summary (ensure two-decimal precision for numeric values)
            summary_path = os.path.join(
                self.config.path_config.model_evaluation_dir,
                "evaluation_summary.json"
            )
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Analysis summary saved to: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Analysis summary saving failed: {e}")
    
    def run_complete_evaluation(self, dataset_path: Optional[str] = None, 
                              model_types: List[str] = None) -> Dict[str, Any]:
        """
        Run complete comprehensive model evaluation pipeline
        
        Args:
            dataset_path: Path to dataset. If None, uses default path from config.
            model_types: List of model types to evaluate. If None, uses default models.
            
        Returns:
            Dictionary containing all evaluation results
        """
        try:
            self.logger.info("🚀 Starting Comprehensive Model Evaluation Pipeline")
            self.logger.info("=" * 100)
            
            # Step 1: Load and prepare data
            if not self.load_and_prepare_data(dataset_path):
                self.logger.error("Data preparation failed. Exiting.")
                return {}
            
            # Step 2: Train and evaluate models
            if not self.train_and_evaluate_models(model_types):
                self.logger.error("Model training failed. Exiting.")
                return {}
            
            # Step 3: Generate comprehensive analysis
            analysis_results = self.generate_comprehensive_analysis()
            
            # Final summary
            self.logger.info("=" * 100)
            self.logger.info("🎉 COMPREHENSIVE MODEL EVALUATION COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 100)
            
            if 'comparison_df' in analysis_results and not analysis_results['comparison_df'].empty:
                best_model = analysis_results['comparison_df'].iloc[0]
                self.logger.info(f"🏆 Best Model (Composite Score): {best_model['Model']}")
                self.logger.info(f"🎯 Composite Score: {best_model['Composite_Score']:.2f}")
                self.logger.info(f"📊 PR-AUC: {best_model['PR-AUC']:.2f}")
                self.logger.info(f"🎯 Precision: {best_model['Precision']:.2f}")
                self.logger.info(f"📈 Recall: {best_model['Recall']:.2f}")
                self.logger.info(f"⚖️ F1-Score: {best_model['F1-Score']:.2f}")
                self.logger.info(f"📊 ROC-AUC: {best_model['ROC-AUC']:.2f}")
                self.logger.info(f"🎯 Accuracy: {best_model['Accuracy']:.2f}")
            
            self.logger.info(f"📁 All results saved to: {self.config.path_config.model_evaluation_dir}")
            self.logger.info(f"📊 Visualizations saved to: {self.config.path_config.visualizations_dir}")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive evaluation pipeline failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}


def main():
    """Main function for running comprehensive model evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive Model Evaluation for Pfizer EMR Alert System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python comprehensive_model_evaluation.py
  
  # Run with custom dataset
  python comprehensive_model_evaluation.py --dataset data/model_ready/model_ready_dataset.csv
  
  # Run with specific models
  python comprehensive_model_evaluation.py --models xgboost random_forest logistic_regression
  
  # Run with custom configuration
  python comprehensive_model_evaluation.py --config config/custom_config.yaml
        """
    )
    
    parser.add_argument('--dataset', type=str, 
                       help='Path to dataset file')
    parser.add_argument('--models', nargs='+', 
                       default=['xgboost', 'random_forest', 'logistic_regression', 
                               'gradient_boosting', 'svm', 'naive_bayes'],
                       help='List of models to evaluate')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Create evaluator
        evaluator = ComprehensiveModelEvaluator()
        
        # Run comprehensive evaluation
        results = evaluator.run_complete_evaluation(
            dataset_path=args.dataset,
            model_types=args.models
        )
        
        if results:
            print("\n✅ Comprehensive model evaluation completed successfully!")
            print(f"📁 Results saved to: {evaluator.config.path_config.model_evaluation_dir}")
        else:
            print("\n❌ Comprehensive model evaluation failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
