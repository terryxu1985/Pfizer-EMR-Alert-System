"""
Evaluation pipeline for model comparison and assessment
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from ..config.optimized_config import OptimizedModelConfig as ModelConfig
from ..utils.data_loader import DataLoader
from ..utils.preprocessor import DataPreprocessor
from ..core.model_factory import ModelFactory
from ..core.evaluator import ModelEvaluator


class EvaluationPipeline:
    """Evaluation pipeline"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize evaluation pipeline
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.preprocessor = DataPreprocessor(config)
        self.evaluator = ModelEvaluator(config)
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Results
        self.model_results = []
        self.comparison_results = None
    
    def load_and_prepare_data(self, dataset_path: Optional[str] = None):
        """
        Load and prepare data
        
        Args:
            dataset_path: Dataset path
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 1: Load and prepare data")
        self.logger.info("=" * 60)
        
        # Load data
        df = self.data_loader.load_dataset(dataset_path)
        
        # Validate data
        if not self.data_loader.validate_dataset():
            raise ValueError("Dataset validation failed")
        
        # Get production environment dataset with target inversion if configured
        # Target encoding: TARGET=1="Not Prescribed Drug A" (Alert Required), TARGET=0="Prescribed Drug A" (No Alert)
        invert_target = getattr(self.config, 'invert_target', False)
        X, y = self.data_loader.get_clean_dataset(invert_target=invert_target)
        
        self.logger.info(f"Target inversion: {invert_target}")
        if invert_target:
            self.logger.info("🎯 Target variable inverted")
            self.logger.info("⚠️  Warning: Inverted target may not align with business logic")
        else:
            self.logger.info("🎯 Using original target encoding (recommended)")
            self.logger.info("✅ TARGET=1 → 'Not Prescribed Drug A' → Alert Required (Positive Class)")
            self.logger.info("✅ TARGET=0 → 'Prescribed Drug A' → No Alert Required (Negative Class)")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.split_data(X, y)
        
        # Fit and transform training data
        self.X_train, self.y_train = self.preprocessor.fit_transform(self.X_train, self.y_train)
        
        # Transform test data
        self.X_test = self.preprocessor.transform(self.X_test)
        
        # Handle class imbalance
        self.X_train, self.y_train = self.preprocessor.handle_class_imbalance(
            self.X_train, self.y_train
        )
        
        self.logger.info(f"Data preparation completed")
        self.logger.info(f"Training set shape: {self.X_train.shape}")
        self.logger.info(f"Test set shape: {self.X_test.shape}")
    
    def train_and_evaluate_models(self, model_types: List[str]) -> List[Dict[str, Any]]:
        """
        Train and evaluate multiple models
        
        Args:
            model_types: List of model types to evaluate
            
        Returns:
            List of model results
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 2: Train and evaluate multiple models")
        self.logger.info("=" * 60)
        
        self.model_results = []
        
        for model_type in model_types:
            self.logger.info(f"\nTrain model: {model_type}")
            
            try:
                # Create trainer
                trainer = ModelFactory.create_trainer(self.config, model_type)
                trainer.feature_columns = self.preprocessor.feature_columns
                
                # Train model
                trainer.train(self.X_train, self.y_train)
                
                # Evaluate model
                result = self.evaluator.evaluate_single_model(trainer, self.X_test, self.y_test)
                
                self.model_results.append(result)
                
                self.logger.info(f"✅ {model_type} Training and evaluation completed")
                
            except Exception as e:
                self.logger.error(f"❌ {model_type} Training failed: {e}")
                continue
        
        self.logger.info(f"\nSuccessfully trained and evaluated {len(self.model_results)} models")
        
        return self.model_results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare model performance
        
        Returns:
            Comparison results DataFrame
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 3: Compare model performance")
        self.logger.info("=" * 60)
        
        if not self.model_results:
            raise ValueError("No model results available for comparison, please call train_and_evaluate_models() first")
        
        # Compare models
        self.comparison_results = self.evaluator.compare_models(self.model_results)
        
        self.logger.info("Model comparison completed")
        
        return self.comparison_results
    
    def create_visualizations(self):
        """Create visualization charts"""
        self.logger.info("=" * 60)
        self.logger.info("Step 4: Create visualization charts")
        self.logger.info("=" * 60)
        
        if not self.model_results:
            raise ValueError("No model results available for visualization, please call train_and_evaluate_models() first")
        
        # Create charts
        self.evaluator.create_visualizations(self.model_results)
        
        self.logger.info("Visualization charts creation completed")
    
    def generate_report(self) -> str:
        """
        Generate evaluation report
        
        Returns:
            Report content
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 5: Generate evaluation report")
        self.logger.info("=" * 60)
        
        if not self.model_results:
            raise ValueError("No model results available for report, please call train_and_evaluate_models() first")
        
        # Generate report
        report_content = self.evaluator.generate_detailed_report(self.model_results)
        
        self.logger.info("Evaluation report generation completed")
        
        return report_content
    
    def save_results(self) -> pd.DataFrame:
        """
        Save all results
        
        Returns:
            Detailed results DataFrame
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 6: Save results")
        self.logger.info("=" * 60)
        
        if not self.model_results:
            raise ValueError("No model results available for saving, please call train_and_evaluate_models() first")
        
        # Save results
        detailed_results = self.evaluator.save_results(self.model_results)
        
        self.logger.info("Results saving completed")
        
        return detailed_results
    
    def run_complete_evaluation(self, dataset_path: Optional[str] = None,
                              model_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline
        
        Args:
            dataset_path: Dataset path
            model_types: List of model types to evaluate
            
        Returns:
            Complete evaluation results
        """
        self.logger.info("🔍 Starting complete model evaluation pipeline")
        self.logger.info("=" * 80)
        
        # Default model types
        if model_types is None:
            model_types = ['xgboost', 'random_forest', 'logistic_regression', 
                          'gradient_boosting', 'svm', 'naive_bayes']
        
        try:
            # 1. Load and prepare data
            self.load_and_prepare_data(dataset_path)
            
            # 2. Train and evaluate models
            self.train_and_evaluate_models(model_types)
            
            # 3. Compare models
            comparison_df = self.compare_models()
            
            # 4. Create visualizations
            self.create_visualizations()
            
            # 5. Generate report
            report_content = self.generate_report()
            
            # 6. Save results
            detailed_results = self.save_results()
            
            # Summarize results
            complete_results = {
                'model_results': self.model_results,
                'comparison_results': comparison_df,
                'detailed_results': detailed_results,
                'report_content': report_content,
                'best_model': comparison_df.iloc[0].to_dict() if len(comparison_df) > 0 else None,
                'evaluation_summary': {
                    'total_models': len(self.model_results),
                    'successful_models': len(self.model_results),
                    'best_pr_auc': comparison_df.iloc[0]['PR-AUC'] if len(comparison_df) > 0 else 0,
                    'best_recall': comparison_df.iloc[0]['Recall'] if len(comparison_df) > 0 else 0,
                    'best_precision': comparison_df.iloc[0]['Precision'] if len(comparison_df) > 0 else 0,
                    'best_f1_score': comparison_df.iloc[0]['F1-Score'] if len(comparison_df) > 0 else 0,
                    'best_model_name': comparison_df.iloc[0]['Model'] if len(comparison_df) > 0 else None
                }
            }
            
            self.logger.info("🎉 Complete model evaluation pipeline successfully completed!")
            self.logger.info("=" * 80)
            
            return complete_results
            
        except Exception as e:
            self.logger.error(f"Evaluation pipeline failed: {e}")
            raise
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get evaluation summary
        
        Returns:
            Evaluation summary dictionary
        """
        summary = {
            'config': {
                'test_size': self.config.test_size,
                'use_smote': self.config.use_smote,
                'cv_folds': self.config.cv_folds
            },
            'data_status': {
                'data_loaded': self.data_loader.df is not None,
                'data_prepared': self.X_train is not None,
                'train_shape': self.X_train.shape if self.X_train is not None else None,
                'test_shape': self.X_test.shape if self.X_test is not None else None
            },
            'evaluation_status': {
                'models_evaluated': len(self.model_results),
                'comparison_completed': self.comparison_results is not None,
                'best_model': self.comparison_results.iloc[0]['Model'] if self.comparison_results is not None and len(self.comparison_results) > 0 else None,
                'selection_criteria': 'PR-AUC > Recall > Precision > F1-Score'
            }
        }
        
        return summary
