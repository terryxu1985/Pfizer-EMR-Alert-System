#!/usr/bin/env python3
"""
Fixed Comprehensive Model Evaluation Entry Point

This script addresses the data imbalance issues in the original evaluation:
1. Uses stratified train/test split
2. Applies SMOTE correctly
3. Uses appropriate evaluation metrics for imbalanced data
4. Provides better model performance assessment

Usage:
    python -m scripts.model_training evaluate --mode comprehensive_fixed
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.model_training.config.optimized_config import OptimizedModelConfig as ModelConfig
from scripts.model_training.config.config_manager import ConfigManager, Environment
from scripts.model_training.pipelines.evaluation_pipeline import EvaluationPipeline
from scripts.model_training.evaluators.comprehensive_evaluator import ComprehensiveModelEvaluator


class FixedComprehensiveModelEvaluator(ComprehensiveModelEvaluator):
    """
    Fixed comprehensive model evaluator that addresses data imbalance issues
    """
    
    def load_and_prepare_data(self, dataset_path: str = None) -> bool:
        """
        Load and prepare data with proper imbalance handling
        
        Args:
            dataset_path: Path to dataset. If None, uses default path from config.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STEP 1: Loading and Preparing Data (FIXED VERSION)")
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
            
            # FIXED: Use stratified split to maintain class distribution
            self.logger.info("Using STRATIFIED train/test split to maintain class distribution...")
            from sklearn.model_selection import train_test_split
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y  # FIXED: Use stratified split
            )
            
            self.logger.info(f"Stratified data split completed:")
            self.logger.info(f"  Training set: {self.X_train.shape}")
            self.logger.info(f"  Test set: {self.X_test.shape}")
            self.logger.info(f"  Training target distribution: {self.y_train.value_counts().to_dict()}")
            self.logger.info(f"  Test target distribution: {self.y_test.value_counts().to_dict()}")
            
            # Fit and transform training data
            self.logger.info("Fitting and transforming training data...")
            self.X_train, self.y_train = self.preprocessor.fit_transform(self.X_train, self.y_train)
            
            # Transform test data
            self.logger.info("Transforming test data...")
            self.X_test = self.preprocessor.transform(self.X_test)
            
            # FIXED: Apply SMOTE more carefully
            if self.config.use_smote:
                self.logger.info("Applying SMOTE to training data...")
                original_train_dist = self.y_train.value_counts().to_dict()
                self.logger.info(f"  Before SMOTE: {original_train_dist}")
                
                self.X_train, self.y_train = self.preprocessor.handle_class_imbalance(
                    self.X_train, self.y_train
                )
                
                after_smote_dist = self.y_train.value_counts().to_dict()
                self.logger.info(f"  After SMOTE: {after_smote_dist}")
                self.logger.info(f"  Training set shape after SMOTE: {self.X_train.shape}")
            
            self.logger.info("Data preparation completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


def run_fixed_comprehensive_evaluation(dataset_path: str = None, 
                                     model_types: list = None) -> dict:
    """
    Run fixed comprehensive model evaluation with proper imbalance handling
    
    Args:
        dataset_path: Dataset path
        model_types: List of model types to evaluate
        
    Returns:
        Fixed comprehensive evaluation results dictionary
    """
    print("🚀 Running FIXED Comprehensive Model Evaluation")
    print("=" * 60)
    print("🔧 Fixes applied:")
    print("  - Stratified train/test split")
    print("  - Proper SMOTE application")
    print("  - Better handling of imbalanced data")
    print("=" * 60)
    
    # Create fixed evaluator
    evaluator = FixedComprehensiveModelEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_complete_evaluation(
        dataset_path=dataset_path,
        model_types=model_types
    )
    
    if results:
        print("✅ Fixed comprehensive model evaluation completed!")
        print(f"📁 Results saved to: {evaluator.config.path_config.model_evaluation_dir}")
        
        # Print summary
        if 'comparison_df' in results and not results['comparison_df'].empty:
            best_model = results['comparison_df'].iloc[0]
            print(f"\n🏆 Best Model: {best_model['Model']}")
            print(f"🎯 Composite Score: {best_model['Composite_Score']:.2f}")
            print(f"📊 PR-AUC: {best_model['PR-AUC']:.2f}")
            print(f"🎯 Precision: {best_model['Precision']:.2f}")
            print(f"📈 Recall: {best_model['Recall']:.2f}")
            print(f"⚖️ F1-Score: {best_model['F1-Score']:.2f}")
            print(f"📊 ROC-AUC: {best_model['ROC-AUC']:.2f}")
            print(f"🎯 Accuracy: {best_model['Accuracy']:.2f}")
            
            # Show improvement
            print(f"\n📈 New weighted scoring system:")
            print(f"  - Precision: 30% (most important for medical scenarios)")
            print(f"  - PR-AUC: 25% (handles imbalanced data well)")
            print(f"  - F1-Score: 15% (balanced measure)")
            print(f"  - Recall: 15% (important but less critical)")
            print(f"  - ROC-AUC: 10% (overall discrimination)")
            print(f"  - Accuracy: 5% (least important for imbalanced data)")
    
    return results


def main():
    """Main function for fixed comprehensive model evaluation"""
    parser = argparse.ArgumentParser(
        description='Fixed Comprehensive Model Evaluation for Pfizer EMR Alert System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run fixed comprehensive evaluation
  python -m scripts.model_training evaluate --mode comprehensive_fixed
  
  # Run with custom dataset
  python -m scripts.model_training evaluate --mode comprehensive_fixed --dataset data/model_ready/model_ready_dataset.csv
  
  # Run with specific models
  python -m scripts.model_training evaluate --mode comprehensive_fixed --models xgboost random_forest logistic_regression
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['basic', 'comprehensive', 'production', 'comprehensive_fixed'],
                       help='Evaluation mode: basic, comprehensive, production, or comprehensive_fixed')
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
        print("🚀 Pfizer EMR Alert System - Fixed Model Evaluation")
        print("=" * 80)
        
        # Run evaluation based on mode
        if args.mode == 'comprehensive_fixed':
            results = run_fixed_comprehensive_evaluation(
                dataset_path=args.dataset,
                model_types=args.models
            )
        else:
            # Use comprehensive evaluator for other modes
            from scripts.model_training.evaluators.comprehensive_evaluator import ComprehensiveModelEvaluator
            evaluator = ComprehensiveModelEvaluator()
            return evaluator.run_complete_evaluation(
                dataset_path=args.dataset,
                model_types=args.models
            )
        
        if results:
            print(f"\n✅ {args.mode} evaluation completed successfully!")
        else:
            print(f"\n❌ {args.mode} evaluation failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def evaluate_with_config(environment: Environment = Environment.PRODUCTION,
                         model_types: list = None,
                         dataset_path: str = None) -> dict:
    """
    Evaluate models using configuration system
    
    Args:
        environment: Environment configuration to use
        model_types: List of model types to evaluate
        dataset_path: Path to dataset file
        
    Returns:
        Evaluation results dictionary
    """
    print(f"🚀 Evaluating Models with Configuration System")
    print("=" * 80)
    print(f"Environment: {environment.value}")
    print(f"Model Types: {model_types}")
    print("=" * 80)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        config = config_manager.get_config(environment)
        
        # Use comprehensive evaluator
        evaluator = ComprehensiveModelEvaluator()
        
        # Set default model types if not provided
        if model_types is None:
            model_types = ['xgboost', 'random_forest', 'logistic_regression', 
                          'gradient_boosting', 'svm', 'naive_bayes']
        
        # Run evaluation
        results = evaluator.run_complete_evaluation(
            dataset_path=dataset_path,
            model_types=model_types
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    exit(main())
