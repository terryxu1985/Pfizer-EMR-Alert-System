#!/usr/bin/env python3
"""
Configuration-based Model Training Script

This script uses the configuration system to train models with proper settings
and addresses the data imbalance issues identified earlier.

Usage:
    python -m scripts.model_training train --environment production
    python -m scripts.model_training train --environment development --model-type xgboost
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.model_training.config.config_manager import ConfigManager, Environment
from scripts.model_training.config.optimized_config import ModelType
from scripts.model_training.pipelines.training_pipeline import TrainingPipeline
from scripts.model_training.pipelines.evaluation_pipeline import EvaluationPipeline


def train_with_config(environment: Environment = Environment.PRODUCTION, 
                     model_type: str = 'xgboost',
                     dataset_path: str = None) -> dict:
    """
    Train model using configuration system
    
    Args:
        environment: Environment configuration to use
        model_type: Type of model to train
        dataset_path: Path to dataset file
        
    Returns:
        Training results dictionary
    """
    print(f"🚀 Training Model with Configuration System")
    print("=" * 80)
    print(f"Environment: {environment.value}")
    print(f"Model Type: {model_type}")
    print("=" * 80)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Get configuration for specified environment
        print(f"Loading {environment.value} configuration...")
        config = config_manager.get_config(environment)
        
        # Override model type if specified
        if model_type is not None and model_type != config.model_type:
            print(f"Overriding model type: {config.model_type} -> {model_type}")
            config.model_type = model_type
        
        # Validate configuration
        print("Validating configuration...")
        validation_results = config_manager.validate_config(config)
        
        if not validation_results['valid']:
            print("❌ Configuration validation failed:")
            for error in validation_results['errors']:
                print(f"  - {error}")
            return {}
        
        if validation_results['warnings']:
            print("⚠️  Configuration warnings:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
        
        print("✅ Configuration validation passed")
        
        # Display configuration summary
        print(f"\n📋 Configuration Summary:")
        print(f"  Model Type: {config.model_type}")
        print(f"  Model Version: {config.model_version}")
        print(f"  Test Size: {config.test_size}")
        print(f"  Use SMOTE: {config.use_smote}")
        print(f"  CV Folds: {config.cv_folds}")
        print(f"  Primary Metric: {config.primary_metric}")
        print(f"  Production Features: {len(config.feature_config.production_features)}")
        print(f"  Features to Remove: {len(config.feature_config.get_features_to_remove())}")
        
        # Get hyperparameters for the model
        hyperparams = config.get_model_hyperparameters(config.model_type)
        print(f"\n🔧 {config.model_type.upper()} Hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
        
        # Create training pipeline
        print(f"\n🏗️  Creating training pipeline...")
        pipeline = TrainingPipeline(config)
        
        # Run training
        print(f"\n🚀 Starting model training...")
        results = pipeline.run_complete_training(dataset_path=dataset_path)
        
        if results:
            print(f"\n✅ Model training completed successfully!")
            
            # Display training results
            evaluation_results = results.get('evaluation_results', {})
            metrics = evaluation_results.get('metrics', {})
            
            print(f"\n📊 Training Results:")
            print(f"  Model: {pipeline.trainer._get_model_name()}")
            print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A'):.2f}")
            print(f"  F1-Score: {metrics.get('f1', 'N/A'):.2f}")
            print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.2f}")
            print(f"  Precision: {metrics.get('precision', 'N/A'):.2f}")
            print(f"  Recall: {metrics.get('recall', 'N/A'):.2f}")
            
            # Cross-validation results
            cv_results = evaluation_results.get('cv_results', {})
            if cv_results:
                print(f"\n📈 Cross-Validation Results:")
                print(f"  CV ROC-AUC Mean: {cv_results.get('cv_mean', 'N/A'):.2f}")
                print(f"  CV ROC-AUC Std: {cv_results.get('cv_std', 'N/A'):.2f}")
            
            # Data information
            print(f"\n📋 Data Information:")
            print(f"  Training Samples: {len(pipeline.X_train)}")
            print(f"  Test Samples: {len(pipeline.X_test)}")
            print(f"  Features Used: {len(pipeline.preprocessor.feature_columns)}")
            print(f"  Training Target Distribution: {pipeline.y_train.value_counts().to_dict()}")
            print(f"  Test Target Distribution: {pipeline.y_test.value_counts().to_dict()}")
            
            return results
        else:
            print(f"\n❌ Model training failed!")
            return {}
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return {}


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
    print(f"🔍 Evaluating Models with Configuration System")
    print("=" * 80)
    print(f"Environment: {environment.value}")
    print(f"Model Types: {model_types or 'All'}")
    print("=" * 80)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Get configuration for specified environment
        print(f"Loading {environment.value} configuration...")
        config = config_manager.get_config(environment)
        
        # Default model types
        if model_types is None:
            model_types = ['xgboost', 'random_forest', 'logistic_regression', 
                          'gradient_boosting', 'svm', 'naive_bayes']
        
        print(f"Models to evaluate: {model_types}")
        
        # Create evaluation pipeline
        print(f"\n🏗️  Creating evaluation pipeline...")
        pipeline = EvaluationPipeline(config)
        
        # Run evaluation
        print(f"\n🚀 Starting model evaluation...")
        results = pipeline.run_complete_evaluation(
            dataset_path=dataset_path,
            model_types=model_types
        )
        
        if results:
            print(f"\n✅ Model evaluation completed successfully!")
            
            # Display evaluation results
            if 'best_model' in results and results['best_model']:
                best_model = results['best_model']
                print(f"\n🏆 Best Model (Selected by PR-AUC > Recall > Precision > F1-Score):")
                print(f"  Model: {best_model['Model']}")
                print(f"  PR-AUC: {best_model['PR-AUC']:.2f}")
                print(f"  Recall: {best_model['Recall']:.2f}")
                print(f"  Precision: {best_model['Precision']:.2f}")
                print(f"  F1-Score: {best_model['F1-Score']:.2f}")
                print(f"  ROC-AUC: {best_model['ROC-AUC']:.2f}")
                print(f"  Accuracy: {best_model['Accuracy']:.2f}")
            
            return results
        else:
            print(f"\n❌ Model evaluation failed!")
            return {}
            
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """Main function for configuration-based training"""
    parser = argparse.ArgumentParser(
        description='Configuration-based Model Training for Pfizer EMR Alert System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XGBoost model with production configuration
  python -m scripts.model_training train --environment production --model-type xgboost
  
  # Train with development configuration (faster)
  python -m scripts.model_training train --environment development
  
  # Evaluate all models with production configuration
  python -m scripts.model_training evaluate --environment production
  
  # Train with custom dataset
  python -m scripts.model_training train --dataset data/model_ready/model_ready_dataset.csv
        """
    )
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate'],
                       help='Mode: train single model or evaluate multiple models')
    parser.add_argument('--environment', type=str, default='production',
                       choices=['development', 'testing', 'staging', 'production'],
                       help='Environment configuration to use')
    parser.add_argument('--model-type', type=str, default=None,
                       choices=['xgboost', 'random_forest', 'logistic_regression', 
                               'gradient_boosting', 'svm', 'naive_bayes'],
                       help='Model type to train (only for train mode). If not specified, uses config default.')
    parser.add_argument('--models', nargs='+',
                       default=['xgboost', 'random_forest', 'logistic_regression', 
                               'gradient_boosting', 'svm', 'naive_bayes'],
                       help='List of models to evaluate (only for evaluate mode)')
    parser.add_argument('--dataset', type=str,
                       help='Path to dataset file')
    
    args = parser.parse_args()
    
    try:
        # Convert environment string to enum
        environment = Environment(args.environment)
        
        if args.mode == 'train':
            # Train single model
            results = train_with_config(
                environment=environment,
                model_type=args.model_type,
                dataset_path=args.dataset
            )
        else:
            # Evaluate multiple models
            results = evaluate_with_config(
                environment=environment,
                model_types=args.models,
                dataset_path=args.dataset
            )
        
        if results:
            print(f"\n🎉 {args.mode.title()} completed successfully!")
            return 0
        else:
            print(f"\n❌ {args.mode.title()} failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
