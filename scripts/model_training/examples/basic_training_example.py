"""
Example usage of the new modular model training system
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.model_training import ModelConfig, TrainingPipeline, EvaluationPipeline, ModelFactory


def example_training():
    """Example of training a single model"""
    print("=" * 60)
    print("Example 1: Training a single model")
    print("=" * 60)
    
    # Create configuration
    config = ModelConfig()
    config.model_type = 'xgboost'
    
    # Create training pipeline
    pipeline = TrainingPipeline(config)
    
    # Run training
    results = pipeline.run_complete_training()
    
    print(f"Training completed! Best ROC-AUC: {results['evaluation_results']['metrics'].get('roc_auc', 'N/A')}")
    
    return results


def example_comparison():
    """Model comparison example"""
    print("=" * 60)
    print("Example 2: Comparing multiple models")
    print("=" * 60)
    
    # Create configuration
    config = ModelConfig()
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(config)
    
    # Compare multiple models
    model_types = ['xgboost', 'random_forest', 'logistic_regression']
    results = pipeline.run_complete_evaluation(model_types=model_types)
    
    print(f"Comparison completed! Best model: {results['best_model']['Model']}")
    print(f"Best ROC-AUC: {results['best_model']['ROC-AUC']:.2f}")
    
    return results


def example_production_evaluation():
    """Production environment evaluation example (without data leakage)"""
    print("=" * 60)
    print("Example 3: Production environment model evaluation (without data leakage)")
    print("=" * 60)
    
    # Create configuration
    config = ModelConfig()
    
    # Ensure using production features
    print("Data leakage protection configuration:")
    print(f"  Features to remove: {config.feature_config.get_features_to_remove()}")
    print(f"  Features to keep: {config.feature_config.production_features}")
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(config)
    
    # Run clean evaluation
    model_types = ['xgboost', 'random_forest', 'logistic_regression']
    results = pipeline.run_complete_evaluation(model_types=model_types)
    
    print(f"Production environment evaluation completed! Best model: {results['best_model']['Model']}")
    print(f"Best ROC-AUC: {results['best_model']['ROC-AUC']:.2f}")
    print("This is the real performance after removing data leakage!")
    
    return results


def example_custom_config():
    """Custom configuration example"""
    print("=" * 60)
    print("Example 4: Custom configuration")
    print("=" * 60)
    
    # Create configuration
    config = ModelConfig()
    
    # Custom hyperparameters
    config.update_hyperparameters('xgboost', {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6
    })
    
    # Custom feature configuration
    config.feature_config.production_features = [
        'PATIENT_AGE', 'PATIENT_GENDER', 'RISK_IMMUNO', 'RISK_CVD'
    ]
    
    print("Custom configuration:")
    print(f"  XGBoost parameters: {config.get_model_hyperparameters('xgboost')}")
    print(f"  Feature count: {len(config.feature_config.production_features)}")
    
    # Create training pipeline
    pipeline = TrainingPipeline(config)
    
    # Run training
    results = pipeline.run_complete_training()
    
    return results


def example_model_factory():
    """Model factory example"""
    print("=" * 60)
    print("Example 5: Model factory")
    print("=" * 60)
    
    # Create configuration
    config = ModelConfig()
    
    # Get available models
    available_models = ModelFactory.get_available_models()
    print(f"Available models: {available_models}")
    
    # Create trainers for different models
    for model_type in ['xgboost', 'random_forest', 'logistic_regression']:
        if ModelFactory.is_model_supported(model_type):
            trainer = ModelFactory.create_trainer(config, model_type)
            print(f"Created {model_type} trainer: {trainer._get_model_name()}")


def main():
    """Main function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 New modular model training system usage examples")
    print("=" * 80)
    
    try:
        # Run examples
        example_model_factory()
        example_custom_config()
        
        # Note: The following examples require actual data files
        # example_training()
        # example_comparison()
        # example_production_evaluation()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
