# Model Training System Guide

## 📋 Overview

The Pfizer EMR Alert System's model training framework provides a comprehensive, modular, and production-ready machine learning pipeline for training, evaluating, and deploying predictive models. The system supports multiple algorithms, automated hyperparameter optimization, and dynamic model selection.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Model Training System                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Pipelines     │  │   Trainers      │  │   Config    │ │
│  │                 │  │                 │  │             │ │
│  │ • Training      │  │ • XGBoost       │  │ • Manager   │ │
│  │ • Evaluation    │  │ • Random Forest │  │ • YAML      │ │
│  │ • Model Select  │  │ • Logistic Reg  │  │ • Validation│ │
│  │                 │  │ • Gradient Boost│  │             │ │
│  │                 │  │ • SVM           │  │             │ │
│  │                 │  │ • Naive Bayes   │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Utilities     │  │   Hyperparam    │  │   Core      │ │
│  │                 │  │   Tuning        │  │             │ │
│  │ • Data Loader   │  │ • Optuna        │  │ • Factory   │ │
│  │ • Preprocessor  │  │ • Bayesian Opt  │  │ • Evaluator │ │
│  │ • Metrics       │  │ • Grid Search   │  │ • Base      │ │
│  │ • Feature Sel   │  │ • Auto Config   │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
scripts/model_training/
├── __init__.py                    # Module exports
├── __main__.py                    # CLI entry point
├── train.py                       # Main training script
├── model_selection.py             # Dynamic model selection
├── evaluate.py                    # Model evaluation script
├── select_scheduled.py            # Scheduled model selection
│
├── config/                        # Configuration system
│   ├── __init__.py
│   ├── config_manager.py          # Configuration management
│   ├── optimized_config.py        # Configuration structure
│   ├── environment_config.yaml    # Environment-specific configs
│   ├── CONFIG_GUIDE.md            # Configuration documentation
│   └── README.md
│
├── core/                          # Core framework components
│   ├── __init__.py
│   ├── base_trainer.py            # Abstract trainer base class
│   ├── model_factory.py           # Model creation factory
│   └── evaluator.py               # Model evaluation framework
│
├── trainers/                      # Model-specific trainers
│   ├── __init__.py
│   ├── xgboost_trainer.py         # XGBoost implementation
│   ├── random_forest_trainer.py  # Random Forest implementation
│   ├── logistic_regression_trainer.py
│   ├── gradient_boosting_trainer.py
│   ├── svm_trainer.py
│   └── naive_bayes_trainer.py
│
├── pipelines/                     # Training workflows
│   ├── __init__.py
│   ├── training_pipeline.py      # Complete training workflow
│   └── evaluation_pipeline.py    # Model evaluation workflow
│
├── evaluators/                    # Evaluation components
│   ├── __init__.py
│   └── comprehensive_evaluator.py # Comprehensive evaluation
│
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessor.py            # Data preprocessing
│   ├── metrics.py                 # Evaluation metrics
│   └── feature_selector.py        # Feature selection utilities
│
├── hyperparameter_tuning/         # Hyperparameter optimization
│   ├── __init__.py
│   ├── tuner.py                   # Optuna-based tuning
│   └── HYPERPARAMETER_TUNING_GUIDE.md
│
├── examples/                      # Usage examples
│   ├── basic_training_example.py  # Basic training examples
│   └── production_training_example.py
│
└── deployment/                    # Deployment utilities
    └── cron_jobs.txt              # Scheduled job configurations
```

## 🚀 Quick Start

### Basic Training

```python
from scripts.model_training.config.config_manager import ConfigManager, Environment
from scripts.model_training.pipelines.training_pipeline import TrainingPipeline

# Load configuration
config_manager = ConfigManager()
config = config_manager.get_config(Environment.PRODUCTION)

# Create training pipeline
pipeline = TrainingPipeline(config)

# Run complete training workflow
results = pipeline.run_complete_training()

print(f"Training completed! ROC-AUC: {results['evaluation_results']['metrics']['roc_auc']:.3f}")
```

### Model Comparison

```python
from scripts.model_training.pipelines.evaluation_pipeline import EvaluationPipeline

# Create evaluation pipeline
pipeline = EvaluationPipeline(config)

# Compare multiple models
model_types = ['xgboost', 'random_forest', 'logistic_regression', 'gradient_boosting']
results = pipeline.run_complete_evaluation(model_types=model_types)

# Get best model
best_model = results['best_model']
print(f"Best model: {best_model['Model']}")
print(f"Best PR-AUC: {best_model['PR-AUC']:.3f}")
```

### Command Line Usage

```bash
# Train single model with production configuration
python -m scripts.model_training train --environment production --model-type xgboost

# Evaluate all models
python -m scripts.model_training evaluate --environment production

# Train with custom dataset
python -m scripts.model_training train --dataset data/model_ready/model_ready_dataset.csv

# Run dynamic model selection
python scripts/model_training/model_selection.py
```

## 🔧 Core Components

### 1. Configuration System

The configuration system provides environment-specific settings and hyperparameters:

**Key Features:**
- Multi-environment support (development, testing, staging, production)
- YAML-based configuration with Python structure
- Automatic validation and error checking
- Hyperparameter management for all model types

**Usage:**
```python
from scripts.model_training.config.config_manager import ConfigManager, Environment

config_manager = ConfigManager()
config = config_manager.get_config(Environment.PRODUCTION)

# Access configuration
print(f"Model Type: {config.model_type}")
print(f"Test Size: {config.test_size}")
print(f"CV Folds: {config.cv_folds}")

# Get model hyperparameters
hyperparams = config.get_model_hyperparameters('xgboost')
print(f"XGBoost params: {hyperparams}")
```

### 2. Training Pipelines

#### TrainingPipeline
Complete training workflow including data loading, preprocessing, training, and evaluation.

**Key Methods:**
- `run_complete_training()`: Execute full training workflow
- `load_and_preprocess_data()`: Load and preprocess dataset
- `train_model()`: Train the specified model
- `evaluate_model()`: Evaluate trained model

**Example:**
```python
pipeline = TrainingPipeline(config)

# Run complete training
results = pipeline.run_complete_training(dataset_path="data/model_ready_dataset.csv")

# Access results
metrics = results['evaluation_results']['metrics']
cv_results = results['evaluation_results']['cv_results']
```

#### EvaluationPipeline
Comprehensive model evaluation and comparison system.

**Key Methods:**
- `run_complete_evaluation()`: Evaluate multiple models
- `compare_models()`: Compare model performance
- `select_best_model()`: Select best performing model

**Example:**
```python
pipeline = EvaluationPipeline(config)

# Evaluate multiple models
results = pipeline.run_complete_evaluation(
    model_types=['xgboost', 'random_forest', 'logistic_regression']
)

# Get comparison results
comparison_df = results['comparison_df']
best_model = results['best_model']
```

### 3. Model Trainers

Each model type has a dedicated trainer implementing the `BaseTrainer` interface:

#### Available Trainers

| Trainer | Model | Key Features |
|---------|-------|--------------|
| `XGBoostTrainer` | XGBoost | Early stopping, class balancing, regularization |
| `RandomForestTrainer` | Random Forest | Feature importance, bootstrap sampling |
| `LogisticRegressionTrainer` | Logistic Regression | L1/L2 regularization, feature scaling |
| `GradientBoostingTrainer` | Gradient Boosting | Learning rate, subsampling |
| `SVMTrainer` | Support Vector Machine | Kernel selection, C parameter |
| `NaiveBayesTrainer` | Naive Bayes | Gaussian/Multinomial variants |

#### Trainer Interface

All trainers implement the following interface:

```python
class BaseTrainer(ABC):
    def train(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]
    def evaluate(self, X_test, y_test) -> Dict[str, Any]
    def predict(self, X) -> np.ndarray
    def predict_proba(self, X) -> np.ndarray
    def save_model(self, filepath: str) -> bool
    def load_model(self, filepath: str) -> bool
    def get_feature_importance(self) -> Dict[str, float]
```

#### Usage Example

```python
from scripts.model_training.core.model_factory import ModelFactory

# Create trainer
trainer = ModelFactory.create_trainer(config, 'xgboost')

# Train model
results = trainer.train(X_train, y_train, X_test, y_test)

# Get metrics
metrics = results['metrics']
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"F1-Score: {metrics['f1']:.3f}")

# Get feature importance
importance = trainer.get_feature_importance()
```

### 4. Hyperparameter Tuning

Advanced hyperparameter optimization using Optuna Bayesian optimization:

#### XGBoostHyperparameterTuner

**Key Methods:**
- `bayesian_optimization()`: Bayesian optimization with TPE sampler
- `grid_search_optimization()`: Exhaustive grid search
- `evaluate_optimized_model()`: Evaluate optimized parameters
- `update_configuration()`: Update config with optimized parameters

**Example:**
```python
from scripts.model_training.hyperparameter_tuning import XGBoostHyperparameterTuner

tuner = XGBoostHyperparameterTuner()

# Perform Bayesian optimization
results = tuner.bayesian_optimization(
    X_train, y_train,
    n_trials=100,
    cv_folds=5,
    early_stopping_rounds=20
)

# Evaluate optimized model
evaluation = tuner.evaluate_optimized_model(
    X_train, y_train, X_test, y_test,
    results['best_params']
)

# Update configuration if improvement is significant
if evaluation['metrics']['pr_auc'] > baseline_pr_auc + 0.01:
    tuner.update_configuration(results['best_params'], Environment.PRODUCTION)
```

### 5. Dynamic Model Selection

Automated model selection system that evaluates all available models and selects the best performer:

#### DynamicModelSelector

**Key Methods:**
- `evaluate_all_models()`: Evaluate all available models
- `select_best_model()`: Select best model based on criteria
- `update_production_config()`: Update production configuration
- `run_dynamic_selection()`: Complete selection workflow

**Selection Criteria (Priority Order):**
1. PR-AUC (Primary metric for imbalanced data)
2. Recall (Sensitivity for medical alerts)
3. Precision (Reducing false positives)
4. F1-Score (Balanced performance)

**Example:**
```python
from scripts.model_training.model_selection import DynamicModelSelector

selector = DynamicModelSelector()

# Run complete selection process
results = selector.run_dynamic_selection(
    environment=Environment.PRODUCTION,
    update_config=True
)

if results['success']:
    print(f"Selected model: {results['selected_model']['Model']}")
    print(f"PR-AUC: {results['selected_model']['PR-AUC']:.3f}")
```

## 📊 Evaluation Metrics

The system tracks comprehensive performance metrics:

### Primary Metrics
- **PR-AUC (Average Precision)**: Primary metric for imbalanced medical data
- **ROC-AUC**: Overall discriminative ability
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy

### Secondary Metrics
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **Specificity**: True negative rate
- **Cross-Validation Scores**: Stability assessment

### Cross-Validation
- **Method**: Stratified 5-fold cross-validation
- **Stratification**: Maintains class distribution across folds
- **Early Stopping**: Integrated with XGBoost for efficiency
- **Metrics**: Mean ± Standard deviation

## 🔄 Workflow Examples

### Complete Training Workflow

```python
#!/usr/bin/env python3
"""
Complete model training workflow example
"""

import logging
from pathlib import Path
from scripts.model_training.config.config_manager import ConfigManager, Environment
from scripts.model_training.pipelines.training_pipeline import TrainingPipeline

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.get_config(Environment.PRODUCTION)
    
    # Create training pipeline
    pipeline = TrainingPipeline(config)
    
    # Run complete training
    logger.info("Starting model training...")
    results = pipeline.run_complete_training()
    
    if results:
        # Display results
        metrics = results['evaluation_results']['metrics']
        logger.info(f"Training completed successfully!")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        logger.info(f"PR-AUC: {metrics['pr_auc']:.3f}")
        logger.info(f"F1-Score: {metrics['f1']:.3f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
        
        # Cross-validation results
        cv_results = results['evaluation_results']['cv_results']
        if cv_results:
            logger.info(f"CV ROC-AUC: {cv_results['cv_mean']:.3f} ± {cv_results['cv_std']:.3f}")
    else:
        logger.error("Training failed!")

if __name__ == "__main__":
    main()
```

### Model Comparison Workflow

```python
#!/usr/bin/env python3
"""
Model comparison workflow example
"""

import logging
from scripts.model_training.config.config_manager import ConfigManager, Environment
from scripts.model_training.pipelines.evaluation_pipeline import EvaluationPipeline

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.get_config(Environment.PRODUCTION)
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(config)
    
    # Define models to compare
    model_types = [
        'xgboost', 
        'random_forest', 
        'logistic_regression',
        'gradient_boosting',
        'svm',
        'naive_bayes'
    ]
    
    # Run evaluation
    logger.info("Starting model comparison...")
    results = pipeline.run_complete_evaluation(model_types=model_types)
    
    if results:
        # Display results
        best_model = results['best_model']
        logger.info(f"Model comparison completed!")
        logger.info(f"Best model: {best_model['Model']}")
        logger.info(f"PR-AUC: {best_model['PR-AUC']:.3f}")
        logger.info(f"Recall: {best_model['Recall']:.3f}")
        logger.info(f"Precision: {best_model['Precision']:.3f}")
        logger.info(f"F1-Score: {best_model['F1-Score']:.3f}")
        
        # Display full comparison
        comparison_df = results['comparison_df']
        logger.info("\nFull comparison results:")
        print(comparison_df.to_string(index=False))
    else:
        logger.error("Model comparison failed!")

if __name__ == "__main__":
    main()
```

### Hyperparameter Optimization Workflow

```python
#!/usr/bin/env python3
"""
Hyperparameter optimization workflow example
"""

import logging
from pathlib import Path
from scripts.model_training.hyperparameter_tuning import XGBoostHyperparameterTuner
from scripts.model_training.config.config_manager import ConfigManager, Environment
from scripts.model_training.utils.data_loader import DataLoader
from scripts.model_training.utils.preprocessor import DataPreprocessor

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize components
    project_root = Path(__file__).parent.parent.parent
    tuner = XGBoostHyperparameterTuner(project_root)
    config_manager = ConfigManager(project_root)
    
    # Load configuration
    config = config_manager.get_config(Environment.PRODUCTION)
    
    # Load and preprocess data
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor(config)
    
    X, y = data_loader.get_clean_dataset()
    X_processed, y_processed = preprocessor.preprocess_data(X, y)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_processed, y_processed)
    
    logger.info(f"Data loaded: {X_train.shape[0]} training samples")
    
    # Perform optimization
    logger.info("Starting Bayesian optimization...")
    results = tuner.bayesian_optimization(
        X_train, y_train,
        n_trials=100,
        cv_folds=5,
        early_stopping_rounds=20,
        timeout=3600  # 1 hour
    )
    
    # Evaluate optimized model
    logger.info("Evaluating optimized model...")
    evaluation = tuner.evaluate_optimized_model(
        X_train, y_train, X_test, y_test,
        results['best_params']
    )
    
    # Print results
    logger.info(f"Best CV Score: {results['best_score']:.4f}")
    logger.info(f"Test PR-AUC: {evaluation['metrics']['pr_auc']:.4f}")
    logger.info(f"Best Parameters: {results['best_params']}")
    
    # Update configuration if significant improvement
    baseline_pr_auc = 0.88  # Example baseline
    pr_auc_improvement = evaluation['metrics']['pr_auc'] - baseline_pr_auc
    if pr_auc_improvement > 0.01:
        logger.info("Significant improvement detected! Updating configuration...")
        success = tuner.update_configuration(
            results['best_params'],
            Environment.PRODUCTION
        )
        if success:
            logger.info("Configuration updated successfully!")
        else:
            logger.error("Failed to update configuration")
    else:
        logger.info("Improvement not significant enough to update configuration")

if __name__ == "__main__":
    main()
```

## 🛠️ Advanced Features

### 1. Custom Model Integration

To add a new model type:

```python
from scripts.model_training.core.base_trainer import BaseTrainer
from scripts.model_training.core.model_factory import register_trainer

class CustomModelTrainer(BaseTrainer):
    def _create_model(self):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**self.config.get_model_hyperparameters('extra_trees'))
    
    def _get_model_name(self):
        return 'Extra Trees'

# Register the new trainer
register_trainer('extra_trees', CustomModelTrainer)
```

### 2. Custom Evaluation Metrics

```python
from scripts.model_training.core.evaluator import ModelEvaluator

class CustomEvaluator(ModelEvaluator):
    def calculate_custom_metrics(self, y_true, y_pred, y_proba):
        # Add custom clinical metrics
        metrics = super().calculate_custom_metrics(y_true, y_pred, y_proba)
        
        # Custom metric: Clinical relevance score
        metrics['clinical_relevance'] = self.calculate_clinical_relevance(y_true, y_pred)
        
        return metrics
    
    def calculate_clinical_relevance(self, y_true, y_pred):
        # Implementation of clinical relevance metric
        pass
```

### 3. Custom Preprocessing

```python
from scripts.model_training.utils.preprocessor import DataPreprocessor

class CustomPreprocessor(DataPreprocessor):
    def custom_feature_engineering(self, X):
        # Add custom feature engineering
        X['custom_feature'] = X['feature1'] * X['feature2']
        return X
    
    def preprocess_data(self, X, y):
        # Apply custom preprocessing
        X = self.custom_feature_engineering(X)
        return super().preprocess_data(X, y)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Configuration Loading Errors
**Problem**: Configuration not loading properly
**Solution**:
```python
# Check configuration file exists
from pathlib import Path
config_path = Path("scripts/model_training/config/environment_config.yaml")
print(f"Config exists: {config_path.exists()}")

# Validate configuration
validation_results = config_manager.validate_config(config)
if not validation_results['valid']:
    print(f"Validation errors: {validation_results['errors']}")
```

#### 2. Data Loading Issues
**Problem**: Dataset not found or corrupted
**Solution**:
```python
# Check dataset path
dataset_path = Path("data/model_ready/model_ready_dataset.csv")
print(f"Dataset exists: {dataset_path.exists()}")

# Check data format
import pandas as pd
try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Data loading error: {e}")
```

#### 3. Model Training Failures
**Problem**: Model training fails with errors
**Solution**:
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data preprocessing
try:
    X_processed, y_processed = preprocessor.preprocess_data(X, y)
    print(f"Preprocessed data shape: {X_processed.shape}")
except Exception as e:
    print(f"Preprocessing error: {e}")
    import traceback
    traceback.print_exc()
```

#### 4. Memory Issues
**Problem**: Out of memory during training
**Solution**:
- Reduce dataset size for testing
- Use smaller hyperparameters (fewer estimators)
- Enable early stopping
- Use development environment configuration

### Performance Tips

1. **Use Development Environment**: For testing and development, use `Environment.DEVELOPMENT` with reduced complexity
2. **Enable Early Stopping**: Always use early stopping for XGBoost to prevent overfitting
3. **Parallel Processing**: Use `n_jobs=-1` for parallel training
4. **Caching**: Configuration manager caches results for better performance
5. **Logging**: Use appropriate logging levels to balance detail and performance

## 📚 Related Documentation

- [Configuration Guide](config/CONFIG_GUIDE.md): Detailed configuration system documentation
- [Hyperparameter Tuning Guide](hyperparameter_tuning/HYPERPARAMETER_TUNING_GUIDE.md): Advanced optimization documentation
- [Backend Models Guide](../../backend/ml_models/MODELS_GUIDE.md): Production model documentation
- [API Guide](../../backend/api/API_GUIDE.md): API integration documentation

## 🤝 Contributing

### Adding New Models

1. **Create Trainer Class**: Implement `BaseTrainer` interface
2. **Register Trainer**: Use `register_trainer()` in model factory
3. **Add Configuration**: Add hyperparameters to configuration files
4. **Update Documentation**: Update this guide with new model information
5. **Add Tests**: Create unit tests for the new trainer

### Adding New Features

1. **Design Interface**: Follow existing patterns and interfaces
2. **Implement Core Logic**: Add implementation in appropriate module
3. **Add Configuration**: Update configuration system if needed
4. **Update Pipelines**: Integrate with training/evaluation pipelines
5. **Document Changes**: Update relevant documentation

## 📋 Best Practices

### 1. Configuration Management
- Always use `ConfigManager` for configuration access
- Validate configuration before use
- Use environment-specific configurations
- Keep Python defaults and YAML values in sync

### 2. Model Training
- Use appropriate environment for development vs production
- Enable early stopping for tree-based models
- Use cross-validation for robust evaluation
- Save model artifacts with versioning

### 3. Evaluation
- Use multiple metrics for comprehensive evaluation
- Always evaluate on held-out test set
- Track cross-validation stability
- Document performance baselines

### 4. Hyperparameter Optimization
- Start with Bayesian optimization for efficiency
- Use appropriate search space bounds
- Validate results on independent test set
- Update configuration only for significant improvements

## 🆘 Support

### Getting Help

1. **Check Logs**: Review training logs for detailed error information
2. **Validate Configuration**: Use `validate_config()` to check configuration
3. **Test Components**: Test individual components (data loading, preprocessing)
4. **Review Examples**: Check example scripts for usage patterns
5. **Check Documentation**: Review related documentation guides

### Common Commands

```bash
# Test configuration loading
python3 -c "from scripts.model_training.config.config_manager import ConfigManager; print('Config OK')"

# Run basic training example
python3 scripts/model_training/examples/basic_training_example.py

# Test model training
python -m scripts.model_training train --environment development --model-type xgboost

# Run model comparison
python -m scripts.model_training evaluate --environment production

# Check hyperparameter tuning
python3 scripts/model_training/hyperparameter_tuning/tuner.py
```

---

**Last Updated**: October 25, 2025  
**Status**: ✅ Production Ready  
**Version**: 2.1.0
