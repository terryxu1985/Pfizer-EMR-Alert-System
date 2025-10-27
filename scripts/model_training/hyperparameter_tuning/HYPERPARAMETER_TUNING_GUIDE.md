# Hyperparameter Tuning Guide

## 📋 Overview

The Pfizer EMR Alert System includes a comprehensive hyperparameter tuning framework that provides advanced optimization capabilities for machine learning models, with a primary focus on XGBoost optimization.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Hyperparameter Tuning System                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────┐  ┌───────────────────────┐   │
│  │     tuner.py             │  │ Configuration        │   │
│  │  (Core Tuning Logic)     │◄─┤ Integration           │   │
│  │                          │  │                      │   │
│  │ • Bayesian Optimization │  │ • ConfigManager      │   │
│  │ • Grid Search            │  │ • Environment Config │   │
│  │ • Cross-validation       │  │ • Parameter Updates   │   │
│  │ • Early Stopping         │  │ • Results Persistence │   │
│  │ • Performance Tracking  │  │                      │   │
│  └──────────────────────────┘  └───────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Module Structure

```
hyperparameter_tuning/
├── __init__.py              # Module exports
├── tuner.py                 # Core tuning implementation
└── HYPERPARAMETER_TUNING_GUIDE.md  # This guide
```

## 🚀 Quick Start

### Basic Usage

```python
from scripts.model_training.hyperparameter_tuning import XGBoostHyperparameterTuner
from scripts.model_training.config.config_manager import ConfigManager, Environment

# Initialize tuner
tuner = XGBoostHyperparameterTuner()

# Load configuration and data
config_manager = ConfigManager()
config = config_manager.get_config(Environment.PRODUCTION)

# Load and preprocess data
data_loader = DataLoader(config)
preprocessor = DataPreprocessor(config)

X, y = data_loader.get_clean_dataset()
X_processed, y_processed = preprocessor.preprocess_data(X, y)
X_train, X_test, y_train, y_test = preprocessor.split_data(X_processed, y_processed)

# Perform Bayesian optimization
results = tuner.bayesian_optimization(
    X_train, y_train,
    n_trials=50,
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

## 🔧 Core Components

### XGBoostHyperparameterTuner Class

The main class providing comprehensive hyperparameter tuning capabilities.

#### Key Methods

##### `bayesian_optimization()`
Performs Bayesian optimization using Optuna with TPE sampler.

**Parameters:**
- `X`: Training features (pandas DataFrame)
- `y`: Training labels (pandas Series)
- `n_trials`: Number of optimization trials (default: 100)
- `cv_folds`: Cross-validation folds (default: 5)
- `early_stopping_rounds`: Early stopping rounds (default: 20)
- `timeout`: Maximum optimization time in seconds (optional)

**Returns:**
```python
{
    'best_params': dict,      # Optimized hyperparameters
    'best_score': float,     # Best cross-validation score
    'study': optuna.Study,   # Complete optimization study
    'method': 'bayesian'     # Optimization method used
}
```

**Example:**
```python
results = tuner.bayesian_optimization(
    X_train, y_train,
    n_trials=30,
    cv_folds=5,
    timeout=1800  # 30 minutes
)
```

##### `grid_search_optimization()`
Performs exhaustive grid search optimization.

**Parameters:**
- `X`: Training features
- `y`: Training labels
- `param_grid`: Dictionary defining parameter search space
- `cv_folds`: Cross-validation folds (default: 5)
- `early_stopping_rounds`: Early stopping rounds (default: 20)

**Example:**
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9]
}

results = tuner.grid_search_optimization(
    X_train, y_train, param_grid
)
```

##### `evaluate_optimized_model()`
Evaluates optimized model performance on test set.

**Parameters:**
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data
- `params`: Optimized hyperparameters

**Returns:**
```python
{
    'model': XGBClassifier,           # Trained model
    'metrics': dict,                 # Performance metrics
    'predictions': array,            # Test predictions
    'probabilities': array           # Prediction probabilities
}
```

##### `update_configuration()`
Updates configuration files with optimized parameters.

**Parameters:**
- `optimized_params`: Dictionary of optimized parameters
- `environment`: Target environment (default: PRODUCTION)

**Returns:** `bool` - Success status

## 📊 Optimization Methods

### 1. Bayesian Optimization (Recommended)

**Advantages:**
- Efficient exploration of parameter space
- Balances exploration vs exploitation
- Handles high-dimensional parameter spaces well
- Uses Optuna with TPE sampler

**Best for:**
- Complex parameter spaces
- Limited computational resources
- Production optimization runs

**Parameter Search Space:**
```python
{
    'n_estimators': [100, 500],
    'learning_rate': [0.01, 0.3],  # log scale
    'max_depth': [3, 10],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'reg_alpha': [0.0, 1.0],
    'reg_lambda': [0.0, 2.0],
    'min_child_weight': [1, 10],
    'gamma': [0.0, 1.0]
}
```

### 2. Grid Search

**Advantages:**
- Exhaustive search guarantees optimal solution
- Deterministic results
- Easy to understand and interpret

**Best for:**
- Small parameter spaces
- When computational resources are abundant
- Research and experimentation

## 🎯 Performance Metrics

The tuning system uses **Average Precision (PR-AUC)** as the primary optimization metric because:

- **Imbalanced Data**: EMR data typically has class imbalance
- **Clinical Relevance**: Precision-recall is more relevant for medical alerts
- **Robustness**: Less sensitive to class distribution changes

**Additional Metrics Tracked:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

## 🔄 Cross-Validation Strategy

### Stratified K-Fold Cross-Validation
- **Folds**: 5 (configurable)
- **Stratification**: Maintains class distribution across folds
- **Shuffle**: Random shuffling with fixed seed (42)
- **Early Stopping**: Integrated with XGBoost early stopping

### Early Stopping Integration
- **Validation Set**: Uses holdout validation within each fold
- **Patience**: Configurable early stopping rounds
- **Metric**: Uses validation loss for early stopping
- **Efficiency**: Prevents overfitting and reduces training time

## 📈 Results Management

### Automatic Result Persistence

The tuner automatically saves optimization results to:
```
logs/hyperparameter_optimization/
├── optuna_study_bayesian_YYYYMMDD_HHMMSS.pkl
├── optimization_summary_bayesian_YYYYMMDD_HHMMSS.json
└── optuna_optimization_YYYYMMDD_HHMMSS.json
```

### Result Files

#### Study Object (`.pkl`)
- Complete Optuna study object
- Contains all trial information
- Can be loaded for further analysis

#### Summary File (`.json`)
```json
{
    "timestamp": "20251025_214955",
    "method": "bayesian",
    "best_params": {...},
    "best_score": 0.8234,
    "n_trials": 50,
    "best_trial_number": 23
}
```

## ⚙️ Configuration Integration

### Automatic Configuration Updates

When significant improvement is detected (PR-AUC improvement > 0.01), the tuner automatically:

1. **Updates YAML Configuration**: Modifies `environment_config.yaml`
2. **Environment-Specific**: Updates production environment parameters
3. **Backup**: Preserves original configuration
4. **Validation**: Ensures parameter validity

### Configuration Structure

```yaml
hyperparameters:
  xgboost:
    n_estimators: 799
    learning_rate: 0.2610570842357371
    max_depth: 7
    subsample: 0.9130034518395074
    colsample_bytree: 0.8549084927445061
    reg_alpha: 0.3040562063700757
    reg_lambda: 1.9851311518315806
    min_child_weight: 3
    gamma: 0.029685899839029244
    random_state: 42
    n_jobs: -1
    eval_metric: logloss
    early_stopping_rounds: 20

environments:
  production:
    hyperparameters:
      xgboost: *id001  # References optimized parameters
```

## 🛠️ Advanced Usage

### Custom Parameter Search Space

```python
# Define custom search space for Bayesian optimization
def custom_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        # Add more custom parameters...
    }
    # Custom evaluation logic
    return custom_score_function(params)
```

### Multi-Objective Optimization

```python
# Example: Optimize for both PR-AUC and training speed
def multi_objective(trial):
    params = get_params_from_trial(trial)
    
    # Train model
    model = train_model(params)
    
    # Calculate multiple objectives
    pr_auc = calculate_pr_auc(model)
    training_time = calculate_training_time(model)
    
    # Return tuple for multi-objective optimization
    return pr_auc, -training_time  # Maximize PR-AUC, minimize time
```

### Custom Evaluation Metrics

```python
# Add custom metrics to evaluation
def evaluate_with_custom_metrics(model, X_test, y_test):
    # Standard metrics
    metrics = calculate_standard_metrics(model, X_test, y_test)
    
    # Custom clinical metrics
    metrics['clinical_relevance'] = calculate_clinical_relevance(model, X_test)
    metrics['alert_frequency'] = calculate_alert_frequency(model, X_test)
    
    return metrics
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Memory Issues
**Problem**: Out of memory during optimization
**Solution**: 
- Reduce `n_trials`
- Use smaller `cv_folds`
- Enable early stopping
- Use smaller parameter search space

#### 2. Slow Optimization
**Problem**: Optimization takes too long
**Solution**:
- Set `timeout` parameter
- Reduce `n_trials`
- Use fewer CV folds
- Enable pruning in Optuna

#### 3. No Improvement Found
**Problem**: Optimization doesn't improve baseline
**Solution**:
- Check parameter search space bounds
- Verify data preprocessing
- Ensure baseline model is properly configured
- Consider different optimization methods

### Performance Tips

1. **Start Small**: Begin with fewer trials (20-30) for initial exploration
2. **Use Early Stopping**: Always enable early stopping to prevent overfitting
3. **Monitor Progress**: Use Optuna's progress bar and logging
4. **Save Intermediate Results**: Results are automatically saved
5. **Validate Results**: Always evaluate on held-out test set

## 📚 Examples

### Complete Optimization Workflow

```python
#!/usr/bin/env python3
"""
Complete hyperparameter optimization workflow
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
        n_trials=50,
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

## 🔗 Integration Points

### With Model Training Pipeline
```python
# Integration with training pipeline
from scripts.model_training.trainers.xgboost_trainer import XGBoostTrainer

# Use optimized parameters in training
optimized_config = config
optimized_config.hyperparameter_config.hyperparameters['xgboost'] = best_params

trainer = XGBoostTrainer(optimized_config)
trainer.train(X_train, y_train, X_test, y_test)
```

### With Model Selection
```python
# Integration with model selection system
from scripts.model_training.model_selection import ModelSelector

# Optimize multiple models
models_to_optimize = ['xgboost', 'random_forest', 'gradient_boosting']
optimized_models = {}

for model_type in models_to_optimize:
    tuner = get_tuner_for_model(model_type)
    results = tuner.optimize(X_train, y_train)
    optimized_models[model_type] = results['best_params']
```

## 📋 Best Practices

### 1. Optimization Strategy
- **Start with Bayesian**: Use Bayesian optimization for initial exploration
- **Validate Results**: Always evaluate on held-out test set
- **Compare Methods**: Try both Bayesian and grid search for critical models
- **Monitor Overfitting**: Use early stopping and cross-validation

### 2. Parameter Space Design
- **Reasonable Bounds**: Set realistic parameter ranges
- **Log Scale**: Use log scale for learning rates and regularization
- **Prior Knowledge**: Incorporate domain knowledge into bounds
- **Iterative Refinement**: Narrow bounds based on initial results

### 3. Computational Efficiency
- **Resource Management**: Set appropriate timeouts and trial limits
- **Parallel Processing**: Use `n_jobs=-1` for parallel training
- **Early Stopping**: Always enable early stopping
- **Pruning**: Use Optuna pruning for inefficient trials

### 4. Result Management
- **Version Control**: Track optimization runs and results
- **Documentation**: Document parameter choices and rationale
- **Backup**: Keep backup of original configurations
- **Monitoring**: Monitor model performance after deployment

## 🆘 Support

### Getting Help

1. **Check Logs**: Review optimization logs in `logs/hyperparameter_optimization/`
2. **Validate Configuration**: Ensure configuration is valid before optimization
3. **Test Import**: Verify module imports work correctly
4. **Review Examples**: Check example scripts for usage patterns

### Common Commands

```bash
# Test module import
python3 -c "from scripts.model_training.hyperparameter_tuning import XGBoostHyperparameterTuner"

# Run optimization example
python3 scripts/model_training/hyperparameter_tuning/tuner.py

# Check optimization results
ls -la logs/hyperparameter_optimization/
```

---

**Last Updated**: 2025-01-25  
**Status**: ✅ Production Ready  
**Version**: 1.0.0
