# Configuration System Guide

## 📋 Overview

The Pfizer EMR Alert System uses a **two-file configuration architecture** that separates configuration **structure/logic** from configuration **values**.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Configuration System                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────┐  ┌───────────────────────┐   │
│  │  optimized_config.py     │  │ environment_config.    │   │
│  │  (Structure & Logic)     │◄─┤ yaml (Values)         │   │
│  │                          │  │                        │   │
│  │ • Data structures        │  │ • Actual parameters    │   │
│  │ • Validation logic       │  │ • Multi-environment    │   │
│  │ • Load/Save methods      │  │ • Easy to edit         │   │
│  │ • Default values         │  │ • Version controlled   │   │
│  │   (fallback)             │  │   (overrides Python)   │   │
│  └──────────────────────────┘  └───────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📁 File Descriptions

### 1. `optimized_config.py` - **Structure & Logic**

**Purpose**: Defines HOW configuration is structured and managed

**Contains**:
- Data classes (`@dataclass`) defining configuration structure
- Validation logic for configuration parameters
- Methods to load/save YAML files
- Default values (used only as fallback)
- Configuration management utilities

**When to edit**:
- ✅ Adding new configuration parameters
- ✅ Changing validation rules
- ✅ Adding new configuration methods
- ❌ Changing parameter values (use YAML instead)

**Example**:
```python
@dataclass
class OptimizedModelConfig:
    model_type: str = 'random_forest'  # Default/fallback only
    test_size: float = 0.2
    # ... more structure
```

### 2. `environment_config.yaml` - **Values**

**Purpose**: Stores ACTUAL configuration values

**Contains**:
- Base configuration values for all models
- Hyperparameters for each model type
- Environment-specific overrides (dev/test/staging/prod)
- Feature lists and paths

**When to edit**:
- ✅ Changing hyperparameters
- ✅ Adjusting model settings
- ✅ Adding environment-specific configs
- ✅ Most configuration changes!

**Example**:
```yaml
model:
  model_type: random_forest    # This overrides Python default
  test_size: 0.2
  use_smote: false
  
environments:
  production:
    hyperparameters:
      random_forest:
        n_estimators: 200
```

## 🔄 Configuration Priority

Configuration values are loaded in the following priority (high to low):

```
1. Environment-specific overrides in YAML  ← Highest priority
   └─ environments.production.hyperparameters
   
2. Base configuration values in YAML
   └─ model, features, hyperparameters
   
3. Default values in optimized_config.py   ← Lowest priority (fallback)
   └─ Used only if not specified in YAML
```

### Priority Example

```python
# Python default
model_type: str = 'random_forest'

# YAML base config  ← Overrides Python
model:
  model_type: gradient_boosting

# YAML production environment  ← Overrides base config
environments:
  production:
    model_type: xgboost  # This wins in production!
```

## 🚀 Usage Examples

### Basic Usage

```python
from scripts.model_training.config.optimized_config import (
    OptimizedModelConfig, 
    Environment
)

# Load configuration from YAML
config = OptimizedModelConfig.load_from_yaml(
    'environment_config.yaml',
    environment=Environment.PRODUCTION
)

# Access configuration
print(f"Model Type: {config.model_type}")
print(f"Test Size: {config.test_size}")

# Get model-specific hyperparameters
hyperparams = config.get_model_hyperparameters('random_forest')
```

### Using ConfigManager (Recommended)

```python
from scripts.model_training.config.config_manager import ConfigManager

# Initialize manager
config_manager = ConfigManager()

# Get production config
config = config_manager.get_production_config()

# Validate configuration
validation_results = config_manager.validate_config(config)
if validation_results['valid']:
    print("Configuration is valid!")
```

### Creating Custom Configuration

```python
# Start with production config and override
config = config_manager.create_custom_config(
    base_environment=Environment.PRODUCTION,
    overrides={
        'test_size': 0.3,
        'cv_folds': 10,
        'hyperparameters': {
            'random_forest': {'n_estimators': 500}
        }
    }
)
```

## 🎯 Best Practices

### ✅ DO

1. **Edit YAML for value changes**: Most configuration changes should be in `environment_config.yaml`
2. **Use environments**: Create environment-specific overrides for dev/test/staging/prod
3. **Validate after changes**: Use `validate_config()` to ensure configuration is valid
4. **Keep defaults in sync**: If you change Python defaults, update YAML to match
5. **Document parameter choices**: Add comments in YAML explaining why values were chosen
6. **Use ConfigManager**: Provides validation, caching, and cleaner API

### ❌ DON'T

1. **Don't hardcode model assumptions**: Let `model_selection.py` choose the best model
2. **Don't edit Python for value changes**: Use YAML instead (easier to maintain)
3. **Don't duplicate configuration**: Use environment overrides instead of copying
4. **Don't skip validation**: Always validate after making changes
5. **Don't guess at parameters**: Document the reasoning behind parameter choices

## 🌍 Multi-Environment Configuration

### Environment Structure

```yaml
# Base configuration (applies to all environments)
model:
  test_size: 0.2
  cv_folds: 5

# Environment-specific overrides
environments:
  development:
    cv_folds: 3              # Faster for dev
    hyperparameters:
      xgboost:
        n_estimators: 100    # Less complex
  
  production:
    cv_folds: 5              # Full validation
    hyperparameters:
      xgboost:
        n_estimators: 300    # Full complexity
```

### Switching Environments

```python
# Via ConfigManager
dev_config = config_manager.get_development_config()
test_config = config_manager.get_testing_config()
staging_config = config_manager.get_staging_config()
prod_config = config_manager.get_production_config()

# Via environment variable
import os
os.environ['EMR_ENVIRONMENT'] = 'production'
config = config_manager.get_config()  # Auto-detects environment
```

## 🔧 Configuration Parameters

### Model Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_type` | str | Model to train | `'random_forest'` |
| `model_version` | str | Model version | `'2.1.0'` |
| `random_state` | int | Random seed | `42` |
| `test_size` | float | Test set proportion | `0.2` |
| `validation_size` | float | Validation set proportion | `0.2` |
| `stratify` | bool | Maintain class distribution | `true` |
| `invert_target` | bool | Invert target variable | `false` |

### SMOTE Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `use_smote` | bool | Enable SMOTE globally | `false` |
| `smote_target_ratio` | float | Target minority ratio | `0.3` |
| `smote_k_neighbors` | int | K neighbors for SMOTE | `2` |
| `model_specific_smote` | dict | Per-model SMOTE settings | See YAML |

**Note**: Some models (e.g., Gradient Boosting) handle class imbalance natively and perform better WITHOUT SMOTE.

### Cross-Validation

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `cv_folds` | int | Number of CV folds | `5` |
| `cv_random_state` | int | CV random seed | `42` |

### Metrics

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `primary_metric` | str | Primary evaluation metric | `'average_precision'` |
| `metrics` | list | All metrics to compute | See YAML |

## 📊 Hyperparameter Tuning

### Current Strategy

All hyperparameters are optimized to **prevent overfitting** based on analysis:

```yaml
hyperparameters:
  random_forest:
    n_estimators: 50          # Reduced complexity
    max_depth: 6              # Limited depth
    min_samples_split: 30     # More conservative splits
    min_samples_leaf: 15      # Larger leaf nodes
    
  xgboost:
    n_estimators: 150         # Reduced from 300
    reg_alpha: 0.2            # Increased L1 regularization
    reg_lambda: 1.5           # Increased L2 regularization
```

### Environment-Specific Tuning

- **Development**: Fast iteration, reduced complexity
- **Testing**: Minimal resources, quick validation
- **Staging**: Near-production settings
- **Production**: Full complexity, optimal performance

## 🔍 Model Selection

### Important Principle

**The configuration should NOT hardcode which model is "best".**

Instead:
1. Configuration provides reasonable defaults for ALL models
2. `model_selection.py` trains and evaluates all models
3. Best model is selected based on actual performance metrics
4. Selected model is used for production

```python
# ❌ BAD: Hardcoding model assumption
model:
  model_type: gradient_boosting  # Assuming this is best

# ✅ GOOD: Let model selection decide
model:
  model_type: random_forest  # Default for training
  # Actual production model determined by model_selection.py
```

## 🐛 Troubleshooting

### Configuration Not Loading

```python
# Check if file exists
from pathlib import Path
config_path = Path("environment_config.yaml")
print(f"Exists: {config_path.exists()}")

# Try loading with error details
try:
    config = OptimizedModelConfig.load_from_yaml(config_path)
except Exception as e:
    print(f"Error: {e}")
```

### Validation Errors

```python
# Validate and see detailed results
validation = config_manager.validate_config(config)
print(f"Valid: {validation['valid']}")
print(f"Errors: {validation['errors']}")
print(f"Warnings: {validation['warnings']}")
```

### Parameter Not Taking Effect

Check priority order:
1. Is there an environment-specific override?
2. Is the YAML value correct?
3. Is the Python default different?
4. Is caching causing stale values?

```python
# Clear cache and reload
config_manager.clear_cache()
config = config_manager.get_config(cache=False)
```

## 📝 Change Log

### When Adding New Parameters

1. **Add to Python** (`optimized_config.py`):
   ```python
   new_parameter: float = 0.5  # Default/fallback
   ```

2. **Add to YAML** (`environment_config.yaml`):
   ```yaml
   model:
     new_parameter: 0.5  # Base value
   ```

3. **Update this documentation**: Add to parameter tables

4. **Validate**: Test that loading works correctly

## 📚 Related Files

- `config_manager.py`: High-level configuration management
- `base_config.py`: Legacy configuration (deprecated)
- `../../train.py`: Training script using configuration
- `../../model_selection.py`: Model selection system

## 🤝 Contributing

When modifying configuration:

1. **Update both files** if adding new parameters
2. **Keep values in sync** between Python defaults and YAML base
3. **Document your changes** with comments
4. **Validate** before committing
5. **Test all environments** (dev/test/staging/prod)

## ❓ Questions?

For questions or issues with configuration:
1. Check this guide first
2. Review inline comments in `optimized_config.py` and `environment_config.yaml`
3. Use `validate_config()` to identify issues
4. Contact the development team

---

**Last Updated**: 2025-10-21  
**Version**: 2.1.0

