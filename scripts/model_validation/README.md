# Model Validation Tools

This directory contains comprehensive model validation tools for the EMR Alert System.

## Overview

The model validation tools ensure consistency between training and prediction environments by validating:
- Feature column alignment
- Categorical value consistency  
- Model artifact integrity
- Data format compatibility

## Tools

### 1. `model_validator.py` - Main Validation Tool

A comprehensive model validation utility that provides:

- **Feature Consistency Validation**: Ensures test data has all required features
- **Categorical Value Validation**: Verifies categorical values are within training range
- **Training Value Inspection**: Shows all categorical classes used during training
- **Comprehensive Reporting**: Generates detailed validation reports

#### Usage

```bash
# Basic validation
python model_validator.py

# With test data
python model_validator.py --test-data data/test_data.csv

# With specific columns
python model_validator.py --test-columns col1 col2 col3

# With custom model path
python model_validator.py --model-path /path/to/model
```

#### Programmatic Usage

```python
from scripts.model_validation.model_validator import ModelValidator

# Initialize validator
validator = ModelValidator()

# Validate features
validation_report = validator.generate_validation_report(
    test_data=test_df,
    test_columns=list(test_df.columns)
)

# Print summary
validator.print_validation_summary(validation_report)
```

### 2. `check_features.py` - Feature Consistency Check

Legacy-compatible script for checking feature alignment between model and test data.

```bash
python check_features.py
```

### 3. `check_training_values.py` - Training Values Check

Legacy-compatible script for inspecting categorical training values.

```bash
python check_training_values.py
```

## Validation Types

### Feature Consistency
- Ensures all model features are present in test data
- Identifies missing or extra features
- Reports feature count differences

### Categorical Value Validation
- Verifies categorical values are within training range
- Identifies unknown/novel categorical values
- Reports coverage statistics

### Model Artifact Validation
- Validates model file integrity
- Checks label encoder consistency
- Verifies feature column alignment

## Output

All tools provide detailed console output and can be integrated into CI/CD pipelines. The main validator returns appropriate exit codes:
- `0`: All validations passed
- `1`: One or more validations failed

## Integration

These tools can be integrated into:
- **Pre-deployment validation**: Ensure model consistency before deployment
- **CI/CD pipelines**: Automated validation in deployment workflows  
- **Data quality checks**: Validate incoming data format
- **Model monitoring**: Regular consistency checks in production

## Dependencies

- `pandas`: Data manipulation
- `pickle`: Model artifact loading
- `pathlib`: File path handling
- Project configuration from `config.settings`
