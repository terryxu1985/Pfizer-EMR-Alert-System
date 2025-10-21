# Configuration Directory

This directory contains all configuration files and settings for the Pfizer EMR Alert System. It provides centralized configuration management for the entire application.

## Files Overview

### `settings.py`
The main configuration file that defines all system settings, paths, and configurations.

**Key Configuration Sections:**
- **Base Paths**: Defines directory structure for data, models, logs, and scripts
- **Model Configuration**: Settings for ML model files and metadata
- **Feature Configuration**: Defines categorical columns, excluded features, and target variables
- **API Configuration**: FastAPI settings including host, port, and debug mode
- **Logging Configuration**: Log levels, file rotation, and output settings
- **Database Configuration**: Database connection settings
- **Security Configuration**: Authentication and security settings

**Key Functions:**
- `get_latest_model_path()`: Automatically discovers the latest trained model
- `get_model_version_info()`: Retrieves model metadata and version information
- `get_log_level()`: Determines appropriate log level based on environment

### `requirements.txt`
Development and testing dependencies for data quality assessment and reporting.

**Key Dependencies:**
- **Data Processing**: pandas, numpy, openpyxl
- **PDF Generation**: reportlab
- **Visualization**: matplotlib
- **Utilities**: python-dateutil

### `requirements_production.txt`
Production dependencies for the EMR Alert System API and ML components.

**Key Dependencies:**
- **Core ML**: pandas, numpy, scikit-learn, xgboost, imbalanced-learn
- **API Framework**: fastapi, uvicorn, pydantic
- **Data Processing**: openpyxl
- **Utilities**: python-dateutil, pathlib2
- **Enhanced Logging**: structlog (optional)

## Configuration Management

### Environment Variables
The system supports configuration through environment variables:

- `API_HOST`: API server host (default: "0.0.0.0")
- `API_PORT`: API server port (default: 8000)
- `DEBUG`: Enable debug mode (default: "False")
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_TO_CONSOLE`: Enable console logging (default: "True")
- `DATABASE_URL`: Database connection string
- `DATABASE_ECHO`: Enable SQL query logging
- `SECRET_KEY`: Secret key for authentication

### Model Discovery
The system automatically discovers trained models using the following priority:
1. `backend/ml_models/models/` (primary location)
2. `scripts/model_training/models/` (training output)
3. `reports/model_evaluation/` (evaluation reports)
4. `backend/ml_models/` (fallback)

### Feature Configuration
The system includes built-in feature management:
- **Categorical Features**: Automatically encoded features
- **Data Leakage Features**: Features excluded to prevent data leakage
- **Excluded Features**: Features not suitable for ML (dates, IDs)
- **Target Column**: The prediction target variable

## Usage

### Importing Configuration
```python
from config.settings import (
    BASE_DIR, MODEL_DIR, LOG_DIR,
    MODEL_CONFIG, FEATURE_CONFIG, API_CONFIG,
    get_latest_model_path, get_model_version_info
)
```

### Getting Model Information
```python
# Get latest model path
model_path = get_latest_model_path()

# Get model version info
version_info = get_model_version_info()
print(f"Model version: {version_info['version']}")
print(f"Training date: {version_info['training_date']}")
```

### Environment Setup
```bash
# Set environment variables
export API_HOST="0.0.0.0"
export API_PORT="8000"
export DEBUG="True"
export LOG_LEVEL="DEBUG"

# Install dependencies
pip install -r requirements.txt  # For development
pip install -r requirements_production.txt  # For production
```

## Directory Structure
```
config/
├── README.md                    # This file
├── settings.py                  # Main configuration file
├── requirements.txt             # Development dependencies
└── requirements_production.txt  # Production dependencies
```

## Security Notes

- **Secret Key**: Change the default secret key in production
- **Database URL**: Use secure database connections in production
- **Debug Mode**: Disable debug mode in production environments
- **Logging**: Configure appropriate log levels for production

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure models are trained and placed in the correct directory
2. **Import Errors**: Verify all dependencies are installed using the appropriate requirements file
3. **Path Issues**: Check that BASE_DIR is correctly set relative to the config directory
4. **Environment Variables**: Ensure all required environment variables are set

### Validation
The configuration system includes automatic validation:
- Directory creation for required paths
- Model metadata validation
- Environment variable type checking
- Log level validation

## Contributing

When modifying configuration:
1. Update this README if adding new configuration options
2. Maintain backward compatibility when possible
3. Add appropriate environment variable support
4. Update both development and production requirements files
5. Test configuration changes in both environments
