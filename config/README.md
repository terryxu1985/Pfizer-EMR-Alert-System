# Configuration Guide - Pfizer EMR Alert System

## 📋 Overview

This directory contains all configuration files and settings for the Pfizer EMR Alert System. The configuration system provides centralized, environment-aware settings management for the entire application stack.

## 📁 Directory Structure

```
config/
├── README.md                    # This configuration guide
├── settings.py                  # Main configuration module with all system settings
├── requirements.txt             # Development dependencies
└── requirements_production.txt  # Production dependencies (includes security & monitoring)
```

## 🔧 Core Configuration Files

### `settings.py`
The heart of the configuration system, providing centralized settings management.

#### Key Features
- **Environment-aware**: Automatically adapts based on runtime environment
- **Dynamic model discovery**: Automatically finds latest trained models
- **Path management**: Intelligent directory structure handling
- **Security-first**: Environment variable support for sensitive data

#### Configuration Sections

##### 1. Base Paths
```python
BASE_DIR: Project root directory
DATA_DIR: Data storage (raw, processed, model_ready)
MODEL_DIR: ML model storage
LOG_DIR: Logging directory
SCRIPTS_MODEL_DIR: Training script outputs
```

##### 2. Model Configuration
- **Model Name**: `xgboost_emr_alert`
- **Version**: `2.2.0`
- **Model File**: XGBoost classifier with optimized hyperparameters
- **Performance**: PR-AUC=0.90, Precision=0.88, Recall=0.75, F1=0.81
- **Supporting Files**: Scaler, encoders, feature columns, metadata, preprocessor

##### 3. Feature Configuration
Based on `model_feature_dictionary.xlsx`:
- **Categorical Features**: 7 features (gender, experience level, state, type, season, location, insurance)
- **Data Leakage Features**: Excluded to prevent information leakage
- **Excluded Features**: Dates, IDs, and target variable
- **Target**: Binary classification (treatment likelihood)

##### 4. API Configuration
- **Title**: Pfizer EMR Alert System API
- **Version**: 2.2.0
- **Default Host**: 0.0.0.0
- **Default Port**: 8000
- **Debug Mode**: Configurable via environment variable

##### 5. Logging Configuration
- **Format**: Timestamp, logger name, level, message
- **File**: `logs/emr_alert_system.log`
- **Rotation**: Daily with 10MB max size
- **Retention**: 30 days with 5 backup files
- **Console Output**: Configurable

##### 6. Database Configuration
- **Default**: SQLite for development
- **Connection**: Configurable via DATABASE_URL
- **Query Logging**: Optional via DATABASE_ECHO

##### 7. Security Configuration
- **Secret Key**: Environment variable (required in production)
- **Algorithm**: HS256 for JWT tokens
- **Token Expiry**: 30 minutes

### `requirements.txt`
Development and testing dependencies for the full system.

**Categories:**
- Core ML libraries (pandas, numpy, scikit-learn, xgboost)
- Hyperparameter optimization (optuna)
- API framework (fastapi, uvicorn, pydantic)
- Data processing (openpyxl, requests)
- Visualization (matplotlib, seaborn, reportlab)
- Configuration management (PyYAML)
- Development tools (structlog)

### `requirements_production.txt`
Production dependencies with additional security and monitoring components.

**Production Extras:**
- Enhanced logging with structlog
- Production WSGI server (gunicorn)
- Security hardening
- Performance optimizations

## 🌍 Environment Variables

### API Configuration
```bash
export API_HOST="0.0.0.0"           # Server host
export API_PORT="8000"              # Server port
export DEBUG="False"                # Debug mode (True/False)
```

### Logging Configuration
```bash
export LOG_LEVEL="INFO"             # DEBUG, INFO, WARNING, ERROR, CRITICAL
export LOG_TO_CONSOLE="True"        # Enable console output
```

### Database Configuration
```bash
export DATABASE_URL="sqlite:///./emr_alert.db"  # Database connection string
export DATABASE_ECHO="False"        # Enable SQL query logging
```

### Security Configuration
```bash
export SECRET_KEY="your-secret-key-here"  # JWT secret key (required in production)
```

## 🔍 Key Functions

### `get_latest_model_path()`
Automatically discovers the latest trained model with intelligent fallback:
1. Primary: `backend/ml_models/models/`
2. Fallback 1: `scripts/model_training/models/`
3. Fallback 2: `reports/model_evaluation/`
4. Fallback 3: `backend/ml_models/`

Returns the most recently modified model directory.

### `get_model_version_info(model_path)`
Retrieves comprehensive model metadata:
- Version number
- Training date
- Model type
- Feature count
- Performance metrics
- Model parameters

Returns a dictionary with all available metadata or defaults for missing information.

### `get_log_level()`
Determines appropriate logging level based on:
1. Explicit `LOG_LEVEL` environment variable
2. Debug mode (DEBUG if enabled, else INFO)
3. Default to INFO if neither set

## 💻 Usage Examples

### Basic Import
```python
from config.settings import (
    BASE_DIR,
    MODEL_DIR,
    LOG_DIR,
    MODEL_CONFIG,
    FEATURE_CONFIG,
    API_CONFIG
)
```

### Get Latest Model
```python
from config.settings import get_latest_model_path, get_model_version_info

# Get model path
model_path = get_latest_model_path()
print(f"Model location: {model_path}")

# Get version info
info = get_model_version_info()
print(f"Model: {info['model_type']}")
print(f"Version: {info['version']}")
print(f"Training Date: {info['training_date']}")
print(f"Features: {info['feature_count']}")
print(f"Performance: {info['performance_metrics']}")
```

### Configure Paths
```python
from config.settings import DATA_DIR, MODEL_DIR

# Access data files
data_file = DATA_DIR / "model_ready" / "model_ready_dataset.csv"

# Access models
model_file = MODEL_DIR / "models" / "xgboost_model.pkl"
```

### Feature Configuration
```python
from config.settings import FEATURE_CONFIG

# Categorical features
categorical = FEATURE_CONFIG["categorical_columns"]

# Excluded features
excluded = (
    FEATURE_CONFIG["data_leakage_features"] + 
    FEATURE_CONFIG["excluded_features"]
)

# Target column
target = FEATURE_CONFIG["target_column"]
```

## 🔐 Security Best Practices

### Development Environment
- Use default secret keys
- Enable debug mode for troubleshooting
- SQLite database is acceptable
- Console logging enabled

### Production Environment
⚠️ **CRITICAL**: Must configure before deployment

1. **Secret Key**: Generate strong random key
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   Export as `SECRET_KEY`

2. **Debug Mode**: Always disable
   ```bash
   export DEBUG="False"
   ```

3. **Logging**: Set appropriate level
   ```bash
   export LOG_LEVEL="WARNING"  # Less verbose
   export LOG_TO_CONSOLE="False"  # Disable console logs
   ```

4. **Database**: Use production-grade database
   ```bash
   export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
   ```

5. **HTTPS**: Use reverse proxy (nginx, Apache) with SSL certificates

6. **API Security**: Implement rate limiting, CORS restrictions, and API keys

## 🐛 Troubleshooting

### Model Not Found
**Symptoms**: `Model file not found` error

**Solutions**:
1. Verify model files exist in `backend/ml_models/models/`
2. Check file permissions
3. Run training scripts to generate models
4. Verify MODEL_CONFIG file names match actual files

### Import Errors
**Symptoms**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
1. Install dependencies: `pip install -r requirements.txt`
2. Verify virtual environment is activated
3. Check Python version (3.9+ required)
4. Reinstall dependencies if corrupted

### Path Issues
**Symptoms**: File not found errors, incorrect paths

**Solutions**:
1. Verify BASE_DIR points to project root
2. Check directory structure matches expected layout
3. Ensure all parent directories exist
4. Run from project root directory

### Environment Variables Not Working
**Symptoms**: Changes not taking effect

**Solutions**:
1. Verify variable names are correct
2. Restart application after setting variables
3. Check for typos in variable names
4. Ensure no spaces around equals sign
5. Export variables before starting application

### Permission Errors
**Symptoms**: Cannot create directories or write files

**Solutions**:
1. Check directory permissions
2. Run with appropriate user permissions
3. Fix ownership: `chown -R user:group config/`
4. Set write permissions: `chmod -R u+w config/`

## 📊 Configuration Validation

The configuration system includes automatic validation:

- ✅ **Directory Creation**: Auto-creates required directories
- ✅ **Model Metadata**: Validates model metadata integrity
- ✅ **Environment Variables**: Type checking for env vars
- ✅ **Log Levels**: Validates against allowed values
- ✅ **Path Resolution**: Ensures all paths are absolute and valid

## 🔄 Configuration Updates

When modifying configuration:

### For System Settings (`settings.py`)
1. Update relevant configuration dictionary
2. Add any new environment variable support
3. Document changes in this README
4. Test in both development and production environments
5. Update version number if making breaking changes

### For Dependencies (`requirements.txt`)
1. Add new package to appropriate file
2. Specify version constraints (use `>=` for minimum)
3. Update both development and production files if needed
4. Run `pip install -r requirements.txt` to test
5. Document why the dependency is needed

### Version Changes
- **Patch** (2.2.0 → 2.2.1): Bug fixes, no breaking changes
- **Minor** (2.2.0 → 2.3.0): New features, backward compatible
- **Major** (2.2.0 → 3.0.0): Breaking changes, migration required

## 🤝 Contributing

When contributing to configuration:

1. **Read First**: Review existing configuration structure
2. **Test Locally**: Verify changes work in your environment
3. **Document**: Update this README with new features
4. **Backward Compatibility**: Maintain compatibility when possible
5. **Security**: Never commit secrets or sensitive data
6. **Version Control**: Add meaningful commit messages

## 📚 Additional Resources

- **API Guide**: `backend/api/API_GUIDE.md`
- **Backend Guide**: `backend/BACKEND_GUIDE.md`
- **Data Guide**: `data/DATA_GUIDE.md`
- **Model Training**: `scripts/model_training/MODEL_TRAINING_GUIDE.md`
- **Main README**: `../README.md`
- **Quick Start**: `../QUICK_START.md`

## 📝 Configuration Checklist

### Development Setup
- [ ] Install development dependencies
- [ ] Configure local paths
- [ ] Set up development database
- [ ] Enable debug mode
- [ ] Configure local logging

### Production Deployment
- [ ] Install production dependencies
- [ ] Generate and set SECRET_KEY
- [ ] Disable debug mode
- [ ] Configure production database
- [ ] Set appropriate log level
- [ ] Configure HTTPS/SSL
- [ ] Set up monitoring and alerting
- [ ] Test all environment variables
- [ ] Review security settings
- [ ] Set up backup procedures

---

**Last Updated**: November 2025  
**Configuration Version**: 2.2.0  
**Maintainer**: Pfizer EMR Alert System Team
