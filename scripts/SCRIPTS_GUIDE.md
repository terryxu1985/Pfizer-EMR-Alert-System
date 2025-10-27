# Scripts Directory Guide

## 📋 Overview

The `scripts/` directory contains utility scripts for data processing, model training, system debugging, and system startup across the Pfizer EMR Alert System. These scripts provide essential tooling for data pipeline management, model development, validation, and deployment.

## 🏗️ Directory Structure

```
scripts/
├── data_cleaning/              # Data cleaning and quality assurance
│   ├── data_cleaning.py
│   └── data_quality_check_reporting.py
│
├── debug/                      # Debugging utilities
│   ├── debug_api.py
│   ├── debug_health.py
│   └── debug_prediction.py
│
├── feature_engineering/        # Feature engineering pipeline
│   └── build_model_ready_dataset.py
│
├── model_training/             # Model training framework (see MODEL_TRAINING_GUIDE.md)
│   ├── train.py
│   ├── evaluate.py
│   ├── model_selection.py
│   ├── trainers/
│   ├── pipelines/
│   ├── hyperparameter_tuning/
│   └── ...
│
├── model_validation/           # Model validation tools
│   ├── model_validator.py
│   ├── check_features.py
│   ├── check_training_values.py
│   └── README.md
│
└── startup/                    # System startup scripts
    ├── run_api_only.py
    ├── run_complete_system.py
    ├── run_quick_start.py
    └── README.md
```

---

## 📊 Data Cleaning Scripts

### Location: `data_cleaning/`

### 1. `data_cleaning.py`

**Purpose:** Comprehensive data cleaning pipeline for EMR datasets

**What it does:**
- Cleans three related datasets: `fact_txn.xlsx`, `dim_physician.xlsx`, `dim_patient.xlsx`
- Implements format standardization across all fields
- Handles missing values with appropriate strategies
- Validates referential integrity between tables
- Performs logical consistency checks
- Generates comprehensive quality assessment reports

**Key Features:**
- Format standardization (dates, codes, names)
- Missing value imputation
- Data type validation and conversion
- Referential integrity validation
- Outlier detection and handling
- Comprehensive cleaning statistics tracking

**Usage:**
```bash
python scripts/data_cleaning/data_cleaning.py
```

**Outputs:**
- `data/processed/fact_txn_cleaned.csv`
- `data/processed/dim_patient_cleaned.csv`
- `data/processed/dim_physician_cleaned.csv`
- Data quality reports in `reports/data_quality/`

### 2. `data_quality_check_reporting.py`

**Purpose:** Automated data quality assessment and reporting

**What it does:**
- Generates detailed data quality reports
- Identifies data issues and inconsistencies
- Creates visualization of data quality metrics
- Tracks data quality over time

**Usage:**
```bash
python scripts/data_cleaning/data_quality_check_reporting.py
```

**Outputs:**
- PDF data quality reports
- Quality metrics summary
- Data profiling visualizations

---

## 🔍 Debug Scripts

### Location: `debug/`

### 1. `debug_api.py`

**Purpose:** Debug and test API endpoints

**What it does:**
- Tests model loading functionality
- Validates prediction API with test data
- Checks API response format
- Identifies API configuration issues

**Usage:**
```bash
python scripts/debug/debug_api.py
```

**Features:**
- Model loading verification
- Prediction endpoint testing
- Error handling validation
- Configuration debugging

### 2. `debug_health.py`

**Purpose:** Health check debugging for system components

**What it does:**
- Tests health check endpoints
- Validates service availability
- Checks dependency status
- Monitors system health metrics

**Usage:**
```bash
python scripts/debug/debug_health.py
```

### 3. `debug_prediction.py`

**Purpose:** Debug prediction functionality

**What it does:**
- Tests prediction logic independently
- Validates model input/output formats
- Checks feature engineering pipeline
- Debugs prediction errors

**Usage:**
```bash
python scripts/debug/debug_prediction.py
```

---

## 🔧 Feature Engineering Scripts

### Location: `feature_engineering/`

### 1. `build_model_ready_dataset.py`

**Purpose:** Transform cleaned data into model-ready format

**What it does:**
- Merges fact and dimension tables
- Implements business rules (cohort selection, filtering)
- Creates derived features (risk scores, age-based features)
- Applies target variable logic
- Enforces feature dictionary specifications

**Key Business Rules:**
- **Cohort**: One row per patient based on earliest DISEASE_X diagnosis
- **Age Filter**: Keep patients with age >= 12 at diagnosis
- **Target Logic**: TARGET = 1 if NOT prescribed Drug A
- **Risk Conditions**: Flags for key comorbidities
- **Contraindication Level**: Highest level before diagnosis
- **Physician Experience**: Lifetime counts and treatment rates
- **Time Features**: Season, symptom-to-diagnosis duration

**Inputs:**
- `data/processed/fact_txn_cleaned.csv`
- `data/processed/dim_patient_cleaned.csv`
- `data/processed/dim_physician_cleaned.csv`
- `data/model_ready/model_feature_dictionary.xlsx`

**Output:**
- `data/model_ready/model_ready_dataset.csv`

**Usage:**
```bash
python scripts/feature_engineering/build_model_ready_dataset.py
```

**Features:**
- Enforces column order from feature dictionary
- Coerces data types where feasible
- Handles missing physician IDs (treated as NaN)
- Implements business logic consistently
- Generates audit trail

---

## 🤖 Model Training Scripts

### Location: `model_training/`

**Comprehensive Guide:** See `scripts/model_training/MODEL_TRAINING_GUIDE.md`

### Quick Overview:

The model training framework provides:
- **Multiple Algorithms**: XGBoost, Random Forest, Logistic Regression, Gradient Boosting, SVM, Naive Bayes
- **Automated Hyperparameter Tuning**: Optuna-based Bayesian optimization
- **Dynamic Model Selection**: Automated best model selection based on performance
- **Comprehensive Evaluation**: Multiple metrics and validation strategies
- **Production Pipelines**: End-to-end training workflows

**Main Scripts:**
- `train.py`: Main training script
- `evaluate.py`: Model evaluation
- `model_selection.py`: Dynamic model selection
- `select_scheduled.py`: Scheduled model selection

**Key Components:**
- `trainers/`: Algorithm-specific implementations
- `pipelines/`: Training and evaluation workflows
- `hyperparameter_tuning/`: Automated optimization
- `utils/`: Data loading, preprocessing, metrics
- `config/`: Configuration management

---

## ✅ Model Validation Scripts

### Location: `model_validation/`

**Detailed Guide:** See `scripts/model_validation/README.md`

### 1. `model_validator.py`

**Purpose:** Comprehensive model validation tool

**What it does:**
- Validates feature consistency between training and prediction
- Ensures categorical values are within training range
- Inspects training categorical classes
- Generates detailed validation reports
- Validates model artifact integrity

**Features:**
- Feature column alignment validation
- Categorical value range checking
- Unknown value detection
- Model artifact verification
- Comprehensive reporting

**Usage:**
```bash
# Basic validation
python scripts/model_validation/model_validator.py

# With test data
python scripts/model_validation/model_validator.py --test-data data/test_data.csv

# With custom model path
python scripts/model_validation/model_validator.py --model-path /path/to/model
```

**Programmatic Usage:**
```python
from scripts.model_validation.model_validator import ModelValidator

validator = ModelValidator()
report = validator.generate_validation_report(test_data=test_df)
validator.print_validation_summary(report)
```

### 2. `check_features.py`

**Purpose:** Legacy feature consistency check

**Usage:**
```bash
python scripts/model_validation/check_features.py
```

### 3. `check_training_values.py`

**Purpose:** Inspect categorical training values

**Usage:**
```bash
python scripts/model_validation/check_training_values.py
```

---

## 🚀 Startup Scripts

### Location: `startup/`

**Detailed Guide:** See `scripts/startup/README.md`

### 1. `run_api_only.py`

**Purpose:** Start FastAPI backend service only

**Features:**
- FastAPI server on port 8000
- Hot reload for development
- Health check endpoint (`/health`)
- API documentation at `/docs`

**Usage:**
```bash
python scripts/startup/run_api_only.py
```

**What it does:**
- Starts uvicorn server with hot reload
- Loads API routes and models
- Provides interactive API documentation
- Enables real-time development

### 2. `run_complete_system.py`

**Purpose:** Start both API and UI servers

**Features:**
- API server on port 8000
- UI server on port 8080
- Automatic browser opening
- Process management and cleanup
- Multi-step doctor data input
- Error handling and graceful shutdown

**Usage:**
```bash
python scripts/startup/run_complete_system.py
```

**What it does:**
- Launches both backend API and frontend UI
- Opens browser automatically
- Handles background processes
- Provides process cleanup on exit
- Enables full system testing

### 3. `run_quick_start.py`

**Purpose:** Convenience wrapper for complete system

**Usage:**
```bash
python scripts/startup/run_quick_start.py
```

---

## 🔄 Workflow Integration

### Data Pipeline Workflow

```
Raw Data (data/raw/)
    ↓
[data_cleaning.py]
    ↓
Cleaned Data (data/processed/)
    ↓
[build_model_ready_dataset.py]
    ↓
Model-Ready Data (data/model_ready/)
    ↓
[train.py]
    ↓
Trained Models (backend/ml_models/models/)
    ↓
[model_validator.py]
    ↓
Validated Model for Production
```

### Training Workflow

```
1. Data Preparation
   - Run data_cleaning.py
   - Run build_model_ready_dataset.py

2. Model Training
   - Run train.py or use training pipelines
   - Or run select_scheduled.py for automated selection

3. Model Validation
   - Run model_validator.py
   - Check training values with check_training_values.py

4. System Deployment
   - Use run_complete_system.py for full system
   - Or run_api_only.py for API testing
```

---

## 📝 Usage Examples

### Complete Data Pipeline

```bash
# 1. Clean raw data
python scripts/data_cleaning/data_cleaning.py

# 2. Build model-ready dataset
python scripts/feature_engineering/build_model_ready_dataset.py

# 3. Train models
python scripts/model_training/train.py

# 4. Validate model
python scripts/model_validation/model_validator.py
```

### System Startup

```bash
# Option 1: Complete system (API + UI)
python scripts/startup/run_complete_system.py

# Option 2: API only
python scripts/startup/run_api_only.py

# Option 3: Quick start
python scripts/startup/run_quick_start.py
```

### Debugging

```bash
# Debug API
python scripts/debug/debug_api.py

# Debug health checks
python scripts/debug/debug_health.py

# Debug predictions
python scripts/debug/debug_prediction.py
```

---

## 🔗 Integration with Other Components

### Backend Integration
- Scripts integrate with `backend/api/` for API functionality
- Use `backend/ml_models/models/` for model storage
- Connect to `backend/data_access/` for data operations

### Frontend Integration
- Startup scripts launch `frontend/server/emr_ui_server.py`
- Serve UI on port 8080 alongside API

### Configuration
- Scripts use `config/settings.py` for configuration
- Training scripts use `model_training/config/` for model-specific configs
- Environment variables supported for overrides

### Docker Integration
- Startup scripts are used in Docker containers
- `docker-compose.yml` references startup scripts
- Same scripts work in containerized and native environments

---

## 📚 Additional Documentation

- **Model Training**: `scripts/model_training/MODEL_TRAINING_GUIDE.md`
- **Hyperparameter Tuning**: `scripts/model_training/hyperparameter_tuning/HYPERPARAMETER_TUNING_GUIDE.md`
- **Training Config**: `scripts/model_training/config/CONFIG_GUIDE.md`
- **Model Validation**: `scripts/model_validation/README.md`
- **System Startup**: `scripts/startup/README.md`

---

## 🛠️ Best Practices

### Data Cleaning
1. Always backup raw data before cleaning
2. Review quality reports before proceeding to feature engineering
3. Document any manual interventions

### Feature Engineering
1. Verify feature dictionary is up to date
2. Check derived features match business logic
3. Validate target variable distribution

### Model Training
1. Use version control for model configurations
2. Document hyperparameters and results
3. Run validation after training
4. Keep training logs for reproducibility

### Model Validation
1. Validate models before production deployment
2. Check for data drift regularly
3. Monitor prediction consistency

### Debugging
1. Use debug scripts to isolate issues
2. Check logs for detailed error messages
3. Verify environment and dependencies

---

## 🚨 Troubleshooting

### Common Issues

**Data Cleaning:**
- **Issue**: Missing referential integrity
- **Solution**: Check foreign key relationships in raw data

**Feature Engineering:**
- **Issue**: Feature dictionary mismatch
- **Solution**: Update feature dictionary or verify column names

**Model Training:**
- **Issue**: Missing features in test data
- **Solution**: Run model validation to identify missing features

**API Startup:**
- **Issue**: Port already in use
- **Solution**: Change port in configuration or stop existing service

---

## 📦 Dependencies

All scripts share common dependencies from `config/requirements.txt`:
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning
- `xgboost`: Gradient boosting models
- `fastapi`: API framework
- `uvicorn`: ASGI server
- Additional dependencies per script type

---

## 🔐 Security Considerations

- Scripts validate input data to prevent injection
- Model artifacts are version-controlled
- Configuration secrets should use environment variables
- Debug scripts should not be used in production

---

## 📞 Support

For issues or questions:
1. Check relevant subdirectory README files
2. Review script docstrings and comments
3. Consult main project documentation
4. Review logs in `logs/` directory

---

*Last updated: See git commit history*
