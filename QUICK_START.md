# Pfizer EMR Alert System - Quick Start Guide

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Ensure Python 3.9+ is installed
python --version

# Install dependencies
pip install -r config/requirements_production.txt
```

### 2. Data Preparation (Optional)

```bash
# If you need to reprocess data (includes quality assessment)
python scripts/data_cleaning/data_cleaning.py
python scripts/data_cleaning/build_model_ready_dataset.py
```

### 3. Model Training (Optional)

```bash
# If you need to retrain the model
python scripts/model_training/train_production_model.py
```

### 4. Start API Service

```bash
# Method 1: Use startup script
python run_api_only.py

# Method 2: Use uvicorn directly
uvicorn backend.api.api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test API

```bash
# Run API tests
python tests/test_api.py

# Or visit API documentation
# http://localhost:8000/docs
```

## 📁 Project Structure Overview

```
Pfizer-EMR Alert System/
├── src/                          # Source code
│   ├── api/                      # API services
│   │   ├── api.py               # FastAPI application
│   │   ├── api_models.py        # Pydantic models
│   │   ├── config.py            # Configuration management
│   │   └── model_manager.py     # Model management
│   ├── data_processing/         # Data processing modules
│   │   ├── __init__.py          # Module initialization
│   │   └── data_processor.py    # Data preprocessing utilities
│   └── ml_models/              # Machine learning models
│       ├── feature_columns.pkl
│       ├── label_encoders.pkl
│       ├── model_metadata.pkl
│       ├── standard_scaler.pkl
│       └── xgboost_emr_alert_model.pkl
├── scripts/                      # Script files
│   ├── data_cleaning/           # Data cleaning scripts
│   ├── model_training/          # Model training scripts
│   ├── deployment/              # Deployment scripts
│   └── debug/                   # Debug scripts
├── config/                       # Configuration files
├── data/                         # Data files
├── tests/                        # Test files
├── docs/                         # Documentation
└── reports/                      # Reports
```

## 💻 Usage Examples

### Using Data Processing Module

```python
from src.data_processing.data_processor import DataProcessor
import pandas as pd

# Initialize data processor
processor = DataProcessor()

# Load training data
train_df = pd.read_csv('data/model_ready/model_ready_dataset.csv')

# Fit and transform training data
X_train = processor.fit_transform(train_df.drop('TARGET', axis=1))
y_train = train_df['TARGET']

# Transform new data
new_data = pd.read_csv('new_patient_data.csv')
X_new = processor.transform(new_data)
```

### Using API Module

```python
from src.api.model_manager import ModelManager

# Initialize model manager
model_manager = ModelManager()

# Load trained model
model_manager.load_model()

# Make prediction
prediction = model_manager.predict(patient_data)
```

## 🔧 Common Commands

### Data Cleaning
```bash
# Clean raw data and generate quality assessment reports
python scripts/data_cleaning/data_cleaning.py

# Check data quality (legacy script - now integrated above)
python scripts/data_cleaning/data_quality_check_reporting.py

# Build model dataset
python scripts/data_cleaning/build_model_ready_dataset.py
```

### Model Training
```bash
# Train production model
python scripts/model_training/train_production_model.py

# Run model evaluation
python scripts/model_training/clean_model_evaluation.py
```

### API Service
```bash
# Start service
python run_api_only.py

# Health check
curl http://localhost:8000/health

# Prediction test
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"patient_age": 65, "patient_gender": "M", ...}'
```

### Debugging
```bash
# API debugging
python scripts/debug/debug_api.py

# Health check debugging
python scripts/debug/debug_health.py

# Prediction debugging
python scripts/debug/debug_prediction.py
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t pfizer-emr-alert .

# Run API only service
docker-compose --profile api-only up

# Run complete system (API + UI)
docker-compose --profile complete up

# Run microservices architecture
docker-compose --profile microservices up

# Manual run container
docker run -p 8000:8000 \
  -e PYTHONPATH=/app \
  -e PYTHONUNBUFFERED=1 \
  -e ENVIRONMENT=docker \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data/storage:/app/data/storage \
  pfizer-emr-alert python run_api_only.py
```

## 📊 API Endpoints

- **Health Check**: `GET /health`
- **Single Patient Prediction**: `POST /predict`
- **Batch Prediction**: `POST /predict/batch`
- **Model Information**: `GET /model/info`
- **Feature Information**: `GET /model/features`
- **API Documentation**: `GET /docs`

## 🔍 Troubleshooting

### Common Issues

1. **Module Import Error**
   ```bash
   # Ensure running from project root directory
   cd "/Users/st2025/Documents/GitHub/Project/Pfizer-EMR Alert System"
   python run_api_only.py
   ```

2. **Missing Model Files**
   ```bash
   # Retrain the model
   python scripts/model_training/train_production_model.py
   ```

3. **Port Already in Use**
   ```bash
   # Use different port
   uvicorn src.api.api:app --port 8001
   ```

4. **Dependency Issues**
   ```bash
   # Reinstall dependencies
   pip install -r config/requirements_production.txt --force-reinstall
   ```

### Log Viewing

```bash
# View system logs
tail -f logs/emr_alert_system.log

# View API logs
python run_api_only.py --log-level debug
```

## 📚 More Documentation

- [Project Structure Guide](docs/PROJECT_STRUCTURE.md)
- [Production Deployment](docs/README_PRODUCTION.md)
- [Data Cleaning Guide](docs/DATA_CLEANING_GUIDE.md)
- [API Documentation](http://localhost:8000/docs)

## 🆘 Get Help

If you encounter issues, please:

1. Check log files
2. Verify configuration files
3. Run debug scripts
4. Review relevant documentation
5. Contact the development team
