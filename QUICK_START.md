# Pfizer EMR Alert System - Quick Start Guide

Welcome to the Pfizer EMR Alert System! This guide provides a quick overview to help you get up and running with the system.

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- At least 4GB RAM
- 500MB free disk space

### 1. Environment Setup

```bash
# Ensure Python 3.9+ is installed
python --version

# Navigate to project root (if not already there)
cd "/Users/st2025/Documents/GitHub/Project/Pfizer-EMR Alert System"

# Install dependencies
pip install -r config/requirements.txt
```

### 2. Data Preparation (First-Time Setup)

The first time you run the system, you will need to prepare the data:

```bash
# Clean and validate raw data
python scripts/data_cleaning/data_cleaning.py

# Generate model-ready dataset
python scripts/feature_engineering/build_model_ready_dataset.py
```

**Note**: The system includes pre-processed data in `data/model_ready/`, making this step optional if you wish to use the existing data.

### 3. Model Training (First-Time Setup)

To train a new model or retrain an existing one:

```bash
# Train the production model
python scripts/model_training/train_production_model.py
```

**Note**: The system includes pre-trained models in `backend/ml_models/models/`, making this step optional if you wish to use the existing models.

### 4. Start the API Service

Choose one of the following methods:

```bash
# Method 1: Use the startup script (Recommended)
python scripts/startup/run_api_only.py

# Method 2: Use uvicorn directly
uvicorn backend.api.api:app --host 0.0.0.0 --port 8000 --reload

# Method 3: Use the convenience script from project root
python scripts/startup/run_api_only.py
```

The API will be available at:
- **API Base**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`
- **Alternative Documentation**: `http://localhost:8000/redoc`

### 5. (Optional) Start the Web UI

To run the complete system with the web interface:

```bash
python scripts/startup/run_complete_system.py
```

The web UI will be available at `http://localhost:8001`

### 6. Test the System

```bash
# Test the API endpoints
python tests/test_api_endpoint.py

# Or use curl commands
curl http://localhost:8000/health

# Or visit the interactive documentation
# http://localhost:8000/docs
```

## 📁 Project Structure Overview

```
Pfizer-EMR Alert System/
├── backend/                      # Backend source code
│   ├── api/                      # FastAPI REST API
│   │   ├── api.py               # Main API application
│   │   ├── api_models.py        # Pydantic models
│   │   ├── config.py            # API configuration
│   │   ├── model_manager.py     # Model loading & prediction
│   │   └── API_GUIDE.md         # API documentation
│   ├── data_access/              # Data access layer
│   │   ├── emr_data_loader.py   # EMR data loading
│   │   ├── patient_repository.py # Patient data repository
│   │   ├── system_data_manager.py # System data management
│   │   └── DATA_ACCESS_GUIDE.md  # Data access guide
│   ├── data_processing/         # Data processing pipeline
│   │   ├── data_processor.py    # Data preprocessing
│   │   └── DATA_PROCESSING_GUIDE.md
│   ├── feature_engineering/     # Feature engineering
│   │   ├── emr_feature_processor.py # Feature processing
│   │   └── FEATURE_ENGINEERING_GUIDE.md
│   ├── ml_models/               # ML models storage
│   │   ├── models/              # Trained models (*.pkl)
│   │   └── MODELS_GUIDE.md      # Models documentation
│   └── monitoring/               # Monitoring & alerts
│       ├── alert_system.py      # Alert generation
│       ├── drift_detector.py    # Data drift detection
│       ├── metrics_collector.py # Metrics collection
│       └── performance_monitor.py # Performance monitoring
├── scripts/                      # Scripts and tools
│   ├── startup/                  # System startup scripts
│   │   ├── run_api_only.py      # API-only mode
│   │   ├── run_complete_system.py # Full system
│   │   └── run_quick_start.py   # Quick demo
│   ├── data_cleaning/           # Data cleaning scripts
│   ├── model_training/          # Model training scripts
│   ├── model_validation/        # Model validation
│   └── debug/                   # Debugging tools
├── config/                       # Configuration files
│   ├── settings.py              # Application settings
│   └── requirements.txt         # Python dependencies
├── data/                         # Data directory
│   ├── raw/                     # Raw input data
│   ├── processed/               # Processed data
│   ├── model_ready/             # Model-ready datasets
│   ├── storage/                 # Runtime data storage
│   └── versioning/              # Data versioning
├── tests/                        # Test suite
│   ├── test_api_endpoint.py     # API endpoint tests
│   └── input/                   # Test input files
├── frontend/                     # Web UI frontend
│   ├── server/                  # Flask UI server
│   ├── static/                  # Static assets (CSS, JS)
│   └── templates/               # HTML templates
├── reports/                      # Reports and analysis
│   └── model_evaluation/        # Model evaluation reports
├── logs/                         # System logs
├── bin/                         # Executable scripts
└── docker-compose.yml           # Docker orchestration
```

## 💻 Usage Examples

### Making Predictions via the API

```python
import requests
import json

# Example patient data
patient_data = {
    "patient_age": 65,
    "patient_gender": "M",
    "physician_specialty": "Cardiology",
    # ... other features
}

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json=patient_data
)

result = response.json()
print(f"Alert Probability: {result['alert_probability']}")
print(f"Alert Status: {result['alert_status']}")
```

### Using Model Manager Directly (Python)

```python
from backend.api.model_manager import ModelManager
import pandas as pd

# Initialize the model manager
model_manager = ModelManager()

# Load the trained model
model_manager.load_model()

# Make predictions with a pandas DataFrame
patient_df = pd.DataFrame([{
    'patient_age': 65,
    'patient_gender': 'M',
    # ... other features
}])

prediction = model_manager.predict(patient_df)
print(prediction)
```

### Batch Predictions via the API

```python
import requests

# Batch of patient data
batch_data = [
    {"patient_age": 65, "patient_gender": "M", ...},
    {"patient_age": 45, "patient_gender": "F", ...},
    # ... more patients
]

# Make a batch prediction request
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"patients": batch_data}
)

results = response.json()
for i, result in enumerate(results['predictions']):
    print(f"Patient {i+1}: {result['alert_status']}")
```

## 🔧 Common Commands

### Data Management

```bash
# Clean raw data and generate quality assessment reports
python scripts/data_cleaning/data_cleaning.py

# Build a model-ready dataset
python scripts/feature_engineering/build_model_ready_dataset.py

# Check data quality (detailed reporting)
python scripts/data_cleaning/data_quality_check_reporting.py
```

### Model Operations

```bash
# Train a new production model
python scripts/model_training/train_production_model.py

# Run model evaluation and comparison
python scripts/model_validation/comprehensive_model_evaluation.py

# Visualize model feature importance
python scripts/model_training/visualize_xgboost_importance.py
```

### System Startup

```bash
# Start API service only (recommended for production)
python scripts/startup/run_api_only.py

# Start the complete system (API + Web UI)
python scripts/startup/run_complete_system.py

# Quick start demo mode
python scripts/startup/run_quick_start.py

# Or use the bin scripts
bash bin/start_api.sh
bash bin/start_complete.sh
```

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Get model information
curl http://localhost:8000/model/info

# Get feature information
curl http://localhost:8000/model/features

# Test single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @tests/input/single_patient.json

# Test batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d @tests/input/batch_patients.json
```

### Debugging

```bash
# API debugging
python scripts/debug/debug_api.py

# Health check debugging
python scripts/debug/debug_health.py

# Prediction debugging
python scripts/debug/debug_prediction.py

# View system logs
tail -f logs/emr_alert_system.log
```

## 🐳 Docker Deployment

The system includes Docker support for straightforward deployment and containerization.

### Build and Run with Docker Compose

```bash
# Build and run with docker-compose
bash docker-setup.sh

# Or run manually:
docker-compose up

# Run in detached mode
docker-compose up -d
```

### Available Docker Profiles

```bash
# API-only service (lightweight)
docker-compose --profile api-only up

# Complete system (API + Web UI)
docker-compose --profile complete up

# Stop all services
bash bin/stop_all.sh
```

### Manual Docker Commands

```bash
# Build the Docker image
docker build -t pfizer-emr-alert .

# Run the container
docker run -p 8000:8000 \
  -e PYTHONPATH=/app \
  -e PYTHONUNBUFFERED=1 \
  -e ENVIRONMENT=docker \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data/storage:/app/data/storage \
  pfizer-emr-alert

# Check Docker health
bash docker-health-check.sh
```

## 📊 API Endpoints

The system exposes a RESTful API with the following key endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check and status |
| `/predict` | POST | Single patient prediction |
| `/predict/batch` | POST | Batch prediction for multiple patients |
| `/model/info` | GET | Retrieve model metadata and version information |
| `/model/features` | GET | Retrieve the list of features required by the model |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/redoc` | GET | Alternative API documentation (ReDoc) |

### Interactive Documentation

Once the API is running, visit:
- **Swagger UI**: `http://localhost:8000/docs` (recommended)
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces enable you to:
- View all available endpoints
- Test API calls directly from your browser
- Examine request/response schemas
- Review authentication requirements

## 🔍 Troubleshooting

### Common Issues

#### 1. Module Import Error
**Problem**: `ModuleNotFoundError` or import errors

**Solution**:
```bash
# Ensure you are running from the project root directory
cd "/Users/st2025/Documents/GitHub/Project/Pfizer-EMR Alert System"

# Verify and set the Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Run the startup script
python scripts/startup/run_api_only.py
```

#### 2. Missing Model Files
**Problem**: Model files not found in `backend/ml_models/models/`

**Solution**:
```bash
# Check if models exist
ls -la backend/ml_models/models/

# If missing, retrain the model
python scripts/model_training/train_production_model.py
```

#### 3. Port Already in Use
**Problem**: `Address already in use` error

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uvicorn backend.api.api:app --host 0.0.0.0 --port 8001
```

#### 4. Dependency Issues
**Problem**: Missing packages or version conflicts

**Solution**:
```bash
# Reinstall dependencies
pip install -r config/requirements.txt --force-reinstall --no-cache-dir

# Alternatively, create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r config/requirements.txt
```

#### 5. Data Not Found
**Problem**: Data files missing

**Solution**:
```bash
# Check if data exists
ls -la data/model_ready/

# If missing, process the data
python scripts/data_cleaning/data_cleaning.py
python scripts/feature_engineering/build_model_ready_dataset.py
```

### Log Viewing

```bash
# View system logs
tail -f logs/emr_alert_system.log

# View specific log files
tail -f logs/emr_alert_system_backup.log
tail -f logs/dynamic_model_selection.log

# View last model selection run
cat logs/last_model_selection_run.txt
```

### Testing System Health

```bash
# Run health check via API
curl http://localhost:8000/health

# Run Python health check
python scripts/debug/debug_health.py

# Run API endpoint tests
python tests/test_api_endpoint.py
```

## 📚 More Documentation

### In-Depth Guides

- **[API Guide](backend/api/API_GUIDE.md)** - Complete API documentation and usage
- **[Data Access Guide](backend/data_access/DATA_ACCESS_GUIDE.md)** - How to access and manage EMR data
- **[Data Processing Guide](backend/data_processing/DATA_PROCESSING_GUIDE.md)** - Data processing pipeline details
- **[Feature Engineering Guide](backend/feature_engineering/FEATURE_ENGINEERING_GUIDE.md)** - Feature engineering strategies
- **[Models Guide](backend/ml_models/MODELS_GUIDE.md)** - Model architecture and management
- **[Monitoring Guide](backend/monitoring/MONITORING_GUIDE.md)** - System monitoring and alerts

### Project Documentation

- **[Main README](README.md)** - Project overview and architecture
- **[Quick Start Guide](QUICK_START.md)** - This file
- **[Backend Guide](backend/BACKEND_GUIDE.md)** - Backend architecture
- **[Frontend Guide](frontend/FRONTEND_GUIDE.md)** - Frontend implementation
- **[Testing Guide](tests/TESTING_GUIDE.md)** - Testing documentation
- **[Data Guide](data/DATA_GUIDE.md)** - Data structure and usage

### Online Documentation

- **Interactive API Docs**: `http://localhost:8000/docs` (when API is running)
- **ReDoc API Docs**: `http://localhost:8000/redoc` (when API is running)

## 🆘 Get Help

### When You Encounter Issues

1. **Check Log Files**
   - System logs: `logs/emr_alert_system.log`
   - API logs: Review console output when running
   - Model logs: `logs/dynamic_model_selection.log`

2. **Verify Configuration**
   - Settings: `config/settings.py`
   - Requirements: `config/requirements.txt`

3. **Run Debug Scripts**
   - API debug: `python scripts/debug/debug_api.py`
   - Health debug: `python scripts/debug/debug_health.py`
   - Prediction debug: `python scripts/debug/debug_prediction.py`

4. **Review Documentation**
   - Review relevant guide files listed above
   - Explore API documentation at the `/docs` endpoint
   - Consult inline code comments

5. **Contact Support**
   - Review existing issues in the repository
   - Consult project documentation
   - Reach out to the development team

## 🎯 Quick Reference

### Essential Commands

```bash
# Start API
python scripts/startup/run_api_only.py

# Start Complete System
python scripts/startup/run_complete_system.py

# Run Tests
python tests/test_api_endpoint.py

# Health Check
curl http://localhost:8000/health
```

### Key Directories

- **Backend**: `backend/` - All backend code
- **Scripts**: `scripts/` - Utility and operational scripts
- **Data**: `data/` - All data files
- **Models**: `backend/ml_models/models/` - Trained ML models
- **Config**: `config/` - Configuration files
- **Logs**: `logs/` - System logs

---

**Happy Coding!** 🚀
