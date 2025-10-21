# Pfizer EMR Alert System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

## 🏥 Overview

The **Pfizer EMR Alert System** is an AI-powered clinical decision support system designed to optimize Drug A treatment in Disease X patients. This comprehensive system employs a hybrid decision-making approach that combines machine learning models with rule-based logic to provide real-time predictions, medication alerts, and clinical recommendations through an integrated Electronic Medical Record (EMR) interface.

### Key Features

- 🤖 **Hybrid AI System**: Combines machine learning models with rule-based logic for enhanced accuracy
- 📊 **Real-Time Analysis**: Live EMR data processing and feature engineering
- 🚨 **Smart Alerts**: Intelligent medication alerts and contraindication warnings using hybrid decision-making
- 👨‍⚕️ **Clinical Decision Support**: Evidence-based recommendations powered by ML and rule-based systems
- 🔄 **Multi-Service Architecture**: Scalable API and UI services with Docker containerization
- 📱 **Modern Web Interface**: Responsive UI with enhanced doctor input functionality
- 🛡️ **Security & Compliance**: HIPAA-compliant data handling and security hardening

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Docker & Docker Compose** (for containerized deployment)
- **4GB+ RAM** (8GB+ recommended)
- **2GB+ disk space**

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd "Pfizer-EMR Alert System"

# Build and run complete system
./bin/start_docker.sh complete

# Access the application
# UI: http://localhost:8080
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Start complete system (API + UI)
./bin/start_complete.sh

# Or start API only
./bin/start_api.sh

# Stop all services
./bin/stop_all.sh
```

### Option 3: Legacy Scripts (Deprecated)

```bash
# Install dependencies
pip install -r config/requirements_production.txt

# Start API service
python run_api_only.py

# Start complete system (API + UI)
python run_complete_system.py
```

### Option 3: Interactive Demo

```bash
# Run interactive demonstration
./docker-demo.sh
```

## 📋 System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   API Service   │
│   (Port 8080)   │◄──►│   (Port 8000)   │
│                 │    │                 │
│ • Patient Input │    │ • Predictions   │
│ • Results View  │    │ • Data Access   │
│ • Alerts        │    │ • Health Checks │
└─────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │            ┌─────────────────┐
         │            │ Decision Engine │
         │            │                 │
         │            │ • ML Models     │
         │            │ • Rule-Based    │
         │            │   Logic         │
         │            │ • Hybrid        │
         │            │   Decisions     │
         │            └─────────────────┘
         │                       │
         └───────────────────────┼───────────────────────
                                 │
                    ┌─────────────────┐
                    │   Data Layer    │
                    │                 │
                    │ • Patient Data  │
                    │ • EMR Records   │
                    │ • Model Storage │
                    └─────────────────┘
```

#### Hybrid Decision-Making Engine

The system's core intelligence lies in its **Hybrid Decision Engine** that combines:

- **🤖 Machine Learning Models**: Advanced XGBoost algorithms trained on historical patient data to predict treatment likelihood and risk factors
- **📋 Rule-Based Logic**: Clinical guidelines and medical protocols encoded as business rules for safety checks and contraindication detection
- **🔄 Ensemble Decision Making**: Both systems work together to provide comprehensive risk assessment and treatment recommendations

This hybrid approach ensures both **accuracy** (from ML models) and **safety** (from rule-based validation), providing healthcare providers with reliable, evidence-based clinical decision support.

### Backend Architecture

```
backend/
├── api/                    # API Layer - FastAPI application
│   ├── api.py             # Main API endpoints
│   ├── api_models.py      # Pydantic models
│   ├── config.py          # Configuration management
│   └── model_manager.py   # Model lifecycle management
├── data_access/           # Data Access Layer
│   ├── patient_repository.py      # Patient CRUD operations
│   ├── emr_data_loader.py         # EMR data loading
│   └── system_data_manager.py     # System data management
├── data_processing/       # Data Processing Layer
│   └── data_processor.py  # Data preprocessing utilities
├── feature_engineering/   # Feature Engineering Layer
│   └── emr_feature_processor.py   # EMR to features conversion
└── ml_models/             # Model Storage
    └── models/            # Serialized ML models
```

## 🌐 API Endpoints

### Core Prediction Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single patient prediction from raw EMR data |
| `/predict/batch` | POST | Batch predictions for multiple patients |
| `/health` | GET | System health check with model status |

### Patient Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/patients` | GET | Retrieve all stored patients |
| `/patients/{id}` | GET | Get specific patient by ID |
| `/patients` | POST | Add new patient to system |
| `/patients/stats` | GET | Patient database statistics |

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/model/info` | GET | Comprehensive model information |
| `/model/features` | GET | Feature information and importance |
| `/model/reload` | POST | Hot reload model if updated |
| `/model/validate-features` | POST | Validate feature consistency |

## 🐳 Docker Deployment Modes

### 1. Complete System
```bash
./docker-setup.sh complete
```
- Single container with API + UI
- Ports: 8000 (API), 8080 (UI)
- Best for: Development, testing, small deployments

### 2. API-Only Service
```bash
./docker-setup.sh api-only
```
- API service only
- Port: 8000 (API)
- Best for: Backend-only deployments, microservices integration

### 3. Microservices Architecture
```bash
./docker-setup.sh microservices
```
- Separate API and UI containers
- Ports: 8000 (API), 8080 (UI)
- Best for: Production deployments, scalability

### 4. UI-Only Service
```bash
./docker-setup.sh ui-only
```
- UI service only (requires separate API)
- Port: 8080 (UI)
- Best for: Frontend-only deployments

## 📊 Usage Examples

### Raw EMR Prediction

```python
import requests

# Prepare raw EMR data
raw_emr_data = {
    "patient_id": 12345,
    "birth_year": 1960,
    "gender": "M",
    "diagnosis_date": "2024-01-15T10:30:00Z",
    "transactions": [
        {
            "txn_dt": "2024-01-13T08:00:00Z",
            "physician_id": 1001,
            "txn_location_type": "OFFICE",
            "insurance_type": "COMMERCIAL",
            "txn_type": "SYMPTOMS",
            "txn_desc": "FEVER"
        }
    ],
    "physician_info": {
        "physician_id": 1001,
        "state": "NY",
        "physician_type": "Internal Medicine",
        "gender": "F",
        "birth_year": 1975
    }
}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=raw_emr_data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.3f}")
print(f"Alert Recommended: {result['alert_recommended']}")
```

### Batch Prediction

```python
# Prepare multiple patients
batch_data = {
    "patients": [
        {"patient_id": 12345, "birth_year": 1960, "gender": "M", ...},
        {"patient_id": 12346, "birth_year": 1985, "gender": "F", ...}
    ]
}

# Make batch prediction
response = requests.post("http://localhost:8000/predict/batch", json=batch_data)
result = response.json()

print(f"Total patients: {result['total_patients']}")
print(f"Alerts recommended: {result['alerts_recommended']}")
```

### Patient Management

```python
# Get all patients
response = requests.get("http://localhost:8000/patients?source=all&limit=10")
patients = response.json()

# Get patient statistics
response = requests.get("http://localhost:8000/patients/stats")
stats = response.json()

# Add new patient
new_patient = {
    "patient_name": "John Doe",
    "patient_age": 65,
    "patient_gender": "M",
    "physician_id": 1001,
    "diagnosis_date": "2024-10-21T10:00:00",
    "symptoms": ["FEVER", "COUGH"],
    "comorbidities": ["CVD", "Diabetes"]
}

response = requests.post("http://localhost:8000/patients", json=new_patient)
```

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=./backend/ml_models
SECRET_KEY=your-secret-key-here

# Database Configuration
DATABASE_URL=sqlite:///./emr_alert.db
```

### Model Configuration

The system automatically discovers models from multiple locations:
1. `backend/ml_models/models/` - Primary production storage
2. `scripts/model_training/models/` - Training pipeline output
3. `reports/model_evaluation/` - Model evaluation results

## 📈 Monitoring & Health Checks

### Built-in Health Monitoring

```bash
# Comprehensive health check
./docker-health-check.sh

# Check specific components
./docker-health-check.sh api          # API health only
./docker-health-check.sh ui           # UI service only
./docker-health-check.sh containers  # Container status
./docker-health-check.sh resources   # Resource usage

# Continuous monitoring
./docker-health-check.sh monitor
```

### Health Check Endpoints

```bash
# System health
curl http://localhost:8000/health

# Model information
curl http://localhost:8000/model/info

# Feature validation
curl -X POST http://localhost:8000/model/validate-features \
  -H "Content-Type: application/json" \
  -d '["PATIENT_AGE", "PATIENT_GENDER", "RISK_CVD"]'
```

## 🛡️ Security Features

### Container Security
- **Non-root user execution** for enhanced security
- **Minimal base image** (Python slim) to reduce attack surface
- **Read-only model mounts** for data protection
- **Network isolation** with custom Docker networks

### Data Security
- **Input validation** with comprehensive Pydantic models
- **Feature validation** to ensure data integrity
- **Audit logging** for compliance and monitoring
- **HIPAA-compliant** data handling

### API Security
- **CORS configuration** for production use
- **Error handling** that doesn't expose sensitive information
- **Rate limiting** protection against abuse
- **SSL/TLS** support for encrypted communication

## 📁 Project Structure

```
Pfizer-EMR Alert System/
├── bin/                        # 🚀 Startup Scripts (NEW!)
│   ├── start_api.sh           # Start API server only
│   ├── start_complete.sh      # Start complete system (API + UI)
│   ├── start_docker.sh        # Docker management script
│   ├── stop_all.sh            # Stop all services
│   └── README.md              # Scripts documentation
├── backend/                    # Backend services
│   ├── api/                   # API layer
│   ├── data_access/          # Data access layer
│   ├── data_processing/      # Data processing
│   ├── feature_engineering/  # Feature engineering
│   └── ml_models/            # ML models storage
├── frontend/                  # Frontend application
│   ├── templates/            # HTML templates
│   ├── static/               # Static assets
│   └── server/               # UI server
├── config/                   # Configuration files
│   ├── requirements.txt      # Development dependencies
│   ├── requirements_production.txt  # Production dependencies
│   └── settings.py           # System settings
├── data/                     # Data files
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed data
│   ├── model_ready/          # Model-ready datasets
│   └── storage/              # Runtime data storage
├── scripts/                  # Utility scripts
│   ├── startup/             # 🚀 Python startup scripts (NEW!)
│   │   ├── run_api_only.py  # API-only startup
│   │   ├── run_complete_system.py  # Complete system startup
│   │   ├── run_quick_start.py      # Quick start wrapper
│   │   └── README.md        # Startup scripts documentation
│   ├── data_cleaning/        # Data cleaning scripts
│   ├── model_training/       # Model training scripts
│   ├── deployment/           # Deployment scripts
│   └── debug/                # Debug utilities
├── tests/                    # Test files
├── logs/                     # Log files
├── reports/                   # Reports and documentation
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
├── docker-setup.sh          # Docker management script (DEPRECATED)
├── docker-health-check.sh   # Health monitoring script
├── docker-demo.sh           # Interactive demo script
├── MIGRATION_NOTICE.md      # Migration guide to new bin/ scripts
└── README.md                # This file
```

## 🚀 Startup Scripts

The system now includes organized startup scripts in the `bin/` directory for better management:

### Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `start_api.sh` | Start API server only | `./bin/start_api.sh` |
| `start_complete.sh` | Start complete system (API + UI) | `./bin/start_complete.sh` |
| `start_docker.sh` | Docker management | `./bin/start_docker.sh [command]` |
| `stop_all.sh` | Stop all services | `./bin/stop_all.sh` |

### Docker Commands

```bash
# Build and start complete system
./bin/start_docker.sh complete

# Start API only
./bin/start_docker.sh api-only

# Start as microservices
./bin/start_docker.sh microservices

# View logs
./bin/start_docker.sh logs

# Check status
./bin/start_docker.sh status

# Stop containers
./bin/start_docker.sh stop
```

### Local Development

```bash
# Start complete system
./bin/start_complete.sh

# Start API only
./bin/start_api.sh

# Stop everything
./bin/stop_all.sh
```

For detailed documentation, see `bin/README.md`.

### API Testing

```bash
# Run API tests
python tests/test_api_endpoint.py

# Test specific endpoints
curl http://localhost:8000/health
curl http://localhost:8000/model/info

# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 1,
    "age": 45,
    "gender": "M",
    "symptoms": ["fever", "cough"],
    "comorbidities": ["diabetes"],
    "physician_id": 1,
    "diagnosis_date": "2024-01-15"
  }'
```

### Docker Testing

```bash
# Test Docker deployment
./docker-setup.sh test

# Test health monitoring
./docker-health-check.sh

# Test different deployment modes
./docker-setup.sh api-only
./docker-setup.sh microservices
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
lsof -i :8000
lsof -i :8080

# Use different ports or stop conflicting services
```

#### 2. Model Loading Issues
```bash
# Check model files exist
ls -la backend/ml_models/models/

# Verify file permissions
chmod 644 backend/ml_models/models/*.pkl

# Check logs for loading errors
tail -f logs/emr_alert_system.log
```

#### 3. Docker Issues
```bash
# Check Docker status
docker-compose ps

# View container logs
docker-compose logs -f

# Restart containers
docker-compose restart
```

#### 4. API Not Responding
```bash
# Check container health
./docker-health-check.sh

# Test API endpoints
curl http://localhost:8000/health

# Check logs
./docker-setup.sh logs
```

### Debug Commands

```bash
# API debugging
python scripts/debug/debug_api.py

# Health check debugging
python scripts/debug/debug_health.py

# Prediction debugging
python scripts/debug/debug_prediction.py

# Open shell in container
./docker-setup.sh shell
```

## 📚 Documentation

### Comprehensive Guides

- **[Docker Guide](DOCKER_README.md)** - Complete Docker containerization guide
- **[Backend Guide](backend/BACKEND_GUIDE.md)** - Detailed backend architecture and API documentation
- **[Data Processing Guide](backend/data_processing/DATA_PROCESSING_GUIDE.md)** - Data processing workflows
- **[Feature Engineering Guide](backend/feature_engineering/FEATURE_ENGINEERING_GUIDE.md)** - Feature engineering methodology
- **[Models Guide](backend/ml_models/MODELS_GUIDE.md)** - ML models documentation
- **[Quick Start Guide](QUICK_START.md)** - Quick start instructions

### API Documentation

- **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc (ReDoc)
- **Health Check**: http://localhost:8000/health

## 🚀 Production Deployment

### Docker Production Setup

```bash
# Production deployment
./docker-setup.sh microservices

# Configure reverse proxy (nginx)
# Enable SSL/TLS
# Set up monitoring and alerting
# Configure log aggregation
```

### Environment Configuration

```bash
# Production environment variables
export API_HOST=0.0.0.0
export API_PORT=8000
export DEBUG=False
export LOG_LEVEL=INFO
export SECRET_KEY=your-production-secret-key
```

### Performance Optimization

- **Resource Limits**: Configure Docker resource limits
- **Load Balancing**: Use load balancers for high availability
- **Caching**: Implement response caching strategies
- **Monitoring**: Set up comprehensive monitoring and alerting

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd "Pfizer-EMR Alert System"

# Install development dependencies
pip install -r config/requirements.txt

# Run tests
python tests/test_api_endpoint.py

# Start development server
python run_api_only.py
```

### Code Standards

- Follow PEP 8 Python style guidelines
- Use type hints for better code documentation
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is proprietary software developed for Pfizer. All rights reserved.

## 🆘 Support

### Getting Help

1. **Check Documentation**: Review relevant guides in the `docs/` directory
2. **Check Logs**: Examine log files for error details
3. **Run Health Checks**: Use `./docker-health-check.sh` for diagnostics
4. **Test API**: Use `./docker-setup.sh test` for API testing
5. **Contact Team**: Reach out to the development team for assistance

### Useful Commands

```bash
# System status
./docker-setup.sh status

# View logs
./docker-setup.sh logs

# Health monitoring
./docker-health-check.sh

# Interactive demo
./docker-demo.sh

# Clean up resources
./docker-setup.sh cleanup
```

---

## 🎯 Key Benefits

- **🏥 Clinical Decision Support**: AI-powered recommendations for healthcare providers
- **⚡ Real-Time Processing**: Live EMR data analysis and prediction
- **🔒 Security & Compliance**: HIPAA-compliant with enterprise-grade security
- **🐳 Containerized Deployment**: Easy deployment with Docker and Docker Compose
- **📊 Comprehensive Monitoring**: Built-in health checks and monitoring capabilities
- **🔄 Scalable Architecture**: Microservices design for production scalability
- **📱 Modern Interface**: Responsive web UI with enhanced user experience

---

**Pfizer EMR Alert System** - AI-Powered Clinical Decision Support  
*Version 2.0.0*  
*Developed by Dr. Terry Xu*  
*Last Updated: October 2025*
