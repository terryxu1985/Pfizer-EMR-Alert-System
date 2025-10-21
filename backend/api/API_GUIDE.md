# API Module Guide

## Overview

The `backend/api` module provides the FastAPI-based REST API service for the Pfizer EMR Alert System. This module handles all HTTP requests, data validation, model management, and response formatting for the production system.

## Architecture

```
backend/api/
├── api.py                    # Main FastAPI application with all endpoints
├── api_models.py             # Pydantic models for request/response validation
├── config.py                 # API-specific configuration management
├── model_manager.py          # Advanced model management and prediction orchestration
└── API_GUIDE.md              # This guide
```

## Components

### 1. Main API Application (`api.py`)

**Purpose**: Core FastAPI application that defines all REST endpoints and handles HTTP requests.

#### Key Features
- **FastAPI Framework**: High-performance async API with automatic OpenAPI documentation
- **Raw EMR Processing**: Direct processing of EMR transactions for real-time predictions
- **Patient Data Management**: Complete CRUD operations for patient records
- **Comprehensive Error Handling**: Detailed error responses with proper HTTP status codes
- **Input Validation**: Robust Pydantic models with field validation
- **Batch Processing**: Efficient batch prediction endpoints for multiple patients

#### Main Classes
- `FastAPI app`: Main application instance
- `ModelManager`: Dependency injection for model management
- `PatientInput`: Pydantic model for patient creation
- `PatientResponse`: Response model for patient operations

#### Key Endpoints

##### Core Prediction Endpoints
- **`POST /predict`** - Single patient prediction from raw EMR transactions
- **`POST /predict/batch`** - Batch predictions for multiple patients

##### Patient Management Endpoints
- **`GET /patients`** - Retrieve all stored patients
- **`GET /patients/stats`** - Get patient database statistics
- **`POST /patients`** - Add new patient to the system
- **`GET /patients/{patient_id}`** - Retrieve specific patient by ID

##### Model Management Endpoints
- **`GET /model/info`** - Comprehensive model information and metadata
- **`GET /model/features`** - Feature information and importance analysis
- **`POST /model/reload`** - Hot reload model if newer version available
- **`POST /model/validate-features`** - Validate feature consistency with training

##### System Endpoints
- **`GET /`** - Service information and status
- **`GET /health`** - Comprehensive health check with model status

#### Error Handling
```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"Status code: {exc.status_code}",
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )
```

#### Logging Configuration
```python
# Configure logging with rotation
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))

# Create formatter
formatter = logging.Formatter(LOGGING_CONFIG['format'])

# File handler with rotation
file_handler = logging.handlers.RotatingFileHandler(
    LOGGING_CONFIG['file'],
    maxBytes=LOGGING_CONFIG.get('max_bytes', 10 * 1024 * 1024),  # 10MB default
    backupCount=LOGGING_CONFIG.get('backup_count', 5)
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
```

### 2. API Models (`api_models.py`)

**Purpose**: Comprehensive Pydantic models for request/response validation and data serialization.

#### Key Features
- **Type Safety**: Strong typing with Pydantic validation
- **Field Validation**: Custom validators for business logic
- **Enum Support**: Predefined enums for categorical data
- **Response Models**: Structured response formats
- **Error Models**: Standardized error response format

#### Main Model Categories

##### Enums
```python
class Gender(str, Enum):
    """Patient gender enumeration"""
    M = "M"
    F = "F"

class ExperienceLevel(str, Enum):
    """Physician experience level enumeration"""
    LOW = "Low"
    MID = "Mid"
    HIGH = "High"

class TransactionType(str, Enum):
    """Transaction type enumeration"""
    CONDITIONS = "CONDITIONS"
    SYMPTOMS = "SYMPTOMS"
    TREATMENTS = "TREATMENTS"
    CONTRAINDICATIONS = "CONTRAINDICATIONS"
```

##### Request Models
```python
class RawEMRRequest(BaseModel):
    """Request model for raw EMR data prediction"""
    
    # Patient basic information
    patient_id: int = Field(..., description="Patient ID")
    birth_year: int = Field(..., ge=1900, le=2024, description="Patient birth year")
    gender: Gender = Field(..., description="Patient gender")
    
    # Diagnosis information
    diagnosis_date: datetime = Field(..., description="Disease X diagnosis date")
    
    # Transaction records
    transactions: List[TransactionRecord] = Field(..., min_items=1, description="List of transaction records")
    
    # Physician information
    physician_info: PhysicianInfo = Field(..., description="Physician information")
    
    @validator('transactions')
    def validate_transactions(cls, v):
        if len(v) == 0:
            raise ValueError('At least one transaction record is required')
        
        # Check for required transaction types
        txn_types = [txn.txn_type for txn in v]
        if TransactionType.CONDITIONS not in txn_types:
            raise ValueError('At least one CONDITIONS transaction is required')
        
        return v
```

##### Response Models
```python
class RawEMRPredictionResponse(BaseModel):
    """Enhanced response model for raw EMR prediction results"""
    
    prediction: int = Field(..., description="Prediction (0=Not Treated, 1=Treated)")
    probability: float = Field(..., ge=0, le=1, description="Probability of being treated")
    not_prescribed_drug_a: int = Field(..., description="1 if patient was not prescribed Drug A; else 0")
    not_prescribed_drug_a_probability: float = Field(..., ge=0, le=1, description="Probability that patient was not prescribed Drug A")
    alert_recommended: bool = Field(..., description="Whether an alert is recommended")
    
    # Clinical eligibility assessment
    clinical_eligibility: Optional[Dict[str, Any]] = Field(None, description="Clinical eligibility assessment results")
    
    # Feature engineering results
    processed_features: Dict[str, Any] = Field(..., description="Processed features used for prediction")
    feature_engineering_info: Dict[str, Any] = Field(..., description="Feature engineering process information")
    
    # Validation and metadata
    feature_validation: Optional[Dict[str, Any]] = Field(None, description="Feature validation results")
    model_version: Optional[str] = Field(None, description="Model version used for prediction")
    model_type: Optional[str] = Field(None, description="Model type used for prediction")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
```

##### Error Models
```python
class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")
```

### 3. Configuration (`config.py`)

**Purpose**: API-specific configuration management and environment settings.

#### Key Features
- **Model Configuration**: Model file paths and version management
- **Feature Configuration**: Production feature definitions and validation rules
- **API Configuration**: Server settings and deployment options
- **Logging Configuration**: Logging levels and file management

#### Configuration Sections

##### Model Configuration
```python
MODEL_CONFIG = {
    "model_name": "xgboost_emr_alert",
    "model_version": "2.1.0",
    "model_file": "xgboost_model.pkl",
    "scaler_file": "standard_scaler.pkl",
    "label_encoders_file": "label_encoders.pkl",
    "feature_columns_file": "feature_columns.pkl",
    "preprocessor_file": "preprocessor.pkl"
}
```

##### Feature Configuration
```python
FEATURE_CONFIG = {
    "production_features": [
        # Patient demographic and risk features (8 features)
        'PATIENT_AGE', 'PATIENT_GENDER', 'RISK_IMMUNO', 'RISK_CVD',
        'RISK_DIABETES', 'RISK_OBESITY', 'RISK_NUM', 'RISK_AGE_FLAG',
        
        # Physician features (4 features)
        'PHYS_EXPERIENCE_LEVEL', 'PHYSICIAN_STATE', 'PHYSICIAN_TYPE', 'PHYS_TOTAL_DX',
        
        # Visit and temporal features (5 features)
        'SYM_COUNT_5D', 'LOCATION_TYPE', 'INSURANCE_TYPE_AT_DX',
        'SYMPTOM_TO_DIAGNOSIS_DAYS', 'DIAGNOSIS_WITHIN_5DAYS_FLAG',
        
        # Symptom features (12 features)
        'SYMPTOM_ACUTE_PHARYNGITIS', 'SYMPTOM_ACUTE_URI', 'SYMPTOM_CHILLS',
        'SYMPTOM_CONGESTION', 'SYMPTOM_COUGH', 'SYMPTOM_DIARRHEA',
        'SYMPTOM_DIFFICULTY_BREATHING', 'SYMPTOM_FATIGUE', 'SYMPTOM_FEVER',
        'SYMPTOM_HEADACHE', 'SYMPTOM_LOSS_OF_TASTE_OR_SMELL', 'SYMPTOM_MUSCLE_ACHE',
        'SYMPTOM_NAUSEA_AND_VOMITING', 'SYMPTOM_SORE_THROAT',
        
        # Contraindication and engineered features (2 features)
        'PRIOR_CONTRA_LVL', 'DX_SEASON'
    ],
    "categorical_columns": [
        'PATIENT_GENDER', 'PHYS_EXPERIENCE_LEVEL', 'PHYSICIAN_STATE',
        'PHYSICIAN_TYPE', 'LOCATION_TYPE', 'INSURANCE_TYPE_AT_DX', 'DX_SEASON'
    ],
    "data_leakage_features": [
        'PHYS_TREAT_RATE_ALL',  # Doctor's historical treatment rate - data leakage
        'PATIENT_ID',           # Patient ID - no predictive value
        'PHYSICIAN_ID',         # Physician ID - may cause overfitting
        'DISEASEX_DT',          # Diagnosis date - temporal leakage
        'SYMPTOM_ONSET_DT'      # Symptom onset date - temporal leakage
    ],
    "target_column": "TARGET"
}
```

##### API Configuration
```python
API_CONFIG = {
    "title": "EMR Alert System API",
    "description": "API for predicting patient treatment likelihood for Disease X",
    "version": "2.1.0",
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True
}
```

### 4. Model Manager (`model_manager.py`)

**Purpose**: Advanced model management, prediction orchestration, and feature engineering coordination.

#### Key Features
- **Automatic Model Discovery**: Dynamically discovers and loads the latest trained models
- **Version Management**: Tracks model versions, metadata, and performance metrics
- **Hot Reloading**: Supports zero-downtime model updates
- **Feature Validation**: Comprehensive validation against training feature expectations
- **Raw EMR Processing**: Converts raw EMR transactions to model-ready features
- **Clinical Eligibility Assessment**: Rule-based clinical criteria evaluation

#### Main Classes
- `ModelManager`: Main model management class

#### Key Methods

##### Model Lifecycle Management
```python
def load_model(self, model_path: Optional[Path] = None) -> 'ModelManager':
    """Load the trained model and data processor with enhanced error handling"""

def save_model(self, model, data_processor: DataProcessor, 
               model_path: Optional[Path] = None, 
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save the trained model and data processor"""

def reload_model_if_updated(self) -> bool:
    """Check if a newer model is available and reload if necessary"""
```

##### Prediction Methods
```python
def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction on a single record with feature validation"""

def predict_batch_raw_emr(self, raw_data_list: List[Any]) -> Tuple[List[Dict[str, Any]], List[Any], float]:
    """Process and predict multiple raw EMR records"""

def process_raw_emr_data(self, raw_data: RawEMRRequest) -> Tuple[Dict[str, Any], Any]:
    """Process raw EMR data and convert to model features"""
```

##### Validation Methods
```python
def validate_feature_consistency(self, input_features: List[str]) -> Dict[str, Any]:
    """Validate that input features match the trained model's expected features"""

def validate_processed_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
    """Validate processed features using the EMR feature processor"""
```

##### Information Methods
```python
def get_model_info(self) -> Dict[str, Any]:
    """Get comprehensive model information and metadata"""
```

#### Clinical Eligibility Assessment
The model manager implements a comprehensive clinical eligibility assessment that combines AI predictions with rule-based clinical criteria:

```python
# Clinical eligibility check (fully rule-based)
clinical_eligibility = self.emr_feature_processor.check_clinical_eligibility(
    raw_data=raw_data,
    processed_features=processed_features
)

# AI prediction assessment
ai_predicts_missed = (prediction == 1) and (probability >= 0.7)

# Final alert decision
alert_recommended = ai_predicts_missed and meets_clinical_criteria
```

## API Endpoints Documentation

### Core Prediction Endpoints

#### Raw EMR Processing Predictions

##### `POST /predict`
Single patient prediction from raw EMR transactions.

**Request Body:**
```json
{
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
    },
    {
      "txn_dt": "2024-01-14T09:00:00Z",
      "physician_id": 1001,
      "txn_location_type": "OFFICE",
      "insurance_type": "COMMERCIAL",
      "txn_type": "CONDITIONS",
      "txn_desc": "HEART_DISEASE"
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
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.78,
  "not_prescribed_drug_a": 1,
  "not_prescribed_drug_a_probability": 0.78,
  "alert_recommended": true,
  "clinical_eligibility": {
    "meets_criteria": true,
    "age_eligible": true,
    "patient_age": 64,
    "within_5day_window": true,
    "is_high_risk": true,
    "risk_factors_found": ["CVD"],
    "no_severe_contraindication": true,
    "contraindication_level": 0
  },
  "processed_features": {
    "PATIENT_AGE": 64,
    "RISK_CVD": 1,
    "SYM_COUNT_5D": 2
  },
  "feature_engineering_info": {
    "symptom_count": 2,
    "risk_factors_found": ["CVD"],
    "time_to_diagnosis_days": 3,
    "contraindication_level": 0,
    "physician_experience_level": "High",
    "processing_warnings": []
  },
  "feature_validation": {
    "valid": true,
    "missing_features": [],
    "extra_features": []
  },
  "model_version": "2.1.0",
  "model_type": "XGBoost",
  "processing_time_ms": 45.2
}
```

##### `POST /predict/batch`
Batch predictions for multiple patients.

**Request Body:**
```json
{
  "patients": [
    {
      "patient_id": 12345,
      "birth_year": 1960,
      "gender": "M",
      "diagnosis_date": "2024-01-15T10:30:00Z",
      "transactions": [...],
      "physician_info": {...}
    },
    {
      "patient_id": 12346,
      "birth_year": 1985,
      "gender": "F",
      "diagnosis_date": "2024-01-16T14:20:00Z",
      "transactions": [...],
      "physician_info": {...}
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 1,
      "probability": 0.78,
      "alert_recommended": true,
      "processed_features": {...},
      "feature_engineering_info": {...},
      "processing_time_ms": 45.2
    }
  ],
  "total_patients": 2,
  "alerts_recommended": 1,
  "total_processing_time_ms": 90.4
}
```

### Patient Management Endpoints

##### `GET /patients`
Retrieve all stored patients.

**Query Parameters:**
- `source` (optional): Filter by data source (`demo`, `doctor_input`, `all`)
- `limit` (optional): Maximum number of patients to return

**Response:**
```json
{
  "patients": [
    {
      "id": 1001,
      "name": "John Smith",
      "age": 59,
      "gender": "Male",
      "diagnosisDate": "2024-01-15",
      "hasDiseaseX": true,
      "riskLevel": "High Risk",
      "hasAlert": true,
      "comorbidities": ["Heart Disease", "Diabetes"],
      "symptoms": ["Cough", "Fever"],
      "physician": {
        "id": 12345,
        "specialty": "Emergency Medicine",
        "experience": "15 years"
      },
      "dataSource": "Doctor Input",
      "riskScore": 8,
      "createdAt": "2024-01-15T10:30:00",
      "lastUpdated": "2024-01-15T10:30:00"
    }
  ],
  "total_count": 1,
  "data_sources": ["Doctor Input"],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

##### `POST /patients`
Add new patient to the system.

**Request Body:**
```json
{
  "patient_name": "John Doe",
  "patient_age": 65,
  "patient_gender": "M",
  "physician_id": 1001,
  "diagnosis_date": "2024-10-21T10:00:00",
  "symptom_onset_date": "2024-10-19T08:00:00",
  "location_type": "OFFICE",
  "insurance_type": "COMMERCIAL",
  "contraindication_level": "None",
  "comorbidities": ["CVD", "Diabetes"],
  "symptoms": ["FEVER", "COUGH"],
  "additional_notes": "Patient presents with typical symptoms"
}
```

**Response:**
```json
{
  "patient": {
    "id": 1002,
    "name": "John Doe",
    "age": 65,
    "gender": "Male",
    "diagnosisDate": "2024-10-21",
    "hasDiseaseX": true,
    "riskLevel": "High Risk",
    "hasAlert": false,
    "comorbidities": ["CVD", "Diabetes"],
    "symptoms": ["FEVER", "COUGH"],
    "dataSource": "Doctor Input",
    "riskScore": 7,
    "createdAt": "2024-10-21T10:00:00",
    "lastUpdated": "2024-10-21T10:00:00"
  },
  "message": "Patient created successfully and ready for AI analysis",
  "timestamp": "2024-10-21T10:00:00Z"
}
```

### Model Management Endpoints

##### `GET /model/info`
Get comprehensive model information and metadata.

**Response:**
```json
{
  "model_type": "XGBClassifier",
  "feature_count": 31,
  "feature_names": ["PATIENT_AGE", "PATIENT_GENDER", "RISK_CVD", ...],
  "is_loaded": true,
  "model_config": {
    "model_name": "xgboost_emr_alert",
    "model_version": "2.1.0",
    "model_file": "xgboost_model.pkl"
  },
  "current_model_path": "/path/to/model",
  "version_info": {
    "version": "2.1.0",
    "training_date": "2024-10-21T12:03:04",
    "model_type": "XGBoost",
    "performance_metrics": {
      "accuracy": 0.85,
      "precision": 0.82,
      "recall": 0.78,
      "f1_score": 0.80
    }
  },
  "feature_importances": {
    "PATIENT_AGE": 0.15,
    "RISK_CVD": 0.12,
    "SYM_COUNT_5D": 0.10
  }
}
```

##### `POST /model/validate-features`
Validate feature consistency with training.

**Request Body:**
```json
["PATIENT_AGE", "PATIENT_GENDER", "RISK_CVD", "SYM_COUNT_5D"]
```

**Response:**
```json
{
  "validation_result": {
    "valid": true,
    "missing_features": [],
    "extra_features": [],
    "expected_features": ["PATIENT_AGE", "PATIENT_GENDER", "RISK_CVD", ...],
    "input_features": ["PATIENT_AGE", "PATIENT_GENDER", "RISK_CVD", "SYM_COUNT_5D"],
    "feature_count_match": false
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### System Endpoints

##### `GET /health`
Comprehensive health check with model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_type": "XGBClassifier",
    "feature_count": 31,
    "version_info": {
      "version": "2.1.0",
      "training_date": "2024-10-21T12:03:04"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Usage Examples

### Raw EMR Processing Prediction
```python
import requests
from datetime import datetime

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
        },
        {
            "txn_dt": "2024-01-14T09:00:00Z",
            "physician_id": 1001,
            "txn_location_type": "OFFICE",
            "insurance_type": "COMMERCIAL",
            "txn_type": "CONDITIONS",
            "txn_desc": "HEART_DISEASE"
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

# Make raw EMR prediction
response = requests.post("http://localhost:8000/predict", json=raw_emr_data)
result = response.json()

print(f"Patient ID: {result.get('patient_id', 'N/A')}")
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.3f}")
print(f"Alert Recommended: {result['alert_recommended']}")
print(f"Processing Time: {result['processing_time_ms']:.2f}ms")

# Feature engineering details
fe_info = result['feature_engineering_info']
print(f"Symptom Count: {fe_info['symptom_count']}")
print(f"Risk Factors Found: {fe_info['risk_factors_found']}")
print(f"Time to Diagnosis: {fe_info['time_to_diagnosis_days']} days")
print(f"Physician Experience: {fe_info['physician_experience_level']}")
```

### Batch Prediction
```python
import requests

# Prepare multiple patients for batch processing
batch_data = {
    "patients": [
        {
            "patient_id": 12345,
            "birth_year": 1960,
            "gender": "M",
            "diagnosis_date": "2024-01-15T10:30:00Z",
            "transactions": [...],  # Transaction records
            "physician_info": {...}  # Physician information
        },
        {
            "patient_id": 12346,
            "birth_year": 1985,
            "gender": "F",
            "diagnosis_date": "2024-01-16T14:20:00Z",
            "transactions": [...],  # Transaction records
            "physician_info": {...}  # Physician information
        }
    ]
}

# Make batch prediction
response = requests.post("http://localhost:8000/predict/batch", json=batch_data)
result = response.json()

print(f"Total patients processed: {result['total_patients']}")
print(f"Alerts recommended: {result['alerts_recommended']}")
print(f"Total processing time: {result['total_processing_time_ms']:.2f}ms")

# Process individual results
for i, prediction in enumerate(result['predictions']):
    print(f"\nPatient {i+1}:")
    print(f"  Prediction: {prediction['prediction']}")
    print(f"  Probability: {prediction['probability']:.3f}")
    print(f"  Alert Recommended: {prediction['alert_recommended']}")
```

### Patient Management
```python
import requests

# Get all patients
response = requests.get("http://localhost:8000/patients?source=all&limit=10")
patients = response.json()

print(f"Total patients: {len(patients['patients'])}")
for patient in patients['patients'][:5]:
    print(f"Patient {patient['id']}: {patient['name']}")

# Get patient statistics
response = requests.get("http://localhost:8000/patients/stats")
stats = response.json()

print(f"\nPatient Statistics:")
print(f"  Total Patients: {stats['current_patients']['count']}")
print(f"  Data Sources: {stats['current_patients']['data_sources']}")

# Add new patient
new_patient = {
    "patient_name": "John Doe",
    "patient_age": 65,
    "patient_gender": "M",
    "physician_id": 1001,
    "diagnosis_date": "2024-10-21T10:00:00",
    "symptom_onset_date": "2024-10-19T08:00:00",
    "location_type": "OFFICE",
    "insurance_type": "COMMERCIAL",
    "contraindication_level": "None",
    "comorbidities": ["CVD", "Diabetes"],
    "symptoms": ["FEVER", "COUGH"]
}

response = requests.post("http://localhost:8000/patients", json=new_patient)
result = response.json()
print(f"\nPatient added: {result['message']}")
```

### Model Management
```python
import requests

# Get comprehensive model information
response = requests.get("http://localhost:8000/model/info")
model_info = response.json()

print(f"Model Type: {model_info['model_type']}")
print(f"Model Version: {model_info['version_info']['version']}")
print(f"Training Date: {model_info['version_info']['training_date']}")
print(f"Feature Count: {model_info['feature_count']}")
print(f"Model Path: {model_info['current_model_path']}")

# Get feature information and importance
response = requests.get("http://localhost:8000/model/features")
feature_info = response.json()

print(f"\nFeature Importance (Top 5):")
for feature, importance in sorted(feature_info['feature_importances'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {feature}: {importance:.4f}")

# Validate feature consistency
features_to_validate = ["PATIENT_AGE", "PATIENT_GENDER", "RISK_CVD", "SYM_COUNT_5D"]
response = requests.post("http://localhost:8000/model/validate-features", 
                        json=features_to_validate)
validation = response.json()

print(f"\nFeature Validation:")
print(f"  Valid: {validation['validation_result']['valid']}")
if not validation['validation_result']['valid']:
    print(f"  Missing Features: {validation['validation_result']['missing_features']}")
    print(f"  Extra Features: {validation['validation_result']['extra_features']}")

# Reload model if newer version available
response = requests.post("http://localhost:8000/model/reload")
reload_result = response.json()

if reload_result['reloaded']:
    print("\nModel reloaded successfully!")
else:
    print("\nNo newer model found - using current model")
```

## Error Handling

The API provides comprehensive error handling with detailed error responses and proper HTTP status codes:

### Validation Errors (422 Unprocessable Entity)
```json
{
    "error": "Validation Error",
    "detail": "Patient age must be between 0 and 120",
    "timestamp": "2024-01-01T12:00:00Z"
}
```

**Common validation errors:**
- Missing required fields
- Invalid data types
- Values outside acceptable ranges
- Malformed transaction data

### Service Unavailable (503 Service Unavailable)
```json
{
    "error": "Model not loaded",
    "detail": "Status code: 503",
    "timestamp": "2024-01-01T12:00:00Z"
}
```

**Common service unavailable scenarios:**
- Model not loaded or failed to load
- Model files missing or corrupted
- Feature processor not initialized

### Internal Server Error (500 Internal Server Error)
```json
{
    "error": "Internal Server Error",
    "detail": "Feature engineering failed: Invalid transaction format",
    "timestamp": "2024-01-01T12:00:00Z"
}
```

**Common internal errors:**
- Feature engineering failures
- Model prediction errors
- Data processing exceptions

## Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Model Configuration
MODEL_PATH=./backend/ml_models
LOG_LEVEL=INFO

# Security Configuration
SECRET_KEY=your-secret-key-here
```

### Model Discovery
The system automatically discovers models from multiple locations in priority order:
1. **`backend/ml_models/models/`** - Primary production model storage
2. **`scripts/model_training/models/`** - Training pipeline output
3. **`reports/model_evaluation/`** - Model evaluation results
4. **`backend/ml_models/`** - Fallback model directory

## Deployment

### Local Development
```bash
# Install dependencies
pip install -r config/requirements_production.txt

# Set environment variables
export API_HOST=0.0.0.0
export API_PORT=8000
export DEBUG=True
export LOG_LEVEL=DEBUG

# Run the API server
python run_api_only.py

# Or using uvicorn directly
uvicorn backend.api.api:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment
```bash
# Production environment setup
export API_HOST=0.0.0.0
export API_PORT=8000
export DEBUG=False
export LOG_LEVEL=INFO
export SECRET_KEY=your-production-secret-key

# Install production dependencies
pip install -r config/requirements_production.txt

# Run with gunicorn for production
gunicorn backend.api.api:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 --access-logfile - --error-logfile -
```

### Docker Deployment
```bash
# Build the complete system
docker build -t pfizer-emr-alert .

# Run API only service
docker-compose --profile api-only up

# Run complete system (API + UI)
docker-compose --profile complete up

# Run microservices architecture
docker-compose --profile microservices up

# Manual run with specific ports
docker run -p 8000:8000 \
  -e PYTHONPATH=/app \
  -e PYTHONUNBUFFERED=1 \
  -e ENVIRONMENT=docker \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data/storage:/app/data/storage \
  -v $(pwd)/backend/ml_models:/app/backend/ml_models:ro \
  pfizer-emr-alert python run_api_only.py
```

## Monitoring & Observability

### Health Check
```bash
curl http://localhost:8000/health
```

Response includes:
- Service status (healthy/unhealthy)
- Model loaded status
- Model information and metadata
- Response timestamp

### Model Monitoring
```bash
# Get model information
curl http://localhost:8000/model/info

# Check feature consistency
curl -X POST http://localhost:8000/model/validate-features \
  -H "Content-Type: application/json" \
  -d '["PATIENT_AGE", "PATIENT_GENDER", "RISK_CVD"]'

# Reload model if updated
curl -X POST http://localhost:8000/model/reload
```

### API Documentation
- **Interactive Swagger UI**: Available at `http://localhost:8000/docs`
- **ReDoc Documentation**: Available at `http://localhost:8000/redoc`

## Performance Optimization

### System Performance
- **Async FastAPI**: High concurrency and non-blocking I/O
- **Efficient Data Processing**: Optimized preprocessing pipeline
- **Model Caching**: Intelligent model loading and caching strategies
- **Batch Processing**: Efficient handling of multiple requests

### Scalability Features
- **Horizontal Scaling**: Stateless design supports load balancing
- **Resource Optimization**: Efficient memory and CPU usage
- **Connection Pooling**: Optimized database and external connections
- **Caching Strategies**: Response and model metadata caching

## Security Considerations

### Data Protection
- **Input Validation**: Comprehensive validation prevents injection attacks
- **Feature Validation**: Ensures data integrity and consistency
- **Error Handling**: Error messages don't expose sensitive information
- **Audit Logging**: Complete audit trail for compliance

### HIPAA Compliance
- **Data Privacy**: Patient data handling follows HIPAA requirements
- **Access Control**: Proper authentication and authorization
- **Data Encryption**: Sensitive data encrypted in transit and at rest
- **Audit Trails**: Comprehensive logging for compliance reporting

### Network Security
- **CORS Configuration**: Properly configured for production use
- **HTTPS**: SSL/TLS encryption for data in transit
- **Firewall Rules**: Appropriate network access controls
- **API Rate Limiting**: Protection against abuse and DoS attacks

## Best Practices

### 1. Model Management
- **Model Updates**: Train new models in `scripts/` directory
- **Automatic Discovery**: Models are automatically discovered and loaded
- **Hot Reloading**: Use `/model/reload` endpoint to check for updates
- **Version Tracking**: Monitor model versions and performance metrics
- **Backup Strategy**: Maintain multiple model versions for rollback capability

### 2. Feature Consistency
- **Pre-Prediction Validation**: Always validate features before prediction
- **Feature Monitoring**: Use `/model/validate-features` endpoint regularly
- **Warning Handling**: Monitor and address feature validation warnings
- **Data Quality**: Implement data quality checks in your EMR integration

### 3. Error Handling & Resilience
- **Graceful Degradation**: Implement proper error handling in client applications
- **Status Checking**: Check model status before making predictions
- **Retry Logic**: Implement exponential backoff for transient failures
- **Circuit Breakers**: Use circuit breaker patterns for external dependencies

### 4. Performance Optimization
- **Batch Processing**: Use batch predictions for multiple patients
- **Connection Pooling**: Implement proper HTTP connection pooling
- **Response Caching**: Cache model metadata and feature information
- **Data Caching**: Patient repository implements intelligent data caching
- **Load Balancing**: Use load balancers for high availability

## Troubleshooting

### Common Issues

#### 1. Model Loading Issues
**Problem**: Model not loaded or loading failures
```bash
# Check model files exist
ls -la backend/ml_models/models/

# Verify file permissions
chmod 644 backend/ml_models/models/*.pkl

# Check logs for loading errors
tail -f logs/emr_alert_system.log
```

**Solutions**:
- Ensure model files exist in expected locations
- Verify model file permissions (readable)
- Check logs for specific loading errors
- Validate model file integrity

#### 2. Feature Validation Errors
**Problem**: Feature mismatch between training and serving
```python
# Check feature consistency
curl -X POST http://localhost:8000/model/validate-features \
  -H "Content-Type: application/json" \
  -d '["PATIENT_AGE", "PATIENT_GENDER", "RISK_CVD"]'
```

**Solutions**:
- Ensure input features match training features exactly
- Check for missing or extra features
- Verify feature data types and ranges
- Use feature validation endpoint before prediction

#### 3. Prediction Errors
**Problem**: Prediction failures or unexpected results
```python
# Check model status
curl http://localhost:8000/health

# Validate input data
# Ensure all required fields are present and valid
```

**Solutions**:
- Validate input data format and completeness
- Check for missing values in required fields
- Verify feature ranges and data types
- Review error logs for specific failure reasons

### Debugging Tools

#### API Debugging
```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/model/info

# Check API documentation
open http://localhost:8000/docs
```

#### Log Analysis
```bash
# Monitor real-time logs
tail -f logs/emr_alert_system.log

# Search for specific errors
grep "ERROR" logs/emr_alert_system.log

# Check model loading logs
grep "Model loaded" logs/emr_alert_system.log
```

## Dependencies

### Core Dependencies
- **FastAPI**: High-performance web framework
- **Pydantic**: Data validation and settings management
- **uvicorn**: ASGI server for FastAPI
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

### Model Dependencies
- **XGBoost**: Machine learning model framework
- **scikit-learn**: Data preprocessing and validation
- **pickle**: Model serialization

### System Dependencies
- **logging**: Logging functionality
- **pathlib**: File path operations
- **datetime**: Date/time handling
- **typing**: Type hints and annotations

## Testing

### Unit Testing
```python
import pytest
from fastapi.testclient import TestClient
from backend.api.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "unhealthy"]

def test_predict_endpoint():
    test_data = {
        "patient_id": 12345,
        "birth_year": 1960,
        "gender": "M",
        "diagnosis_date": "2024-01-15T10:30:00Z",
        "transactions": [...],
        "physician_info": {...}
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code in [200, 503]  # 503 if model not loaded
```

### Integration Testing
```python
def test_model_loading():
    from backend.api.model_manager import ModelManager
    
    manager = ModelManager()
    assert manager.is_loaded == True
    assert manager.model is not None
    assert manager.data_processor is not None

def test_feature_validation():
    from backend.api.model_manager import ModelManager
    
    manager = ModelManager()
    features = ["PATIENT_AGE", "PATIENT_GENDER", "RISK_CVD"]
    validation = manager.validate_feature_consistency(features)
    assert "valid" in validation
    assert "missing_features" in validation
```

## Future Enhancements

### Model Management
- **Model A/B Testing**: Support for multiple model versions in production
- **Automated Rollback**: Automatic fallback to previous model versions
- **Model Drift Detection**: Real-time monitoring of model performance degradation
- **Continuous Learning**: Online learning capabilities for model updates

### Advanced Features
- **Real-time Monitoring**: Enhanced model performance monitoring dashboard
- **Feature Drift Detection**: Automated detection of feature distribution changes
- **Explainable AI**: Enhanced model interpretability and explanation features
- **Multi-model Ensemble**: Support for ensemble predictions across multiple models

### Integration Enhancements
- **EMR Integration**: Direct integration with major EMR systems
- **Real-time Streaming**: Support for real-time data streaming and processing
- **Advanced Analytics**: Comprehensive analytics and reporting capabilities
- **Mobile API**: Optimized API endpoints for mobile applications

## Technical Specifications

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: Minimum 4GB RAM, Recommended 8GB+ RAM
- **Storage**: Minimum 2GB disk space for models and logs
- **Network**: HTTP/HTTPS access for API endpoints

### Performance Benchmarks
- **Single Prediction**: < 100ms average response time
- **Batch Prediction**: < 500ms for 100 patients
- **Concurrent Requests**: 100+ requests per second
- **Model Loading**: < 5 seconds startup time

## Support & Documentation

### Additional Resources
- **API Documentation**: Available at `/docs` when service is running
- **Backend Guide**: `backend/BACKEND_GUIDE.md`
- **Data Processing Guide**: `backend/data_processing/DATA_PROCESSING_GUIDE.md`
- **Feature Engineering Guide**: `backend/feature_engineering/FEATURE_ENGINEERING_GUIDE.md`
- **Data Access Guide**: `backend/data_access/DATA_ACCESS_GUIDE.md`
- **Models Guide**: `backend/ml_models/MODELS_GUIDE.md`

### Getting Help
- **Logs**: Check `logs/emr_alert_system.log` for detailed error information
- **Health Endpoint**: Use `/health` to verify service status
- **Model Info**: Use `/model/info` to check model status and metadata
- **Feature Validation**: Use `/model/validate-features` to debug feature issues

---

## Document Change Log

### Version 1.0.0 - October 21, 2025

**Initial Release:**
1. **Comprehensive API Documentation**
   - Complete endpoint documentation with request/response examples
   - Detailed model management and prediction workflows
   - Patient management API documentation

2. **Architecture Documentation**
   - Module structure and component responsibilities
   - Configuration management and environment setup
   - Error handling and validation patterns

3. **Usage Examples**
   - Raw EMR processing examples
   - Batch prediction workflows
   - Patient management operations
   - Model management and monitoring

4. **Deployment Guide**
   - Local development setup
   - Production deployment instructions
   - Docker containerization
   - Performance optimization guidelines

5. **Troubleshooting Guide**
   - Common issues and solutions
   - Debugging tools and techniques
   - Log analysis and monitoring

---

**Pfizer EMR Alert System** - API Module Guide  
*Version 1.0.0*  
*Last Updated: October 21, 2025*  
*Developed by Dr. Terry Xu*
