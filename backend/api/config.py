"""
Configuration settings for the EMR Alert System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Model configuration
MODEL_CONFIG = {
    "model_name": "xgboost_emr_alert",
    "model_version": "2.1.0",
    "model_file": "xgboost_model.pkl",
    "scaler_file": "standard_scaler.pkl",
    "label_encoders_file": "label_encoders.pkl",
    "feature_columns_file": "feature_columns.pkl",
    "preprocessor_file": "preprocessor.pkl"
}

# Feature configuration - Updated to match training configuration
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

# API configuration
API_CONFIG = {
    "title": "EMR Alert System API",
    "description": "API for predicting patient treatment likelihood for Disease X",
    "version": "2.1.0",
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOG_DIR / "emr_alert_system.log"
}

# Create directories if they don't exist
for directory in [MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
