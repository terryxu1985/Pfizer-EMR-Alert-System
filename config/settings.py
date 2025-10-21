"""
Configuration settings for Pfizer EMR Alert System
"""
import os
import glob
from pathlib import Path
from typing import Optional, Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "backend" / "ml_models"
SCRIPTS_MODEL_DIR = BASE_DIR / "scripts" / "model_training" / "models"
LOG_DIR = BASE_DIR / "logs"

# Model configuration with dynamic discovery support
# Updated to use XGBoost - best performing model based on model selection (2025-10-21)
# Performance metrics: PR-AUC=0.90, Precision=0.88, Recall=0.75, F1=0.81, CV ROC-AUC=0.88
MODEL_CONFIG = {
    "model_name": "xgboost_emr_alert",
    "model_version": "2.1.0",
    "model_file": "xgboost_model.pkl",
    "scaler_file": "standard_scaler.pkl", 
    "label_encoders_file": "label_encoders.pkl",
    "feature_columns_file": "feature_columns.pkl",
    "metadata_file": "model_metadata.pkl",
    "preprocessor_file": "preprocessor.pkl"
}

def get_latest_model_path() -> Optional[Path]:
    """
    Automatically discover the latest trained model from scripts output
    
    Returns:
        Path to the latest model directory, or None if not found
    """
    # Check backend ml_models directory first (where the actual model files are)
    if (BASE_DIR / "backend" / "ml_models" / "models").exists():
        return BASE_DIR / "backend" / "ml_models" / "models"
    
    # Check scripts model training output
    scripts_output_patterns = [
        BASE_DIR / "scripts" / "model_training" / "models",
        BASE_DIR / "reports" / "model_evaluation",
        BASE_DIR / "backend" / "ml_models"
    ]
    
    for pattern in scripts_output_patterns:
        if pattern.exists():
            # Look for model directories with timestamps or version numbers
            model_dirs = [d for d in pattern.iterdir() if d.is_dir()]
            if model_dirs:
                # Sort by modification time, get the latest
                latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                return latest_dir
    
    # Fallback to default model directory
    if MODEL_DIR.exists():
        return MODEL_DIR
    
    return None

def get_model_version_info(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get model version information from metadata
    
    Args:
        model_path: Path to model directory, if None uses latest model
        
    Returns:
        Dictionary containing model version information
    """
    if model_path is None:
        model_path = get_latest_model_path()
    
    if model_path is None:
        return {"version": "unknown", "training_date": "unknown", "model_type": "unknown"}
    
    metadata_file = model_path / MODEL_CONFIG["metadata_file"]
    
    if metadata_file.exists():
        try:
            import pickle
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            return {
                "version": metadata.get("version", "unknown"),
                "training_date": metadata.get("training_date", "unknown"),
                "model_type": metadata.get("model_type", "unknown"),
                "feature_count": metadata.get("feature_count", 0),
                "performance_metrics": metadata.get("performance_metrics", {})
            }
        except Exception as e:
            print(f"Warning: Could not load model metadata: {e}")
    
    return {"version": "unknown", "training_date": "unknown", "model_type": "unknown"}

# Feature configuration - 严格按照model_feature_dictionary.xlsx定义
FEATURE_CONFIG = {
    "categorical_columns": [
        'PATIENT_GENDER', 'PHYS_EXPERIENCE_LEVEL', 
        'PHYSICIAN_STATE', 'PHYSICIAN_TYPE', 'DX_SEASON', 
        'LOCATION_TYPE', 'INSURANCE_TYPE_AT_DX'
    ],
    "data_leakage_features": [
        'PHYS_TREAT_RATE_ALL',  # Doctor's historical treatment rate - data leakage (标记为NO)
        'PATIENT_ID',           # Patient ID - no predictive value (标记为No)
        'PHYSICIAN_ID'          # Physician ID - may cause overfitting (标记为No)
    ],
    "excluded_features": [
        'DISEASEX_DT',          # Date column - not suitable for ML (标记为No)
        'SYMPTOM_ONSET_DT',     # Date column - not suitable for ML (标记为No)
        'TARGET'                # Target variable - not a feature
    ],
    "target_column": "TARGET"
}

# API configuration
API_CONFIG = {
    "title": "Pfizer EMR Alert System API",
    "description": "API for predicting patient treatment likelihood for Disease X using XGBoost model (best performing with PR-AUC=0.90)",
    "version": "2.1.0",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "debug": os.getenv("DEBUG", "False").lower() == "true"
}

# Logging configuration
def get_log_level():
    """Get appropriate log level based on environment"""
    env_level = os.getenv("LOG_LEVEL", "").upper()
    if env_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        return env_level
    
    # Default based on debug mode
    if os.getenv("DEBUG", "False").lower() == "true":
        return "DEBUG"
    return "INFO"

LOGGING_CONFIG = {
    "level": get_log_level(),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOG_DIR / "emr_alert_system.log",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "rotation": "time",  # daily rotation
    "retention": "30 days",
    "console_output": os.getenv("LOG_TO_CONSOLE", "True").lower() == "true"
}

# Database configuration
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "sqlite:///./emr_alert.db"),
    "echo": os.getenv("DATABASE_ECHO", "False").lower() == "true"
}

# Security configuration
SECURITY_CONFIG = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-here"),
    "algorithm": "HS256",
    "access_token_expire_minutes": 30
}

# Create directories if they don't exist
for directory in [MODEL_DIR, LOG_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
