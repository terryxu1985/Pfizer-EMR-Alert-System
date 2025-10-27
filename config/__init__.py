"""
Configuration module for Pfizer EMR Alert System
"""

__version__ = "2.1.0"

from .settings import (
    BASE_DIR,
    DATA_DIR,
    MODEL_DIR,
    SCRIPTS_MODEL_DIR,
    LOG_DIR,
    MODEL_CONFIG,
    FEATURE_CONFIG,
    API_CONFIG,
    LOGGING_CONFIG,
    DATABASE_CONFIG,
    SECURITY_CONFIG,
    get_latest_model_path,
    get_model_version_info,
    get_log_level
)

__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'MODEL_DIR',
    'SCRIPTS_MODEL_DIR',
    'LOG_DIR',
    'MODEL_CONFIG',
    'FEATURE_CONFIG',
    'API_CONFIG',
    'LOGGING_CONFIG',
    'DATABASE_CONFIG',
    'SECURITY_CONFIG',
    'get_l会给st_model_path',
    'get_model_version_info',
    'get_log_level'
]

