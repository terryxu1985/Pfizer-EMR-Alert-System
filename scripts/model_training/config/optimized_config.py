"""
Optimized Configuration Management System
==========================================

This module provides the configuration STRUCTURE and LOGIC for the model training system.
It works together with environment_config.yaml which stores the actual configuration VALUES.

Configuration Architecture:
---------------------------
1. optimized_config.py (THIS FILE):
   - Defines data structures (dataclasses)
   - Provides configuration validation logic
   - Handles loading/saving YAML files
   - Defines default values as FALLBACK ONLY
   
2. environment_config.yaml:
   - Stores actual configuration values
   - Can be edited without changing code
   - Supports multiple environments (dev/test/staging/prod)
   - Values here OVERRIDE defaults in this file

Configuration Priority (high to low):
-------------------------------------
1. Environment-specific overrides in YAML (environments section)
2. Base configuration values in YAML
3. Default values in this Python file (fallback only)

Usage:
------
# Load configuration from YAML
config = OptimizedModelConfig.load_from_yaml('environment_config.yaml', environment=Environment.PRODUCTION)

# Access configuration
model_type = config.model_type
hyperparams = config.get_model_hyperparameters()

Important Notes:
---------------
- The actual production model should be determined by model_selection.py
- Do NOT hardcode model assumptions in configuration
- Default values here should match YAML for consistency
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from enum import Enum


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ModelType(Enum):
    """Supported model types"""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    NAIVE_BAYES = "naive_bayes"
    SVM = "svm"


@dataclass
class FeatureConfig:
    """Optimized feature configuration with validation"""
    
    # Data leakage features (automatically removed)
    data_leakage_features: List[str] = field(default_factory=lambda: [
        'PHYS_TREAT_RATE_ALL',      # Physician historical treatment rate - severe data leakage
        'PATIENT_ID',               # Patient ID - no predictive value, unique identifier
        'PHYSICIAN_ID',             # Physician ID - may cause overfitting, high cardinality
        'DISEASEX_DT',              # Diagnosis date - temporal leakage
        'SYMPTOM_ONSET_DT'          # Symptom onset date - temporal leakage
    ])
    
    # Temporal leakage features (none - temporal features are clinically valid)
    temporal_leakage_features: List[str] = field(default_factory=list)
    
    # Production features - optimized based on actual data analysis
    production_features: List[str] = field(default_factory=lambda: [
        # Patient demographic and risk features (8 features)
        'PATIENT_AGE', 'PATIENT_GENDER', 'RISK_IMMUNO', 'RISK_CVD',
        'RISK_DIABETES', 'RISK_OBESITY', 'RISK_NUM', 'RISK_AGE_FLAG',
        
        # Physician features (4 features)
        'PHYS_EXPERIENCE_LEVEL', 'PHYSICIAN_STATE', 'PHYSICIAN_TYPE', 'PHYS_TOTAL_DX',
        
        # Visit and temporal features (5 features) - clinically valid temporal features
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
    ])
    
    # Categorical variables
    categorical_features: List[str] = field(default_factory=lambda: [
        'PATIENT_GENDER', 'PHYS_EXPERIENCE_LEVEL', 'PHYSICIAN_STATE',
        'PHYSICIAN_TYPE', 'LOCATION_TYPE', 'INSURANCE_TYPE_AT_DX', 'DX_SEASON'
    ])
    
    # Target variable
    target_column: str = 'TARGET'
    
    def __post_init__(self):
        """Validate feature configuration"""
        self._validate_features()
    
    def _validate_features(self):
        """Validate feature configuration"""
        # Check for duplicates
        all_features = self.production_features + self.data_leakage_features + self.temporal_leakage_features
        if len(all_features) != len(set(all_features)):
            duplicates = [f for f in set(all_features) if all_features.count(f) > 1]
            raise ValueError(f"Duplicate features found: {duplicates}")
        
        # Check categorical features are in production features
        missing_categorical = set(self.categorical_features) - set(self.production_features)
        if missing_categorical:
            raise ValueError(f"Categorical features not in production features: {missing_categorical}")
    
    def get_features_to_remove(self) -> List[str]:
        """Get all features that need to be removed"""
        return self.data_leakage_features + self.temporal_leakage_features


@dataclass
class PathConfig:
    """Optimized path configuration with automatic resolution"""
    
    # Base paths (relative to project root)
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    model_ready_data_dir: str = "data/model_ready"
    model_dir: str = "backend/ml_models"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    
    # Specific file paths
    feature_dictionary_path: str = "data/model_ready/model_feature_dictionary.xlsx"
    model_ready_dataset_path: str = "data/model_ready/model_ready_dataset.csv"
    
    # Computed paths
    model_evaluation_dir: str = field(init=False)
    visualizations_dir: str = field(init=False)
    
    def __post_init__(self):
        """Initialize computed paths"""
        self.model_evaluation_dir = f"{self.reports_dir}/model_evaluation"
        self.visualizations_dir = f"{self.reports_dir}/model_evaluation"
    
    def resolve_paths(self, project_root: Union[str, Path]) -> 'PathConfig':
        """Resolve all paths to absolute paths"""
        project_root = Path(project_root)
        
        # Create a new instance with resolved paths
        resolved = PathConfig()
        resolved.raw_data_dir = str(project_root / self.raw_data_dir)
        resolved.processed_data_dir = str(project_root / self.processed_data_dir)
        resolved.model_ready_data_dir = str(project_root / self.model_ready_data_dir)
        resolved.model_dir = str(project_root / self.model_dir)
        resolved.reports_dir = str(project_root / self.reports_dir)
        resolved.logs_dir = str(project_root / self.logs_dir)
        resolved.feature_dictionary_path = str(project_root / self.feature_dictionary_path)
        resolved.model_ready_dataset_path = str(project_root / self.model_ready_dataset_path)
        resolved.model_evaluation_dir = str(project_root / self.model_evaluation_dir)
        resolved.visualizations_dir = str(project_root / self.visualizations_dir)
        
        return resolved
    
    def get_output_paths(self) -> Dict[str, str]:
        """Get all output paths"""
        return {
            'reports': self.reports_dir,
            'model_evaluation': self.model_evaluation_dir,
            'visualizations': self.visualizations_dir,
            'logs': self.logs_dir,
            'models': self.model_dir
        }
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all data paths"""
        return {
            'raw': self.raw_data_dir,
            'processed': self.processed_data_dir,
            'model_ready': self.model_ready_data_dir,
            'feature_dictionary': self.feature_dictionary_path,
            'dataset': self.model_ready_dataset_path
        }


@dataclass
class HyperparameterConfig:
    """
    Optimized hyperparameter configuration with validation
    
    Note: These are DEFAULT/FALLBACK values synchronized with environment_config.yaml.
    All hyperparameters are optimized to prevent overfitting based on analysis results.
    """
    
    # Default hyperparameters for all models
    # These match the base configuration in environment_config.yaml
    hyperparameters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'xgboost': {
            'n_estimators': 799,         # Optimized value from hyperparameter tuning
            'learning_rate': 0.2610570842357371,  # Optimized value
            'max_depth': 7,              # Optimized value
            'subsample': 0.9130034518395074,  # Optimized value
            'colsample_bytree': 0.8549084927445061,  # Optimized value
            'scale_pos_weight': None,    # Will be calculated dynamically based on data
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1,                # Use all CPU cores
            'early_stopping_rounds': 20, # Enable early stopping for better performance
            'reg_alpha': 0.3040562063700757,  # Optimized L1 regularization
            'reg_lambda': 1.9851311518315806,  # Optimized L2 regularization
            'min_child_weight': 3,       # Optimized value
            'gamma': 0.029685899839029244  # Optimized value
        },
        'random_forest': {
            'n_estimators': 50,           # Reduced to prevent overfitting
            'max_depth': 6,               # Reduced to prevent overfitting  
            'min_samples_split': 30,      # Increased to prevent overfitting
            'min_samples_leaf': 15,       # Increased to prevent overfitting
            'class_weight': 'balanced',   # Handle class imbalance
            'random_state': 42,
            'n_jobs': -1,
            'max_features': 'sqrt'        # Feature sampling
        },
        'logistic_regression': {
            'C': 0.1,                     # Regularization strength
            'penalty': 'l2',              # L2 regularization
            'solver': 'liblinear',        # Efficient for small datasets
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 2000,
            'fit_intercept': True
        },
        'gradient_boosting': {
            'max_iter': 100,              # Conservative base configuration (HistGradientBoosting uses max_iter)
            'learning_rate': 0.05,        # Moderate learning rate
            'max_depth': 6,               # Conservative depth (None for no limit)
            'class_weight': 'balanced',   # Handle class imbalance
            'random_state': 42,
            'max_leaf_nodes': 31,         # Control tree complexity (default is 31)
            'early_stopping': False       # Disable early stopping for base config
        },
        'naive_bayes': {
            'var_smoothing': 1e-09        # Smoothing parameter
        },
        'svm': {
            'C': 0.1,                     # Regularization parameter
            'kernel': 'rbf',              # Radial basis function kernel
            'gamma': 'scale',             # Kernel coefficient
            'class_weight': 'balanced',
            'random_state': 42,
            'probability': True           # Enable probability estimates
        }
    })
    
    def __post_init__(self):
        """Validate hyperparameters"""
        self._validate_hyperparameters()
    
    def _validate_hyperparameters(self):
        """Validate hyperparameter configuration"""
        for model_type, params in self.hyperparameters.items():
            if not isinstance(params, dict):
                raise ValueError(f"Hyperparameters for {model_type} must be a dictionary")
            
            # Validate required parameters for each model type
            if model_type == 'xgboost':
                required_params = ['n_estimators', 'learning_rate', 'max_depth']
                missing = [p for p in required_params if p not in params]
                if missing:
                    raise ValueError(f"Missing required XGBoost parameters: {missing}")
    
    def get_model_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameters for specified model"""
        if model_type not in self.hyperparameters:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.hyperparameters.keys())}")
        
        return self.hyperparameters[model_type].copy()
    
    def update_hyperparameters(self, model_type: str, hyperparameters: Dict[str, Any]):
        """Update model hyperparameters"""
        if model_type not in self.hyperparameters:
            self.hyperparameters[model_type] = {}
        
        self.hyperparameters[model_type].update(hyperparameters)
        self._validate_hyperparameters()


@dataclass
class OptimizedModelConfig:
    """
    Optimized model configuration with comprehensive validation and environment support
    
    Note: These are DEFAULT/FALLBACK values. Actual values should come from environment_config.yaml.
    The defaults here are kept in sync with YAML for consistency.
    
    The actual production model is determined dynamically by model_selection.py based on
    performance metrics, NOT by hardcoding here.
    """
    
    # Basic configuration
    model_type: str = 'random_forest'  # Default model (actual model determined by model_selection.py)
    model_version: str = '2.1.0'
    environment: Environment = Environment.PRODUCTION
    random_state: int = 42
    
    # Data splitting configuration
    test_size: float = 0.2
    validation_size: float = 0.2
    stratify: bool = True
    
    # Target variable configuration
    invert_target: bool = False  # Business-aligned target (false = no inversion)
    
    # Class imbalance handling
    # Note: Whether to use SMOTE depends on model type (see model_specific_smote)
    use_smote: bool = False  # Global default (overridden by model_specific_smote)
    smote_random_state: int = 42
    smote_target_ratio: float = 0.3  # Conservative ratio (1:3 instead of 1:1) to prevent overfitting
    smote_k_neighbors: int = 2  # Conservative k value to prevent overfitting
    
    # Model-specific SMOTE settings
    # IMPORTANT: For this dataset, positive class (TARGET=1) is the MAJORITY class (82.5%)
    # Traditional SMOTE would oversample the minority class (TARGET=0), which is INCORRECT
    # We should NOT use SMOTE in this scenario - rely on class_weight instead
    model_specific_smote: Dict[str, bool] = field(default_factory=lambda: {
        'gradient_boosting': False,  # Handles imbalance natively with class_weight
        'xgboost': False,            # Disabled - positive class is majority
        'random_forest': False,      # Disabled - use class_weight='balanced' instead
        'logistic_regression': False,  # Disabled - use class_weight='balanced' instead
        'svm': False,                # Disabled - use class_weight='balanced' instead
        'naive_bayes': False         # Disabled - positive class is majority
    })
    
    # Cross-validation configuration
    cv_folds: int = 5
    cv_random_state: int = 42
    
    # Evaluation metrics
    primary_metric: str = 'average_precision'  # PR-AUC metric
    metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'average_precision'
    ])
    
    # Sub-configurations
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    path_config: PathConfig = field(default_factory=PathConfig)
    hyperparameter_config: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate the entire configuration"""
        # Validate model type
        try:
            ModelType(self.model_type)
        except ValueError:
            raise ValueError(f"Invalid model type: {self.model_type}. Available: {[e.value for e in ModelType]}")
        
        # Validate data splitting
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        
        if not 0 < self.validation_size < 1:
            raise ValueError(f"validation_size must be between 0 and 1, got {self.validation_size}")
        
        if self.test_size + self.validation_size >= 1:
            raise ValueError(f"test_size + validation_size must be less than 1")
        
        # Validate CV folds
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be at least 2, got {self.cv_folds}")
        
        # Validate primary metric
        if self.primary_metric not in self.metrics:
            raise ValueError(f"primary_metric '{self.primary_metric}' must be in metrics list")
    
    @classmethod
    def load_from_yaml(cls, config_path: Union[str, Path], environment: Optional[Environment] = None) -> 'OptimizedModelConfig':
        """Load configuration from YAML file with environment support"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Handle environment-specific configuration
        if environment:
            env_config = config_data.get('environments', {}).get(environment.value, {})
            # Update base configuration with environment-specific overrides
            if 'model' in config_data:
                config_data['model'].update(env_config)
            else:
                config_data['model'] = env_config
        
        # Create sub-configurations
        feature_config = FeatureConfig(**config_data.get('features', {}))
        path_config = PathConfig(**config_data.get('paths', {}))
        
        # Handle hyperparameters separately to avoid keyword argument issues
        hyperparams_data = config_data.get('hyperparameters', {})
        hyperparameter_config = HyperparameterConfig()
        # Update hyperparameters instead of replacing the entire dictionary
        hyperparameter_config.hyperparameters.update(hyperparams_data)
        
        # Create main configuration
        model_config = config_data.get('model', {})
        model_config['feature_config'] = feature_config
        model_config['path_config'] = path_config
        model_config['hyperparameter_config'] = hyperparameter_config
        
        # Remove hyperparameters from model_config to avoid duplicate
        model_config.pop('hyperparameters', None)
        
        return cls(**model_config)
    
    def save_to_yaml(self, config_path: Union[str, Path]):
        """Save configuration to YAML file"""
        config_data = {
            'model': {
                'model_type': self.model_type,
                'model_version': self.model_version,
                'environment': self.environment.value,
                'random_state': self.random_state,
                'test_size': self.test_size,
                'validation_size': self.validation_size,
                'stratify': self.stratify,
                'invert_target': self.invert_target,
                'use_smote': self.use_smote,
                'smote_random_state': self.smote_random_state,
                'cv_folds': self.cv_folds,
                'cv_random_state': self.cv_random_state,
                'primary_metric': self.primary_metric,
                'metrics': self.metrics
            },
            'features': {
                'data_leakage_features': self.feature_config.data_leakage_features,
                'temporal_leakage_features': self.feature_config.temporal_leakage_features,
                'production_features': self.feature_config.production_features,
                'categorical_features': self.feature_config.categorical_features,
                'target_column': self.feature_config.target_column
            },
            'paths': {
                'raw_data_dir': self.path_config.raw_data_dir,
                'processed_data_dir': self.path_config.processed_data_dir,
                'model_ready_data_dir': self.path_config.model_ready_data_dir,
                'model_dir': self.path_config.model_dir,
                'feature_dictionary_path': self.path_config.feature_dictionary_path,
                'model_ready_dataset_path': self.path_config.model_ready_dataset_path,
                'reports_dir': self.path_config.reports_dir,
                'logs_dir': self.path_config.logs_dir
            },
            'hyperparameters': self.hyperparameter_config.hyperparameters
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def get_model_hyperparameters(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get hyperparameters for specified model"""
        if model_type is None:
            model_type = self.model_type
        
        return self.hyperparameter_config.get_model_hyperparameters(model_type)
    
    def update_hyperparameters(self, model_type: str, hyperparameters: Dict[str, Any]):
        """Update model hyperparameters"""
        self.hyperparameter_config.update_hyperparameters(model_type, hyperparameters)
    
    def resolve_paths(self, project_root: Union[str, Path]) -> 'OptimizedModelConfig':
        """Resolve all paths to absolute paths"""
        resolved_config = OptimizedModelConfig(
            model_type=self.model_type,
            model_version=self.model_version,
            environment=self.environment,
            random_state=self.random_state,
            test_size=self.test_size,
            validation_size=self.validation_size,
            stratify=self.stratify,
            invert_target=self.invert_target,
            use_smote=self.use_smote,
            smote_random_state=self.smote_random_state,
            cv_folds=self.cv_folds,
            cv_random_state=self.cv_random_state,
            primary_metric=self.primary_metric,
            metrics=self.metrics,
            feature_config=self.feature_config,
            path_config=self.path_config.resolve_paths(project_root),
            hyperparameter_config=self.hyperparameter_config
        )
        
        return resolved_config
    
    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment-specific configuration
        
        Note: These are programmatic defaults. Prefer using environment_config.yaml
        for actual configuration values as it's more maintainable.
        """
        env_configs = {
            Environment.DEVELOPMENT: {
                'cv_folds': 3,  # Faster iteration
                'hyperparameters': {
                    # XGBoost uses same optimized parameters as production for consistency
                    'random_forest': {'n_estimators': 50},
                    'gradient_boosting': {'max_iter': 80}  # HistGradientBoosting uses max_iter
                }
            },
            Environment.TESTING: {
                'cv_folds': 3,
                'test_size': 0.3,
                'hyperparameters': {
                    # XGBoost uses same optimized parameters as production for consistency
                    'random_forest': {'n_estimators': 25},
                    'gradient_boosting': {'max_iter': 50}  # HistGradientBoosting uses max_iter
                }
            },
            Environment.STAGING: {
                'cv_folds': 5,
                'hyperparameters': {
                    # XGBoost uses same optimized parameters as production for consistency
                    'random_forest': {'n_estimators': 100},
                    'gradient_boosting': {'max_iter': 150}  # HistGradientBoosting uses max_iter
                }
            },
            Environment.PRODUCTION: {
                'cv_folds': 5,
                # Note: Actual model selection should be done by model_selection.py
                'hyperparameters': {
                    # XGBoost uses optimized parameters from hyperparameter tuning
                    'random_forest': {
                        'n_estimators': 100,
                        'min_samples_split': 25,
                        'min_samples_leaf': 12
                    },
                    'gradient_boosting': {'max_iter': 150}  # HistGradientBoosting uses max_iter
                }
            }
        }
        
        return env_configs.get(self.environment, {})


# Backward compatibility aliases
ModelConfig = OptimizedModelConfig
PathsConfig = PathConfig
