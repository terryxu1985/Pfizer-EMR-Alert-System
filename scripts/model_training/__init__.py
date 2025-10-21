"""
Pfizer EMR Alert System - Model Training Module

This module provides a modular and extensible framework for training machine learning models
for the EMR Alert System. It includes:

- Unified configuration management
- Abstract base classes for trainers and evaluators
- Factory pattern for model creation
- Pipeline-based training workflows
- Comprehensive model evaluation and comparison

Usage:
    from scripts.model_training import TrainingPipeline, ModelConfig
    
    config = ModelConfig()
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
"""

__version__ = "2.1.0"
__author__ = "Pfizer EMR Alert System Team"

# Import optimized configuration classes
from .config.optimized_config import (
    OptimizedModelConfig,
    FeatureConfig,
    PathConfig,
    HyperparameterConfig,
    Environment,
    ModelType
)

# Import configuration manager
from .config.config_manager import (
    ConfigManager,
    config_manager,
    get_config,
    get_development_config,
    get_testing_config,
    get_staging_config,
    get_production_config,
    validate_config
)

# Import pipelines and core components
from .pipelines.training_pipeline import TrainingPipeline
from .pipelines.evaluation_pipeline import EvaluationPipeline
from .core.model_factory import ModelFactory

# Import comprehensive evaluator
from .evaluators.comprehensive_evaluator import ComprehensiveModelEvaluator

# Backward compatibility aliases
ModelConfig = OptimizedModelConfig
PathsConfig = PathConfig

__all__ = [
    # Main configuration classes
    'OptimizedModelConfig',
    'FeatureConfig', 
    'PathConfig',
    'HyperparameterConfig',
    
    # Enums
    'Environment',
    'ModelType',
    
    # Configuration manager
    'ConfigManager',
    'config_manager',
    
    # Convenience functions
    'get_config',
    'get_development_config',
    'get_testing_config', 
    'get_staging_config',
    'get_production_config',
    'validate_config',
    
    # Pipelines and core components
    'TrainingPipeline',
    'EvaluationPipeline',
    'ModelFactory',
    
    # Comprehensive evaluator
    'ComprehensiveModelEvaluator',
    
    # Backward compatibility
    'ModelConfig',
    'PathsConfig'
]
