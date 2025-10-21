"""
Optimized Configuration Management for Model Training
Provides comprehensive, environment-aware configuration management
"""

# Import optimized configuration classes
from .optimized_config import (
    OptimizedModelConfig,
    FeatureConfig,
    PathConfig,
    HyperparameterConfig,
    Environment,
    ModelType
)

# Import configuration manager
from .config_manager import (
    ConfigManager,
    config_manager,
    get_config,
    get_development_config,
    get_testing_config,
    get_staging_config,
    get_production_config,
    validate_config
)

# Backward compatibility aliases
ModelConfig = OptimizedModelConfig
PathsConfig = PathConfig

# Export main classes and functions
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
    
    # Backward compatibility
    'ModelConfig',
    'PathsConfig'
]
