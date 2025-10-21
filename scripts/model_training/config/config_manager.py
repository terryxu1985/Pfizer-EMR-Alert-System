"""
Configuration Manager Utility
Provides easy access to configuration with environment support and validation
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from .optimized_config import OptimizedModelConfig, Environment, ModelType


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager
        
        Args:
            project_root: Project root directory. If None, auto-detect from current file location
        """
        if project_root is None:
            # Auto-detect project root (4 levels up from this file)
            self.project_root = Path(__file__).parent.parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config_cache: Dict[str, OptimizedModelConfig] = {}
    
    def get_config(
        self, 
        environment: Optional[Environment] = None,
        config_file: Optional[Union[str, Path]] = None,
        cache: bool = True
    ) -> OptimizedModelConfig:
        """
        Get configuration for specified environment
        
        Args:
            environment: Environment type (development, testing, staging, production)
            config_file: Custom configuration file path
            cache: Whether to cache the configuration
            
        Returns:
            OptimizedModelConfig instance
        """
        # Determine environment
        if environment is None:
            environment = self._detect_environment()
        
        # Create cache key
        cache_key = f"{environment.value}_{config_file or 'default'}"
        
        # Return cached config if available
        if cache and cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # Determine config file
        if config_file is None:
            config_file = self.project_root / "scripts" / "model_training" / "config" / "environment_config.yaml"
        
        # Load configuration
        try:
            config = OptimizedModelConfig.load_from_yaml(config_file, environment)
            
            # Resolve paths
            config = config.resolve_paths(self.project_root)
            
            # Cache if requested
            if cache:
                self._config_cache[cache_key] = config
            
            self.logger.info(f"Loaded configuration for environment: {environment.value}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_development_config(self) -> OptimizedModelConfig:
        """Get development configuration"""
        return self.get_config(Environment.DEVELOPMENT)
    
    def get_testing_config(self) -> OptimizedModelConfig:
        """Get testing configuration"""
        return self.get_config(Environment.TESTING)
    
    def get_staging_config(self) -> OptimizedModelConfig:
        """Get staging configuration"""
        return self.get_config(Environment.STAGING)
    
    def get_production_config(self) -> OptimizedModelConfig:
        """Get production configuration"""
        return self.get_config(Environment.PRODUCTION)
    
    def validate_config(self, config: OptimizedModelConfig) -> Dict[str, Any]:
        """
        Validate configuration and return validation results
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        try:
            # Validate paths exist
            data_paths = config.path_config.get_data_paths()
            for name, path in data_paths.items():
                if not Path(path).exists():
                    validation_results['warnings'].append(f"Data path does not exist: {name} -> {path}")
            
            # Validate model type
            try:
                ModelType(config.model_type)
            except ValueError:
                validation_results['errors'].append(f"Invalid model type: {config.model_type}")
                validation_results['valid'] = False
            
            # Validate hyperparameters
            try:
                config.get_model_hyperparameters(config.model_type)
            except Exception as e:
                validation_results['errors'].append(f"Invalid hyperparameters: {e}")
                validation_results['valid'] = False
            
            # Check feature configuration
            features_to_remove = config.feature_config.get_features_to_remove()
            production_features = config.feature_config.production_features
            
            # Check for overlap between features to remove and production features
            overlap = set(features_to_remove) & set(production_features)
            if overlap:
                validation_results['errors'].append(f"Feature overlap detected: {overlap}")
                validation_results['valid'] = False
            
            validation_results['info'].append(f"Total production features: {len(production_features)}")
            validation_results['info'].append(f"Features to remove: {len(features_to_remove)}")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {e}")
            validation_results['valid'] = False
        
        return validation_results
    
    def create_custom_config(
        self,
        base_environment: Environment = Environment.PRODUCTION,
        overrides: Optional[Dict[str, Any]] = None
    ) -> OptimizedModelConfig:
        """
        Create custom configuration with overrides
        
        Args:
            base_environment: Base environment to start from
            overrides: Configuration overrides
            
        Returns:
            Custom OptimizedModelConfig instance
        """
        # Get base configuration
        config = self.get_config(base_environment, cache=False)
        
        if overrides:
            # Apply overrides
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                elif key == 'hyperparameters':
                    for model_type, params in value.items():
                        config.update_hyperparameters(model_type, params)
                else:
                    self.logger.warning(f"Unknown configuration key: {key}")
        
        # Re-validate after overrides
        validation_results = self.validate_config(config)
        if not validation_results['valid']:
            self.logger.warning(f"Configuration validation issues: {validation_results['errors']}")
        
        return config
    
    def _detect_environment(self) -> Environment:
        """Auto-detect environment from environment variables"""
        env_var = os.getenv('EMR_ENVIRONMENT', '').lower()
        
        environment_mapping = {
            'dev': Environment.DEVELOPMENT,
            'development': Environment.DEVELOPMENT,
            'test': Environment.TESTING,
            'testing': Environment.TESTING,
            'stage': Environment.STAGING,
            'staging': Environment.STAGING,
            'prod': Environment.PRODUCTION,
            'production': Environment.PRODUCTION
        }
        
        return environment_mapping.get(env_var, Environment.PRODUCTION)
    
    def clear_cache(self):
        """Clear configuration cache"""
        self._config_cache.clear()
        self.logger.info("Configuration cache cleared")
    
    def list_available_configs(self) -> Dict[str, str]:
        """List available configuration files"""
        config_dir = self.project_root / "scripts" / "model_training" / "config"
        config_files = {}
        
        if config_dir.exists():
            for file_path in config_dir.glob("*.yaml"):
                config_files[file_path.stem] = str(file_path)
        
        return config_files


# Global configuration manager instance
config_manager = ConfigManager()


# Convenience functions
def get_config(environment: Optional[Environment] = None) -> OptimizedModelConfig:
    """Get configuration for specified environment"""
    return config_manager.get_config(environment)


def get_development_config() -> OptimizedModelConfig:
    """Get development configuration"""
    return config_manager.get_development_config()


def get_testing_config() -> OptimizedModelConfig:
    """Get testing configuration"""
    return config_manager.get_testing_config()


def get_staging_config() -> OptimizedModelConfig:
    """Get staging configuration"""
    return config_manager.get_staging_config()


def get_production_config() -> OptimizedModelConfig:
    """Get production configuration"""
    return config_manager.get_production_config()


def validate_config(config: OptimizedModelConfig) -> Dict[str, Any]:
    """Validate configuration"""
    return config_manager.validate_config(config)
