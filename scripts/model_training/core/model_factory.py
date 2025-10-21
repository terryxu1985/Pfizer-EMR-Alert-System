"""
Model factory for creating different types of trainers
"""

from typing import Dict, Type, Optional
import logging

from .base_trainer import BaseTrainer
from ..config.optimized_config import OptimizedModelConfig as ModelConfig


class ModelFactory:
    """Model factory class"""
    
    # Registered model types
    _trainers: Dict[str, Type[BaseTrainer]] = {}
    
    @classmethod
    def register_trainer(cls, model_type: str, trainer_class: Type[BaseTrainer]):
        """
        Register trainer class
        
        Args:
            model_type: Model type name
            trainer_class: Trainer class
        """
        cls._trainers[model_type] = trainer_class
        logging.info(f"Registered trainer: {model_type} -> {trainer_class.__name__}")
    
    @classmethod
    def create_trainer(cls, config: ModelConfig, model_type: Optional[str] = None) -> BaseTrainer:
        """
        Create trainer instance
        
        Args:
            config: Model configuration
            model_type: Model type, if None use model_type from config
            
        Returns:
            Trainer instance
            
        Raises:
            ValueError: If model type is not registered
        """
        if model_type is None:
            model_type = config.model_type
        
        if model_type not in cls._trainers:
            available_types = list(cls._trainers.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {available_types}"
            )
        
        trainer_class = cls._trainers[model_type]
        
        # Create config copy and set model type
        config_copy = ModelConfig(
            model_type=model_type,
            model_version=config.model_version,
            random_state=config.random_state,
            test_size=config.test_size,
            validation_size=config.validation_size,
            stratify=config.stratify,
            use_smote=config.use_smote,
            smote_random_state=config.smote_random_state,
            cv_folds=config.cv_folds,
            cv_random_state=config.cv_random_state,
            primary_metric=config.primary_metric,
            metrics=config.metrics,
            feature_config=config.feature_config,
            path_config=config.path_config,
            hyperparameter_config=config.hyperparameter_config
        )
        
        return trainer_class(config_copy)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get all available model types"""
        return list(cls._trainers.keys())
    
    @classmethod
    def is_model_supported(cls, model_type: str) -> bool:
        """Check if model type is supported"""
        return model_type in cls._trainers


# Auto-register model trainers
def auto_register_trainers():
    """Auto-register all available trainers"""
    try:
        from ..trainers.xgboost_trainer import XGBoostTrainer
        ModelFactory.register_trainer('xgboost', XGBoostTrainer)
    except ImportError:
        logging.warning("XGBoost trainer not available")
    
    try:
        from ..trainers.random_forest_trainer import RandomForestTrainer
        ModelFactory.register_trainer('random_forest', RandomForestTrainer)
    except ImportError:
        logging.warning("Random Forest trainer not available")
    
    try:
        from ..trainers.logistic_regression_trainer import LogisticRegressionTrainer
        ModelFactory.register_trainer('logistic_regression', LogisticRegressionTrainer)
    except ImportError:
        logging.warning("Logistic Regression trainer not available")
    
    try:
        from ..trainers.gradient_boosting_trainer import GradientBoostingTrainer
        ModelFactory.register_trainer('gradient_boosting', GradientBoostingTrainer)
    except ImportError:
        logging.warning("Gradient Boosting trainer not available")
    
    try:
        from ..trainers.svm_trainer import SVMTrainer
        ModelFactory.register_trainer('svm', SVMTrainer)
    except ImportError:
        logging.warning("SVM trainer not available")
    
    try:
        from ..trainers.naive_bayes_trainer import NaiveBayesTrainer
        ModelFactory.register_trainer('naive_bayes', NaiveBayesTrainer)
    except ImportError:
        logging.warning("Naive Bayes trainer not available")


# Auto-register at module load time
auto_register_trainers()
