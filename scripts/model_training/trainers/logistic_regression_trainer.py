"""
Logistic Regression model trainer
"""

import logging
from typing import Any
from sklearn.linear_model import LogisticRegression

from ..core.base_trainer import BaseTrainer


class LogisticRegressionTrainer(BaseTrainer):
    """Logistic Regression trainer"""
    
    def _create_model(self) -> Any:
        """Create Logistic Regression model"""
        hyperparameters = self.config.get_model_hyperparameters('logistic_regression')
        
        model = LogisticRegression(**hyperparameters)
        
        self.logger.info(f"Created Logistic Regression model with parameters: {hyperparameters}")
        
        return model
    
    def _get_model_name(self) -> str:
        """Get model name"""
        return "Logistic Regression"
