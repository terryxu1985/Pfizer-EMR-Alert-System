"""
Naive Bayes model trainer
"""

import logging
from typing import Any
from sklearn.naive_bayes import GaussianNB

from ..core.base_trainer import BaseTrainer


class NaiveBayesTrainer(BaseTrainer):
    """Naive Bayes trainer"""
    
    def _create_model(self) -> Any:
        """Create Naive Bayes model"""
        hyperparameters = self.config.get_model_hyperparameters('naive_bayes')
        
        model = GaussianNB(**hyperparameters)
        
        self.logger.info(f"Created Naive Bayes model with parameters: {hyperparameters}")
        
        return model
    
    def _get_model_name(self) -> str:
        """Get model name"""
        return "Naive Bayes"
