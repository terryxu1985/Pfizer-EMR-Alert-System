"""
SVM model trainer
"""

import logging
from typing import Any
from sklearn.svm import SVC

from ..core.base_trainer import BaseTrainer


class SVMTrainer(BaseTrainer):
    """Support Vector Machine trainer"""
    
    def _create_model(self) -> Any:
        """Create SVM model"""
        hyperparameters = self.config.get_model_hyperparameters('svm')
        
        model = SVC(**hyperparameters)
        
        self.logger.info(f"Created SVM model with parameters: {hyperparameters}")
        
        return model
    
    def _get_model_name(self) -> str:
        """Get model name"""
        return "SVM"
