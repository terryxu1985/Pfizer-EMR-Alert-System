"""
Gradient Boosting model trainer using HistGradientBoostingClassifier
which supports class_weight parameter for handling class imbalance
"""

import logging
from typing import Any
from sklearn.ensemble import HistGradientBoostingClassifier

from ..core.base_trainer import BaseTrainer


class GradientBoostingTrainer(BaseTrainer):
    """Gradient Boosting trainer using HistGradientBoostingClassifier"""
    
    def _create_model(self) -> Any:
        """Create Gradient Boosting model with HistGradientBoostingClassifier"""
        hyperparameters = self.config.get_model_hyperparameters('gradient_boosting')
        
        # HistGradientBoostingClassifier supports class_weight parameter
        model = HistGradientBoostingClassifier(**hyperparameters)
        
        self.logger.info(f"Created HistGradientBoosting model with parameters: {hyperparameters}")
        
        return model
    
    def _get_model_name(self) -> str:
        """Get model name"""
        return "Gradient Boosting"
