"""
Random Forest model trainer
"""

import logging
from typing import Any
from sklearn.ensemble import RandomForestClassifier

from ..core.base_trainer import BaseTrainer


class RandomForestTrainer(BaseTrainer):
    """Random Forest trainer"""
    
    def _create_model(self) -> Any:
        """Create Random Forest model"""
        hyperparameters = self.config.get_model_hyperparameters('random_forest')
        
        model = RandomForestClassifier(**hyperparameters)
        
        self.logger.info(f"Created Random Forest model with parameters: {hyperparameters}")
        
        return model
    
    def _get_model_name(self) -> str:
        """Get model name"""
        return "Random Forest"
