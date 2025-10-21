"""
XGBoost model trainer
"""

import logging
from typing import Any, Dict, Optional
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ..core.base_trainer import BaseTrainer


class XGBoostTrainer(BaseTrainer):
    """XGBoost trainer"""
    
    def _create_model(self) -> Any:
        """Create XGBoost model"""
        hyperparameters = self.config.get_model_hyperparameters('xgboost')
        
        # Remove scale_pos_weight from hyperparameters if it exists
        # It will be calculated dynamically during training
        hyperparameters.pop('scale_pos_weight', None)
        
        # Remove early_stopping_rounds from hyperparameters for model creation
        # It will be handled explicitly in train() method with eval_set
        # This prevents errors during sklearn cross_val_score which doesn't support eval_set
        hyperparameters.pop('early_stopping_rounds', None)
        
        model = xgb.XGBClassifier(**hyperparameters)
        
        self.logger.info(f"Created XGBoost model with parameters: {hyperparameters}")
        
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train XGBoost model with dynamic scale_pos_weight calculation and intelligent early stopping
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Starting training {self._get_model_name()} model...")
        
        # Calculate scale_pos_weight dynamically based on training data distribution
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1.0
        
        self.logger.info(f"Training data class distribution: {class_counts.to_dict()}")
        self.logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
        
        # Get hyperparameters and add dynamic scale_pos_weight
        hyperparameters = self.config.get_model_hyperparameters('xgboost')
        hyperparameters['scale_pos_weight'] = scale_pos_weight
        
        # Extract early_stopping_rounds before creating model (it's not a model parameter, it's a fit parameter)
        early_stopping_rounds = hyperparameters.pop('early_stopping_rounds', None)
        
        # Handle early stopping: need validation set
        eval_set = None
        created_val_set = False
        
        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            # Early stopping is enabled
            if X_val is None or y_val is None:
                # No validation set provided, create one from training data
                self.logger.info(f"Early stopping enabled ({early_stopping_rounds} rounds) but no validation set provided.")
                self.logger.info("Splitting training data: 80% train, 20% validation")
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, 
                    test_size=0.2, 
                    random_state=self.config.random_state,
                    stratify=y_train
                )
                created_val_set = True
                
                self.logger.info(f"Created validation set: {X_val.shape[0]} samples")
            
            # Set up evaluation set for early stopping
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.logger.info(f"Using early stopping with {early_stopping_rounds} rounds")
        else:
            # Early stopping disabled
            self.logger.info("Early stopping disabled")
        
        # Create model without early_stopping_rounds (it's not a constructor parameter)
        self.model = xgb.XGBClassifier(**hyperparameters)
        
        # Train model with or without early stopping
        if eval_set is not None:
            # Create model with early_stopping_rounds for this specific training
            # Note: We use set_params to add early_stopping_rounds only for this fit call
            model_with_early_stop = xgb.XGBClassifier(**hyperparameters)
            model_with_early_stop.set_params(early_stopping_rounds=early_stopping_rounds)
            
            model_with_early_stop.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Use the trained model
            self.model = model_with_early_stop
            
            # Log best iteration if early stopping was used
            if hasattr(self.model, 'best_iteration'):
                actual_iterations = self.model.best_iteration + 1
                self.logger.info(f"Early stopping triggered at iteration {actual_iterations} (out of {hyperparameters['n_estimators']} max)")
            else:
                self.logger.info(f"Trained for full {hyperparameters['n_estimators']} iterations (early stopping did not trigger)")
        else:
            self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(X_train, y_train, prefix='train')
        
        # If validation set exists, calculate validation metrics
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self._calculate_metrics(X_val, y_val, prefix='val')
            
            if created_val_set:
                self.logger.info(f"Validation ROC-AUC: {val_metrics.get('val_roc_auc', 'N/A'):.4f}")
        
        # Merge metrics
        self.metrics = {**train_metrics, **val_metrics}
        
        self.logger.info(f"{self._get_model_name()} model training completed")
        self.logger.info(f"Training ROC-AUC: {train_metrics.get('train_roc_auc', 'N/A'):.4f}")
        
        return self.metrics
    
    def _get_model_name(self) -> str:
        """Get model name"""
        return "XGBoost"
