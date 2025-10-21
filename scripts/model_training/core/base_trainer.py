"""
Abstract base class for model trainers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import logging
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, classification_report, confusion_matrix
)

from ..config.optimized_config import OptimizedModelConfig as ModelConfig


class BaseTrainer(ABC):
    """Abstract base class for model trainers"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize trainer
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model = None
        self.metrics = {}
        self.feature_columns = None
        self.label_encoders = {}
        self.scaler = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def _create_model(self) -> Any:
        """Create model instance"""
        pass
    
    @abstractmethod
    def _get_model_name(self) -> str:
        """Get model name"""
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Starting training {self._get_model_name()} model...")
        
        # Create model
        self.model = self._create_model()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(X_train, y_train, prefix='train')
        
        # If validation set exists, calculate validation metrics
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self._calculate_metrics(X_val, y_val, prefix='val')
        
        # Merge metrics
        self.metrics = {**train_metrics, **val_metrics}
        
        self.logger.info(f"{self._get_model_name()} model training completed")
        self.logger.info(f"Training ROC-AUC: {train_metrics.get('train_roc_auc', 'N/A'):.2f}")
        
        return self.metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet, please call train() method first")
        
        self.logger.info(f"Evaluating {self._get_model_name()} model...")
        
        # Predict
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = self._cross_validate(X_test, y_test)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Store prediction results
        metrics['y_pred'] = y_pred
        metrics['y_pred_proba'] = y_pred_proba
        
        self.logger.info(f"Test ROC-AUC: {metrics.get('roc_auc', 'N/A'):.2f}")
        self.logger.info(f"Test PR-AUC: {metrics.get('pr_auc', 'N/A'):.2f}")
        self.logger.info(f"Cross-validation ROC-AUC: {metrics['cv_mean']:.2f} (+/- {metrics['cv_std'] * 2:.2f})")
        
        return metrics
    
    def _calculate_metrics(self, X: pd.DataFrame, y: pd.Series, prefix: str = '') -> Dict[str, float]:
        """Calculate metrics"""
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            f'{prefix}_accuracy': accuracy_score(y, y_pred),
            f'{prefix}_precision': precision_score(y, y_pred),
            f'{prefix}_recall': recall_score(y, y_pred),
            f'{prefix}_f1': f1_score(y, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics[f'{prefix}_roc_auc'] = roc_auc_score(y, y_pred_proba)
            metrics[f'{prefix}_pr_auc'] = average_precision_score(y, y_pred_proba)
        
        return metrics
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Cross-validation"""
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds, 
            shuffle=True, 
            random_state=self.config.cv_random_state
        )
        
        scores = cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring=self.config.primary_metric
        )
        
        return scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict"""
        if self.model is None:
            raise ValueError("Model not trained yet, please call train() method first")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet, please call train() method first")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self._get_model_name()} model does not support probability prediction")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet, please call train() method first")
        
        if not hasattr(self.model, 'feature_importances_'):
            self.logger.warning(f"{self._get_model_name()} model does not support feature importance")
            return None
        
        if self.feature_columns is None:
            self.logger.warning("Feature column names not set, cannot return feature importance")
            return None
        
        importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, model_path: str):
        """Save model with timestamped artifact and update fixed alias.
        Creates a timestamped model file and also updates a fixed-name
        file/symlink for easy consumption in production.
        """
        if self.model is None:
            raise ValueError("Model not trained yet, please call train() method first")
        
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Build filenames
        from datetime import datetime
        import os
        import shutil
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_base_name = f"{self._get_model_name().lower()}"
        # Timestamped artifact
        ts_model_file = model_path / f"{model_base_name}_v{self.config.model_version}_{timestamp}.pkl"
        # Fixed name (backward compatible)
        fixed_model_file = model_path / f"{model_base_name}_model.pkl"
        
        # Save timestamped model
        with open(ts_model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Update fixed alias: prefer symlink; fallback to copy on failure
        try:
            if fixed_model_file.exists() or fixed_model_file.is_symlink():
                fixed_model_file.unlink()
            os.symlink(ts_model_file.name, fixed_model_file)
        except Exception:
            # Fallback: copy file if symlink not permitted
            shutil.copy2(ts_model_file, fixed_model_file)
        
        # Save metadata
        metadata = {
            'model_type': self._get_model_name(),
            'version': self.config.model_version,
            'timestamp': timestamp,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'hyperparameters': self.config.get_model_hyperparameters(self.config.model_type),
            'artifacts': {
                'model_timestamped': ts_model_file.name,
                'model_fixed': fixed_model_file.name
            }
        }
        
        metadata_file = model_path / "model_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"Model saved: ts={ts_model_file.name}, fixed={fixed_model_file.name}")
    
    def load_model(self, model_path: str):
        """Load model"""
        model_path = Path(model_path)
        
        # Load model
        model_file = model_path / f"{self._get_model_name().lower()}_model.pkl"
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        metadata_file = model_path / "model_metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self.feature_columns = metadata.get('feature_columns')
                self.metrics = metadata.get('metrics', {})
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary"""
        summary = {
            'model_name': self._get_model_name(),
            'model_type': self.config.model_type,
            'version': self.config.model_version,
            'metrics': self.metrics,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'hyperparameters': self.config.get_model_hyperparameters(self.config.model_type)
        }
        
        if self.model is not None:
            summary['is_trained'] = True
        else:
            summary['is_trained'] = False
        
        return summary
