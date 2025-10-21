"""
Feature selection utilities
"""

import logging
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from ..config.optimized_config import OptimizedModelConfig as ModelConfig


class FeatureSelector:
    """Feature selector"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize feature selector
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.selected_features = None
        self.feature_scores = None
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'k_best', k: int = 15) -> List[str]:
        """
        Select features
        
        Args:
            X: Feature data
            y: Target variable
            method: Selection method ('k_best', 'mutual_info', 'manual')
            k: Number of features to select
            
        Returns:
            Selected features list
        """
        self.logger.info(f"Using {method} method to select features, selecting {k} features")
        
        if method == 'k_best':
            return self._select_k_best(X, y, k)
        elif method == 'mutual_info':
            return self._select_mutual_info(X, y, k)
        elif method == 'manual':
            return self._select_manual(X)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    def _select_k_best(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Select best features using F-statistic"""
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        self.selected_features = X.columns[selector.get_support()].tolist()
        self.feature_scores = dict(zip(X.columns, selector.scores_))
        
        self.logger.info(f"Selected {len(self.selected_features)} features")
        
        return self.selected_features
    
    def _select_mutual_info(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Select features using mutual information"""
        scores = mutual_info_classif(X, y, random_state=self.config.random_state)
        
        # Get features with highest scores
        feature_scores = list(zip(X.columns, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.selected_features = [feat[0] for feat in feature_scores[:k]]
        self.feature_scores = dict(feature_scores)
        
        self.logger.info(f"Selected {len(self.selected_features)} features")
        
        return self.selected_features
    
    def _select_manual(self, X: pd.DataFrame) -> List[str]:
        """Manually select features (using production features from config)"""
        production_features = self.config.feature_config.production_features
        available_features = [col for col in production_features if col in X.columns]
        
        self.selected_features = available_features
        self.logger.info(f"Manually selected {len(self.selected_features)} features")
        
        return self.selected_features
    
    def get_feature_importance_scores(self) -> Optional[dict]:
        """Get feature importance scores"""
        return self.feature_scores
    
    def get_selected_features(self) -> Optional[List[str]]:
        """Get selected features"""
        return self.selected_features
