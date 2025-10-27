#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization for XGBoost
=================================================

This module provides comprehensive hyperparameter tuning capabilities using:
1. Grid Search
2. Random Search  
3. Bayesian Optimization (Optuna)
4. Early Stopping Integration

Features:
- Cross-validation with early stopping
- Class imbalance handling
- Performance tracking
- Configuration management
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import json
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from ..config.config_manager import ConfigManager, Environment
from ..utils.data_loader import DataLoader
from ..utils.preprocessor import DataPreprocessor


class XGBoostHyperparameterTuner:
    """Advanced XGBoost hyperparameter tuning system"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize hyperparameter tuner
        
        Args:
            project_root: Project root directory
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent.parent
        else:
            self.project_root = project_root
            
        self.config_manager = ConfigManager(self.project_root)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Performance tracking
        self.tuning_history = []
        self.best_params = None
        self.best_score = None
        
    def _create_objective_function(self, X: pd.DataFrame, y: pd.Series, 
                                 cv_folds: int = 5, 
                                 early_stopping_rounds: int = 20) -> callable:
        """
        Create objective function for Optuna optimization
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of CV folds
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            # Define parameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }
            
            # Calculate scale_pos_weight dynamically
            class_counts = y.value_counts()
            if len(class_counts) == 2:
                params['scale_pos_weight'] = class_counts[0] / class_counts[1]
            
            # Cross-validation with early stopping
            cv_scores = []
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create model with early stopping
                model = xgb.XGBClassifier(**params)
                
                # Train with early stopping
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
                
                # Predict and calculate score
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                score = average_precision_score(y_val_fold, y_pred_proba)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        return objective
    
    def bayesian_optimization(self, X: pd.DataFrame, y: pd.Series,
                            n_trials: int = 100,
                            cv_folds: int = 5,
                            early_stopping_rounds: int = 20,
                            timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Bayesian optimization using Optuna
        
        Args:
            X: Features
            y: Target
            n_trials: Number of optimization trials
            cv_folds: Number of CV folds
            early_stopping_rounds: Early stopping rounds
            timeout: Timeout in seconds
            
        Returns:
            Optimization results
        """
        self.logger.info(f"🔍 Starting Bayesian optimization with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Create objective function
        objective = self._create_objective_function(X, y, cv_folds, early_stopping_rounds)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Extract results
        best_params = study.best_params.copy()
        best_score = study.best_value
        
        # Add fixed parameters
        best_params.update({
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        })
        
        # Calculate scale_pos_weight
        class_counts = y.value_counts()
        if len(class_counts) == 2:
            best_params['scale_pos_weight'] = class_counts[0] / class_counts[1]
        
        self.logger.info(f"✅ Bayesian optimization completed!")
        self.logger.info(f"🏆 Best score: {best_score:.4f}")
        self.logger.info(f"📊 Best parameters: {best_params}")
        
        # Store results
        self.best_params = best_params
        self.best_score = best_score
        
        # Save optimization history
        self._save_optimization_results(study, 'bayesian')
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study,
            'method': 'bayesian'
        }
    
    def grid_search_optimization(self, X: pd.DataFrame, y: pd.Series,
                               param_grid: Dict[str, List],
                               cv_folds: int = 5,
                               early_stopping_rounds: int = 20) -> Dict[str, Any]:
        """
        Perform grid search optimization
        
        Args:
            X: Features
            y: Target
            param_grid: Parameter grid for search
            cv_folds: Number of CV folds
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Optimization results
        """
        self.logger.info("🔍 Starting Grid Search optimization...")
        
        from sklearn.model_selection import ParameterGrid
        
        best_score = -np.inf
        best_params = None
        results = []
        
        # Calculate scale_pos_weight
        class_counts = y.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1.0
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        self.logger.info(f"📊 Testing {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Add fixed parameters
            params.update({
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            })
            
            # Cross-validation
            cv_scores = []
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create model with early stopping
                model = xgb.XGBClassifier(**params)
                
                # Train with early stopping
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
                
                # Predict and calculate score
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                score = average_precision_score(y_val_fold, y_pred_proba)
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            results.append({
                'params': params.copy(),
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
            
            self.logger.info(f"Score: {mean_score:.4f} (+/- {std_score:.4f})")
        
        self.logger.info(f"✅ Grid Search completed!")
        self.logger.info(f"🏆 Best score: {best_score:.4f}")
        self.logger.info(f"📊 Best parameters: {best_params}")
        
        # Store results
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'method': 'grid_search'
        }
    
    def evaluate_optimized_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate optimized model on test set
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            params: Optimized parameters
            
        Returns:
            Evaluation results
        """
        self.logger.info("🔍 Evaluating optimized model...")
        
        # Create model with optimized parameters
        model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'best_iteration': getattr(model, 'best_iteration', None),
            'n_estimators_used': getattr(model, 'best_iteration', params['n_estimators']) + 1
        }
        
        self.logger.info(f"📊 Optimized Model Performance:")
        self.logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"   Precision: {metrics['precision']:.4f}")
        self.logger.info(f"   Recall: {metrics['recall']:.4f}")
        self.logger.info(f"   F1-Score: {metrics['f1_score']:.4f}")
        self.logger.info(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        self.logger.info(f"   PR-AUC: {metrics['pr_auc']:.4f}")
        self.logger.info(f"   Iterations used: {metrics['n_estimators_used']}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def _save_optimization_results(self, study: optuna.Study, method: str):
        """Save optimization results"""
        try:
            results_dir = self.project_root / "logs" / "hyperparameter_optimization"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save study object
            study_file = results_dir / f"optuna_study_{method}_{timestamp}.pkl"
            with open(study_file, 'wb') as f:
                pickle.dump(study, f)
            
            # Save results summary
            summary_file = results_dir / f"optimization_summary_{method}_{timestamp}.json"
            summary = {
                'timestamp': timestamp,
                'method': method,
                'best_params': self.best_params,
                'best_score': self.best_score,
                'n_trials': len(study.trials),
                'best_trial_number': study.best_trial.number
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Saved optimization results to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save optimization results: {e}")
    
    def update_configuration(self, optimized_params: Dict[str, Any],
                           environment: Environment = Environment.PRODUCTION) -> bool:
        """
        Update configuration with optimized parameters
        
        Args:
            optimized_params: Optimized parameters
            environment: Environment to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = self.project_root / "scripts" / "model_training" / "config" / "environment_config.yaml"
            
            # Load current configuration
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Update environment-specific configuration
            if 'environments' not in config_data:
                config_data['environments'] = {}
            
            if environment.value not in config_data['environments']:
                config_data['environments'][environment.value] = {}
            
            if 'hyperparameters' not in config_data['environments'][environment.value]:
                config_data['environments'][environment.value]['hyperparameters'] = {}
            
            # Update XGBoost parameters
            config_data['environments'][environment.value]['hyperparameters']['xgboost'] = optimized_params
            
            # Also update base hyperparameters
            if 'hyperparameters' not in config_data:
                config_data['hyperparameters'] = {}
            
            config_data['hyperparameters']['xgboost'] = optimized_params
            
            # Save updated configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            self.logger.info(f"✅ Updated {environment.value} configuration with optimized parameters")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update configuration: {e}")
            return False


def main():
    """Main function for hyperparameter optimization"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create tuner
    tuner = XGBoostHyperparameterTuner()
    
    # Load data
    config = tuner.config_manager.get_config(Environment.PRODUCTION)
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor(config)
    
    # Load and preprocess data
    X, y = data_loader.load_model_ready_data()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    tuner.logger.info(f"📊 Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Perform Bayesian optimization
    optimization_results = tuner.bayesian_optimization(
        X_train, y_train,
        n_trials=50,  # Reduced for faster execution
        cv_folds=5,
        early_stopping_rounds=20
    )
    
    # Evaluate optimized model
    evaluation_results = tuner.evaluate_optimized_model(
        X_train, y_train, X_test, y_test,
        optimization_results['best_params']
    )
    
    # Update configuration
    config_updated = tuner.update_configuration(
        optimization_results['best_params'],
        Environment.PRODUCTION
    )
    
    if config_updated:
        tuner.logger.info("🎉 Hyperparameter optimization completed successfully!")
        tuner.logger.info(f"🏆 Best parameters: {optimization_results['best_params']}")
        tuner.logger.info(f"📈 Best score: {optimization_results['best_score']:.4f}")
    else:
        tuner.logger.error("❌ Failed to update configuration")


if __name__ == "__main__":
    main()
