"""
Training pipeline for model training workflow
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from ..config.optimized_config import OptimizedModelConfig as ModelConfig
from ..utils.data_loader import DataLoader
from ..utils.preprocessor import DataPreprocessor
from ..core.model_factory import ModelFactory
from ..core.model_factory import auto_register_trainers
from ..core.evaluator import ModelEvaluator


class TrainingPipeline:
    """Training pipeline"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize training pipeline
        
        Args:
            config: Model configuration object
        """
        # Ensure all trainers are registered
        auto_register_trainers()
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.preprocessor = DataPreprocessor(config)
        self.evaluator = ModelEvaluator(config)
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Training results
        self.trainer = None
        self.training_results = {}
    
    def load_data(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data
        
        Args:
            dataset_path: Dataset path
            
        Returns:
            Loaded dataset
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 1: Load data")
        self.logger.info("=" * 60)
        
        # Load dataset
        df = self.data_loader.load_dataset(dataset_path)
        
        # Validate dataset
        if not self.data_loader.validate_dataset():
            raise ValueError("Dataset validation failed")
        
        # Get data summary
        summary = self.data_loader.get_data_summary()
        self.logger.info(f"Data summary: {summary}")
        
        return df
    
    def prepare_data(self) -> tuple:
        """
        Prepare training data
        
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 2: Prepare training data")
        self.logger.info("=" * 60)
        
        # Get production environment dataset with target inversion if configured
        # Target encoding: TARGET=1="Not Prescribed Drug A" (Alert Required), TARGET=0="Prescribed Drug A" (No Alert)
        invert_target = getattr(self.config, 'invert_target', False)
        X, y = self.data_loader.get_clean_dataset(invert_target=invert_target)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.split_data(X, y)
        
        # Fit and transform training data
        self.X_train, self.y_train = self.preprocessor.fit_transform(self.X_train, self.y_train)
        
        # Transform test data
        self.X_test = self.preprocessor.transform(self.X_test)
        
        # Handle class imbalance
        self.X_train, self.y_train = self.preprocessor.handle_class_imbalance(
            self.X_train, self.y_train
        )
        
        self.logger.info(f"Training data preparation completed")
        self.logger.info(f"Training set shape: {self.X_train.shape}")
        self.logger.info(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Train model
        
        Args:
            model_type: Model type, if None use the type from config
            
        Returns:
            Training results
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 3: Train model")
        self.logger.info("=" * 60)
        
        # Create trainer
        self.trainer = ModelFactory.create_trainer(self.config, model_type)
        
        # Set feature column names
        self.trainer.feature_columns = self.preprocessor.feature_columns
        
        # Train model
        self.training_results = self.trainer.train(
            self.X_train, self.y_train
        )
        
        self.logger.info(f"Model training completed: {self.trainer._get_model_name()}")
        
        return self.training_results
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate model
        
        Returns:
            Evaluation results
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 4: Evaluate model")
        self.logger.info("=" * 60)
        
        if self.trainer is None:
            raise ValueError("Model not trained yet, please call train_model() first")
        
        # Evaluate model
        evaluation_results = self.evaluator.evaluate_single_model(
            self.trainer, self.X_test, self.y_test
        )
        
        self.logger.info("Model evaluation completed")
        
        return evaluation_results
    
    def save_model(self, model_path: Optional[str] = None):
        """
        Save model
        
        Args:
            model_path: Model save path
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 5: Save model")
        self.logger.info("=" * 60)
        
        if self.trainer is None:
            raise ValueError("Model not trained yet, please call train_model() first")
        
        if model_path is None:
            model_path = self.config.path_config.model_dir
        
        # Always save artifacts inside the canonical 'models' subdirectory
        model_path = Path(model_path) / "models"
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model (handles timestamped + fixed alias internally)
        self.trainer.save_model(model_path)
        
        # Save preprocessor with timestamp and update fixed alias
        from datetime import datetime
        import os
        import shutil
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preprocessor_ts = Path(model_path) / f"preprocessor_v{self.config.model_version}_{timestamp}.pkl"
        preprocessor_fixed = Path(model_path) / "preprocessor.pkl"
        
        # Save timestamped preprocessor
        self.preprocessor.save_preprocessor(str(preprocessor_ts))
        
        # Update fixed alias for preprocessor
        try:
            if preprocessor_fixed.exists() or preprocessor_fixed.is_symlink():
                preprocessor_fixed.unlink()
            os.symlink(preprocessor_ts.name, preprocessor_fixed)
        except Exception:
            shutil.copy2(preprocessor_ts, preprocessor_fixed)
        
        # Augment metadata with preprocessor artifacts if possible
        metadata_file = Path(model_path) / "model_metadata.pkl"
        if metadata_file.exists():
            try:
                import pickle
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                metadata = dict(metadata)
                metadata.setdefault('artifacts', {})
                metadata['artifacts'].update({
                    'preprocessor_timestamped': preprocessor_ts.name,
                    'preprocessor_fixed': preprocessor_fixed.name
                })
                with open(metadata_file, 'wb') as f:
                    pickle.dump(metadata, f)
            except Exception as e:
                self.logger.warning(f"Failed to augment metadata with preprocessor artifacts: {e}")
        
        self.logger.info(
            f"Model and preprocessor saved (ts + fixed aliases) to: {model_path}")
    
    def run_complete_training(self, dataset_path: Optional[str] = None, 
                            model_type: Optional[str] = None,
                            save_model: bool = True) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Args:
            dataset_path: Dataset path
            model_type: Model type
            save_model: Whether to save model
            
        Returns:
            Complete results dictionary
        """
        self.logger.info("🚀 Starting complete training pipeline")
        self.logger.info("=" * 80)
        
        try:
            # 1. Load data
            df = self.load_data(dataset_path)
            
            # 2. Prepare data
            self.prepare_data()
            
            # 3. Train model
            training_results = self.train_model(model_type)
            
            # 4. Evaluate model
            evaluation_results = self.evaluate_model()
            
            # 5. Save model
            if save_model:
                self.save_model()
            
            # Get feature importance
            feature_importances = self.trainer.get_feature_importance()
            
            # Summarize results
            complete_results = {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'feature_importances': feature_importances,
                'model_summary': self.trainer.get_model_summary(),
                'preprocessor_info': self.preprocessor.get_preprocessor_info(),
                'data_summary': self.data_loader.get_data_summary()
            }
            
            self.logger.info("🎉 Complete training pipeline successfully completed!")
            self.logger.info("=" * 80)
            
            return complete_results
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get pipeline summary
        
        Returns:
            Pipeline summary dictionary
        """
        summary = {
            'config': {
                'model_type': self.config.model_type,
                'model_version': self.config.model_version,
                'test_size': self.config.test_size,
                'use_smote': self.config.use_smote
            },
            'data_status': {
                'data_loaded': self.data_loader.df is not None,
                'data_prepared': self.X_train is not None,
                'data_shape': self.X_train.shape if self.X_train is not None else None
            },
            'model_status': {
                'model_trained': self.trainer is not None,
                'model_type': self.trainer._get_model_name() if self.trainer else None
            },
            'preprocessor_status': {
                'is_fitted': self.preprocessor.is_fitted,
                'feature_count': len(self.preprocessor.feature_columns) if self.preprocessor.feature_columns else 0
            }
        }
        
        return summary
