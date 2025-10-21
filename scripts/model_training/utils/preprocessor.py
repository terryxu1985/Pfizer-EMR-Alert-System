"""
Data preprocessing utilities
"""

import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from ..config.optimized_config import OptimizedModelConfig as ModelConfig


class DataPreprocessor:
    """Data preprocessor"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize preprocessor
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.is_fitted = False
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data
        
        Args:
            X: Feature data
            y: Target variable
            
        Returns:
            Preprocessed data
        """
        self.logger.info("Starting data preprocessing...")
        
        # Save feature column names
        self.feature_columns = list(X.columns)
        
        # Create data copy
        X_processed = X.copy()
        
        # Handle missing values
        X_processed = self._handle_missing_values(X_processed)
        
        # Handle categorical variables
        X_processed = self._encode_categorical_variables(X_processed)
        
        self.logger.info(f"Data preprocessing completed, shape: {X_processed.shape}")
        
        return X_processed, y
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with enhanced strategy based on data analysis
        
        Args:
            X: Feature data
            
        Returns:
            Processed data
        """
        self.logger.info("Handling missing values with enhanced strategy...")
        
        missing_count = X.isnull().sum().sum()
        if missing_count == 0:
            self.logger.info("No missing values")
            return X
        
        self.logger.info(f"Found {missing_count} missing values")
        
        # Log missing values by column
        missing_by_col = X.isnull().sum()
        missing_cols = missing_by_col[missing_by_col > 0]
        self.logger.info(f"Missing values by column: {missing_cols.to_dict()}")
        
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                missing_pct = X[col].isnull().sum() / len(X) * 100
                self.logger.info(f"Column {col}: {X[col].isnull().sum()} missing ({missing_pct:.1f}%)")
                
                if col in self.config.feature_config.categorical_features:
                    # For categorical features, use mode or 'Unknown'
                    mode_value = X[col].mode()
                    if len(mode_value) > 0:
                        fill_value = mode_value[0]
                        X[col] = X[col].fillna(fill_value)
                        self.logger.info(f"Filled categorical feature {col} with mode: {fill_value}")
                    else:
                        X[col] = X[col].fillna('Unknown')
                        self.logger.info(f"Filled categorical feature {col} with 'Unknown'")
                else:
                    # For numerical features, use median
                    median_value = X[col].median()
                    X[col] = X[col].fillna(median_value)
                    self.logger.info(f"Filled numerical feature {col} with median: {median_value:.2f}")
        
        # Verify no missing values remain
        remaining_missing = X.isnull().sum().sum()
        if remaining_missing > 0:
            self.logger.warning(f"Still have {remaining_missing} missing values after handling")
        else:
            self.logger.info("All missing values successfully handled")
        
        return X
    
    def _encode_categorical_variables(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            X: Feature data
            
        Returns:
            Encoded data
        """
        self.logger.info("Encoding categorical variables...")
        
        categorical_features = self.config.feature_config.categorical_features
        
        for col in categorical_features:
            if col in X.columns:
                if not self.is_fitted:
                    # Create encoder during training
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    self.label_encoders[col] = le
                    self.logger.info(f"Created label encoder: {col}")
                else:
                    # Use trained encoder during prediction
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        X[col] = X[col].apply(
                            lambda x: x if x in le.classes_ else le.classes_[0]
                        )
                        X[col] = le.transform(X[col])
                    else:
                        self.logger.warning(f"Label encoder for categorical variable {col} not found")
        
        return X
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data
        
        Args:
            X: Feature data
            y: Target variable
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        self.logger.info(f"Splitting data, test set ratio: {self.config.test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self.config.stratify else None
        )
        
        self.logger.info(f"Training set shape: {X_train.shape}")
        self.logger.info(f"Test set shape: {X_test.shape}")
        self.logger.info(f"Training set target distribution: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features (only numerical features)
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Scaled features
        """
        self.logger.info("Scaling features...")
        
        # Create a copy to avoid modifying original data
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy() if X_test is not None else None
        
        # Get numerical features (exclude categorical features)
        categorical_features = self.config.feature_config.categorical_features
        numerical_features = [col for col in X_train.columns if col not in categorical_features]
        
        self.logger.info(f"Scaling {len(numerical_features)} numerical features: {numerical_features}")
        
        # Only fit scaler during training
        if not self.is_fitted:
            if len(numerical_features) > 0:
                X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
            self.is_fitted = True
        else:
            if len(numerical_features) > 0:
                X_train_scaled[numerical_features] = self.scaler.transform(X_train[numerical_features])
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=self.feature_columns,
            index=X_train.index
        )
        
        if X_test is not None:
            if len(numerical_features) > 0:
                X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
            X_test_scaled = pd.DataFrame(
                X_test_scaled,
                columns=self.feature_columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Balanced data
        """
        if not self.config.use_smote:
            self.logger.info("Skipping class imbalance handling")
            return X_train, y_train
        
        self.logger.info("Handling class imbalance...")
        
        # Check class distribution
        class_counts = np.bincount(y_train)
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        self.logger.info(f"Class imbalance ratio: {imbalance_ratio:.1f}:1")
        self.logger.info(f"Original training distribution: {class_counts}")
        
        if imbalance_ratio > 2:  # Only apply SMOTE for severe imbalance
            # Check if SMOTE should be disabled for specific models
            # This is a simple approach - in production, you might want more sophisticated model-specific handling
            use_smote_for_model = getattr(self.config, 'use_smote', True)
            
            # Check if we're in a context where SMOTE should be disabled
            # For now, we'll use a simple heuristic based on the data characteristics
            if not use_smote_for_model:
                self.logger.info("SMOTE disabled - using class_weight only")
                return X_train, y_train
            
            # Use conservative SMOTE - only balance to target ratio instead of 1:1
            target_ratio = getattr(self.config, 'smote_target_ratio', 0.5)
            minority_class_count = class_counts.min()
            majority_class_count = class_counts.max()
            
            # Calculate target minority count (must be >= original minority count)
            target_minority_count = max(
                minority_class_count,  # At least keep original count
                int(majority_class_count * target_ratio)  # Or target ratio
            )
            
            # Find minority class label (class with fewer samples)
            minority_class = 0 if class_counts[0] < class_counts[1] else 1
            
            # Calculate sampling strategy for conservative SMOTE
            # Only oversample minority class to target count
            sampling_strategy = {minority_class: target_minority_count}
            
            self.logger.info(f"SMOTE strategy: oversample class {minority_class} from {minority_class_count} to {target_minority_count}")
            
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.config.smote_random_state,
                k_neighbors=getattr(self.config, 'smote_k_neighbors', 3)
            )
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
            balanced_counts = np.bincount(y_balanced)
            self.logger.info(f"Balanced training distribution: {balanced_counts}")
            self.logger.info(f"Training set size increased from {len(y_train)} to {len(y_balanced)}")
            
            return X_balanced, y_balanced
        else:
            self.logger.info("Class imbalance not severe, using original training data")
            return X_train, y_train
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform training data
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Processed data
        """
        self.logger.info("Fitting preprocessor...")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X_train, y_train)
        
        # Scale features
        X_scaled, _ = self.scale_features(X_processed)
        
        self.logger.info("Preprocessor fitting completed")
        
        return X_scaled, y_processed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data
        
        Args:
            X: Feature data
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet, please call fit_transform() first")
        
        self.logger.info("Transforming new data...")
        
        # Preprocess data
        X_processed, _ = self.preprocess_data(X, None)
        
        # Scale features
        X_scaled, _ = self.scale_features(X_processed)
        
        return X_scaled
    
    def get_preprocessor_info(self) -> Dict[str, Any]:
        """
        Get preprocessor information
        
        Returns:
            Preprocessor information dictionary
        """
        return {
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
            'label_encoders': list(self.label_encoders.keys()),
            'scaler_type': type(self.scaler).__name__
        }
    
    def save_preprocessor(self, path: str):
        """
        Save preprocessor
        
        Args:
            path: Save path
        """
        import pickle
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        self.logger.info(f"Preprocessor saved to: {path}")
    
    def load_preprocessor(self, path: str):
        """
        Load preprocessor
        
        Args:
            path: Load path
        """
        import pickle
        
        with open(path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_columns = preprocessor_data['feature_columns']
        self.is_fitted = preprocessor_data['is_fitted']
        
        self.logger.info(f"Preprocessor loaded from {path}")
