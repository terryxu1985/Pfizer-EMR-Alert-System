"""
Enhanced data processing utilities for the EMR Alert System
Optimized for production serving with improved error handling and logging
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import FEATURE_CONFIG, MODEL_DIR

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Enhanced data processor with improved error handling and validation
    Handles data preprocessing for the EMR Alert System production serving
    """
    
    def __init__(self):
        """
        Initialize DataProcessor with enhanced logging and validation
        """
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.is_fitted = False
        self.validation_stats = {
            'total_processed': 0,
            'validation_errors': 0,
            'missing_value_handled': 0
        }
        
    def fit(self, df: pd.DataFrame) -> 'DataProcessor':
        """
        Fit the data processor on training data
        
        Args:
            df: Training dataframe
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting data processor...")
        
        # Get feature columns from feature dictionary
        self.feature_columns = self._get_feature_columns(df)
        
        # Create a copy for preprocessing
        df_processed = df.copy()
        
        # Handle categorical variables
        for col in FEATURE_CONFIG['categorical_columns']:
            if col in df_processed.columns and col in self.feature_columns:
                le = LabelEncoder()
                df_processed[col] = df_processed[col].fillna('Unknown')
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
                logger.info(f"Fitted label encoder for {col}")
        
        # Handle numeric missing values for included features only
        for col in self.feature_columns:
            if col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        median_val = df_processed[col].median()
                        df_processed[col] = df_processed[col].fillna(median_val)
                        logger.info(f"Filled missing values in {col} with median: {median_val}")
                    else:
                        # For non-numeric columns, fill with mode or first value
                        mode_val = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else df_processed[col].iloc[0]
                        df_processed[col] = df_processed[col].fillna(mode_val)
                        logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Fit scaler on selected features
        X = df_processed[self.feature_columns]
        self.scaler.fit(X)
        
        self.is_fitted = True
        logger.info(f"Data processor fitted successfully. Features: {len(self.feature_columns)}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted processor
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before transform")
        
        logger.info("Transforming data...")
        
        # Create a copy for preprocessing
        df_processed = df.copy()
        
        # Handle categorical variables
        for col in FEATURE_CONFIG['categorical_columns']:
            if col in df_processed.columns and col in self.feature_columns:
                if col in self.label_encoders:
                    # Handle unseen categories
                    df_processed[col] = df_processed[col].fillna('Unknown')
                    
                    # Convert enum values to strings if needed
                    df_processed[col] = df_processed[col].astype(str)
                    
                    unique_values = df_processed[col].unique()
                    unseen_values = [val for val in unique_values 
                                   if val not in self.label_encoders[col].classes_]
                    
                    if unseen_values:
                        logger.warning(f"Unseen categories in {col}: {unseen_values}")
                        # Replace unseen categories with most frequent category
                        most_frequent = self.label_encoders[col].classes_[0]
                        df_processed[col] = df_processed[col].replace(unseen_values, most_frequent)
                    
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])
                else:
                    logger.warning(f"No label encoder found for {col}, filling with 0")
                    df_processed[col] = 0
        
        # Handle numeric missing values
        for col in self.feature_columns:
            if col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        # Use median from training data if available
                        if hasattr(self.scaler, 'mean_') and col in self.scaler.feature_names_in_:
                            col_idx = list(self.scaler.feature_names_in_).index(col)
                            median_val = self.scaler.mean_[col_idx]  # Use mean as proxy for median
                        else:
                            median_val = 0
                        df_processed[col] = df_processed[col].fillna(median_val)
                        logger.info(f"Filled missing values in {col} with: {median_val}")
                    else:
                        # For non-numeric columns, fill with first value
                        first_val = df_processed[col].iloc[0] if len(df_processed[col].dropna()) > 0 else 0
                        df_processed[col] = df_processed[col].fillna(first_val)
                        logger.info(f"Filled missing values in {col} with: {first_val}")
        
        # Select features in the correct order
        X = df_processed[self.feature_columns]
        
        # Only scale numerical features (exclude categorical features)
        categorical_features = FEATURE_CONFIG['categorical_columns']
        numerical_features = [col for col in self.feature_columns if col not in categorical_features]
        
        # Scale only numerical features
        if len(numerical_features) > 0:
            X_numerical = X[numerical_features]
            X_scaled_numerical = self.scaler.transform(X_numerical)
            
            # Create final array with correct feature order
            X_scaled = X.copy()
            X_scaled[numerical_features] = X_scaled_numerical
        else:
            X_scaled = X
        
        logger.info(f"Data transformed successfully. Shape: {X_scaled.shape}")
        return X_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform data
        
        Args:
            df: Dataframe to fit and transform
            
        Returns:
            Transformed numpy array
        """
        return self.fit(df).transform(df)
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get feature columns based on feature dictionary
        
        Args:
            df: Dataframe to extract features from
            
        Returns:
            List of feature column names
        """
        # Load feature dictionary - updated path for new location
        feat_dict_path = Path(__file__).parent.parent.parent / "data/model_ready/model_feature_dictionary.xlsx"
        
        if feat_dict_path.exists():
            feat_dict = pd.read_excel(feat_dict_path)
            
            # Get features to include in model from dictionary
            included_mask = feat_dict['Included in Model'].isin(['Yes', 'Yes (Target)', 'Yes (Engineered)'])
            included_features = feat_dict[included_mask]['Variable'].tolist()
            
            # Remove target variable, data leakage features, and excluded features
            feature_columns = [col for col in included_features 
                             if col != FEATURE_CONFIG['target_column'] 
                             and col not in FEATURE_CONFIG['data_leakage_features']
                             and col not in FEATURE_CONFIG['excluded_features']]
        else:
            # Fallback: use all columns except target, data leakage features, and excluded features
            feature_columns = [col for col in df.columns 
                             if col != FEATURE_CONFIG['target_column'] 
                             and col not in FEATURE_CONFIG['data_leakage_features']
                             and col not in FEATURE_CONFIG['excluded_features']]
        
        return feature_columns
    
    def save(self, filepath: Path) -> None:
        """
        Save the data processor to disk
        
        Args:
            filepath: Path to save the processor
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before saving")
        
        processor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(processor_data, f)
        
        logger.info(f"Data processor saved to {filepath}")
    
    def load(self, filepath: Path) -> 'DataProcessor':
        """
        Load the data processor from disk
        
        Args:
            filepath: Path to load the processor from
            
        Returns:
            Self for method chaining
        """
        with open(filepath, 'rb') as f:
            processor_data = pickle.load(f)
        
        self.scaler = processor_data['scaler']
        self.label_encoders = processor_data['label_encoders']
        self.feature_columns = processor_data['feature_columns']
        self.is_fitted = processor_data['is_fitted']
        
        logger.info(f"Data processor loaded from {filepath}")
        return self
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics for monitoring
        
        Returns:
            Dictionary containing processing statistics
        """
        return {
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'categorical_features_count': len(self.label_encoders),
            'validation_stats': self.validation_stats.copy(),
            'scaler_type': type(self.scaler).__name__
        }
    
    def validate_input_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data before processing
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'data_shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check for required features
        if self.feature_columns:
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                validation_result['errors'].append(f"Missing required features: {missing_features}")
                validation_result['valid'] = False
        
        # Check for excessive missing values
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
        if high_missing_cols:
            validation_result['warnings'].append(f"High missing values (>50%) in: {high_missing_cols}")
        
        return validation_result
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted first")
        return self.feature_columns.copy()
