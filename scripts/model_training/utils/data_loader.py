"""
Data loading utilities
"""

import logging
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from ..config.optimized_config import OptimizedModelConfig as ModelConfig


class DataLoader:
    """Data loader"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize data loader
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.df = None
        self.feature_dictionary = None
    
    def load_dataset(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset
        
        Args:
            dataset_path: Dataset path, if None use path from config
            
        Returns:
            Loaded dataset
        """
        if dataset_path is None:
            dataset_path = self.config.path_config.model_ready_dataset_path
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
        
        self.logger.info(f"Loading dataset: {dataset_path}")
        
        try:
            self.df = pd.read_csv(dataset_path)
            self.logger.info(f"Dataset loaded successfully, shape: {self.df.shape}")
            self.logger.info(f"Target variable distribution: {self.df['TARGET'].value_counts().to_dict()}")
            
            return self.df
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def load_feature_dictionary(self, dict_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load feature dictionary
        
        Args:
            dict_path: Feature dictionary path, if None use path from config
            
        Returns:
            Feature dictionary DataFrame
        """
        if dict_path is None:
            dict_path = self.config.path_config.feature_dictionary_path
        
        dict_path = Path(dict_path)
        
        if not dict_path.exists():
            self.logger.warning(f"Feature dictionary file does not exist: {dict_path}")
            return None
        
        self.logger.info(f"Loading feature dictionary: {dict_path}")
        
        try:
            self.feature_dictionary = pd.read_excel(dict_path)
            self.logger.info(f"Feature dictionary loaded successfully, contains {len(self.feature_dictionary)} features")
            
            return self.feature_dictionary
            
        except Exception as e:
            self.logger.error(f"Failed to load feature dictionary: {e}")
            raise
    
    def get_feature_columns_from_dict(self) -> list:
        """
        Get feature column names from feature dictionary
        
        Returns:
            Feature column names list
        """
        if self.feature_dictionary is None:
            self.load_feature_dictionary()
        
        if self.feature_dictionary is None:
            self.logger.warning("Feature dictionary not loaded, using production features from config")
            return self.config.feature_config.production_features
        
        # Get features included in model from feature dictionary
        included_mask = self.feature_dictionary['Included in Model'].isin([
            'Yes', 'Yes (Target)', 'Yes (Engineered)'
        ])
        included_features = self.feature_dictionary[included_mask]['Variable'].tolist()
        
        # Remove target variable and data leakage features
        features_to_remove = self.config.feature_config.get_features_to_remove()
        production_features = [
            col for col in included_features 
            if col != self.config.feature_config.target_column 
            and col not in features_to_remove
        ]
        
        self.logger.info(f"Retrieved {len(production_features)} production features from feature dictionary")
        
        return production_features
    
    def validate_dataset(self) -> bool:
        """
        Validate dataset with comprehensive data quality checks
        
        Returns:
            Whether validation passed
        """
        if self.df is None:
            self.logger.error("Dataset not loaded")
            return False
        
        validation_passed = True
        
        # Check target variable
        if self.config.feature_config.target_column not in self.df.columns:
            self.logger.error(f"Target variable {self.config.feature_config.target_column} does not exist")
            validation_passed = False
        
        # Check target variable distribution
        target_dist = self.df[self.config.feature_config.target_column].value_counts()
        if len(target_dist) < 2:
            self.logger.error("Target variable has fewer than 2 categories")
            validation_passed = False
        else:
            # Check class imbalance
            class_ratio = target_dist.max() / target_dist.min()
            self.logger.info(f"Class distribution: {target_dist.to_dict()}")
            self.logger.info(f"Class imbalance ratio: {class_ratio:.2f}:1")
            
            if class_ratio > 10:
                self.logger.warning(f"Severe class imbalance detected: {class_ratio:.2f}:1")
            elif class_ratio > 5:
                self.logger.info(f"Moderate class imbalance: {class_ratio:.2f}:1")
        
        # Check data leakage features
        leakage_features = self.config.feature_config.get_features_to_remove()
        present_leakage = [feat for feat in leakage_features if feat in self.df.columns]
        if present_leakage:
            self.logger.warning(f"Data leakage features present in dataset: {present_leakage}")
        
        # Check missing values by feature
        missing_by_feature = self.df.isnull().sum()
        high_missing_features = missing_by_feature[missing_by_feature > len(self.df) * 0.5]
        if len(high_missing_features) > 0:
            self.logger.warning(f"Features with >50% missing values: {high_missing_features.to_dict()}")
        
        # Check feature data types
        numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.df.select_dtypes(include=['object']).columns
        self.logger.info(f"Numeric features: {len(numeric_features)}, Categorical features: {len(categorical_features)}")
        
        # Check for infinite values
        inf_count = np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            self.logger.warning(f"Dataset contains {inf_count} infinite values")
        
        # Check for duplicate rows
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            self.logger.warning(f"Dataset contains {duplicate_count} duplicate rows")
        
        # Check feature variance
        numeric_df = self.df.select_dtypes(include=[np.number])
        zero_variance_features = numeric_df.columns[numeric_df.var() == 0].tolist()
        if zero_variance_features:
            self.logger.warning(f"Features with zero variance: {zero_variance_features}")
        
        if validation_passed:
            self.logger.info("Dataset validation passed")
        else:
            self.logger.error("Dataset validation failed")
        
        return validation_passed
    
    def get_data_summary(self) -> dict:
        """
        Get data summary
        
        Returns:
            Data summary dictionary
        """
        if self.df is None:
            return {}
        
        summary = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'target_distribution': self.df[self.config.feature_config.target_column].value_counts().to_dict(),
            'target_proportion': self.df[self.config.feature_config.target_column].value_counts(normalize=True).to_dict(),
            'missing_values': self.df.isnull().sum().sum(),
            'dtypes': self.df.dtypes.to_dict()
        }
        
        return summary
    
    def get_clean_dataset(self, invert_target: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get production environment dataset with proper missing value handling
        
        Args:
            invert_target: Whether to invert the target variable (y = 1 - TARGET)
                          Current TARGET encoding:
                          - TARGET=1: "Not Prescribed Drug A" (patients who should have received Drug A but did not)
                          - TARGET=0: "Prescribed Drug A" (patients who already received Drug A or are not eligible)
                          
                          Business Logic:
                          - Prediction=1: "Not Prescribed Drug A" → Alert Required
                          - Prediction=0: "Prescribed Drug A" → No Alert Required
                          
                          With invert_target=False (recommended):
                          - TARGET=1 → y=1 (positive class) → Alert Required
                          - TARGET=0 → y=0 (negative class) → No Alert Required
        
        Returns:
            (Feature DataFrame, Target variable Series)
        """
        if self.df is None:
            raise ValueError("Dataset not loaded, please call load_dataset() first")
        
        # Get production features
        production_features = self.get_feature_columns_from_dict()
        
        # Check if features exist
        missing_features = [col for col in production_features if col not in self.df.columns]
        if missing_features:
            self.logger.warning(f"The following features do not exist in dataset: {missing_features}")
            production_features = [col for col in production_features if col in self.df.columns]
        
        # Create production environment dataset
        X = self.df[production_features].copy()
        y = self.df[self.config.feature_config.target_column].copy()
        
        # Apply target inversion if requested
        if invert_target:
            original_distribution = y.value_counts().to_dict()
            y = 1 - y  # Invert target: y = 1 - TARGET
            inverted_distribution = y.value_counts().to_dict()
            
            self.logger.info("🎯 Target variable inverted")
            self.logger.info(f"Original distribution: {original_distribution}")
            self.logger.info(f"Inverted distribution: {inverted_distribution}")
            self.logger.info("⚠️  Warning: Inverted target may not align with business logic")
        else:
            self.logger.info("🎯 Using original target encoding (recommended)")
            self.logger.info("✅ TARGET=1 → 'Not Prescribed Drug A' → Alert Required (Positive Class)")
            self.logger.info("✅ TARGET=0 → 'Prescribed Drug A' → No Alert Required (Negative Class)")
        
        # Handle missing values based on data analysis
        self._handle_missing_values(X)
        
        self.logger.info(f"Production environment dataset shape: X={X.shape}, y={y.shape}")
        self.logger.info(f"Features used: {production_features}")
        
        return X, y
    
    def _handle_missing_values(self, X: pd.DataFrame):
        """
        Handle missing values in the dataset based on data analysis
        
        Args:
            X: Feature DataFrame to process
        """
        self.logger.info("Handling missing values...")
        
        # Log missing values before handling
        missing_before = X.isnull().sum()
        missing_cols = missing_before[missing_before > 0]
        
        if len(missing_cols) > 0:
            self.logger.info(f"Missing values before handling: {missing_cols.to_dict()}")
            
            # Handle missing values based on feature type and clinical meaning
            for col in missing_cols.index:
                if col in self.config.feature_config.categorical_features:
                    # For categorical features, use mode or 'Unknown'
                    mode_value = X[col].mode()
                    if len(mode_value) > 0:
                        X[col].fillna(mode_value[0], inplace=True)
                    else:
                        X[col].fillna('Unknown', inplace=True)
                    self.logger.info(f"Filled categorical feature {col} with mode/Unknown")
                else:
                    # For numerical features, use median
                    median_value = X[col].median()
                    X[col].fillna(median_value, inplace=True)
                    self.logger.info(f"Filled numerical feature {col} with median: {median_value:.2f}")
        
        # Log missing values after handling
        missing_after = X.isnull().sum().sum()
        self.logger.info(f"Missing values after handling: {missing_after}")
        
        if missing_after > 0:
            self.logger.warning(f"Still have {missing_after} missing values after handling")
