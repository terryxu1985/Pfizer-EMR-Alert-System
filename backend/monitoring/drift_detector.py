"""
Data Drift Detection Module

This module provides comprehensive data drift detection capabilities,
including statistical tests, distribution comparisons, and automated alerts.
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, anderson_ksamp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import threading

logger = logging.getLogger(__name__)

@dataclass
class DriftResult:
    """Data drift detection result"""
    timestamp: str
    feature_name: str
    drift_detected: bool
    drift_score: float
    p_value: float
    test_statistic: float
    test_method: str
    severity: str
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class FeatureStatistics:
    """Feature statistics for drift detection"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    null_count: int
    unique_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DataDriftDetector:
    """
    Comprehensive data drift detection system
    """
    
    def __init__(self, reference_data_path: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize data drift detector
        
        Args:
            reference_data_path: Path to reference dataset
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.reference_data_path = reference_data_path
        self.reference_data = None
        self.reference_stats = {}
        self.feature_columns = []
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'ks_test_p_value': 0.05,
            'chi2_test_p_value': 0.05,
            'statistical_distance': 0.1,
            'distribution_similarity': 0.8
        }
        
        # Severity levels
        self.severity_thresholds = {
            'low': 0.05,
            'medium': 0.1,
            'high': 0.2,
            'critical': 0.3
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load reference data if provided
        if self.reference_data_path:
            self.load_reference_data(self.reference_data_path)
        
        logger.info("Data Drift Detector initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'drift_detection_enabled': True,
            'min_samples_for_drift_check': 100,
            'max_features_to_check': 50,
            'statistical_tests': ['ks_test', 'chi2_test', 'wasserstein_distance'],
            'categorical_threshold': 0.05,  # Features with <5% unique values treated as categorical
            'drift_check_interval_hours': 6,
            'reference_data_update_interval_days': 30
        }
    
    def load_reference_data(self, data_path: str) -> bool:
        """
        Load reference dataset for drift detection
        
        Args:
            data_path: Path to reference dataset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if data_path.endswith('.csv'):
                self.reference_data = pd.read_csv(data_path)
            elif data_path.endswith('.pkl'):
                with open(data_path, 'rb') as f:
                    self.reference_data = pickle.load(f)
            else:
                logger.error(f"Unsupported file format: {data_path}")
                return False
            
            # Extract feature columns (exclude target if present)
            self.feature_columns = [col for col in self.reference_data.columns 
                                  if col.lower() not in ['target', 'label', 'y']]
            
            # Calculate reference statistics
            self._calculate_reference_statistics()
            
            logger.info(f"Loaded reference data: {self.reference_data.shape}")
            logger.info(f"Feature columns: {len(self.feature_columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            return False
    
    def _calculate_reference_statistics(self) -> None:
        """Calculate reference statistics for all features"""
        if self.reference_data is None:
            return
        
        self.reference_stats = {}
        
        for feature in self.feature_columns:
            if feature not in self.reference_data.columns:
                continue
            
            feature_data = self.reference_data[feature].dropna()
            
            if len(feature_data) == 0:
                continue
            
            # Determine if feature is categorical
            unique_ratio = len(feature_data.unique()) / len(feature_data)
            is_categorical = unique_ratio < self.config['categorical_threshold']
            
            if is_categorical:
                # Categorical feature statistics
                value_counts = feature_data.value_counts()
                self.reference_stats[feature] = {
                    'type': 'categorical',
                    'value_counts': value_counts.to_dict(),
                    'unique_values': list(feature_data.unique()),
                    'null_count': self.reference_data[feature].isnull().sum(),
                    'total_count': len(feature_data)
                }
            else:
                # Numerical feature statistics
                try:
                    stats_dict = {
                        'type': 'numerical',
                        'mean': float(feature_data.mean()),
                        'std': float(feature_data.std()),
                        'min': float(feature_data.min()),
                        'max': float(feature_data.max()),
                        'median': float(feature_data.median()),
                        'q25': float(feature_data.quantile(0.25)),
                        'q75': float(feature_data.quantile(0.75)),
                        'skewness': float(feature_data.skew()),
                        'kurtosis': float(feature_data.kurtosis()),
                        'null_count': int(self.reference_data[feature].isnull().sum()),
                        'total_count': len(feature_data)
                    }
                    self.reference_stats[feature] = stats_dict
                except Exception as e:
                    logger.warning(f"Failed to calculate statistics for {feature}: {e}")
                    continue
    
    def detect_drift(self, new_data: pd.DataFrame, 
                    features_to_check: Optional[List[str]] = None) -> List[DriftResult]:
        """
        Detect data drift in new data compared to reference data
        
        Args:
            new_data: New data to check for drift
            features_to_check: Specific features to check (if None, checks all)
            
        Returns:
            List of drift detection results
        """
        if self.reference_data is None:
            logger.warning("No reference data loaded for drift detection")
            return []
        
        if len(new_data) < self.config['min_samples_for_drift_check']:
            logger.warning(f"Insufficient samples for drift check: {len(new_data)}")
            return []
        
        features_to_check = features_to_check or self.feature_columns
        features_to_check = features_to_check[:self.config['max_features_to_check']]
        
        drift_results = []
        
        for feature in features_to_check:
            if feature not in new_data.columns:
                continue
            
            if feature not in self.reference_stats:
                continue
            
            # Detect drift for this feature
            drift_result = self._detect_feature_drift(feature, new_data[feature])
            if drift_result:
                drift_results.append(drift_result)
        
        # Detect multivariate drift
        multivariate_drift = self._detect_multivariate_drift(new_data, features_to_check)
        if multivariate_drift:
            drift_results.extend(multivariate_drift)
        
        return drift_results
    
    def _detect_feature_drift(self, feature_name: str, new_feature_data: pd.Series) -> Optional[DriftResult]:
        """
        Detect drift for a single feature
        
        Args:
            feature_name: Name of the feature
            new_feature_data: New feature data
            
        Returns:
            DriftResult if drift detected, None otherwise
        """
        try:
            ref_stats = self.reference_stats[feature_name]
            ref_data = self.reference_data[feature_name].dropna()
            new_data = new_feature_data.dropna()
            
            if len(new_data) == 0:
                return None
            
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            if ref_stats['type'] == 'categorical':
                return self._detect_categorical_drift(feature_name, ref_data, new_data, timestamp)
            else:
                return self._detect_numerical_drift(feature_name, ref_data, new_data, timestamp)
                
        except Exception as e:
            logger.error(f"Error detecting drift for feature {feature_name}: {e}")
            return None
    
    def _detect_categorical_drift(self, feature_name: str, ref_data: pd.Series, 
                                 new_data: pd.Series, timestamp: str) -> Optional[DriftResult]:
        """Detect drift in categorical features using Chi-square test"""
        try:
            # Get value counts for both datasets
            ref_counts = ref_data.value_counts()
            new_counts = new_data.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(new_counts.index)
            
            # Create contingency table
            contingency_table = []
            for category in all_categories:
                ref_count = ref_counts.get(category, 0)
                new_count = new_counts.get(category, 0)
                contingency_table.append([ref_count, new_count])
            
            contingency_table = np.array(contingency_table)
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate drift score (normalized chi-square statistic)
            drift_score = chi2_stat / (len(ref_data) + len(new_data))
            
            # Determine severity
            severity = self._determine_severity(drift_score)
            
            drift_detected = p_value < self.drift_thresholds['chi2_test_p_value']
            
            if drift_detected:
                return DriftResult(
                    timestamp=timestamp,
                    feature_name=feature_name,
                    drift_detected=True,
                    drift_score=drift_score,
                    p_value=p_value,
                    test_statistic=chi2_stat,
                    test_method='chi2_test',
                    severity=severity,
                    message=f"Categorical drift detected in {feature_name}: "
                           f"chi2={chi2_stat:.2f}, p={p_value:.4f}"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in categorical drift detection for {feature_name}: {e}")
            return None
    
    def _detect_numerical_drift(self, feature_name: str, ref_data: pd.Series, 
                               new_data: pd.Series, timestamp: str) -> Optional[DriftResult]:
        """Detect drift in numerical features using multiple statistical tests"""
        try:
            drift_results = []
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = ks_2samp(ref_data, new_data)
            ks_drift_score = ks_stat
            
            # Wasserstein distance (Earth Mover's Distance)
            wasserstein_distance = stats.wasserstein_distance(ref_data, new_data)
            
            # Normalize Wasserstein distance by data range
            data_range = max(ref_data.max(), new_data.max()) - min(ref_data.min(), new_data.min())
            normalized_wasserstein = wasserstein_distance / data_range if data_range > 0 else 0
            
            # Use KS test as primary indicator
            drift_detected = ks_p_value < self.drift_thresholds['ks_test_p_value']
            
            if drift_detected:
                # Determine severity based on KS statistic
                severity = self._determine_severity(ks_stat)
                
                return DriftResult(
                    timestamp=timestamp,
                    feature_name=feature_name,
                    drift_detected=True,
                    drift_score=ks_stat,
                    p_value=ks_p_value,
                    test_statistic=ks_stat,
                    test_method='ks_test',
                    severity=severity,
                    message=f"Numerical drift detected in {feature_name}: "
                           f"KS={ks_stat:.4f}, p={ks_p_value:.4f}, "
                           f"Wasserstein={normalized_wasserstein:.4f}"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in numerical drift detection for {feature_name}: {e}")
            return None
    
    def _detect_multivariate_drift(self, new_data: pd.DataFrame, 
                                  features_to_check: List[str]) -> List[DriftResult]:
        """
        Detect multivariate drift using PCA and distribution comparison
        
        Args:
            new_data: New data
            features_to_check: Features to include in multivariate analysis
            
        Returns:
            List of multivariate drift results
        """
        try:
            # Select numerical features for multivariate analysis
            numerical_features = []
            for feature in features_to_check:
                if (feature in self.reference_stats and 
                    self.reference_stats[feature]['type'] == 'numerical' and
                    feature in new_data.columns):
                    numerical_features.append(feature)
            
            if len(numerical_features) < 2:
                return []
            
            # Prepare data
            ref_data = self.reference_data[numerical_features].dropna()
            new_data_clean = new_data[numerical_features].dropna()
            
            if len(ref_data) < 10 or len(new_data_clean) < 10:
                return []
            
            # Standardize data
            scaler = StandardScaler()
            ref_scaled = scaler.fit_transform(ref_data)
            new_scaled = scaler.transform(new_data_clean)
            
            # Apply PCA
            pca = PCA(n_components=min(5, len(numerical_features)))
            ref_pca = pca.fit_transform(ref_scaled)
            new_pca = pca.transform(new_scaled)
            
            # Calculate multivariate drift using first principal component
            ks_stat, p_value = ks_2samp(ref_pca[:, 0], new_pca[:, 0])
            
            drift_detected = p_value < self.drift_thresholds['ks_test_p_value']
            
            if drift_detected:
                severity = self._determine_severity(ks_stat)
                
                return [DriftResult(
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    feature_name="multivariate_pca",
                    drift_detected=True,
                    drift_score=ks_stat,
                    p_value=p_value,
                    test_statistic=ks_stat,
                    test_method='multivariate_pca',
                    severity=severity,
                    message=f"Multivariate drift detected: "
                           f"KS={ks_stat:.4f}, p={p_value:.4f}, "
                           f"features={len(numerical_features)}"
                )]
            
            return []
            
        except Exception as e:
            logger.error(f"Error in multivariate drift detection: {e}")
            return []
    
    def _determine_severity(self, drift_score: float) -> str:
        """Determine drift severity based on score"""
        if drift_score >= self.severity_thresholds['critical']:
            return 'critical'
        elif drift_score >= self.severity_thresholds['high']:
            return 'high'
        elif drift_score >= self.severity_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def calculate_feature_statistics(self, data: pd.DataFrame, 
                                   feature_name: str) -> Optional[FeatureStatistics]:
        """
        Calculate comprehensive statistics for a feature
        
        Args:
            data: Input data
            feature_name: Name of the feature
            
        Returns:
            FeatureStatistics object
        """
        if feature_name not in data.columns:
            return None
        
        feature_data = data[feature_name].dropna()
        
        if len(feature_data) == 0:
            return None
        
        try:
            return FeatureStatistics(
                mean=float(feature_data.mean()),
                std=float(feature_data.std()),
                min=float(feature_data.min()),
                max=float(feature_data.max()),
                median=float(feature_data.median()),
                q25=float(feature_data.quantile(0.25)),
                q75=float(feature_data.quantile(0.75)),
                skewness=float(feature_data.skew()),
                kurtosis=float(feature_data.kurtosis()),
                null_count=int(data[feature_name].isnull().sum()),
                unique_count=int(len(feature_data.unique()))
            )
        except Exception as e:
            logger.error(f"Error calculating statistics for {feature_name}: {e}")
            return None
    
    def compare_feature_distributions(self, feature_name: str, 
                                    new_data: pd.Series) -> Dict[str, Any]:
        """
        Compare feature distributions between reference and new data
        
        Args:
            feature_name: Name of the feature
            new_data: New feature data
            
        Returns:
            Comparison results dictionary
        """
        if feature_name not in self.reference_stats:
            return {}
        
        ref_data = self.reference_data[feature_name].dropna()
        new_data_clean = new_data.dropna()
        
        if len(ref_data) == 0 or len(new_data_clean) == 0:
            return {}
        
        try:
            # Calculate statistics for both datasets
            ref_stats = self.calculate_feature_statistics(
                self.reference_data, feature_name
            )
            new_stats = self.calculate_feature_statistics(
                pd.DataFrame({feature_name: new_data}), feature_name
            )
            
            if not ref_stats or not new_stats:
                return {}
            
            # Calculate differences
            differences = {}
            for attr in ['mean', 'std', 'min', 'max', 'median']:
                ref_val = getattr(ref_stats, attr)
                new_val = getattr(new_stats, attr)
                differences[f'{attr}_diff'] = new_val - ref_val
                differences[f'{attr}_diff_pct'] = ((new_val - ref_val) / ref_val * 100) if ref_val != 0 else 0
            
            return {
                'feature_name': feature_name,
                'reference_stats': ref_stats.to_dict(),
                'new_stats': new_stats.to_dict(),
                'differences': differences,
                'reference_sample_size': len(ref_data),
                'new_sample_size': len(new_data_clean)
            }
            
        except Exception as e:
            logger.error(f"Error comparing distributions for {feature_name}: {e}")
            return {}
    
    def get_drift_summary(self, drift_results: List[DriftResult]) -> Dict[str, Any]:
        """
        Generate summary of drift detection results
        
        Args:
            drift_results: List of drift detection results
            
        Returns:
            Summary dictionary
        """
        if not drift_results:
            return {
                'total_features_checked': 0,
                'drift_detected_count': 0,
                'severity_counts': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                'most_severe_drift': None,
                'affected_features': []
            }
        
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        affected_features = []
        
        for result in drift_results:
            if result.drift_detected:
                severity_counts[result.severity] += 1
                affected_features.append(result.feature_name)
        
        # Find most severe drift
        most_severe_drift = None
        if drift_results:
            most_severe_drift = max(drift_results, key=lambda x: x.drift_score)
        
        return {
            'total_features_checked': len(set(r.feature_name for r in drift_results)),
            'drift_detected_count': len([r for r in drift_results if r.drift_detected]),
            'severity_counts': severity_counts,
            'most_severe_drift': most_severe_drift.to_dict() if most_severe_drift else None,
            'affected_features': list(set(affected_features))
        }
    
    def update_reference_data(self, new_reference_data: pd.DataFrame) -> None:
        """
        Update reference data and recalculate statistics
        
        Args:
            new_reference_data: New reference dataset
        """
        with self._lock:
            self.reference_data = new_reference_data.copy()
            self.feature_columns = [col for col in self.reference_data.columns 
                                  if col.lower() not in ['target', 'label', 'y']]
            self._calculate_reference_statistics()
            logger.info("Reference data updated successfully")
    
    def save_reference_statistics(self, file_path: str) -> bool:
        """
        Save reference statistics to file
        
        Args:
            file_path: Path to save statistics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(self.reference_stats, f, indent=2, default=str)
            
            logger.info(f"Reference statistics saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save reference statistics: {e}")
            return False
    
    def load_reference_statistics(self, file_path: str) -> bool:
        """
        Load reference statistics from file
        
        Args:
            file_path: Path to statistics file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                self.reference_stats = json.load(f)
            
            logger.info(f"Reference statistics loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load reference statistics: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current drift detector status
        
        Returns:
            Status dictionary
        """
        return {
            'drift_detection_enabled': self.config['drift_detection_enabled'],
            'reference_data_loaded': self.reference_data is not None,
            'reference_data_shape': self.reference_data.shape if self.reference_data is not None else None,
            'feature_count': len(self.feature_columns),
            'reference_stats_count': len(self.reference_stats),
            'drift_thresholds': self.drift_thresholds,
            'severity_thresholds': self.severity_thresholds
        }
