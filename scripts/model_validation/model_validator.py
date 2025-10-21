"""
Model Validation Utilities for EMR Alert System

This module provides comprehensive model validation tools to ensure
model consistency and feature alignment between training and prediction.
"""
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
import pandas as pd
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import MODEL_DIR, MODEL_CONFIG

logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Comprehensive model validation utilities
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize ModelValidator
        
        Args:
            model_path: Path to model directory (defaults to MODEL_DIR)
        """
        self.model_path = model_path or MODEL_DIR
        self.label_encoders = None
        self.feature_columns = None
        self.model_metadata = None
        
    def load_model_artifacts(self) -> None:
        """Load model artifacts for validation"""
        try:
            # Load label encoders
            encoders_file = self.model_path / MODEL_CONFIG['label_encoders_file']
            with open(encoders_file, 'rb') as f:
                self.label_encoders = pickle.load(f)
            logger.info(f"Loaded label encoders from {encoders_file}")
            
            # Load feature columns
            features_file = self.model_path / MODEL_CONFIG['feature_columns_file']
            with open(features_file, 'rb') as f:
                self.feature_columns = pickle.load(f)
            logger.info(f"Loaded feature columns from {features_file}")
            
            # Load model metadata if available
            metadata_file = self.model_path / "model_metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self.model_metadata = pickle.load(f)
                logger.info(f"Loaded model metadata from {metadata_file}")
                
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            raise
    
    def validate_training_values(self) -> Dict[str, List[str]]:
        """
        Validate training values for categorical columns
        
        Returns:
            Dictionary mapping column names to their training values
        """
        if self.label_encoders is None:
            self.load_model_artifacts()
        
        training_values = {}
        for col, encoder in self.label_encoders.items():
            training_values[col] = list(encoder.classes_)
        
        return training_values
    
    def validate_feature_consistency(self, test_columns: List[str]) -> Dict[str, Any]:
        """
        Validate feature consistency between model and test data
        
        Args:
            test_columns: List of columns in test data
            
        Returns:
            Dictionary with validation results
        """
        if self.feature_columns is None:
            self.load_model_artifacts()
        
        model_features = set(self.feature_columns)
        test_features = set(test_columns)
        
        missing_features = model_features - test_features
        extra_features = test_features - model_features
        
        return {
            'model_features': list(model_features),
            'test_features': list(test_features),
            'missing_features': list(missing_features),
            'extra_features': list(extra_features),
            'is_consistent': len(missing_features) == 0 and len(extra_features) == 0,
            'model_feature_count': len(model_features),
            'test_feature_count': len(test_features)
        }
    
    def validate_categorical_values(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate categorical values in test data against training values
        
        Args:
            test_data: DataFrame with test data
            
        Returns:
            Dictionary with validation results
        """
        if self.label_encoders is None:
            self.load_model_artifacts()
        
        validation_results = {}
        
        for col, encoder in self.label_encoders.items():
            if col in test_data.columns:
                test_values = set(test_data[col].dropna().unique())
                training_values = set(encoder.classes_)
                
                unknown_values = test_values - training_values
                
                validation_results[col] = {
                    'training_values': list(training_values),
                    'test_values': list(test_values),
                    'unknown_values': list(unknown_values),
                    'has_unknown_values': len(unknown_values) > 0,
                    'coverage': len(test_values & training_values) / len(training_values) if training_values else 0
                }
            else:
                validation_results[col] = {
                    'error': f"Column {col} not found in test data"
                }
        
        return validation_results
    
    def generate_validation_report(self, test_data: Optional[pd.DataFrame] = None, 
                                 test_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Args:
            test_data: Optional DataFrame with test data
            test_columns: Optional list of test data columns
            
        Returns:
            Comprehensive validation report
        """
        if self.label_encoders is None or self.feature_columns is None:
            self.load_model_artifacts()
        
        report = {
            'model_path': str(self.model_path),
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Feature consistency validation
        if test_columns:
            feature_validation = self.validate_feature_consistency(test_columns)
            report['feature_validation'] = feature_validation
        
        # Training values validation
        training_values = self.validate_training_values()
        report['training_values'] = training_values
        
        # Categorical values validation
        if test_data is not None:
            categorical_validation = self.validate_categorical_values(test_data)
            report['categorical_validation'] = categorical_validation
        
        # Overall validation status
        overall_valid = True
        if test_columns:
            overall_valid &= report['feature_validation']['is_consistent']
        
        if test_data is not None:
            categorical_results = report['categorical_validation']
            for col_result in categorical_results.values():
                if isinstance(col_result, dict) and col_result.get('has_unknown_values', False):
                    overall_valid = False
        
        report['overall_valid'] = overall_valid
        
        return report
    
    def print_validation_summary(self, report: Dict[str, Any]) -> None:
        """
        Print a formatted validation summary
        
        Args:
            report: Validation report from generate_validation_report
        """
        print("=" * 60)
        print("MODEL VALIDATION REPORT")
        print("=" * 60)
        print(f"Model Path: {report['model_path']}")
        print(f"Validation Time: {report['validation_timestamp']}")
        print(f"Overall Status: {'✅ VALID' if report['overall_valid'] else '❌ INVALID'}")
        print()
        
        # Feature validation
        if 'feature_validation' in report:
            fv = report['feature_validation']
            print("FEATURE CONSISTENCY:")
            print(f"  Model Features: {fv['model_feature_count']}")
            print(f"  Test Features: {fv['test_feature_count']}")
            
            if fv['missing_features']:
                print(f"  ❌ Missing Features ({len(fv['missing_features'])}):")
                for feature in fv['missing_features']:
                    print(f"    - {feature}")
            
            if fv['extra_features']:
                print(f"  ⚠️  Extra Features ({len(fv['extra_features'])}):")
                for feature in fv['extra_features']:
                    print(f"    - {feature}")
            
            if fv['is_consistent']:
                print("  ✅ Feature consistency: PASSED")
            else:
                print("  ❌ Feature consistency: FAILED")
            print()
        
        # Training values
        print("TRAINING VALUES:")
        for col, values in report['training_values'].items():
            print(f"  {col}: {len(values)} values")
            if len(values) <= 10:  # Show values if not too many
                print(f"    Values: {values}")
        print()
        
        # Categorical validation
        if 'categorical_validation' in report:
            print("CATEGORICAL VALUES VALIDATION:")
            for col, result in report['categorical_validation'].items():
                if 'error' in result:
                    print(f"  ❌ {col}: {result['error']}")
                else:
                    status = "✅" if not result['has_unknown_values'] else "❌"
                    print(f"  {status} {col}: Coverage {result['coverage']:.2%}")
                    if result['unknown_values']:
                        print(f"    Unknown values: {result['unknown_values']}")
        print()


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Validation Tool")
    parser.add_argument("--model-path", type=str, help="Path to model directory")
    parser.add_argument("--test-data", type=str, help="Path to test data CSV file")
    parser.add_argument("--test-columns", nargs="+", help="List of test data columns")
    
    args = parser.parse_args()
    
    # Initialize validator
    model_path = Path(args.model_path) if args.model_path else None
    validator = ModelValidator(model_path)
    
    # Load test data if provided
    test_data = None
    test_columns = args.test_columns
    
    if args.test_data:
        test_data = pd.read_csv(args.test_data)
        test_columns = list(test_data.columns)
    
    # Generate and print validation report
    report = validator.generate_validation_report(test_data, test_columns)
    validator.print_validation_summary(report)
    
    # Exit with error code if validation failed
    if not report['overall_valid']:
        sys.exit(1)


if __name__ == "__main__":
    main()
