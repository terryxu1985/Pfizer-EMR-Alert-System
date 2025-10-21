"""
Model management utilities for the EMR Alert System
Enhanced with automatic model discovery and version management
"""
import pickle
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import MODEL_CONFIG, MODEL_DIR, get_latest_model_path, get_model_version_info
from ..data_processing.data_processor import DataProcessor
from ..feature_engineering.emr_feature_processor import EMRFeatureProcessor
from ..api.api_models import RawEMRRequest

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Enhanced model manager with automatic discovery and version management
    Manages model loading, saving, and prediction for the EMR Alert System
    """
    
    def __init__(self, auto_discover: bool = True):
        """
        Initialize ModelManager with optional automatic model discovery
        
        Args:
            auto_discover: Whether to automatically discover the latest model
        """
        self.model = None
        self.data_processor = DataProcessor()
        self.emr_feature_processor = EMRFeatureProcessor()
        self.model_metadata = {}
        self.is_loaded = False
        self.auto_discover = auto_discover
        self.current_model_path = None
        self.model_version_info = {}
        
        # Load model automatically if discovery is enabled
        if self.auto_discover:
            self._auto_load_latest_model()
    
    def _auto_load_latest_model(self) -> bool:
        """
        Automatically discover and load the latest trained model
        
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            latest_model_path = get_latest_model_path()
            if latest_model_path:
                logger.info(f"Auto-discovering latest model at: {latest_model_path}")
                self.load_model(latest_model_path)
                return True
            else:
                logger.warning("No trained model found for auto-discovery")
                return False
        except Exception as e:
            logger.error(f"Failed to auto-discover model: {str(e)}")
            return False
    
    def validate_feature_consistency(self, input_features: List[str]) -> Dict[str, Any]:
        """
        Validate that input features match the trained model's expected features
        
        Args:
            input_features: List of feature names from input data
            
        Returns:
            Dictionary containing validation results
        """
        if not self.is_loaded:
            return {
                "valid": False,
                "error": "Model not loaded",
                "missing_features": [],
                "extra_features": [],
                "expected_features": []
            }
        
        expected_features = set(self.data_processor.feature_columns or [])
        input_features_set = set(input_features)
        
        missing_features = list(expected_features - input_features_set)
        extra_features = list(input_features_set - expected_features)
        
        validation_result = {
            "valid": len(missing_features) == 0,
            "missing_features": missing_features,
            "extra_features": extra_features,
            "expected_features": list(expected_features),
            "input_features": list(input_features_set),
            "feature_count_match": len(expected_features) == len(input_features_set)
        }
        
        if not validation_result["valid"]:
            logger.warning(f"Feature validation failed. Missing: {missing_features}")
            logger.warning(f"Extra features: {extra_features}")
        else:
            logger.info("Feature validation passed")
        
        return validation_result
    
    def reload_model_if_updated(self) -> bool:
        """
        Check if a newer model is available and reload if necessary
        
        Returns:
            True if model was reloaded, False otherwise
        """
        try:
            latest_model_path = get_latest_model_path()
            if latest_model_path and latest_model_path != self.current_model_path:
                logger.info(f"Newer model detected at: {latest_model_path}")
                self.load_model(latest_model_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to check for model updates: {str(e)}")
            return False
        
    def load_model(self, model_path: Optional[Path] = None) -> 'ModelManager':
        """
        Load the trained model and data processor with enhanced error handling
        
        Args:
            model_path: Path to the model directory (optional)
            
        Returns:
            Self for method chaining
        """
        if model_path is None:
            model_path = get_latest_model_path() or MODEL_DIR
        
        logger.info(f"Loading model from {model_path}")
        
        try:
            # Validate model directory exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory does not exist: {model_path}")
            
            # Load model metadata first
            self.model_version_info = get_model_version_info(model_path)
            logger.info(f"Model version info: {self.model_version_info}")
            
            # Load model - try Random Forest first, then fallback to XGBoost
            model_file = model_path / MODEL_CONFIG['model_file']
            if not model_file.exists():
                # Fallback to XGBoost model if Random Forest not found
                fallback_model_file = model_path / "xgboost_emr_alert_model.pkl"
                if fallback_model_file.exists():
                    model_file = fallback_model_file
                    logger.warning(f"Random Forest model not found, using XGBoost fallback: {model_file}")
                else:
                    raise FileNotFoundError(f"Model file not found: {model_file}")
            
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {model_file}")
            
            # Load data processor - try preprocessor file first, then fallback to scaler file
            preprocessor_file = model_path / "preprocessor.pkl"
            processor_file = model_path / MODEL_CONFIG['scaler_file']
            
            # Try to load using the new preprocessor format first
            if preprocessor_file.exists():
                try:
                    with open(preprocessor_file, 'rb') as f:
                        preprocessor_data = pickle.load(f)
                    
                    # Load the preprocessor data into the data processor
                    self.data_processor.scaler = preprocessor_data['scaler']
                    self.data_processor.label_encoders = preprocessor_data['label_encoders']
                    self.data_processor.feature_columns = preprocessor_data['feature_columns']
                    self.data_processor.is_fitted = preprocessor_data['is_fitted']
                    
                    logger.info(f"Data processor loaded from {preprocessor_file} (preprocessor format)")
                except Exception as e:
                    logger.warning(f"Failed to load preprocessor file: {e}, falling back to scaler file")
                    if not processor_file.exists():
                        raise FileNotFoundError(f"Neither preprocessor nor scaler file found")
                    
                    with open(processor_file, 'rb') as f:
                        preprocessor_data = pickle.load(f)
                    
                    # Load the preprocessor data into the data processor
                    self.data_processor.scaler = preprocessor_data['scaler']
                    self.data_processor.label_encoders = preprocessor_data['label_encoders']
                    self.data_processor.feature_columns = preprocessor_data['feature_columns']
                    self.data_processor.is_fitted = preprocessor_data['is_fitted']
                    
                    logger.info(f"Data processor loaded from {processor_file} (scaler format)")
            else:
                if not processor_file.exists():
                    raise FileNotFoundError(f"Neither preprocessor nor scaler file found")
                
                # Fallback to old format
                self.data_processor.load(processor_file)
                logger.info(f"Data processor loaded from {processor_file} (old format)")
            
            # Load label encoders only if not already provided by preprocessor
            if not getattr(self.data_processor, 'label_encoders', None):
                encoders_file = model_path / MODEL_CONFIG['label_encoders_file']
                if encoders_file.exists():
                    with open(encoders_file, 'rb') as f:
                        self.data_processor.label_encoders = pickle.load(f)
                    logger.info(f"Label encoders loaded from {encoders_file}")
                else:
                    # Preprocessor is authoritative; skip separate encoders file
                    logger.warning(
                        f"Label encoders file not found: {encoders_file}. Using encoders bundled in preprocessor if available."
                    )
            
            # Skip loading feature columns file since preprocessor already contains the correct features
            logger.info(f"Feature columns already loaded from preprocessor: {len(self.data_processor.feature_columns)} features")
            
            # Load additional metadata if available
            metadata_file = model_path / MODEL_CONFIG['metadata_file']
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self.model_metadata = pickle.load(f)
                logger.info(f"Model metadata loaded from {metadata_file}")
            
            # Set fitted flag
            self.data_processor.is_fitted = True
            
            # Store current model path for version checking
            self.current_model_path = model_path
            
            self.is_loaded = True
            logger.info("Model and data processor loaded successfully")
            logger.info(f"Model type: {self.model_version_info.get('model_type', 'Unknown')}")
            logger.info(f"Model version: {self.model_version_info.get('version', 'Unknown')}")
            logger.info(f"Feature count: {len(self.data_processor.feature_columns)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            raise
        
        return self
    
    def save_model(self, model, data_processor: DataProcessor, 
                   model_path: Optional[Path] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the trained model and data processor
        
        Args:
            model: Trained model object
            data_processor: Fitted data processor
            model_path: Path to save the model (optional)
            metadata: Model metadata (optional)
        """
        if model_path is None:
            model_path = MODEL_DIR
        
        # Ensure model_path is a Path object
        model_path = Path(model_path)
        
        # Create directory if it doesn't exist
        model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {model_path}")
        
        try:
            # Save model
            model_file = model_path / MODEL_CONFIG['model_file']
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_file}")
            
            # Save data processor
            processor_file = model_path / MODEL_CONFIG['scaler_file']
            data_processor.save_preprocessor(processor_file)
            logger.info(f"Data processor saved to {processor_file}")
            
            # Save label encoders
            encoders_file = model_path / MODEL_CONFIG['label_encoders_file']
            with open(encoders_file, 'wb') as f:
                pickle.dump(data_processor.label_encoders, f)
            logger.info(f"Label encoders saved to {encoders_file}")
            
            # Save feature columns
            features_file = model_path / MODEL_CONFIG['feature_columns_file']
            with open(features_file, 'wb') as f:
                pickle.dump(data_processor.feature_columns, f)
            logger.info(f"Feature columns saved to {features_file}")
            
            # Save metadata
            if metadata:
                metadata_file = model_path / "model_metadata.pkl"
                with open(metadata_file, 'wb') as f:
                    pickle.dump(metadata, f)
                logger.info(f"Model metadata saved to {metadata_file}")
            
            logger.info("Model and data processor saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            data: Input data as pandas DataFrame
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")
        
        logger.info(f"Making predictions on data with shape: {data.shape}")
        
        try:
            # Transform data
            X_transformed = self.data_processor.transform(data)
            
            # Make predictions
            predictions = self.model.predict(X_transformed)
            probabilities = self.model.predict_proba(X_transformed)
            
            logger.info(f"Predictions completed. Shape: {predictions.shape}")
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction on a single record with feature validation
        
        Args:
            data: Single record as dictionary
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")
        
        logger.info("Making prediction on single record")
        
        try:
            # Validate feature consistency
            input_features = list(data.keys())
            validation_result = self.validate_feature_consistency(input_features)
            
            if not validation_result["valid"]:
                logger.warning("Feature validation failed, attempting prediction with available features")
                logger.warning(f"Missing features: {validation_result['missing_features']}")
                logger.warning(f"Extra features: {validation_result['extra_features']}")
            
            # Convert to DataFrame
            df = pd.DataFrame([data])
            
            # Make prediction
            predictions, probabilities = self.predict(df)
            
            # Extract results
            prediction = int(predictions[0])
            # Probability of class 1, where class 1 = TARGET=1 = "Not Prescribed Drug A" (alert candidates)
            probability = float(probabilities[0][1])
            
            # ========== Clinical Eligibility Check (Fully Rule-Based) ==========
            # Get clinical eligibility assessment results (passed from process_raw_emr_data)
            clinical_eligibility = data.get('_clinical_eligibility', {})
            
            # If no clinical eligibility data, create default values (not eligible for alert)
            if not clinical_eligibility:
                logger.warning("No clinical eligibility data found, defaulting to not eligible")
                clinical_eligibility = {
                    'meets_criteria': False,
                    'age_eligible': False,
                    'patient_age': 0,
                    'within_5day_window': False,
                    'is_high_risk': False,
                    'risk_factors_found': [],
                    'no_severe_contraindication': False,
                    'contraindication_level': 0
                }
            
            # Extract clinical eligibility decision result
            meets_clinical_criteria = clinical_eligibility.get('meets_criteria', False)
            
            # ========== AI Prediction Assessment ==========
            ai_predicts_missed = (prediction == 1) and (probability >= 0.7)
            
            # ========== Final Alert Decision ==========
            # Only recommend alert when [AI predicts missed] AND [meets clinical criteria]
            alert_recommended = ai_predicts_missed and meets_clinical_criteria
            
            # ========== Detailed Logging ==========
            if ai_predicts_missed and not meets_clinical_criteria:
                reasons = []
                if not clinical_eligibility.get('age_eligible', False):
                    reasons.append(f"age={clinical_eligibility.get('patient_age', 0)}<12")
                if not clinical_eligibility.get('within_5day_window', False):
                    reasons.append(f"outside_5day_window(days={clinical_eligibility.get('symptom_to_diagnosis_days', 'unknown')})")
                if not clinical_eligibility.get('is_high_risk', False):
                    reasons.append("not_high_risk")
                if not clinical_eligibility.get('no_severe_contraindication', False):
                    reasons.append(f"severe_contraindication(level={clinical_eligibility.get('contraindication_level', 0)})")
                
                logger.info(
                    f"AI predicted missed prescription (prob={probability:.3f}) but patient does NOT meet "
                    f"clinical criteria. Reasons: {', '.join(reasons)}"
                )
            
            if alert_recommended:
                logger.info(
                    f"✅ ALERT RECOMMENDED: age={clinical_eligibility.get('patient_age')}, "
                    f"risk_factors={clinical_eligibility.get('risk_factors_found')}, "
                    f"AI prob={probability:.3f}"
                )

            result = {
                # Keep prediction aligned with TARGET semantics: 1 = Not Prescribed Drug A (needs intervention)
                'prediction': prediction,
                'probability': probability,
                # Business-facing fields directly aligned to class 1 (no inversion)
                'not_prescribed_drug_a': prediction,
                'not_prescribed_drug_a_probability': probability,
                # Alert recommendation: AI prediction AND clinical eligibility (fully rule-based)
                'alert_recommended': alert_recommended,
                # Clinical eligibility details (fully rule-based assessment)
                'clinical_eligibility': clinical_eligibility,
                'feature_validation': validation_result,
                'model_version': self.model_version_info.get('version', 'unknown'),
                'model_type': self.model_version_info.get('model_type', 'unknown')
            }
            
            logger.info(
                f"Prediction completed: pred={prediction}, prob={probability:.3f}, "
                f"clinical_eligible={meets_clinical_criteria}, alert={alert_recommended}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error making single prediction: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information and metadata
        
        Returns:
            Dictionary with model information including version details
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded first")
        
        info = {
            'model_type': type(self.model).__name__,
            'feature_count': len(self.data_processor.feature_columns),
            'feature_names': self.data_processor.feature_columns,
            'is_loaded': self.is_loaded,
            'model_config': MODEL_CONFIG,
            'current_model_path': str(self.current_model_path) if self.current_model_path else None,
            'version_info': self.model_version_info,
            'metadata': self.model_metadata
        }
        
        # Add model-specific information
        if hasattr(self.model, 'feature_importances_'):
            # Convert numpy types to Python types for JSON serialization
            importances = [float(x) for x in self.model.feature_importances_]
            info['feature_importances'] = dict(zip(
                self.data_processor.feature_columns, 
                importances
            ))
        
        # Add Random Forest specific parameters
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth
            
        if hasattr(self.model, 'min_samples_split'):
            info['min_samples_split'] = self.model.min_samples_split
            
        if hasattr(self.model, 'min_samples_leaf'):
            info['min_samples_leaf'] = self.model.min_samples_leaf
        
        # Add performance metrics if available
        if self.model_metadata and 'performance_metrics' in self.model_metadata:
            info['performance_metrics'] = self.model_metadata['performance_metrics']
        
        return info
    
    def process_raw_emr_data(self, raw_data: RawEMRRequest) -> Tuple[Dict[str, Any], Any]:
        """
        Process raw EMR data and convert to model features
        
        Args:
            raw_data: Raw EMR request data
            
        Returns:
            Tuple of (processed_features_dict, feature_engineering_info)
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before processing raw EMR data")
        
        logger.info(f"Processing raw EMR data for patient {raw_data.patient_id}")
        
        try:
            # Use EMR feature processor to convert raw data to features
            processed_features, fe_info = self.emr_feature_processor.process_raw_emr_data(raw_data)
            
            # Check clinical eligibility based on rules (fully rule-based, independent of model features)
            clinical_eligibility = self.emr_feature_processor.check_clinical_eligibility(
                raw_data=raw_data,
                processed_features=processed_features
            )
            
            # Add clinical eligibility assessment results to processed features (for alert decision)
            processed_features['_clinical_eligibility'] = clinical_eligibility
            
            logger.info(f"Successfully processed {len(processed_features)} features for patient {raw_data.patient_id}")
            logger.info(f"Clinical eligibility: meets_criteria={clinical_eligibility['meets_criteria']}, "
                       f"age={clinical_eligibility['patient_age']}, "
                       f"high_risk={clinical_eligibility['is_high_risk']}, "
                       f"risk_factors={clinical_eligibility['risk_factors_found']}")
            
            return processed_features, fe_info
            
        except Exception as e:
            logger.error(f"Error processing raw EMR data for patient {raw_data.patient_id}: {str(e)}")
            raise
    
    def validate_processed_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate processed features using the EMR feature processor
        
        Args:
            features: Dictionary of processed features
            
        Returns:
            Validation results dictionary
        """
        try:
            validation_result = self.emr_feature_processor.validate_features(features)
            
            # Also validate against model expectations
            feature_names = list(features.keys())
            model_validation = self.validate_feature_consistency(feature_names)
            
            # Combine validation results
            combined_validation = {
                **validation_result,
                "model_consistency": model_validation,
                "overall_valid": validation_result["valid"] and model_validation["valid"]
            }
            
            return combined_validation
            
        except Exception as e:
            logger.error(f"Error validating processed features: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "warnings": [],
                "errors": [str(e)]
            }
    
    def predict_batch_raw_emr(self, raw_data_list: List[Any]) -> Tuple[List[Dict[str, Any]], List[Any], float]:
        """
        Process and predict multiple raw EMR records
        
        Args:
            raw_data_list: List of RawEMRRequest objects
            
        Returns:
            Tuple of (prediction_results_list, feature_engineering_info_list, total_processing_time_ms)
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before processing raw EMR data")
        
        start_time = datetime.now()
        results = []
        fe_info_list = []
        
        logger.info(f"Processing batch of {len(raw_data_list)} raw EMR records")
        
        try:
            for raw_data in raw_data_list:
                # Process raw EMR data to features
                processed_features, fe_info = self.process_raw_emr_data(raw_data)
                
                # Validate features
                feature_validation = self.validate_processed_features(processed_features)
                
                # Make prediction
                prediction_result = self.predict_single(processed_features)
                
                # Add feature engineering info to prediction result
                prediction_result.update({
                    'processed_features': processed_features,
                    'feature_engineering_info': {
                        "symptom_count": fe_info.symptom_count,
                        "risk_factors_found": fe_info.risk_factors_found,
                        "time_to_diagnosis_days": fe_info.time_to_diagnosis_days,
                        "contraindication_level": fe_info.contraindication_level,
                        "physician_experience_level": fe_info.physician_experience_level,
                        "processing_warnings": fe_info.processing_warnings
                    },
                    'feature_validation': feature_validation
                })
                
                results.append(prediction_result)
                fe_info_list.append(fe_info)
            
            # Calculate total processing time
            total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Batch raw EMR processing completed: {len(results)} patients in {total_processing_time:.2f}ms")
            
            return results, fe_info_list, total_processing_time
            
        except Exception as e:
            logger.error(f"Error processing batch raw EMR data: {str(e)}")
            raise