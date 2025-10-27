"""
FastAPI application for the EMR Alert System
"""
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .model_manager import ModelManager
from .api_models import (
    HealthResponse, ErrorResponse,
    RawEMRRequest, RawEMRPredictionResponse, BatchRawEMRRequest, BatchRawEMRPredictionResponse,
    PerformanceMetricsResponse, DriftAlertResponse, DataDriftResult, MonitoringStatusResponse,
    MetricsSummaryResponse, AlertSummaryResponse, MetricValueResponse, AggregatedMetricsResponse
)
from ..data_access.patient_repository import patient_repository
from ..data_access.emr_data_loader import emr_data_loader
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import API_CONFIG, LOGGING_CONFIG

# Pydantic models for patient input
class PatientInput(BaseModel):
    patient_name: str
    patient_age: int
    patient_gender: str
    physician_id: int
    diagnosis_date: str
    symptom_onset_date: Optional[str] = None
    location_type: str
    insurance_type: str
    contraindication_level: str
    comorbidities: List[str] = []
    symptoms: List[str] = []
    additional_notes: Optional[str] = None

class PatientResponse(BaseModel):
    patient: dict
    message: str
    timestamp: str

# Configure logging with rotation
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))

# Create formatter
formatter = logging.Formatter(LOGGING_CONFIG['format'])

# File handler with rotation
file_handler = logging.handlers.RotatingFileHandler(
    LOGGING_CONFIG['file'],
    maxBytes=LOGGING_CONFIG.get('max_bytes', 10 * 1024 * 1024),  # 10MB default
    backupCount=LOGGING_CONFIG.get('backup_count', 5)
)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)

# Add console handler only if configured
if LOGGING_CONFIG.get('console_output', True):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Configure root logger to prevent duplicate logs
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG['level']), handlers=[])

# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG['title'],
    description=API_CONFIG['description'],
    version=API_CONFIG['version']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager instance
model_manager = ModelManager()

# Initialize monitoring systems
try:
    from ..monitoring import PerformanceMonitor, DataDriftDetector, AlertSystem, MetricsCollector
    
    # Initialize monitoring components
    performance_monitor = PerformanceMonitor()
    drift_detector = DataDriftDetector()
    alert_system = AlertSystem()
    metrics_collector = MetricsCollector()
    
    monitoring_enabled = True
    logger.info("Monitoring systems initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize monitoring systems: {e}")
    performance_monitor = None
    drift_detector = None
    alert_system = None
    metrics_collector = None
    monitoring_enabled = False

def get_model_manager() -> ModelManager:
    """Dependency to get the model manager"""
    return model_manager

def get_monitoring_systems():
    """Dependency to get monitoring systems"""
    return {
        'performance_monitor': performance_monitor,
        'drift_detector': drift_detector,
        'alert_system': alert_system,
        'metrics_collector': metrics_collector,
        'monitoring_enabled': monitoring_enabled
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        logger.info("Starting EMR Alert System API...")
        # Only load model if not already loaded
        if not model_manager.is_loaded:
            model_manager.load_model()
            logger.info("Model loaded successfully")
        else:
            logger.info("Model already loaded, skipping reload")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "EMR Alert System API",
        "version": API_CONFIG['version'],
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """Health check endpoint"""
    try:
        model_info = manager.get_model_info() if manager.is_loaded else None
        
        return HealthResponse(
            status="healthy" if manager.is_loaded else "unhealthy",
            model_loaded=manager.is_loaded,
            model_info=model_info,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_info=None,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )


@app.get("/model/info")
async def get_model_info(manager: ModelManager = Depends(get_model_manager)):
    """Get model information and metadata"""
    try:
        if not manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        model_info = manager.get_model_info()
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving model information"
        )

@app.get("/model/features")
async def get_feature_info(manager: ModelManager = Depends(get_model_manager)):
    """Get feature information and importance"""
    try:
        if not manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        model_info = manager.get_model_info()
        
        feature_info = {
            'feature_names': model_info.get('feature_names', []),
            'feature_count': model_info.get('feature_count', 0),
            'feature_importances': model_info.get('feature_importances', {}),
            'model_type': model_info.get('model_type', 'Unknown'),
            'version_info': model_info.get('version_info', {}),
            'current_model_path': model_info.get('current_model_path')
        }
        
        return feature_info
        
    except Exception as e:
        logger.error(f"Error getting feature info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving feature information"
        )

@app.post("/model/reload")
async def reload_model(manager: ModelManager = Depends(get_model_manager)):
    """Reload model if a newer version is available"""
    try:
        reloaded = manager.reload_model_if_updated()
        
        if reloaded:
            logger.info("Model successfully reloaded")
            return {
                "message": "Model reloaded successfully",
                "reloaded": True,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        else:
            return {
                "message": "No newer model found",
                "reloaded": False,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reloading model"
        )

@app.post("/model/validate-features")
async def validate_features(
    features: List[str],
    manager: ModelManager = Depends(get_model_manager)
):
    """Validate feature consistency with the loaded model"""
    try:
        if not manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        validation_result = manager.validate_feature_consistency(features)
        
        return {
            "validation_result": validation_result,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Error validating features: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error validating features"
        )

@app.post("/predict", response_model=RawEMRPredictionResponse)
async def predict(
    request: RawEMRRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    EMR Prediction endpoint
    
    This is the main prediction endpoint that accepts raw EMR transaction data
    and performs real-time feature engineering to convert it to model-ready 
    features for prediction.
    
    For batch predictions, use /predict/batch endpoint.
    """
    start_time = datetime.utcnow()
    
    try:
        if not manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        logger.info(f"Processing raw EMR data for patient {request.patient_id}")
        
        # Process raw EMR data to features
        processed_features, fe_info = manager.process_raw_emr_data(request)
        
        # Validate features
        feature_validation = manager.validate_processed_features(processed_features)
        
        # Make prediction
        prediction_result = manager.predict_single(processed_features)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Prepare feature engineering info
        fe_info_dict = {
            "symptom_count": fe_info.symptom_count,
            "risk_factors_found": fe_info.risk_factors_found,
            "time_to_diagnosis_days": fe_info.time_to_diagnosis_days,
            "contraindication_level": fe_info.contraindication_level,
            "physician_experience_level": fe_info.physician_experience_level,
            "processing_warnings": fe_info.processing_warnings
        }
        
        # Create response
        response = RawEMRPredictionResponse(
            prediction=prediction_result['prediction'],
            probability=prediction_result['probability'],
            not_prescribed_drug_a=prediction_result['not_prescribed_drug_a'],
            not_prescribed_drug_a_probability=prediction_result['not_prescribed_drug_a_probability'],
            alert_recommended=prediction_result['alert_recommended'],
            clinical_eligibility=prediction_result.get('clinical_eligibility', {}),
            processed_features=processed_features,
            feature_engineering_info=fe_info_dict,
            feature_validation=feature_validation,
            model_version=prediction_result.get('model_version', 'unknown'),
            model_type=prediction_result.get('model_type', 'unknown'),
            processing_time_ms=processing_time
        )
        
        logger.info(f"Raw EMR prediction completed for patient {request.patient_id} in {processing_time:.2f}ms")
        
        return response
        
    except ValueError as e:
        logger.error(f"Feature engineering error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature engineering failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Raw EMR prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )

@app.get("/patients")
async def get_patients():
    """
    Get list of patients for UI display
    Returns patient data from various sources including:
    - Real EMR data (when available)
    - Generated data based on model training patterns
    - Sample demonstration data
    """
    try:
        # Get patients from the patient data manager
        patients = patient_repository.get_patients()
        
        return {
            "patients": patients,
            "total_count": len(patients),
            "data_sources": list(set(p.get("dataSource", "Unknown") for p in patients)),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting patients: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving patient data"
        )

@app.get("/patients/stats")
async def get_patient_statistics():
    """
    Get statistics about patient data sources and availability
    """
    try:
        # Get real EMR data statistics
        real_stats = emr_data_loader.get_patient_statistics()
        
        # Get current patient data info
        current_patients = patient_repository.get_patients()
        
        return {
            "real_emr_data": real_stats,
            "current_patients": {
                "count": len(current_patients),
                "data_sources": list(set(p.get("dataSource", "Unknown") for p in current_patients))
            },
            "data_availability": {
                "real_emr_available": "error" not in real_stats,
                "generated_data_available": True,
                "sample_data_available": True
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting patient statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving patient statistics"
        )

@app.post("/patients", response_model=PatientResponse)
async def create_patient(patient_input: PatientInput):
    """
    Create a new patient from doctor input and automatically store in database
    """
    try:
        # Convert input to patient record format
        patient_record = patient_repository.create_patient_from_input(patient_input)
        
        logger.info(f"Created new patient: {patient_record['name']} (ID: {patient_record['id']})")
        
        return PatientResponse(
            patient=patient_record,
            message="Patient created successfully and ready for AI analysis",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Error creating patient: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating patient"
        )

@app.get("/patients/{patient_id}")
async def get_patient(patient_id: int):
    """
    Get a specific patient by ID
    """
    try:
        patient = patient_repository.get_patient_by_id(patient_id)
        
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient with ID {patient_id} not found"
            )
        
        return {
            "patient": patient,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient {patient_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving patient data"
        )

@app.post("/predict/batch", response_model=BatchRawEMRPredictionResponse)
async def predict_batch(
    request: BatchRawEMRRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Batch EMR prediction endpoint
    
    This endpoint accepts multiple raw EMR records and performs batch processing
    with real-time feature engineering for each patient.
    """
    start_time = datetime.utcnow()
    
    try:
        if not manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        logger.info(f"Processing batch of {len(request.patients)} raw EMR records")
        
        # Process batch raw EMR data
        prediction_results, fe_info_list, processing_time = manager.predict_batch_raw_emr(request.patients)
        
        # Format results
        formatted_results = []
        alerts_count = 0
        
        for result in prediction_results:
            formatted_result = RawEMRPredictionResponse(
                prediction=result['prediction'],
                probability=result['probability'],
                not_prescribed_drug_a=result['not_prescribed_drug_a'],
                not_prescribed_drug_a_probability=result['not_prescribed_drug_a_probability'],
                alert_recommended=result['alert_recommended'],
                processed_features=result['processed_features'],
                feature_engineering_info=result['feature_engineering_info'],
                feature_validation=result['feature_validation'],
                model_version=result.get('model_version', 'unknown'),
                model_type=result.get('model_type', 'unknown'),
                processing_time_ms=result.get('processing_time_ms', 0)
            )
            
            formatted_results.append(formatted_result)
            
            if result['alert_recommended']:
                alerts_count += 1
        
        # Calculate total processing time
        total_processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response = BatchRawEMRPredictionResponse(
            predictions=formatted_results,
            total_patients=len(formatted_results),
            alerts_recommended=alerts_count,
            total_processing_time_ms=total_processing_time
        )
        
        logger.info(f"Batch unified prediction completed: {len(formatted_results)} patients, {alerts_count} alerts in {total_processing_time:.2f}ms")
        
        return response
        
    except ValueError as e:
        logger.error(f"Batch feature engineering error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch feature engineering failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Batch unified prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch prediction"
        )


# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@app.get("/monitoring/status", response_model=MonitoringStatusResponse)
async def get_monitoring_status(monitoring_systems: dict = Depends(get_monitoring_systems)):
    """Get comprehensive monitoring system status"""
    try:
        if not monitoring_systems['monitoring_enabled']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring systems not available"
            )
        
        return MonitoringStatusResponse(
            monitoring_enabled=monitoring_systems['monitoring_enabled'],
            performance_monitor_status=monitoring_systems['performance_monitor'].get_status(),
            drift_detector_status=monitoring_systems['drift_detector'].get_status(),
            alert_system_status=monitoring_systems['alert_system'].get_status(),
            metrics_collector_status=monitoring_systems['metrics_collector'].get_status(),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving monitoring status"
        )

@app.get("/monitoring/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(monitoring_systems: dict = Depends(get_monitoring_systems)):
    """Get current performance metrics"""
    try:
        if not monitoring_systems['monitoring_enabled']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring systems not available"
            )
        
        current_metrics = monitoring_systems['performance_monitor'].calculate_current_metrics()
        
        if not current_metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No performance metrics available"
            )
        
        return PerformanceMetricsResponse(
            timestamp=current_metrics.timestamp,
            pr_auc=current_metrics.pr_auc,
            precision=current_metrics.precision,
            recall=current_metrics.recall,
            f1_score=current_metrics.f1_score,
            accuracy=current_metrics.accuracy,
            prediction_count=current_metrics.prediction_count,
            error_count=current_metrics.error_count,
            avg_response_time_ms=current_metrics.avg_response_time_ms,
            model_version=current_metrics.model_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving performance metrics"
        )

@app.get("/monitoring/drift-alerts", response_model=List[DriftAlertResponse])
async def get_drift_alerts(
    hours: int = 24,
    alert_level: Optional[str] = None,
    monitoring_systems: dict = Depends(get_monitoring_systems)
):
    """Get drift alerts for specified time period"""
    try:
        if not monitoring_systems['monitoring_enabled']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring systems not available"
            )
        
        alerts = monitoring_systems['performance_monitor'].get_drift_alerts(
            hours=hours,
            alert_level=alert_level
        )
        
        return [
            DriftAlertResponse(
                id=alert['id'],
                timestamp=alert['timestamp'],
                metric=alert['metric'],
                baseline_value=alert['baseline_value'],
                current_value=alert['current_value'],
                decline_percentage=alert['decline_percentage'],
                alert_level=alert['alert_level'],
                message=alert['message']
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Error getting drift alerts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving drift alerts"
        )

@app.get("/monitoring/metrics-summary", response_model=MetricsSummaryResponse)
async def get_metrics_summary(
    hours: int = 24,
    monitoring_systems: dict = Depends(get_monitoring_systems)
):
    """Get comprehensive metrics summary"""
    try:
        if not monitoring_systems['monitoring_enabled']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring systems not available"
            )
        
        summary = monitoring_systems['performance_monitor'].get_performance_summary(hours=hours)
        
        return MetricsSummaryResponse(
            time_period_hours=summary.get('time_period_hours', hours),
            total_predictions=summary.get('total_predictions', 0),
            avg_response_time_ms=summary.get('avg_response_time_ms', 0.0),
            labeled_predictions=summary.get('labeled_predictions', 0),
            recent_metrics=summary.get('recent_metrics', []),
            recent_alerts=summary.get('recent_alerts', []),
            current_baseline=summary.get('current_baseline', {}),
            drift_thresholds=summary.get('drift_thresholds', {})
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving metrics summary"
        )

@app.get("/monitoring/alerts/summary", response_model=AlertSummaryResponse)
async def get_alert_summary(
    hours: int = 24,
    monitoring_systems: dict = Depends(get_monitoring_systems)
):
    """Get alert summary statistics"""
    try:
        if not monitoring_systems['monitoring_enabled']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring systems not available"
            )
        
        summary = monitoring_systems['alert_system'].get_alert_statistics(hours=hours)
        
        return AlertSummaryResponse(
            total_alerts=summary.get('total_alerts', 0),
            severity_counts=summary.get('severity_counts', {}),
            type_counts=summary.get('type_counts', {}),
            acknowledged_count=summary.get('acknowledged_count', 0),
            resolved_count=summary.get('resolved_count', 0),
            time_period_hours=summary.get('time_period_hours', hours)
        )
        
    except Exception as e:
        logger.error(f"Error getting alert summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving alert summary"
        )

@app.get("/monitoring/metrics/{metric_name}", response_model=List[MetricValueResponse])
async def get_metric_values(
    metric_name: str,
    hours: int = 24,
    tags: Optional[Dict[str, str]] = None,
    monitoring_systems: dict = Depends(get_monitoring_systems)
):
    """Get metric values for a specific metric"""
    try:
        if not monitoring_systems['monitoring_enabled']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring systems not available"
            )
        
        values = monitoring_systems['metrics_collector'].get_metric_values(
            metric_name=metric_name,
            hours=hours,
            tags=tags
        )
        
        return [
            MetricValueResponse(
                timestamp=value['timestamp'],
                metric_name=value['metric_name'],
                value=value['value'],
                tags=value['tags'],
                source=value['source']
            )
            for value in values
        ]
        
    except Exception as e:
        logger.error(f"Error getting metric values: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving metric values"
        )

@app.get("/monitoring/metrics/{metric_name}/aggregated", response_model=List[AggregatedMetricsResponse])
async def get_aggregated_metrics(
    metric_name: str,
    interval: str = "1h",
    hours: int = 24,
    monitoring_systems: dict = Depends(get_monitoring_systems)
):
    """Get aggregated metrics for a specific metric and interval"""
    try:
        if not monitoring_systems['monitoring_enabled']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring systems not available"
            )
        
        aggregated = monitoring_systems['metrics_collector'].get_aggregated_metrics(
            metric_name=metric_name,
            interval=interval,
            hours=hours
        )
        
        return [
            AggregatedMetricsResponse(
                timestamp=agg['timestamp'],
                metric_name=agg['metric_name'],
                time_window=agg['time_window'],
                count=agg['count'],
                sum_value=agg['sum_value'],
                min_value=agg['min_value'],
                max_value=agg['max_value'],
                mean_value=agg['mean_value'],
                std_value=agg['std_value'],
                p50_value=agg['p50_value'],
                p95_value=agg['p95_value'],
                p99_value=agg['p99_value'],
                tags=agg['tags']
            )
            for agg in aggregated
        ]
        
    except Exception as e:
        logger.error(f"Error getting aggregated metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving aggregated metrics"
        )

@app.post("/monitoring/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    monitoring_systems: dict = Depends(get_monitoring_systems)
):
    """Acknowledge an alert"""
    try:
        if not monitoring_systems['monitoring_enabled']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring systems not available"
            )
        
        success = monitoring_systems['alert_system'].acknowledge_alert(alert_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        return {"message": f"Alert {alert_id} acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error acknowledging alert"
        )

@app.post("/monitoring/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    monitoring_systems: dict = Depends(get_monitoring_systems)
):
    """Resolve an alert"""
    try:
        if not monitoring_systems['monitoring_enabled']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring systems not available"
            )
        
        success = monitoring_systems['alert_system'].resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        return {"message": f"Alert {alert_id} resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error resolving alert"
        )

@app.post("/monitoring/drift-detection")
async def trigger_drift_detection(
    data: Dict[str, Any],
    monitoring_systems: dict = Depends(get_monitoring_systems)
):
    """Trigger manual drift detection on provided data"""
    try:
        if not monitoring_systems['monitoring_enabled']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring systems not available"
            )
        
        # Convert data to DataFrame
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Perform drift detection
        drift_results = monitoring_systems['drift_detector'].detect_drift(df)
        
        # Process alerts for detected drift
        for result in drift_results:
            if result.drift_detected:
                monitoring_systems['alert_system'].process_data_drift_alert(result.to_dict())
        
        return {
            "drift_results": [result.to_dict() for result in drift_results],
            "summary": monitoring_systems['drift_detector'].get_drift_summary(drift_results)
        }
        
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error performing drift detection"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"Status code: {exc.status_code}",
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "emr_alert_system.api:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=API_CONFIG['debug']
    )
