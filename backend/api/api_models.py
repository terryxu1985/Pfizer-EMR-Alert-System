"""
Pydantic models for the EMR Alert System API
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

class Gender(str, Enum):
    """Patient gender enumeration"""
    M = "M"
    F = "F"

class ExperienceLevel(str, Enum):
    """Physician experience level enumeration"""
    LOW = "Low"
    MID = "Mid"
    HIGH = "High"

class PhysicianType(str, Enum):
    """Physician type enumeration"""
    INTERNAL_MEDICINE = "Internal Medicine"
    EMERGENCY = "Emergency"
    PULMONOLOGY = "Pulmonology"
    FAMILY_MEDICINE = "Family Medicine"
    INFECTIOUS_DISEASE = "Infectious Disease"

class Season(str, Enum):
    """Diagnosis season enumeration"""
    SPRING = "Spring"
    SUMMER = "Summer"
    FALL = "Fall"
    WINTER = "Winter"

class LocationType(str, Enum):
    """Location type enumeration"""
    OFFICE = "OFFICE"
    EMERGENCY_ROOM = "EMERGENCY ROOM - HOSPITAL"
    HOSPITAL_OUTPATIENT = "HOSPITAL OUTPATIENT"
    INDEPENDENT_LABORATORY = "INDEPENDENT LABORATORY"
    CLINIC_FREESTANDING = "CLINIC - FREESTANDING"

class InsuranceType(str, Enum):
    """Insurance type enumeration"""
    COMMERCIAL = "COMMERCIAL"
    MEDICARE = "MEDICARE"
    MEDICAID = "MEDICAID"
    UNSPECIFIED = "UNSPECIFIED"

class TransactionType(str, Enum):
    """Transaction type enumeration"""
    CONDITIONS = "CONDITIONS"
    SYMPTOMS = "SYMPTOMS"
    TREATMENTS = "TREATMENTS"
    CONTRAINDICATIONS = "CONTRAINDICATIONS"

class TransactionRecord(BaseModel):
    """Individual transaction record"""
    txn_dt: datetime = Field(..., description="Transaction date")
    physician_id: Optional[int] = Field(None, description="Physician ID")
    txn_location_type: str = Field(..., description="Transaction location type")
    insurance_type: str = Field(..., description="Insurance type")
    txn_type: TransactionType = Field(..., description="Transaction type")
    txn_desc: str = Field(..., description="Transaction description")

class PhysicianInfo(BaseModel):
    """Physician information"""
    physician_id: Optional[int] = Field(None, description="Physician ID")
    state: str = Field(..., description="Physician state")
    physician_type: str = Field(..., description="Physician specialty type")
    gender: str = Field(..., description="Physician gender")
    birth_year: Optional[int] = Field(None, description="Physician birth year")

class RawEMRRequest(BaseModel):
    """Request model for raw EMR data prediction"""
    
    # Patient basic information
    patient_id: int = Field(..., description="Patient ID")
    birth_year: int = Field(..., ge=1900, le=2024, description="Patient birth year")
    gender: Gender = Field(..., description="Patient gender")
    
    # Diagnosis information
    diagnosis_date: datetime = Field(..., description="Disease X diagnosis date")
    
    # Transaction records
    transactions: List[TransactionRecord] = Field(..., min_items=1, description="List of transaction records")
    
    # Physician information
    physician_info: PhysicianInfo = Field(..., description="Physician information")
    
    @validator('transactions')
    def validate_transactions(cls, v):
        if len(v) == 0:
            raise ValueError('At least one transaction record is required')
        
        # Check for required transaction types
        txn_types = [txn.txn_type for txn in v]
        if TransactionType.CONDITIONS not in txn_types:
            raise ValueError('At least one CONDITIONS transaction is required')
        
        return v
    
    @validator('diagnosis_date')
    def validate_diagnosis_date(cls, v):
        if v > datetime.now():
            raise ValueError('Diagnosis date cannot be in the future')
        return v

class PatientRequest(BaseModel):
    """Request model for patient prediction"""
    
    # Patient demographics
    PATIENT_AGE: int = Field(..., ge=0, le=120, description="Patient age in years", alias="patient_age")
    PATIENT_GENDER: Gender = Field(..., description="Patient gender", alias="patient_gender")
    
    # Risk factors
    RISK_IMMUNO: int = Field(..., ge=0, le=1, description="Immunocompromised condition (0=No, 1=Yes)", alias="risk_immuno")
    RISK_CVD: int = Field(..., ge=0, le=1, description="Cardiovascular disease (0=No, 1=Yes)", alias="risk_cvd")
    RISK_DIABETES: int = Field(..., ge=0, le=1, description="Diabetes mellitus (0=No, 1=Yes)", alias="risk_diabetes")
    RISK_OBESITY: int = Field(..., ge=0, le=1, description="Obesity BMI≥30 (0=No, 1=Yes)", alias="risk_obesity")
    
    # Additional risk features
    RISK_NUM: int = Field(..., ge=0, description="Count of risk factors", alias="risk_num")
    RISK_AGE_FLAG: int = Field(..., ge=0, le=1, description="Age ≥ 65 indicator (0=No, 1=Yes)", alias="risk_age_flag")
    
    # Physician information
    PHYS_EXPERIENCE_LEVEL: ExperienceLevel = Field(..., description="Physician experience level", alias="phys_experience_level")
    PHYSICIAN_STATE: str = Field(..., min_length=2, max_length=2, description="Physician practice state (2-letter code)", alias="physician_state")
    PHYSICIAN_TYPE: PhysicianType = Field(..., description="Physician specialty type", alias="physician_type")
    PHYS_TOTAL_DX: int = Field(..., ge=0, description="Physician total diagnosis count", alias="phys_total_dx")
    
    # Visit information
    SYM_COUNT_5D: int = Field(..., ge=0, description="Symptom count within first 5 days", alias="sym_count_5d")
    DX_SEASON: Season = Field(..., description="Season of diagnosis", alias="dx_season")
    LOCATION_TYPE: LocationType = Field(..., description="Type of practice location", alias="location_type")
    INSURANCE_TYPE_AT_DX: InsuranceType = Field(..., description="Insurance type at diagnosis", alias="insurance_type_at_dx")
    
    # Temporal features
    SYMPTOM_TO_DIAGNOSIS_DAYS: float = Field(..., ge=0, le=30, description="Days between symptom onset and diagnosis", alias="symptom_to_diagnosis_days")
    DIAGNOSIS_WITHIN_5DAYS_FLAG: int = Field(..., ge=0, le=1, description="Diagnosis within 5 days flag (0=No, 1=Yes)", alias="diagnosis_within_5days_flag")
    
    # Symptom features
    SYMPTOM_ACUTE_PHARYNGITIS: int = Field(..., ge=0, le=1, description="Acute pharyngitis symptom (0=No, 1=Yes)", alias="symptom_acute_pharyngitis")
    SYMPTOM_ACUTE_URI: int = Field(..., ge=0, le=1, description="Acute upper respiratory infection symptom (0=No, 1=Yes)", alias="symptom_acute_uri")
    SYMPTOM_CHILLS: int = Field(..., ge=0, le=1, description="Chills symptom (0=No, 1=Yes)", alias="symptom_chills")
    SYMPTOM_CONGESTION: int = Field(..., ge=0, le=1, description="Congestion symptom (0=No, 1=Yes)", alias="symptom_congestion")
    SYMPTOM_COUGH: int = Field(..., ge=0, le=1, description="Cough symptom (0=No, 1=Yes)", alias="symptom_cough")
    SYMPTOM_DIARRHEA: int = Field(..., ge=0, le=1, description="Diarrhea symptom (0=No, 1=Yes)", alias="symptom_diarrhea")
    SYMPTOM_DIFFICULTY_BREATHING: int = Field(..., ge=0, le=1, description="Difficulty breathing symptom (0=No, 1=Yes)", alias="symptom_difficulty_breathing")
    SYMPTOM_FATIGUE: int = Field(..., ge=0, le=1, description="Fatigue symptom (0=No, 1=Yes)", alias="symptom_fatigue")
    SYMPTOM_FEVER: int = Field(..., ge=0, le=1, description="Fever symptom (0=No, 1=Yes)", alias="symptom_fever")
    SYMPTOM_HEADACHE: int = Field(..., ge=0, le=1, description="Headache symptom (0=No, 1=Yes)", alias="symptom_headache")
    SYMPTOM_LOSS_OF_TASTE_OR_SMELL: int = Field(..., ge=0, le=1, description="Loss of taste or smell symptom (0=No, 1=Yes)", alias="symptom_loss_of_taste_or_smell")
    SYMPTOM_MUSCLE_ACHE: int = Field(..., ge=0, le=1, description="Muscle ache symptom (0=No, 1=Yes)", alias="symptom_muscle_ache")
    SYMPTOM_NAUSEA_AND_VOMITING: int = Field(..., ge=0, le=1, description="Nausea and vomiting symptom (0=No, 1=Yes)", alias="symptom_nausea_and_vomiting")
    SYMPTOM_SORE_THROAT: int = Field(..., ge=0, le=1, description="Sore throat symptom (0=No, 1=Yes)", alias="symptom_sore_throat")
    
    # Contraindication
    PRIOR_CONTRA_LVL: int = Field(..., ge=0, le=3, description="Contraindication level (0=None, 1=Mild, 2=Moderate, 3=Severe)", alias="prior_contra_lvl")
    
    @validator('PATIENT_AGE')
    def validate_age(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Patient age must be between 0 and 120')
        return v
    
    @validator('SYMPTOM_TO_DIAGNOSIS_DAYS')
    def validate_symptom_days(cls, v):
        if v < 0 or v > 30:
            raise ValueError('Symptom to diagnosis days must be between 0 and 30')
        return v
    
    @validator('PHYSICIAN_STATE')
    def validate_state_code(cls, v):
        if len(v) != 2 or not v.isalpha():
            raise ValueError('Physician state must be a 2-letter code')
        return v.upper()

class PredictionResponse(BaseModel):
    """Enhanced response model for prediction results with validation info"""
    
    prediction: int = Field(..., description="Prediction (0=Not Treated, 1=Treated)")
    probability: float = Field(..., ge=0, le=1, description="Probability of being treated")
    alert_recommended: bool = Field(..., description="Whether an alert is recommended (True when prediction=1 AND probability>=0.7)")
    feature_validation: Optional[Dict[str, Any]] = Field(None, description="Feature validation results")
    model_version: Optional[str] = Field(None, description="Model version used for prediction")
    model_type: Optional[str] = Field(None, description="Model type used for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.78,
                "alert_recommended": True,
                "feature_validation": {
                    "valid": True,
                    "missing_features": [],
                    "extra_features": []
                },
                "model_version": "1.0.0",
                "model_type": "XGBoost"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    
    patients: List[PatientRequest] = Field(..., min_items=1, max_items=1000, description="List of patient records")
    
    @validator('patients')
    def validate_patients_list(cls, v):
        if len(v) == 0:
            raise ValueError('At least one patient record is required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 patient records allowed per batch')
        return v

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_patients: int = Field(..., description="Total number of patients processed")
    alerts_recommended: int = Field(..., description="Number of patients with alerts recommended")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 1,
                        "probability": 0.78,
                        "alert_recommended": True
                    }
                ],
                "total_patients": 1,
                "alerts_recommended": 1
            }
        }

class RawEMRPredictionResponse(BaseModel):
    """Enhanced response model for raw EMR prediction results"""
    
    prediction: int = Field(..., description="Prediction (0=Not Treated, 1=Treated)")
    probability: float = Field(..., ge=0, le=1, description="Probability of being treated")
    # Preferred business naming (Drug A specific)
    not_prescribed_drug_a: int = Field(..., description="1 if patient was not prescribed Drug A; else 0")
    not_prescribed_drug_a_probability: float = Field(..., ge=0, le=1, description="Probability that patient was not prescribed Drug A")
    alert_recommended: bool = Field(..., description="Whether an alert is recommended (True when prediction=1 AND probability>=0.7)")
    
    # Clinical eligibility assessment (rule-based)
    clinical_eligibility: Optional[Dict[str, Any]] = Field(None, description="Clinical eligibility assessment results including age, time window, risk factors, and contraindications")
    
    # Feature engineering results
    processed_features: Dict[str, Any] = Field(..., description="Processed features used for prediction")
    feature_engineering_info: Dict[str, Any] = Field(..., description="Feature engineering process information")
    
    # Validation and metadata
    feature_validation: Optional[Dict[str, Any]] = Field(None, description="Feature validation results")
    model_version: Optional[str] = Field(None, description="Model version used for prediction")
    model_type: Optional[str] = Field(None, description="Model type used for prediction")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.78,
                "not_prescribed_drug_a": 1,
                "not_prescribed_drug_a_probability": 0.78,
                "alert_recommended": True,
                "processed_features": {
                    "PATIENT_AGE": 65,
                    "RISK_IMMUNO": 0,
                    "RISK_CVD": 1,
                    "SYM_COUNT_5D": 2
                },
                "feature_engineering_info": {
                    "symptom_count": 2,
                    "risk_factors_found": ["CVD"],
                    "time_to_diagnosis_days": 3
                },
                "feature_validation": {
                    "valid": True,
                    "missing_features": [],
                    "extra_features": []
                },
                "model_version": "1.0.0",
                "model_type": "XGBoost",
                "processing_time_ms": 45.2
            }
        }

class BatchRawEMRRequest(BaseModel):
    """Request model for batch raw EMR predictions"""
    
    patients: List[RawEMRRequest] = Field(..., min_items=1, max_items=100, description="List of raw EMR patient records")
    
    @validator('patients')
    def validate_patients_list(cls, v):
        if len(v) == 0:
            raise ValueError('At least one patient record is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 patient records allowed per batch')
        return v

class BatchRawEMRPredictionResponse(BaseModel):
    """Response model for batch raw EMR predictions"""
    
    predictions: List[RawEMRPredictionResponse] = Field(..., description="List of prediction results")
    total_patients: int = Field(..., description="Total number of patients processed")
    alerts_recommended: int = Field(..., description="Number of patients with alerts recommended")
    total_processing_time_ms: Optional[float] = Field(None, description="Total processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 1,
                        "probability": 0.78,
                        "alert_recommended": True,
                        "processed_features": {
                            "PATIENT_AGE": 65,
                            "RISK_IMMUNO": 0,
                            "RISK_CVD": 1,
                            "SYM_COUNT_5D": 2
                        },
                        "feature_engineering_info": {
                            "symptom_count": 2,
                            "risk_factors_found": ["CVD"],
                            "time_to_diagnosis_days": 3
                        },
                        "processing_time_ms": 45.2
                    }
                ],
                "total_patients": 1,
                "alerts_recommended": 1,
                "total_processing_time_ms": 45.2
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")
    timestamp: str = Field(..., description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_info": {
                    "model_type": "XGBClassifier",
                    "feature_count": 15
                },
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Patient age must be between 0 and 120",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
