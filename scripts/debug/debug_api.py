"""
Debug script for the EMR Alert System API
"""
import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.api.model_manager import ModelManager

def test_model_loading():
    """Test model loading"""
    try:
        print("Testing model loading...")
        manager = ModelManager()
        manager.load_model()
        print("Model loaded successfully!")
        
        # Test prediction with correct field names
        test_data = {
            "PATIENT_AGE": 72,
            "PATIENT_GENDER": "M",
            "RISK_IMMUNO": 0,
            "RISK_CVD": 1,
            "RISK_DIABETES": 1,
            "RISK_OBESITY": 0,
            "RISK_NUM": 2,
            "RISK_AGE_FLAG": 1,
            "PHYS_EXPERIENCE_LEVEL": "Senior",
            "PHYSICIAN_STATE": "CA",
            "PHYSICIAN_TYPE": "Internal Medicine",
            "PHYS_TOTAL_DX": 10,
            "SYM_COUNT_5D": 3,
            "DX_SEASON": "Winter",
            "LOCATION_TYPE": "Urban",
            "INSURANCE_TYPE_AT_DX": "Medicare",
            "SYMPTOM_TO_DIAGNOSIS_DAYS": 2.5,
            "DIAGNOSIS_WITHIN_5DAYS_FLAG": 1,
            "PRIOR_CONTRA_LVL": 1,
            # Add all required symptom fields
            "SYMPTOM_ACUTE_PHARYNGITIS": 0,
            "SYMPTOM_ACUTE_URI": 0,
            "SYMPTOM_CHILLS": 1,
            "SYMPTOM_CONGESTION": 1,
            "SYMPTOM_COUGH": 1,
            "SYMPTOM_DIARRHEA": 0,
            "SYMPTOM_DIFFICULTY_BREATHING": 0,
            "SYMPTOM_FATIGUE": 1,
            "SYMPTOM_FEVER": 1,
            "SYMPTOM_HEADACHE": 1,
            "SYMPTOM_LOSS_OF_TASTE_OR_SMELL": 0,
            "SYMPTOM_MUSCLE_ACHE": 1,
            "SYMPTOM_NAUSEA_AND_VOMITING": 0,
            "SYMPTOM_SORE_THROAT": 1
        }
        
        result = manager.predict_single(test_data)
        print(f"Test prediction: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()
