"""
Debug the prediction endpoint
"""
import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.api.model_manager import ModelManager
from backend.api.api_models import PatientRequest

def test_prediction():
    """Test prediction logic"""
    try:
        print("Testing prediction logic...")
        manager = ModelManager()
        manager.load_model()
        
        print("Model loaded successfully")
        
        # Test with Pydantic model using values that exist in training data
        patient_data = {
            "patient_age": 72,
            "patient_gender": "M",
            "risk_immuno": 0,
            "risk_cvd": 1,
            "risk_diabetes": 1,
            "risk_obesity": 0,
            "risk_num": 2,
            "risk_age_flag": 1,
            "phys_experience_level": "Low",  # Use "Low" which exists in training data
            "physician_state": "CA",
            "physician_type": "Internal Medicine",
            "phys_total_dx": 10,
            "sym_count_5d": 3,
            "dx_season": "Spring",  # Use "Spring" which exists in training data
            "location_type": "OFFICE",  # Use "OFFICE" which exists in training data
            "insurance_type_at_dx": "MEDICARE",
            "symptom_to_diagnosis_days": 2.5,
            "diagnosis_within_5days_flag": 1,
            "prior_contra_lvl": 1,
            # Add all required symptom fields
            "symptom_acute_pharyngitis": 0,
            "symptom_acute_uri": 0,
            "symptom_chills": 1,
            "symptom_congestion": 1,
            "symptom_cough": 1,
            "symptom_diarrhea": 0,
            "symptom_difficulty_breathing": 0,
            "symptom_fatigue": 1,
            "symptom_fever": 1,
            "symptom_headache": 1,
            "symptom_loss_of_taste_or_smell": 0,
            "symptom_muscle_ache": 1,
            "symptom_nausea_and_vomiting": 0,
            "symptom_sore_throat": 1
        }
        
        # Create Pydantic model
        patient_request = PatientRequest(**patient_data)
        print(f"Pydantic model created: {patient_request}")
        
        # Convert to dict with by_alias=False
        patient_dict = patient_request.dict(by_alias=False)
        print(f"Converted dict: {patient_dict}")
        
        # Test prediction
        result = manager.predict_single(patient_dict)
        print(f"Prediction result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
