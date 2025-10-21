"""
Patient repository for the EMR Alert System
This module provides patient data access from various sources including:
- Sample data for demonstration
- Real EMR data (when available)
- Generated patient data based on model training data
"""
import json
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path
from .emr_data_loader import emr_data_loader

class PatientRepository:
    """Repository for patient data from multiple sources"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent
        self.patient_cache = []
        self.last_updated = None
        self.persistent_file = self.data_dir.parent.parent / "data" / "storage" / "patients" / "doctor_input_patients.json"
        self.doctor_input_patients = []
        
        # Load persistent data on initialization
        self._load_persistent_data()
    
    def _load_persistent_data(self):
        """Load persistent patient data from JSON file"""
        try:
            if self.persistent_file.exists():
                with open(self.persistent_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle both old and new data formats
                    if 'patients' in data:
                        # New format with metadata
                        self.doctor_input_patients = data.get('patients', [])
                        metadata = data.get('metadata', {})
                        print(f"✅ Loaded {len(self.doctor_input_patients)} persistent patients (v{metadata.get('version', 'unknown')})")
                    else:
                        # Old format (backward compatibility)
                        self.doctor_input_patients = data.get('doctor_input_patients', [])
                        print(f"✅ Loaded {len(self.doctor_input_patients)} persistent patients (legacy format)")
            else:
                print("📝 No persistent patient data found, starting fresh")
                self.doctor_input_patients = []
        except Exception as e:
            print(f"❌ Error loading persistent data: {str(e)}")
            self.doctor_input_patients = []
    
    def _save_persistent_data(self):
        """Save patient data to JSON file"""
        try:
            # Ensure directory exists
            self.persistent_file.parent.mkdir(parents=True, exist_ok=True)
            
            # New format with metadata
            data = {
                'metadata': {
                    'version': '1.0',
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'total_patients': len(self.doctor_input_patients),
                    'data_source': 'Doctor Input UI',
                    'file_type': 'patient_data'
                },
                'patients': self.doctor_input_patients
            }
            
            with open(self.persistent_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {len(self.doctor_input_patients)} patients to persistent storage")
        except Exception as e:
            print(f"❌ Error saving persistent data: {str(e)}")
    
    def get_patients(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get patient data from available sources
        
        Priority order:
        1. Doctor input patients (from UI) - ONLY THESE ARE SHOWN
        2. Sample demonstration data (if no doctor input patients exist)
        """
        if use_cache and self.patient_cache and self.last_updated:
            # Return cached data if it's recent (within 5 minutes)
            if (datetime.now() - self.last_updated).seconds < 300:
                return self.patient_cache
        
        # ONLY return doctor input patients (from UI)
        patients = []
        if hasattr(self, 'doctor_input_patients') and self.doctor_input_patients:
            patients = self.doctor_input_patients
            print(f"✅ Returning {len(patients)} doctor input patients")
        else:
            # If no doctor input patients, show a small sample for demonstration
            patients = self._get_sample_patients()
            print(f"ℹ️ No doctor input patients, showing {len(patients)} sample patients")
        
        # Cache the results
        self.patient_cache = patients
        self.last_updated = datetime.now()
        
        return patients
    
    def _load_real_patient_data(self) -> List[Dict[str, Any]]:
        """Load patient data from real EMR sources"""
        try:
            # Load real EMR data with proper error handling
            real_patients = emr_data_loader.load_real_patients(max_patients=500)  # Load 500 real patients
            
            if real_patients:
                print(f"✅ Loaded {len(real_patients)} real patients from EMR data")
                return real_patients
            else:
                print("⚠️ No real EMR data available, falling back to generated data")
                return []
                
        except Exception as e:
            print(f"❌ Error loading real EMR data: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _generate_patients_from_model_patterns(self) -> List[Dict[str, Any]]:
        """Generate patient data based on model training patterns"""
        patients = []
        
        # Generate patients with different risk profiles based on model training data
        risk_profiles = [
            {
                "risk_level": "High Risk",
                "age_range": (65, 85),
                "comorbidity_probability": 0.8,
                "symptom_count_range": (3, 5),
                "alert_probability": 0.7
            },
            {
                "risk_level": "Medium Risk", 
                "age_range": (45, 64),
                "comorbidity_probability": 0.5,
                "symptom_count_range": (2, 4),
                "alert_probability": 0.3
            },
            {
                "risk_level": "Low Risk",
                "age_range": (25, 44),
                "comorbidity_probability": 0.2,
                "symptom_count_range": (1, 3),
                "alert_probability": 0.1
            }
        ]
        
        names = [
            ("John", "Smith"), ("Sarah", "Johnson"), ("Michael", "Brown"),
            ("Emily", "Davis"), ("David", "Wilson"), ("Lisa", "Anderson"),
            ("Robert", "Taylor"), ("Jennifer", "Thomas"), ("William", "Jackson"),
            ("Mary", "White")
        ]
        
        comorbidities_pool = [
            "Heart Disease", "Diabetes", "Hypertension", "Obesity",
            "Chronic Kidney Disease", "COPD", "Asthma", "Liver Disease"
        ]
        
        symptoms_pool = [
            "Cough", "Fever", "Shortness of Breath", "Fatigue",
            "Chest Pain", "Headache", "Nausea", "Body Aches"
        ]
        
        specialties = [
            "Emergency Medicine", "Internal Medicine", "Pulmonology",
            "Family Medicine", "Infectious Disease"
        ]
        
        for i, (first_name, last_name) in enumerate(names[:5]):  # Generate 5 patients
            profile = random.choice(risk_profiles)
            age = random.randint(*profile["age_range"])
            
            # Generate comorbidities
            comorbidities = []
            if random.random() < profile["comorbidity_probability"]:
                num_comorbidities = random.randint(1, 3)
                comorbidities = random.sample(comorbidities_pool, 
                                            min(num_comorbidities, len(comorbidities_pool)))
            
            # Generate symptoms
            num_symptoms = random.randint(*profile["symptom_count_range"])
            symptoms = random.sample(symptoms_pool, min(num_symptoms, len(symptoms_pool)))
            
            # Determine alert status
            has_alert = random.random() < profile["alert_probability"]
            
            # Generate diagnosis date (within last 30 days)
            diagnosis_date = datetime.now() - timedelta(days=random.randint(1, 30))
            
            patient = {
                "id": 2000 + i,  # Start from 2000 to distinguish from sample data
                "name": f"{first_name} {last_name}",
                "age": age,
                "gender": random.choice(["Male", "Female"]),
                "diagnosisDate": diagnosis_date.strftime("%Y-%m-%d"),
                "hasDiseaseX": True,
                "riskLevel": profile["risk_level"],
                "hasAlert": has_alert,
                "comorbidities": comorbidities,
                "symptoms": symptoms,
                "physician": {
                    "id": 20000 + i,
                    "specialty": random.choice(specialties),
                    "experience": f"{random.randint(5, 25)} years"
                },
                "dataSource": "Generated from Model Patterns"
            }
            
            patients.append(patient)
        
        return patients
    
    def _get_sample_patients(self) -> List[Dict[str, Any]]:
        """Get sample patient data for demonstration"""
        return [
            {
                "id": 1001,
                "name": "John Smith",
                "age": 59,
                "gender": "Male",
                "diagnosisDate": "2024-01-15",
                "hasDiseaseX": True,
                "riskLevel": "High Risk",
                "hasAlert": True,
                "comorbidities": ["Heart Disease", "Diabetes"],
                "symptoms": ["Cough", "Fever", "Shortness of Breath"],
                "physician": {
                    "id": 12345,
                    "specialty": "Emergency Medicine",
                    "experience": "15 years"
                },
                "dataSource": "Sample Data"
            },
            {
                "id": 1002,
                "name": "Sarah Johnson",
                "age": 45,
                "gender": "Female",
                "diagnosisDate": "2024-01-20",
                "hasDiseaseX": True,
                "riskLevel": "Medium Risk",
                "hasAlert": False,
                "comorbidities": ["Hypertension"],
                "symptoms": ["Cough", "Fatigue"],
                "physician": {
                    "id": 12346,
                    "specialty": "Internal Medicine",
                    "experience": "12 years"
                },
                "dataSource": "Sample Data"
            },
            {
                "id": 1003,
                "name": "Michael Brown",
                "age": 72,
                "gender": "Male",
                "diagnosisDate": "2024-01-18",
                "hasDiseaseX": True,
                "riskLevel": "High Risk",
                "hasAlert": True,
                "comorbidities": ["Heart Disease", "Diabetes", "Hypertension"],
                "symptoms": ["Cough", "Fever", "Chest Pain", "Shortness of Breath"],
                "physician": {
                    "id": 12347,
                    "specialty": "Pulmonology",
                    "experience": "20 years"
                },
                "dataSource": "Sample Data"
            }
        ]
    
    def get_patient_by_id(self, patient_id: int) -> Dict[str, Any]:
        """Get a specific patient by ID"""
        patients = self.get_patients()
        for patient in patients:
            if patient["id"] == patient_id:
                return patient
        return None

    def create_patient_from_input(self, patient_input) -> Dict[str, Any]:
        """Create a new patient record from doctor input"""
        import random
        from datetime import datetime
        
        # Generate unique patient ID
        new_patient_id = max([p.get("id", 0) for p in self.get_patients()] + [0]) + 1
        
        # Calculate risk level based on comorbidities and symptoms
        risk_score = 0
        risk_score += len(patient_input.comorbidities) * 2
        risk_score += len(patient_input.symptoms)
        
        # Age-based risk adjustment
        if patient_input.patient_age >= 65:
            risk_score += 3
        elif patient_input.patient_age >= 50:
            risk_score += 1
            
        # Contraindication level adjustment
        contraindication_multiplier = {
            "Low": 1.0,
            "Medium": 1.5,
            "High": 2.0
        }
        risk_score *= contraindication_multiplier.get(patient_input.contraindication_level, 1.0)
        
        if risk_score >= 8:
            risk_level = "High Risk"
        elif risk_score >= 5:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
        
        # Determine if patient should have alert based on risk factors
        has_alert = risk_score >= 6 or (
            patient_input.patient_age >= 65 and len(patient_input.comorbidities) >= 2
        )
        
        # Map gender values to display format
        gender_display = {
            "M": "Male",
            "F": "Female"
        }.get(patient_input.patient_gender, patient_input.patient_gender)
        
        # Create patient record with enhanced data
        patient_record = {
            "id": new_patient_id,
            "name": patient_input.patient_name,
            "age": patient_input.patient_age,
            "gender": gender_display,
            "diagnosisDate": patient_input.diagnosis_date,
            "hasDiseaseX": True,  # All patients have Disease X by default
            "riskLevel": risk_level,
            "hasAlert": has_alert,
            "comorbidities": patient_input.comorbidities,
            "symptoms": patient_input.symptoms,
            "physician": {
                "id": patient_input.physician_id,
                "specialty": random.choice(["Internal Medicine", "Emergency Medicine", "Family Medicine", "Pulmonology"]),
                "experience": f"{random.randint(5, 25)} years"
            },
            "dataSource": "Doctor Input",
            "rawPatientId": new_patient_id,
            "lastTransactionDate": patient_input.diagnosis_date,
            "transactionCount": 1,
            "locationType": patient_input.location_type,
            "insuranceType": patient_input.insurance_type,
            "contraindicationLevel": patient_input.contraindication_level,
            "symptomOnsetDate": getattr(patient_input, 'symptom_onset_date', None),
            "additionalNotes": getattr(patient_input, 'additional_notes', ''),
            "riskScore": risk_score,
            "createdAt": datetime.now().isoformat(),
            "lastUpdated": datetime.now().isoformat()
        }
        
        # Add to persistent storage
        self.doctor_input_patients.append(patient_record)
        
        # Save to persistent storage
        self._save_persistent_data()
        
        # Clear cache to force reload
        self.patient_cache = []
        self.last_updated = None
        
        return patient_record
    
    def add_patient(self, patient_data: Dict[str, Any]) -> bool:
        """Add a new patient (for future EMR integration)"""
        # This would integrate with real EMR systems
        # For now, just add to cache
        patients = self.get_patients(use_cache=False)  # Refresh cache
        patients.append(patient_data)
        self.patient_cache = patients
        self.last_updated = datetime.now()
        return True
    
    def update_patient(self, patient_id: int, updates: Dict[str, Any]) -> bool:
        """Update patient data (for future EMR integration)"""
        patients = self.get_patients(use_cache=False)
        for i, patient in enumerate(patients):
            if patient["id"] == patient_id:
                patients[i].update(updates)
                self.patient_cache = patients
                self.last_updated = datetime.now()
                return True
        return False

# Global instance
patient_repository = PatientRepository()
