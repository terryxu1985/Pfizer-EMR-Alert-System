"""
EMR data loader for EMR Alert System
Integrates with actual EMR data sources and model training data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Helper function to convert numpy types to Python native types
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

logger = logging.getLogger(__name__)

class EMRDataLoader:
    """Loads patient data from EMR sources and model training data"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.processed_dir = self.data_dir / "processed"
        self.model_ready_dir = self.data_dir / "model_ready"
        
        # Cache for loaded data
        self._patient_data_cache = None
        self._physician_data_cache = None
        self._transaction_data_cache = None
        self._last_cache_time = None
    
    def load_real_patients(self, max_patients: int = 100) -> List[Dict[str, Any]]:
        """
        Load real patient data from EMR sources
        
        Args:
            max_patients: Maximum number of patients to load
            
        Returns:
            List of patient dictionaries with real EMR data
        """
        try:
            # Load data from CSV files
            patients_df = self._load_patient_dataframe()
            physicians_df = self._load_physician_dataframe()
            transactions_df = self._load_transaction_dataframe()
            
            if patients_df is None or patients_df.empty:
                logger.warning("No patient data found in EMR sources")
                return []
            
            # Get recent patients with Disease X indicators
            disease_x_patients = self._filter_disease_x_patients(transactions_df, max_patients)
            
            # Build patient records
            real_patients = []
            for _, patient_row in disease_x_patients.iterrows():
                patient_id = patient_row['PATIENT_ID']
                
                # Get patient demographics
                patient_info = patients_df[patients_df['PATIENT_ID'] == patient_id]
                if patient_info.empty:
                    continue
                
                patient_demographics = patient_info.iloc[0]
                
                # Get physician info
                physician_info = physicians_df[physicians_df['PHYSICIAN_ID'] == patient_row['PHYSICIAN_ID']]
                
                # Get patient's recent transactions
                patient_transactions = transactions_df[transactions_df['PATIENT_ID'] == patient_id]
                
                # Build comprehensive patient record
                patient_record = self._build_patient_record(
                    patient_demographics,
                    physician_info,
                    patient_transactions,
                    patient_row
                )
                
                real_patients.append(patient_record)
            
            logger.info(f"Loaded {len(real_patients)} real patients from EMR data")
            return real_patients
            
        except Exception as e:
            logger.error(f"Error loading real patient data: {str(e)}")
            return []
    
    def _load_patient_dataframe(self) -> Optional[pd.DataFrame]:
        """Load patient demographic data"""
        try:
            patient_file = self.processed_dir / "dim_patient_cleaned.csv"
            if patient_file.exists():
                df = pd.read_csv(patient_file)
                # Calculate age from birth year
                current_year = datetime.now().year
                df['AGE'] = current_year - df['BIRTH_YEAR']
                return df
            return None
        except Exception as e:
            logger.error(f"Error loading patient data: {str(e)}")
            return None
    
    def _load_physician_dataframe(self) -> Optional[pd.DataFrame]:
        """Load physician data"""
        try:
            physician_file = self.processed_dir / "dim_physician_cleaned.csv"
            if physician_file.exists():
                return pd.read_csv(physician_file)
            return None
        except Exception as e:
            logger.error(f"Error loading physician data: {str(e)}")
            return None
    
    def _load_transaction_dataframe(self) -> Optional[pd.DataFrame]:
        """Load transaction/fact data"""
        try:
            transaction_file = self.processed_dir / "fact_txn_cleaned.csv"
            if transaction_file.exists():
                df = pd.read_csv(transaction_file)
                # Convert date column
                df['TXN_DT'] = pd.to_datetime(df['TXN_DT'])
                return df
            return None
        except Exception as e:
            logger.error(f"Error loading transaction data: {str(e)}")
            return None
    
    def _filter_disease_x_patients(self, transactions_df: pd.DataFrame, max_patients: int) -> pd.DataFrame:
        """Filter patients with Disease X indicators"""
        try:
            # Look for Disease X related conditions/symptoms (based on actual data)
            disease_x_keywords = [
                'DISEASE_X', 'IMMUNOCOMPROMISED', 'DIABETES', 'HEART_DISEASE',
                'HYPERTENSION', 'COUGH', 'FEVER', 'SHORTNESS_OF_BREATH',
                'CHEST_PAIN', 'FATIGUE', 'OBESITY'
            ]
            
            # Filter transactions with Disease X indicators
            disease_x_transactions = transactions_df[
                transactions_df['TXN_DESC'].str.contains('|'.join(disease_x_keywords), case=False, na=False)
            ]
            
            print(f"Found {len(disease_x_transactions)} Disease X related transactions")
            
            # Get recent patients (last 365 days instead of 90)
            recent_date = datetime.now() - timedelta(days=365)
            recent_transactions = disease_x_transactions[
                disease_x_transactions['TXN_DT'] >= recent_date
            ]
            
            print(f"Found {len(recent_transactions)} recent Disease X transactions")
            
            if recent_transactions.empty:
                # If no recent data, use all Disease X transactions
                recent_transactions = disease_x_transactions
                print(f"Using all Disease X transactions: {len(recent_transactions)}")
            
            # Get unique patients with most recent activity
            recent_patients = recent_transactions.sort_values('TXN_DT').groupby('PATIENT_ID').last().reset_index()
            
            print(f"Found {len(recent_patients)} unique Disease X patients")
            
            return recent_patients.head(max_patients)
            
        except Exception as e:
            logger.error(f"Error filtering Disease X patients: {str(e)}")
            return pd.DataFrame()
    
    def _build_patient_record(self, patient_demographics: pd.Series, 
                            physician_info: pd.DataFrame, 
                            patient_transactions: pd.DataFrame,
                            latest_transaction: pd.Series) -> Dict[str, Any]:
        """Build comprehensive patient record"""
        
        # Extract comorbidities from transactions
        comorbidities = []
        if not patient_transactions.empty:
            condition_transactions = patient_transactions[
                patient_transactions['TXN_TYPE'] == 'Conditions'
            ]
            comorbidities = condition_transactions['TXN_DESC'].unique().tolist()
        
        # Extract symptoms
        symptoms = []
        if not patient_transactions.empty:
            symptom_transactions = patient_transactions[
                patient_transactions['TXN_TYPE'] == 'Symptoms'
            ]
            symptoms = symptom_transactions['TXN_DESC'].unique().tolist()
        
        # Determine risk level based on comorbidities and age
        age = patient_demographics['AGE']
        risk_level = self._calculate_risk_level(age, len(comorbidities), len(symptoms))
        
        # Determine alert status based on model predictions
        has_alert = self._should_alert(age, comorbidities, symptoms)
        
        # Physician information
        physician_data = {
            "id": latest_transaction['PHYSICIAN_ID'],
            "specialty": "General Medicine",  # Default, could be enhanced
            "experience": "10+ years"  # Default, could be enhanced
        }
        
        if not physician_info.empty:
            physician_data["specialty"] = physician_info.iloc[0].get('SPECIALTY', 'General Medicine')
        
        # Build patient record with proper type conversion
        patient_record = {
            "id": int(patient_demographics['PATIENT_ID']),
            "name": f"Patient {patient_demographics['PATIENT_ID']}",  # Anonymized
            "age": int(age),
            "gender": str(patient_demographics['GENDER']),
            "diagnosisDate": latest_transaction['TXN_DT'].strftime("%Y-%m-%d"),
            "hasDiseaseX": True,
            "riskLevel": risk_level,
            "hasAlert": bool(has_alert),
            "comorbidities": [str(comorbidity) for comorbidity in comorbidities[:5]],  # Convert to strings
            "symptoms": [str(symptom) for symptom in symptoms[:5]],  # Convert to strings
            "physician": physician_data,
            "dataSource": "Real EMR Data",
            "rawPatientId": int(patient_demographics['PATIENT_ID']),
            "lastTransactionDate": latest_transaction['TXN_DT'].strftime("%Y-%m-%d"),
            "transactionCount": int(len(patient_transactions))
        }
        
        # Convert numpy types to Python native types
        return convert_numpy_types(patient_record)
    
    def _calculate_risk_level(self, age: int, comorbidity_count: int, symptom_count: int) -> str:
        """Calculate risk level based on patient characteristics"""
        risk_score = 0
        
        # Age factor
        if age >= 65:
            risk_score += 3
        elif age >= 45:
            risk_score += 2
        else:
            risk_score += 1
        
        # Comorbidity factor
        risk_score += min(comorbidity_count, 3)
        
        # Symptom factor
        risk_score += min(symptom_count, 2)
        
        if risk_score >= 6:
            return "High Risk"
        elif risk_score >= 4:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def _should_alert(self, age: int, comorbidities: List[str], symptoms: List[str]) -> bool:
        """Determine if patient should trigger alert"""
        # High-risk conditions
        high_risk_conditions = ['IMMUNOCOMPROMISED', 'DIABETES', 'HEART_DISEASE']
        has_high_risk = any(condition in comorbidities for condition in high_risk_conditions)
        
        # Age factor
        is_elderly = age >= 65
        
        # Symptom severity
        severe_symptoms = ['SHORTNESS_OF_BREATH', 'CHEST_PAIN', 'FEVER']
        has_severe_symptoms = any(symptom in symptoms for symptom in severe_symptoms)
        
        return is_elderly and (has_high_risk or has_severe_symptoms)
    
    def get_patient_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded patient data"""
        try:
            patients_df = self._load_patient_dataframe()
            if patients_df is None or patients_df.empty:
                return {"error": "No patient data available"}
            
            stats = {
                "total_patients": int(len(patients_df)),
                "age_distribution": {
                    "min": int(patients_df['AGE'].min()),
                    "max": int(patients_df['AGE'].max()),
                    "mean": float(patients_df['AGE'].mean())
                },
                "gender_distribution": patients_df['GENDER'].value_counts().to_dict(),
                "data_last_updated": datetime.now().isoformat()
            }
            
            # Convert numpy types to Python native types
            return convert_numpy_types(stats)
            
        except Exception as e:
            logger.error(f"Error getting patient statistics: {str(e)}")
            return {"error": str(e)}

# Global instance
emr_data_loader = EMRDataLoader()
