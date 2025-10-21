"""
EMR Feature Processor for real-time feature engineering
Converts raw EMR transaction data to model-ready features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

from ..api.api_models import RawEMRRequest, TransactionRecord, PhysicianInfo

logger = logging.getLogger(__name__)

@dataclass
class FeatureEngineeringInfo:
    """Information about feature engineering process"""
    symptom_count: int
    risk_factors_found: List[str]
    time_to_diagnosis_days: Optional[float]
    contraindication_level: int
    physician_experience_level: str
    processing_warnings: List[str]

class EMRFeatureProcessor:
    """
    Real-time feature engineering for EMR data
    Converts raw EMR transactions to model-ready features
    """
    
    def __init__(self):
        """Initialize the feature processor"""
        # Domain constants
        self.DX_CODE = "DISEASE_X"
        self.DRUG_A = "DRUG A"
        self.MIN_AGE_YRS = 12
        self.TREAT_WINDOW_DAYS = 5
        self.PHYS_EXP_THRESHOLDS = {
            "High": 200,
            "Medium": 20
        }
        
        # Risk condition mappings
        self.RISK_CONDITIONS = {
            "RISK_IMMUNO": ["IMMUNOCOMPROMISED"],
            "RISK_CVD": ["HEART_DISEASE", "HYPERTENSION", "STROKE"],
            "RISK_DIABETES": ["DIABETES"],
            "RISK_OBESITY": ["OBESITY"]
        }
        
        # Contraindication level mapping
        self.CONTRAINDICATION_LEVELS = {
            "LOW_CONTRAINDICATION": 1,
            "MEDIUM_CONTRAINDICATION": 2,
            "HIGH_CONTRAINDICATION": 3
        }
        
        # 完整的高风险条件定义（基于Drug A临床标准，不依赖训练特征）
        self.CLINICAL_HIGH_RISK_CONDITIONS = {
            "chronic_lung_disease": [
                "COPD", "CHRONIC_OBSTRUCTIVE_PULMONARY_DISEASE", 
                "ASTHMA", "CHRONIC_BRONCHITIS", "EMPHYSEMA", "PULMONARY"
            ],
            "cardiovascular_disease": [
                "HEART_DISEASE", "HEART_FAILURE", "CORONARY_ARTERY_DISEASE",
                "HYPERTENSION", "STROKE", "CARDIAC", "CHF", "MYOCARDIAL"
            ],
            "cancer": [
                "CANCER", "MALIGNANCY", "CARCINOMA", "LYMPHOMA", 
                "LEUKEMIA", "CHEMOTHERAPY", "ONCOLOGY", "TUMOR"
            ],
            "immunocompromised": [
                "IMMUNOCOMPROMISED", "IMMUNOSUPPRESSED", "IMMUNODEFICIENCY",
                "ORGAN_TRANSPLANT", "TRANSPLANT", "HIV", "AIDS"
            ],
            "obesity": [
                "OBESITY", "OBESE", "BMI_30", "OVERWEIGHT"
            ],
            "diabetes": [
                "DIABETES", "DIABETIC", "TYPE_1_DIABETES", "TYPE_2_DIABETES",
                "HYPERGLYCEMIA"
            ],
            "smoking": [
                "SMOKING", "SMOKER", "TOBACCO_USE", "TOBACCO", "NICOTINE"
            ]
        }
        
        logger.info("EMR Feature Processor initialized")
    
    def process_raw_emr_data(self, raw_data: RawEMRRequest) -> Tuple[Dict[str, Any], FeatureEngineeringInfo]:
        """
        Process raw EMR data and convert to model features
        
        Args:
            raw_data: Raw EMR request data
            
        Returns:
            Tuple of (processed_features_dict, feature_engineering_info)
        """
        logger.info(f"Processing raw EMR data for patient {raw_data.patient_id}")
        
        try:
            # Initialize feature engineering info
            fe_info = FeatureEngineeringInfo(
                symptom_count=0,
                risk_factors_found=[],
                time_to_diagnosis_days=None,
                contraindication_level=0,
                physician_experience_level="Low",
                processing_warnings=[]
            )
            
            # Convert transactions to DataFrame for easier processing
            transactions_df = self._transactions_to_dataframe(raw_data.transactions)
            
            # Process patient features
            patient_features = self._process_patient_features(raw_data, fe_info)
            
            # Process symptom features
            symptom_features = self._process_symptom_features(
                raw_data.patient_id, raw_data.diagnosis_date, transactions_df, fe_info
            )
            
            # Process detailed symptom features (14 individual symptom features)
            detailed_symptom_features = self._process_detailed_symptom_features(
                raw_data.patient_id, raw_data.diagnosis_date, transactions_df, fe_info
            )
            
            # Process risk features
            risk_features = self._process_risk_features(
                raw_data.patient_id, raw_data.diagnosis_date, transactions_df, fe_info
            )
            
            # Process contraindication features
            contra_features = self._process_contraindication_features(
                raw_data.patient_id, raw_data.diagnosis_date, transactions_df, fe_info
            )
            
            # Process physician features
            physician_features = self._process_physician_features(raw_data.physician_info, fe_info)
            
            # Process temporal features
            temporal_features = self._process_temporal_features(
                raw_data.diagnosis_date, symptom_features, fe_info
            )
            
            # Process visit features
            visit_features = self._process_visit_features(raw_data, transactions_df)
            
            # Combine all features
            all_features = {
                **patient_features,
                **symptom_features,
                **detailed_symptom_features,
                **risk_features,
                **contra_features,
                **physician_features,
                **temporal_features,
                **visit_features
            }
            
            logger.info(f"Successfully processed {len(all_features)} features for patient {raw_data.patient_id}")
            
            return all_features, fe_info
            
        except Exception as e:
            logger.error(f"Error processing raw EMR data for patient {raw_data.patient_id}: {str(e)}")
            raise ValueError(f"Feature engineering failed: {str(e)}")
    
    def _transactions_to_dataframe(self, transactions: List[TransactionRecord]) -> pd.DataFrame:
        """Convert transaction records to DataFrame"""
        data = []
        for txn in transactions:
            data.append({
                'TXN_DT': txn.txn_dt,
                'PATIENT_ID': txn.physician_id,  # This will be updated with actual patient_id
                'PHYSICIAN_ID': txn.physician_id,
                'TXN_LOCATION_TYPE': txn.txn_location_type,
                'INSURANCE_TYPE': txn.insurance_type,
                'TXN_TYPE': txn.txn_type.value,
                'TXN_DESC': txn.txn_desc
            })
        
        return pd.DataFrame(data)
    
    def _process_patient_features(self, raw_data: RawEMRRequest, fe_info: FeatureEngineeringInfo) -> Dict[str, Any]:
        """Process patient demographic features"""
        # Calculate age at diagnosis
        patient_age = raw_data.diagnosis_date.year - raw_data.birth_year
        
        # Age validation
        if patient_age < self.MIN_AGE_YRS:
            fe_info.processing_warnings.append(f"Patient age {patient_age} is below minimum {self.MIN_AGE_YRS}")
        
        return {
            'PATIENT_AGE': patient_age,
            'PATIENT_GENDER': raw_data.gender.value,
            'RISK_AGE_FLAG': 1 if patient_age >= 65 else 0
        }
    
    def _process_symptom_features(self, patient_id: int, diagnosis_date: datetime, 
                                 transactions_df: pd.DataFrame, fe_info: FeatureEngineeringInfo) -> Dict[str, Any]:
        """Process symptom-related features"""
        # Filter symptom transactions
        symptom_txns = transactions_df[transactions_df['TXN_TYPE'] == 'SYMPTOMS'].copy()
        
        if symptom_txns.empty:
            fe_info.processing_warnings.append("No symptom transactions found")
            return {
                'SYM_COUNT_5D': 0,
                'SYMPTOM_ONSET_DT': None,
                'SYMPTOM_TO_DIAGNOSIS_DAYS': None,
                'DIAGNOSIS_WITHIN_5DAYS_FLAG': 1  # Default to 1 if no symptoms
            }
        
        # Filter symptoms on or before diagnosis date
        symptom_txns = symptom_txns[symptom_txns['TXN_DT'] <= diagnosis_date]
        
        if symptom_txns.empty:
            fe_info.processing_warnings.append("No symptoms found before diagnosis date")
            return {
                'SYM_COUNT_5D': 0,
                'SYMPTOM_ONSET_DT': None,
                'SYMPTOM_TO_DIAGNOSIS_DAYS': None,
                'DIAGNOSIS_WITHIN_5DAYS_FLAG': 1
            }
        
        # Most recent symptom date
        symptom_onset_dt = symptom_txns['TXN_DT'].max()
        
        # Count symptoms within 5 days of diagnosis
        five_days_before = diagnosis_date - timedelta(days=5)
        recent_symptoms = symptom_txns[
            symptom_txns['TXN_DT'].between(five_days_before, diagnosis_date, inclusive='both')
        ]
        sym_count_5d = len(recent_symptoms)
        
        # Calculate time to diagnosis
        time_to_diagnosis = (diagnosis_date - symptom_onset_dt).days
        diagnosis_within_5days = 1 if time_to_diagnosis <= 5 else 0
        
        # Update feature engineering info
        fe_info.symptom_count = sym_count_5d
        fe_info.time_to_diagnosis_days = time_to_diagnosis
        
        return {
            'SYM_COUNT_5D': sym_count_5d,
            'SYMPTOM_ONSET_DT': symptom_onset_dt,
            'SYMPTOM_TO_DIAGNOSIS_DAYS': time_to_diagnosis,
            'DIAGNOSIS_WITHIN_5DAYS_FLAG': diagnosis_within_5days
        }
    
    def _process_detailed_symptom_features(self, patient_id: int, diagnosis_date: datetime, 
                                          transactions_df: pd.DataFrame, fe_info: FeatureEngineeringInfo) -> Dict[str, Any]:
        """Process detailed individual symptom features (14 features)"""
        # Filter symptom transactions
        symptom_txns = transactions_df[transactions_df['TXN_TYPE'] == 'SYMPTOMS'].copy()
        symptom_txns = symptom_txns[symptom_txns['TXN_DT'] <= diagnosis_date]
        
        # Initialize all symptom features to 0
        detailed_symptom_features = {
            'SYMPTOM_ACUTE_PHARYNGITIS': 0,
            'SYMPTOM_ACUTE_URI': 0,
            'SYMPTOM_CHILLS': 0,
            'SYMPTOM_CONGESTION': 0,
            'SYMPTOM_COUGH': 0,
            'SYMPTOM_DIARRHEA': 0,
            'SYMPTOM_DIFFICULTY_BREATHING': 0,
            'SYMPTOM_FATIGUE': 0,
            'SYMPTOM_FEVER': 0,
            'SYMPTOM_HEADACHE': 0,
            'SYMPTOM_LOSS_OF_TASTE_OR_SMELL': 0,
            'SYMPTOM_MUSCLE_ACHE': 0,
            'SYMPTOM_NAUSEA_AND_VOMITING': 0,
            'SYMPTOM_SORE_THROAT': 0
        }
        
        if symptom_txns.empty:
            fe_info.processing_warnings.append("No symptom transactions found for detailed symptom features")
            return detailed_symptom_features
        
        # Symptom mapping from transaction descriptions to feature names
        symptom_mapping = {
            'ACUTE_PHARYNGITIS': 'SYMPTOM_ACUTE_PHARYNGITIS',
            'ACUTE_URI': 'SYMPTOM_ACUTE_URI',
            'CHILLS': 'SYMPTOM_CHILLS',
            'CONGESTION': 'SYMPTOM_CONGESTION',
            'COUGH': 'SYMPTOM_COUGH',
            'DIARRHEA': 'SYMPTOM_DIARRHEA',
            'DIFFICULTY_BREATHING': 'SYMPTOM_DIFFICULTY_BREATHING',
            'FATIGUE': 'SYMPTOM_FATIGUE',
            'FEVER': 'SYMPTOM_FEVER',
            'HEADACHE': 'SYMPTOM_HEADACHE',
            'LOSS_OF_TASTE_OR_SMELL': 'SYMPTOM_LOSS_OF_TASTE_OR_SMELL',
            'MUSCLE_ACHE': 'SYMPTOM_MUSCLE_ACHE',
            'NAUSEA_AND_VOMITING': 'SYMPTOM_NAUSEA_AND_VOMITING',
            'SORE_THROAT': 'SYMPTOM_SORE_THROAT'
        }
        
        # Process each symptom transaction
        for _, txn in symptom_txns.iterrows():
            symptom_desc = txn['TXN_DESC'].upper().strip()
            
            # Try exact match first
            if symptom_desc in symptom_mapping:
                feature_name = symptom_mapping[symptom_desc]
                detailed_symptom_features[feature_name] = 1
                continue
            
            # Try partial match for compound symptoms
            for symptom_key, feature_name in symptom_mapping.items():
                if symptom_key in symptom_desc:
                    detailed_symptom_features[feature_name] = 1
                    break
        
        # Count how many symptoms were found
        symptom_count = sum(detailed_symptom_features.values())
        if symptom_count == 0:
            fe_info.processing_warnings.append("No specific symptoms identified in detailed symptom analysis")
        else:
            logger.info(f"Found {symptom_count} specific symptoms for patient {patient_id}")
        
        return detailed_symptom_features
    
    def _process_risk_features(self, patient_id: int, diagnosis_date: datetime,
                             transactions_df: pd.DataFrame, fe_info: FeatureEngineeringInfo) -> Dict[str, Any]:
        """Process comorbidity risk features"""
        # Filter condition transactions
        condition_txns = transactions_df[transactions_df['TXN_TYPE'] == 'CONDITIONS'].copy()
        condition_txns = condition_txns[condition_txns['TXN_DT'] <= diagnosis_date]
        
        risk_features = {}
        risk_factors_found = []
        
        # Process each risk category
        for risk_name, condition_list in self.RISK_CONDITIONS.items():
            matching_conditions = condition_txns[condition_txns['TXN_DESC'].isin(condition_list)]
            risk_value = 1 if not matching_conditions.empty else 0
            risk_features[risk_name] = risk_value
            
            if risk_value == 1:
                risk_factors_found.append(risk_name.replace('RISK_', ''))
        
        # Calculate total risk count
        risk_features['RISK_NUM'] = sum(risk_features.values())
        
        # Update feature engineering info
        fe_info.risk_factors_found = risk_factors_found
        
        return risk_features
    
    def _process_contraindication_features(self, patient_id: int, diagnosis_date: datetime,
                                          transactions_df: pd.DataFrame, fe_info: FeatureEngineeringInfo) -> Dict[str, Any]:
        """Process contraindication features"""
        # Filter contraindication transactions
        contra_txns = transactions_df[transactions_df['TXN_TYPE'] == 'CONTRAINDICATIONS'].copy()
        contra_txns = contra_txns[contra_txns['TXN_DT'] <= diagnosis_date]
        
        if contra_txns.empty:
            fe_info.contraindication_level = 0
            return {'PRIOR_CONTRA_LVL': 0}
        
        # Map contraindication descriptions to levels
        contra_txns['LVL'] = contra_txns['TXN_DESC'].map(self.CONTRAINDICATION_LEVELS).fillna(0)
        
        # Get highest contraindication level
        max_contra_level = int(contra_txns['LVL'].max())
        fe_info.contraindication_level = max_contra_level
        
        return {'PRIOR_CONTRA_LVL': max_contra_level}
    
    def _process_physician_features(self, physician_info: PhysicianInfo, fe_info: FeatureEngineeringInfo) -> Dict[str, Any]:
        """Process physician-related features"""
        # For real-time processing, we use simplified physician features
        # In a full implementation, these would be calculated from historical data
        
        physician_features = {
            'PHYSICIAN_STATE': physician_info.state,
            'PHYSICIAN_TYPE': physician_info.physician_type,
            'PHYS_TOTAL_DX': 50,  # Default value - would be calculated from historical data
        }
        
        # Determine experience level based on total diagnoses
        total_dx = physician_features['PHYS_TOTAL_DX']
        if total_dx >= self.PHYS_EXP_THRESHOLDS["High"]:
            experience_level = "High"
        elif total_dx >= self.PHYS_EXP_THRESHOLDS["Medium"]:
            experience_level = "Medium"
        else:
            experience_level = "Low"
        
        physician_features['PHYS_EXPERIENCE_LEVEL'] = experience_level
        fe_info.physician_experience_level = experience_level
        
        return physician_features
    
    def _process_temporal_features(self, diagnosis_date: datetime, symptom_features: Dict[str, Any],
                                 fe_info: FeatureEngineeringInfo) -> Dict[str, Any]:
        """Process temporal features including DX_SEASON derived from diagnosis month"""
        month = diagnosis_date.month
        if month in (12, 1, 2):
            season = "WINTER"
        elif month in (3, 4, 5):
            season = "SPRING"
        elif month in (6, 7, 8):
            season = "SUMMER"
        else:
            season = "FALL"
        return {
            'DX_SEASON': season
        }
    
    def _process_visit_features(self, raw_data: RawEMRRequest, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Process visit-related features"""
        # Get diagnosis transaction for visit features
        diagnosis_txn = transactions_df[transactions_df['TXN_DESC'] == self.DX_CODE]
        
        if diagnosis_txn.empty:
            # Use first transaction as fallback
            first_txn = transactions_df.iloc[0]
            location_type = first_txn['TXN_LOCATION_TYPE']
            insurance_type = first_txn['INSURANCE_TYPE']
        else:
            location_type = diagnosis_txn.iloc[0]['TXN_LOCATION_TYPE']
            insurance_type = diagnosis_txn.iloc[0]['INSURANCE_TYPE']
        
        return {
            'LOCATION_TYPE': location_type,
            'INSURANCE_TYPE_AT_DX': insurance_type
        }
    
    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate processed features
        
        Args:
            features: Dictionary of processed features
            
        Returns:
            Validation results dictionary
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'feature_count': len(features),
            'missing_features': [],
            'invalid_values': []
        }
        
        # Check for required features (33 features total - matching the trained model)
        required_features = [
            # Patient features (3)
            'PATIENT_AGE', 'PATIENT_GENDER', 'RISK_AGE_FLAG',
            # Risk features (5)
            'RISK_IMMUNO', 'RISK_CVD', 'RISK_DIABETES', 'RISK_OBESITY', 'RISK_NUM',
            # Physician features (4)
            'PHYS_EXPERIENCE_LEVEL', 'PHYSICIAN_STATE', 'PHYSICIAN_TYPE', 'PHYS_TOTAL_DX',
            # Symptom and temporal features (4)
            'SYM_COUNT_5D', 'SYMPTOM_TO_DIAGNOSIS_DAYS', 'DIAGNOSIS_WITHIN_5DAYS_FLAG', 'DX_SEASON',
            # Visit features (2)
            'LOCATION_TYPE', 'INSURANCE_TYPE_AT_DX',
            # Detailed symptom features (14)
            'SYMPTOM_ACUTE_PHARYNGITIS', 'SYMPTOM_ACUTE_URI', 'SYMPTOM_CHILLS',
            'SYMPTOM_CONGESTION', 'SYMPTOM_COUGH', 'SYMPTOM_DIARRHEA',
            'SYMPTOM_DIFFICULTY_BREATHING', 'SYMPTOM_FATIGUE', 'SYMPTOM_FEVER',
            'SYMPTOM_HEADACHE', 'SYMPTOM_LOSS_OF_TASTE_OR_SMELL', 'SYMPTOM_MUSCLE_ACHE',
            'SYMPTOM_NAUSEA_AND_VOMITING', 'SYMPTOM_SORE_THROAT',
            # Other features (1)
            'PRIOR_CONTRA_LVL'
        ]
        
        for feature in required_features:
            if feature not in features:
                validation_result['missing_features'].append(feature)
                validation_result['valid'] = False
        
        # Check for invalid values
        if 'PATIENT_AGE' in features:
            age = features['PATIENT_AGE']
            if age < 0 or age > 120:
                validation_result['invalid_values'].append(f"PATIENT_AGE: {age}")
                validation_result['warnings'].append(f"Patient age {age} is outside normal range")
        
        if 'SYMPTOM_TO_DIAGNOSIS_DAYS' in features and features['SYMPTOM_TO_DIAGNOSIS_DAYS'] is not None:
            days = features['SYMPTOM_TO_DIAGNOSIS_DAYS']
            if days < 0 or days > 365:
                validation_result['invalid_values'].append(f"SYMPTOM_TO_DIAGNOSIS_DAYS: {days}")
                validation_result['warnings'].append(f"Symptom to diagnosis days {days} is unusual")
        
        return validation_result
    
    def check_clinical_eligibility(self, raw_data: 'RawEMRRequest', 
                                   processed_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        完全基于临床规则检查患者是否符合Drug A处方资格
        不依赖模型训练特征，直接从原始交易数据判断
        
        Drug A 临床标准：
        1. 患者确诊疾病X ✅ (已确认，否则不会进入预测流程)
        2. 年龄 ≥ 12岁
        3. 距症状出现时间 ≤ 5天
        4. 高风险患者：年龄 ≥ 65岁 或 有以下任何基础疾病：
           - 慢性肺部疾病（COPD、哮喘）
           - 心血管疾病
           - 癌症
           - 免疫系统受损
           - 肥胖 (BMI ≥ 30)
           - 糖尿病
           - 吸烟
        5. 没有严重禁忌症
        
        Args:
            raw_data: Raw EMR request data
            processed_features: Processed features dict
            
        Returns:
            Dictionary with clinical eligibility assessment
        """
        # 从原始数据获取患者年龄
        birth_year = raw_data.birth_year
        diagnosis_year = raw_data.diagnosis_date.year
        patient_age = diagnosis_year - birth_year
        
        # 转换交易数据为DataFrame
        transactions_df = self._transactions_to_dataframe(raw_data.transactions)
        
        # ========== 规则1: 年龄 ≥ 12岁 ==========
        age_eligible = patient_age >= 12
        
        # ========== 规则2: 距症状出现时间 ≤ 5天 ==========
        # 从交易数据中找症状记录
        symptom_txns = transactions_df[transactions_df['TXN_TYPE'] == 'SYMPTOMS'].copy()
        symptom_txns = symptom_txns[symptom_txns['TXN_DT'] <= raw_data.diagnosis_date]
        
        if symptom_txns.empty:
            # 没有症状记录，默认为符合（保守处理）
            within_5day_window = True
            symptom_to_diagnosis_days = None
        else:
            # 最早症状日期
            earliest_symptom_dt = symptom_txns['TXN_DT'].min()
            symptom_to_diagnosis_days = (raw_data.diagnosis_date - earliest_symptom_dt).days
            within_5day_window = symptom_to_diagnosis_days <= 5
        
        # ========== 规则3: 高风险患者 ==========
        # 3a. 年龄 ≥ 65岁
        is_high_risk_by_age = patient_age >= 65
        
        # 3b. 检查基础疾病（从原始交易数据）
        condition_txns = transactions_df[transactions_df['TXN_TYPE'] == 'CONDITIONS'].copy()
        condition_txns = condition_txns[condition_txns['TXN_DT'] <= raw_data.diagnosis_date]
        
        risk_factors_found = []
        risk_conditions_details = {}
        
        if is_high_risk_by_age:
            risk_factors_found.append("age_65_or_older")
        
        # 检查每个高风险条件类别
        if not condition_txns.empty:
            for category, condition_keywords in self.CLINICAL_HIGH_RISK_CONDITIONS.items():
                found_in_category = False
                matched_conditions = []
                
                for keyword in condition_keywords:
                    matching = condition_txns[
                        condition_txns['TXN_DESC'].str.contains(keyword, case=False, na=False)
                    ]
                    if not matching.empty:
                        found_in_category = True
                        matched_conditions.extend(matching['TXN_DESC'].unique().tolist())
                
                if found_in_category:
                    risk_factors_found.append(category)
                    risk_conditions_details[category] = list(set(matched_conditions))
        
        # 高风险判断：年龄≥65 或 有任何基础疾病
        is_high_risk = len(risk_factors_found) > 0
        
        # ========== 规则4: 没有严重禁忌症 ==========
        # 从交易数据中检查禁忌症
        contra_txns = transactions_df[transactions_df['TXN_TYPE'] == 'CONTRAINDICATIONS'].copy()
        contra_txns = contra_txns[contra_txns['TXN_DT'] <= raw_data.diagnosis_date]
        
        if contra_txns.empty:
            contraindication_level = 0
            no_severe_contraindication = True
            contraindication_details = []
        else:
            # 映射禁忌症等级
            contra_txns['LVL'] = contra_txns['TXN_DESC'].map(self.CONTRAINDICATION_LEVELS).fillna(0)
            contraindication_level = int(contra_txns['LVL'].max())
            no_severe_contraindication = (contraindication_level < 3)
            contraindication_details = contra_txns['TXN_DESC'].unique().tolist()
        
        # ========== 综合判断 ==========
        meets_clinical_criteria = (
            age_eligible and
            within_5day_window and
            is_high_risk and
            no_severe_contraindication
        )
        
        return {
            'meets_criteria': meets_clinical_criteria,
            'age_eligible': age_eligible,
            'patient_age': patient_age,
            'within_5day_window': within_5day_window,
            'symptom_to_diagnosis_days': symptom_to_diagnosis_days,
            'is_high_risk': is_high_risk,
            'is_high_risk_by_age': is_high_risk_by_age,
            'risk_factors_found': risk_factors_found,
            'risk_conditions_details': risk_conditions_details,
            'no_severe_contraindication': no_severe_contraindication,
            'contraindication_level': contraindication_level,
            'contraindication_details': contraindication_details
        }
