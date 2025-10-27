# 📊 Pfizer EMR Alert System - Data Directory

This directory contains all data files for the Pfizer EMR Alert System, organized in a structured pipeline from raw data to model-ready datasets.

## 📁 Directory Structure

```
data/
├── raw/                    # Original raw data files (Excel format)
├── processed/              # Cleaned and processed data files (CSV format)
├── model_ready/           # Final dataset ready for machine learning
├── storage/               # Temporary storage for system operations
└── DATA_GUIDE.md         # This documentation file
```

## 📋 Data Pipeline Overview

### 1. Raw Data (`raw/`)
Contains the original source data files in Excel format, directly imported from the EMR system:

| File | Size | Description |
|------|------|-------------|
| `dim_patient.xlsx` | 89K | Patient dimension table with demographics |
| `dim_physician.xlsx` | 723K | Physician dimension table with provider information |
| `fact_txn.xlsx` | 4.6M | Transaction fact table with medical records |
| `model_table.xlsx` | 17K | Model-specific table with target variables |
| `Data Dictionary.xlsx` | 19K | Comprehensive data dictionary |

### 2. Processed Data (`processed/`)
Contains cleaned and standardized data files in CSV format, ready for analysis and feature engineering:

| File | Rows | Description |
|------|------|-------------|
| `dim_patient_cleaned.csv` | 4,021 | Cleaned patient demographics and risk factors |
| `dim_physician_cleaned.csv` | 25,344 | Cleaned physician information and statistics |
| `fact_txn_cleaned.csv` | 115,042 | Cleaned medical transactions and diagnoses |

### 3. Model Ready Data (`model_ready/`)
Contains the final dataset optimized for machine learning with engineered features and proper formatting:

| File | Rows | Description |
|------|------|-------------|
| `model_ready_dataset.csv` | 3,613 | Final dataset with engineered features |
| `model_feature_dictionary.md` | - | Comprehensive feature documentation |
| `model_feature_dictionary.xlsx` | 12K | Feature dictionary in Excel format |

## 🎯 Target Variable

### Primary Target: Drug A Treatment Prediction

**TARGET**: Binary classification variable (0/1) indicating whether a patient received Drug A treatment within the critical 5-day window following symptom onset.

#### Target Variable Definition
- **`1` (Positive Class)**: Patient was **NOT** prescribed Drug A treatment (alert candidates requiring intervention)
- **`0` (Negative Class)**: Patient **WAS** prescribed Drug A treatment (no alert needed)

#### Clinical Context
Drug A is a time-sensitive therapeutic intervention that must be administered within 5 days of symptom onset to achieve optimal clinical outcomes. The target variable identifies patients who **should have received** Drug A but were **not prescribed** it, based on:

- **Patient Risk Factors**: Age, comorbidities, contraindications
- **Clinical Presentation**: Symptom patterns, severity indicators
- **Temporal Factors**: Time from symptom onset to diagnosis
- **Physician Characteristics**: Experience level, specialty, treatment patterns

#### Target Distribution
- **Class Balance**: Approximately balanced dataset for robust model training
- **Temporal Sensitivity**: Critical 5-day treatment window for Drug A efficacy
- **Clinical Relevance**: Directly impacts patient outcomes and treatment protocols

#### Model Objective
The machine learning model aims to identify patients who **missed** Drug A treatment to:
1. **Generate Clinical Alerts**: Flag patients who should have received Drug A but were not prescribed it
2. **Prevent Treatment Gaps**: Identify missed opportunities for timely intervention
3. **Improve Patient Outcomes**: Reduce under-treatment and improve clinical decision-making
4. **Support Quality Assurance**: Help physicians identify cases requiring immediate attention

## 🏗️ Data Schema

### Patient Features (`dim_patient_cleaned.csv`)
- **PATIENT_ID**: Unique patient identifier (anonymized)
- **BIRTH_YEAR**: Patient's birth year (used to calculate age at diagnosis)
- **GENDER**: Patient gender (M/F/U for Male/Female/Unknown)

### Physician Features (`dim_physician_cleaned.csv`)
- **PHYSICIAN_ID**: Unique physician identifier (anonymized)
- **PHYS_TREAT_RATE_ALL**: Physician's historical Drug A treatment rate (0-1)
- **PHYS_TOTAL_DX**: Total diagnosis count by physician (experience indicator)
- **PHYS_EXPERIENCE_LEVEL**: Categorized experience level (Low/Mid/Senior)
- **PHYSICIAN_STATE**: Practice state/geographic region
- **PHYSICIAN_TYPE**: Clinical specialty classification

### Transaction Features (`fact_txn_cleaned.csv`)
- **TXN_DT**: Transaction date (clinical encounter date)
- **PATIENT_ID**: Patient identifier (foreign key reference)
- **PHYSICIAN_ID**: Physician identifier (foreign key reference)
- **TXN_LOCATION_TYPE**: Healthcare facility type (Hospital/Clinic/Urgent Care)
- **INSURANCE_TYPE**: Patient's insurance coverage classification
- **TXN_TYPE**: Transaction category (Conditions/Symptoms/Contraindications)
- **TXN_DESC**: Detailed clinical transaction description

### Model Features (`model_ready_dataset.csv`)
The final dataset includes comprehensive engineered features optimized for machine learning:

#### Patient Risk Factors
- **PATIENT_AGE**: Patient age at diagnosis (continuous variable)
- **PATIENT_GENDER**: Patient gender (categorical: M/F/U)
- **RISK_IMMUNO**: Immunocompromised condition indicator (binary: 0/1)
- **RISK_CVD**: Cardiovascular disease risk flag (binary: 0/1)
- **RISK_DIABETES**: Diabetes mellitus risk flag (binary: 0/1)
- **RISK_OBESITY**: Obesity indicator (BMI ≥ 30) (binary: 0/1)
- **RISK_NUM**: Total count of risk factors (continuous: 0-4)
- **RISK_AGE_FLAG**: Advanced age indicator (≥65 years) (binary: 0/1)

#### Physician Characteristics
- **PHYS_TREAT_RATE_ALL**: Historical Drug A treatment rate (continuous: 0-1)
- **PHYS_TOTAL_DX**: Total diagnosis count (experience proxy)
- **PHYS_EXPERIENCE_LEVEL**: Categorized experience level (Low/Mid/Senior)
- **PHYSICIAN_STATE**: Practice state/geographic region
- **PHYSICIAN_TYPE**: Clinical specialty classification

#### Temporal & Visit Features
- **SYM_COUNT_5D**: Symptom count within first 5 days of presentation
- **DX_SEASON**: Season of diagnosis (Spring/Summer/Fall/Winter)
- **LOCATION_TYPE**: Healthcare facility type at diagnosis
- **INSURANCE_TYPE_AT_DX**: Insurance coverage type at time of diagnosis
- **SYMPTOM_TO_DIAGNOSIS_DAYS**: Days elapsed between symptom onset and diagnosis
- **DIAGNOSIS_WITHIN_5DAYS_FLAG**: Critical timing indicator (diagnosis ≤5 days) (binary: 0/1)

#### Contraindication Features
- **PRIOR_CONTRA_LVL**: Prior contraindication severity level (categorical: None/Low/Medium/High)

## 🔄 Data Processing Workflow

The data pipeline follows a systematic approach to transform raw EMR data into machine learning-ready features:

1. **Raw Data Ingestion**: 
   - Import Excel files from `raw/` directory
   - Validate data integrity and completeness
   - Establish data lineage tracking

2. **Data Cleaning & Standardization**: 
   - Handle missing values using appropriate imputation strategies
   - Standardize data formats and encoding schemes
   - Remove duplicate records and validate referential integrity
   - Perform data type validation and conversion

3. **Feature Engineering**:
   - Calculate derived clinical features (age, risk scores)
   - Create temporal features (seasonality, time windows)
   - Generate risk indicators and clinical flags
   - Implement domain-specific transformations

4. **Model Preparation**:
   - Join tables using appropriate keys
   - Filter records based on clinical criteria
   - Create balanced datasets for training
   - Apply feature scaling and encoding

## 🚀 Usage Examples

### Loading Processed Data
```python
import pandas as pd

# Load processed data
patients = pd.read_csv('data/processed/dim_patient_cleaned.csv')
physicians = pd.read_csv('data/processed/dim_physician_cleaned.csv')
transactions = pd.read_csv('data/processed/fact_txn_cleaned.csv')

# Load model-ready dataset
model_data = pd.read_csv('data/model_ready/model_ready_dataset.csv')
```

### Data Exploration
```python
# Basic statistics
print(model_data.describe())

# Target distribution
print(model_data['TARGET'].value_counts())

# Missing values
print(model_data.isnull().sum())
```

## 📊 Data Quality Metrics

### Dataset Overview
- **Total Patients**: 4,021 unique patients in the cohort
- **Total Physicians**: 25,344 healthcare providers
- **Total Transactions**: 115,042 clinical encounters
- **Model Records**: 3,613 patients with complete feature sets
- **Target Distribution**: Balanced dataset optimized for Drug A treatment prediction

### Data Completeness
- **Patient Demographics**: 99.8% complete
- **Physician Information**: 97.2% complete  
- **Transaction Records**: 98.5% complete
- **Clinical Features**: 96.7% complete

### Temporal Coverage
- **Study Period**: Multi-year longitudinal data
- **Seasonal Representation**: Balanced across all seasons
- **Treatment Window**: Critical 5-day window well-represented

## ⚠️ Important Notes

### Data Privacy & Security
1. **Patient Privacy**: All patient and physician identifiers are anonymized and should remain protected in production environments
2. **HIPAA Compliance**: Data handling follows healthcare privacy regulations and best practices

### Feature Considerations
3. **Feature Correlation**: Some features exhibit high correlation (e.g., RISK_NUM vs individual risk flags) - consider multicollinearity in model selection
4. **Temporal Sensitivity**: Critical 5-day treatment window features are essential for Drug A efficacy prediction
5. **Missing Data**: Some physician information may contain missing values (-1 or NaN) requiring appropriate handling strategies

### Clinical Context
6. **Treatment Window**: Drug A effectiveness is highly dependent on timely administration within the 5-day window
7. **Risk Stratification**: Patient risk factors significantly influence treatment decisions and outcomes
8. **Physician Variability**: Treatment patterns vary significantly across physician experience levels and specialties

## 🔧 Data Processing Scripts

- **Data Cleaning**: `scripts/data_cleaning/data_cleaning.py`
- **Feature Engineering**: `scripts/feature_engineering/build_model_ready_dataset.py`
- **Quality Assessment**: `scripts/data_cleaning/data_quality_check_reporting.py`

## 📈 Data Reports

Quality assessment reports are available in `reports/data_quality/`:
- Data quality assessment summary
- Individual table quality reports
- Visualization outputs

## 🔄 Data Updates

When updating data:
1. Place new raw files in `raw/` directory
2. Run data cleaning scripts
3. Regenerate processed files
4. Update model-ready dataset
5. Validate data quality reports

## 📞 Support

For data-related questions or issues:
1. Check data quality reports
2. Review feature dictionary
3. Consult data processing scripts
4. Contact the development team

---

## 🔄 Data Version Control

### Version Management System

The project implements a complete data version control system that tracks the version and integrity of all data files through metadata.

#### Core Features

- **✅ MD5 Checksum** - Each file has a unique checksum for integrity verification
- **✅ Version Metadata** - Track dataset version, creation time, and statistics
- **✅ Integrity Verification** - Automatically detect if files have been modified or corrupted
- **✅ Model Compatibility** - Ensure data version matches model version

#### Quick Start

```bash
# 1. Generate data manifest (first use or after data updates)
python data/versioning/manifest.py

# 2. Verify data integrity
python data/versioning/verify.py

# 3. Check model-data compatibility
python data/versioning/compatibility.py
```

#### Version Files

- **`.metadata/DATA_VERSIONS.json`** - Data version manifest (includes MD5, statistics, model compatibility)
- **`DATA_VERSIONS_README.md`** - Version description and usage guide

#### Current Version Information

- **Version**: 1.0.0
- **Total Size**: 14.91 MB
- **File Count**: 11 files
- **Compatible Model**: XGBoost v2.1.0

For detailed documentation, please refer to:
- `data/versioning/README.md` - Complete usage guide
- `data/DATA_VERSIONS_README.md` - Version description

---

*Last Updated: October 2024*
*Data Pipeline Version: 1.0*
*Data Version: 1.0.0*
