# EMR Feature Engineering Module

## Overview

The EMR Feature Engineering Module transforms raw Electronic Medical Record (EMR) transaction data into standardized, model-ready features for the Pfizer EMR Alert System. This module processes real-time patient data and generates 32 standardized features used by machine learning models to predict patient outcomes and optimize treatment recommendations.

## Architecture

```
backend/feature_engineering/
├── __init__.py
├── emr_feature_processor.py    # Core feature engineering class
└── FEATURE_ENGINEERING.md      # This documentation
```

## Key Components

### EMRFeatureProcessor Class

The primary class responsible for all feature engineering operations:

- **Real-time Processing**: Converts raw EMR requests into structured feature sets
- **Feature Validation**: Ensures data quality, completeness, and consistency
- **Domain Knowledge Integration**: Incorporates medical domain expertise into feature creation
- **Error Handling**: Provides robust error handling with graceful degradation

## Feature Specification

The module generates **32 features** organized across 7 clinical categories:

### 1. Patient Demographics (3 features)
- `PATIENT_AGE`: Patient age at time of diagnosis (years)
- `PATIENT_GENDER`: Patient gender (Male/Female)
- `RISK_AGE_FLAG`: High-risk age indicator (1 if age ≥65 years, 0 otherwise)

### 2. Symptom Timeline Features (4 features)
- `SYM_COUNT_5D`: Number of symptoms recorded within 5 days prior to diagnosis
- `SYMPTOM_ONSET_DT`: Date when first symptom appeared
- `SYMPTOM_TO_DIAGNOSIS_DAYS`: Duration between symptom onset and diagnosis (days)
- `DIAGNOSIS_WITHIN_5DAYS_FLAG`: Binary indicator for diagnosis within 5-day treatment window

### 3. Detailed Symptom Indicators (14 features)
Individual binary flags (0/1) for specific clinical symptoms:
- `SYMPTOM_ACUTE_PHARYNGITIS`: Acute pharyngitis present
- `SYMPTOM_ACUTE_URI`: Acute upper respiratory infection present
- `SYMPTOM_CHILLS`: Chills present
- `SYMPTOM_CONGESTION`: Nasal congestion present
- `SYMPTOM_COUGH`: Cough present
- `SYMPTOM_DIARRHEA`: Diarrhea present
- `SYMPTOM_DIFFICULTY_BREATHING`: Respiratory difficulty present
- `SYMPTOM_FATIGUE`: Fatigue or weakness present
- `SYMPTOM_FEVER`: Fever present
- `SYMPTOM_HEADACHE`: Headache present
- `SYMPTOM_LOSS_OF_TASTE_OR_SMELL`: Anosmia or ageusia present
- `SYMPTOM_MUSCLE_ACHE`: Myalgia present
- `SYMPTOM_NAUSEA_AND_VOMITING`: Gastrointestinal symptoms present
- `SYMPTOM_SORE_THROAT`: Pharyngitis present

### 4. Clinical Risk Factors (5 features)
- `RISK_IMMUNO`: Immunocompromised status indicator
- `RISK_CVD`: Cardiovascular disease indicator
- `RISK_DIABETES`: Diabetes mellitus indicator
- `RISK_OBESITY`: Obesity indicator (BMI-based)
- `RISK_NUM`: Total count of risk factors present (0-4)

### 5. Physician Characteristics (4 features)
- `PHYS_EXPERIENCE_LEVEL`: Physician experience classification (Low/Medium/High)
- `PHYSICIAN_STATE`: Geographic location of practice
- `PHYSICIAN_TYPE`: Physician specialty or type
- `PHYS_TOTAL_DX`: Total number of diagnoses made by physician (historical)

### 6. Clinical Visit Context (2 features)
- `LOCATION_TYPE`: Healthcare facility type (Emergency, Outpatient, etc.)
- `INSURANCE_TYPE_AT_DX`: Insurance coverage type at time of diagnosis

### 7. Medication Contraindications (1 feature)
- `PRIOR_CONTRA_LVL`: Prior contraindication severity level (0-3)

## Usage

### Basic Implementation

```python
from backend.feature_engineering.emr_feature_processor import EMRFeatureProcessor
from backend.api.api_models import RawEMRRequest

# Initialize the feature processor
processor = EMRFeatureProcessor()

# Process raw EMR data into features
features, fe_metadata = processor.process_raw_emr_data(raw_emr_request)

# Validate feature completeness and quality
validation_result = processor.validate_features(features)
```

### Input Requirements

The processor expects a `RawEMRRequest` object containing:
- **Patient Demographics**: Age, gender, birth year
- **Diagnosis Information**: Diagnosis date and codes
- **Transaction Records**: Complete symptom history, conditions, and contraindications
- **Physician Information**: Physician ID, experience, and practice details
- **Visit Context**: Location type and insurance information

### Output Structure

Returns a tuple containing:
1. **Features Dictionary**: Dictionary with 32 standardized features ready for ML model consumption
2. **Feature Engineering Metadata**: Processing metadata including warnings, timestamps, and data quality flags

## Core Methods

### `process_raw_emr_data(raw_data: RawEMRRequest) -> Tuple[Dict, Dict]`

Primary processing method that transforms raw EMR data into structured features.

**Parameters:**
- `raw_data` (RawEMRRequest): Raw EMR request object containing patient data

**Returns:**
- Tuple containing:
  - `features` (Dict): Dictionary of 32 processed features
  - `metadata` (Dict): Feature engineering metadata and warnings

**Raises:**
- Handles all exceptions internally with graceful degradation
- Logs errors and warnings to system logs

### `validate_features(features: Dict[str, Any]) -> Dict`

Validates processed features for completeness, data types, and business logic consistency.

**Parameters:**
- `features` (Dict): Dictionary of processed features to validate

**Returns:**
- Validation results dictionary containing:
  - `is_valid` (bool): Overall validation status
  - `missing_features` (List): List of missing required features
  - `warnings` (List): Data quality warnings
  - `errors` (List): Critical validation errors

## Domain-Specific Constants

The processor utilizes clinically-relevant constants:

```python
# Disease and Treatment Configuration
DX_CODE = "DISEASE_X"              # Target disease identifier
DRUG_A = "DRUG A"                  # Primary treatment drug
MIN_AGE_YRS = 12                   # Minimum eligible patient age
TREAT_WINDOW_DAYS = 5              # Optimal treatment window (days)

# Physician Experience Classification
PHYS_EXP_THRESHOLDS = {
    "High": 200,      # ≥200 total diagnoses
    "Medium": 20      # 20-199 diagnoses
}                     # <20 diagnoses = "Low"
```

## Risk Condition Mappings

Clinical condition codes mapped to standardized risk categories:

```python
RISK_CONDITIONS = {
    "RISK_IMMUNO": [
        "IMMUNOCOMPROMISED"
    ],
    "RISK_CVD": [
        "HEART_DISEASE",
        "HYPERTENSION",
        "STROKE"
    ],
    "RISK_DIABETES": [
        "DIABETES"
    ],
    "RISK_OBESITY": [
        "OBESITY"
    ]
}
```

## Contraindication Severity Levels

Contraindication conditions mapped to severity scores:

```python
CONTRAINDICATION_LEVELS = {
    "LOW_CONTRAINDICATION": 1,      # Minor risk
    "MEDIUM_CONTRAINDICATION": 2,   # Moderate risk
    "HIGH_CONTRAINDICATION": 3      # Significant risk
}
```

## Data Quality Assurance

The processor implements comprehensive data quality controls:

### Validation Checks
- **Age Validation**: Ensures patient age is within clinically reasonable bounds (12-120 years)
- **Temporal Consistency**: Validates symptom onset precedes diagnosis date
- **Feature Completeness**: Verifies all 32 required features are present
- **Data Type Validation**: Ensures features match expected data types

### Missing Data Handling
- **Default Values**: Provides clinically appropriate defaults for missing data
- **Imputation Logic**: Uses domain knowledge for intelligent missing value handling
- **Warning System**: Logs all missing data instances for quality monitoring

### Error Recovery
- **Graceful Degradation**: Continues processing with warnings when non-critical data is missing
- **Exception Handling**: Catches and logs all processing exceptions
- **Fallback Values**: Uses safe defaults to prevent pipeline failures

## Integration Architecture

### Input Integration
- **API Models**: Consumes `RawEMRRequest` from `backend.api.api_models`
- **Transaction Records**: Processes `TransactionRecord` objects with temporal data
- **Physician Information**: Integrates `PhysicianInfo` objects for provider features

### Output Integration
- **Model Manager**: Features consumed by `backend.api.model_manager` for prediction
- **ML Pipeline**: Features feed directly into trained XGBoost models
- **Validation Layer**: Features validated before model inference

## Performance Characteristics

### Optimization Strategies
- **Real-time Processing**: Optimized for sub-100ms feature generation latency
- **Memory Efficiency**: Streaming data processing minimizes memory footprint
- **Computation Reuse**: Caches intermediate calculations where possible
- **Minimal Logging**: Production logging optimized for performance

### Scalability
- **Stateless Design**: Enables horizontal scaling across multiple instances
- **No External Dependencies**: Self-contained processing eliminates network latency
- **Batch Support**: Can process individual or batch requests efficiently

## Testing and Validation

### Automated Testing
- **Feature Completeness Tests**: Validates all 32 features are generated
- **Data Type Tests**: Verifies correct feature data types
- **Range Validation Tests**: Checks numeric features are within expected ranges
- **Business Logic Tests**: Ensures clinical rules are properly implemented

### Integration Testing
- **End-to-End Tests**: Validates feature flow through entire pipeline
- **Mock Data Tests**: Uses synthetic test cases for edge case coverage
- **Production Data Tests**: Validates against historical patient data

## Technical Dependencies

### Required Libraries
- **pandas** (≥1.3.0): DataFrame operations and data manipulation
- **numpy** (≥1.21.0): Numerical computations and array operations
- **datetime**: Date and time calculations
- **logging**: Application logging and monitoring
- **dataclasses**: Type-safe data structure definitions
- **typing**: Type hints and annotations

### Internal Dependencies
- `backend.api.api_models`: API request/response models
- `backend.data_access`: Data loading utilities
- `config.settings`: System configuration

## Monitoring and Observability

### Logging
- **Processing Logs**: Detailed logs for each feature engineering request
- **Warning Logs**: Data quality warnings and processing issues
- **Error Logs**: Exception tracking and error conditions
- **Performance Logs**: Processing time and resource utilization

### Metrics
- **Processing Latency**: Time to generate features per request
- **Data Quality Scores**: Percentage of complete vs. imputed features
- **Warning Rates**: Frequency of data quality warnings
- **Error Rates**: Feature engineering failure rates

## Future Enhancements

### Planned Improvements
- **Advanced Temporal Features**: Time-series feature engineering for symptom progression
- **Dynamic Feature Selection**: Automatic feature selection based on model performance
- **Feature Caching**: Redis-based caching for frequently accessed patient features
- **Streaming Pipeline**: Kafka-based streaming for real-time feature updates
- **Enhanced Monitoring**: Grafana dashboards for feature quality monitoring
- **Feature Versioning**: Support for multiple feature schema versions
- **A/B Testing Support**: Feature variants for experimentation

### Research Opportunities
- **Automated Feature Engineering**: ML-based feature discovery
- **Domain-Specific Embeddings**: Neural network-based feature representations
- **Causal Feature Engineering**: Causal inference-based feature selection

## Troubleshooting

### Common Issues

**Issue**: Missing features in output
- **Cause**: Incomplete input data
- **Solution**: Check `fe_metadata` for warnings; ensure all required fields in input

**Issue**: Invalid feature values
- **Cause**: Out-of-range input data
- **Solution**: Review validation logs; check data quality at source

**Issue**: Slow processing time
- **Cause**: Large transaction history
- **Solution**: Optimize transaction filtering; consider caching

## Support and Documentation

For additional information, refer to:
- **API Documentation**: `backend/api/README.md`
- **Model Training Documentation**: `scripts/model_training/README.md`
- **System Architecture**: Main project `README.md`
- **System Logs**: Check `logs/emr_alert_system.log` for debugging

## Contributing

When modifying this module:
1. Maintain the 32-feature contract (do not add/remove features without model retraining)
2. Preserve backward compatibility with existing API contracts
3. Add comprehensive unit tests for new functionality
4. Update this documentation with any changes
5. Follow medical domain best practices for clinical feature engineering

## Version History

- **v2.1.0**: Current version with 32 standardized features
- **v2.0.0**: Added contraindication features
- **v1.0.0**: Initial release with core feature set

