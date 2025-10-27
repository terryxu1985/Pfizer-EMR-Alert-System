# Prediction Results Storage

This directory contains prediction results from the `/predict` API endpoint.

## File Naming Convention

Prediction files are named using the following format:
```
prediction_{prediction_value}_{timestamp}.json
```

Where:
- `prediction_value`: The prediction result (0 or 1)
- `timestamp`: Format YYYYMMDD_HHMMSS

Example: `prediction_1_20251021_124218.json`

## File Structure

Each prediction file contains the complete API response including:

### Core Prediction Data
- `prediction`: Binary prediction result (0=Not Treated, 1=Treated)
- `probability`: Probability score (0.0-1.0)
- `not_prescribed_drug_a`: 1 if patient was not prescribed Drug A; else 0
- `not_prescribed_drug_a_probability`: Probability that patient was not prescribed Drug A
- `alert_recommended`: Whether an alert is recommended (True when prediction=1 AND probability>=0.7)

### Clinical Assessment
- `clinical_eligibility`: Rule-based clinical eligibility assessment
  - `meets_criteria`: Overall eligibility status
  - `age_eligible`: Age-based eligibility
  - `within_5day_window`: Within treatment window
  - `is_high_risk`: High-risk patient status
  - `risk_factors_found`: List of identified risk factors
  - `risk_conditions_details`: Detailed risk condition mapping

### Feature Engineering
- `processed_features`: Model-ready features extracted from raw EMR data
- `feature_engineering_info`: Information about the feature engineering process
- `feature_validation`: Validation results for feature consistency

### Metadata
- `model_version`: Version of the model used
- `model_type`: Type of model (e.g., XGBoost)
- `processing_time_ms`: Processing time in milliseconds

## Usage

These files can be used for:
1. **Audit Trail**: Tracking all prediction requests and results
2. **Model Performance Analysis**: Analyzing prediction patterns over time
3. **Debugging**: Investigating specific prediction cases
4. **Compliance**: Maintaining records for regulatory requirements
5. **Research**: Analyzing patient outcomes and model accuracy

## Data Retention

Consider implementing a data retention policy based on:
- Regulatory requirements
- Storage capacity
- Analysis needs
- Privacy considerations

## Security

These files contain sensitive patient data and should be:
- Stored securely with appropriate access controls
- Encrypted if required by compliance standards
- Regularly backed up
- Properly disposed of according to data retention policies
