# Data Processing Module

The `backend/data_processing` module provides comprehensive data preprocessing and feature engineering utilities for the Pfizer EMR Alert System. This module is designed for production serving with robust error handling, validation, and monitoring capabilities.

## 📁 Module Structure

```
backend/data_processing/
├── __init__.py          # Module initialization
├── data_processor.py    # Core DataProcessor class
└── README.md           # This documentation
```

## 🎯 Overview

The Data Processing module is responsible for transforming raw EMR data into machine learning-ready features. It handles data standardization, categorical encoding, missing value imputation, and feature selection based on the system's feature dictionary.

## 🔧 Core Components

### DataProcessor Class

The main class that orchestrates all data preprocessing operations.

#### Key Responsibilities:
- **Data Standardization**: Uses StandardScaler for numerical feature normalization
- **Categorical Encoding**: Handles categorical variables with LabelEncoder
- **Missing Value Imputation**: Intelligent filling of missing values
- **Feature Selection**: Automatic feature selection based on feature dictionary
- **Data Validation**: Input data quality checks and validation

## 🚀 Quick Start

### Basic Usage

```python
from backend.data_processing import DataProcessor
import pandas as pd

# Initialize processor
processor = DataProcessor()

# Load your data
df = pd.read_csv('your_data.csv')

# Fit and transform data
X_processed = processor.fit_transform(df)

# Or fit and transform separately
processor.fit(df)
X_processed = processor.transform(df)
```

### Production Usage

```python
# Load pre-fitted processor
processor = DataProcessor()
processor.load(Path('models/preprocessor.pkl'))

# Process new data
X_new = processor.transform(new_df)
```

## 📊 Feature Configuration

The system uses **32 features** total:

### Feature Distribution
- **Total Features**: 32
- **Categorical Features**: 7
- **Numerical Features**: 25

### Categorical Features (7)
- `PATIENT_GENDER` - Patient gender
- `PHYS_EXPERIENCE_LEVEL` - Physician experience level
- `PHYSICIAN_STATE` - Physician state
- `PHYSICIAN_TYPE` - Physician type
- `DX_SEASON` - Diagnosis season
- `LOCATION_TYPE` - Location type
- `INSURANCE_TYPE_AT_DX` - Insurance type at diagnosis

### Numerical Features (25)
**Patient Demographics & Risk Factors:**
- `PATIENT_AGE` - Patient age
- `RISK_IMMUNO` - Immunocompromised risk
- `RISK_CVD` - Cardiovascular disease risk
- `RISK_DIABETES` - Diabetes risk
- `RISK_OBESITY` - Obesity risk
- `RISK_NUM` - Number of risk factors
- `RISK_AGE_FLAG` - Age risk flag

**Physician Information:**
- `PHYS_TOTAL_DX` - Total diagnoses by physician

**Symptom Timeline:**
- `SYMPTOM_TO_DIAGNOSIS_DAYS` - Days from symptom to diagnosis
- `DIAGNOSIS_WITHIN_5DAYS_FLAG` - Diagnosis within 5 days flag
- `SYM_COUNT_5D` - Symptom count within 5 days
- `PRIOR_CONTRA_LVL` - Prior contact level

**Specific Symptoms (15 features):**
- `SYMPTOM_ACUTE_PHARYNGITIS` - Acute pharyngitis
- `SYMPTOM_ACUTE_URI` - Acute upper respiratory infection
- `SYMPTOM_CHILLS` - Chills
- `SYMPTOM_CONGESTION` - Congestion
- `SYMPTOM_COUGH` - Cough
- `SYMPTOM_DIARRHEA` - Diarrhea
- `SYMPTOM_DIFFICULTY_BREATHING` - Difficulty breathing
- `SYMPTOM_FATIGUE` - Fatigue
- `SYMPTOM_FEVER` - Fever
- `SYMPTOM_HEADACHE` - Headache
- `SYMPTOM_LOSS_OF_TASTE_OR_SMELL` - Loss of taste or smell
- `SYMPTOM_MUSCLE_ACHE` - Muscle ache
- `SYMPTOM_NAUSEA_AND_VOMITING` - Nausea and vomiting
- `SYMPTOM_SORE_THROAT` - Sore throat

## 🔄 Data Processing Pipeline

### 1. Feature Selection
- Reads feature dictionary from `data/model_ready/model_feature_dictionary.xlsx`
- Automatically excludes target variables, data leakage features, and excluded features
- Ensures feature consistency across training and prediction

### 2. Categorical Encoding
- Handles categorical variables using LabelEncoder
- Manages unseen categories gracefully
- Replaces unknown categories with most frequent category

### 3. Missing Value Imputation
- **Numerical features**: Fills with median values
- **Categorical features**: Fills with mode values
- **Unseen categories**: Replaces with most frequent category

### 4. Data Standardization
- Applies StandardScaler to all numerical features
- Ensures features are on the same scale for ML algorithms

### 5. Output Generation
- Returns standardized numpy array ready for ML models
- Maintains feature order consistency

## 🛡️ Error Handling & Validation

### Data Validation
The processor includes comprehensive validation:

```python
# Validate input data
validation_result = processor.validate_input_data(df)
if not validation_result['valid']:
    print("Validation errors:", validation_result['errors'])
```

**Validation Checks:**
- Required features presence
- High missing value detection (>50%)
- Data type consistency
- Feature count matching

### Error Handling
- **Unseen Categories**: Gracefully handles new categorical values
- **Missing Encoders**: Provides fallback for missing label encoders
- **Data Leakage**: Automatically excludes problematic features
- **Logging**: Comprehensive logging for debugging and monitoring

## 💾 Persistence & State Management

### Saving Processor State
```python
processor.fit(training_data)
processor.save(Path('models/preprocessor.pkl'))
```

### Loading Processor State
```python
processor = DataProcessor()
processor.load(Path('models/preprocessor.pkl'))
```

### State Information
The processor maintains:
- Fitted scaler parameters
- Label encoders for categorical features
- Feature column mapping
- Processing statistics

## 📈 Monitoring & Statistics

### Processing Statistics
```python
stats = processor.get_processing_stats()
print(f"Feature count: {stats['feature_count']}")
print(f"Categorical features: {stats['categorical_features_count']}")
print(f"Is fitted: {stats['is_fitted']}")
```

### Validation Statistics
- Total processed records
- Validation errors count
- Missing values handled count

## 🔗 Integration Points

### With Model Manager
The DataProcessor integrates seamlessly with the ModelManager:

```python
from backend.api.model_manager import ModelManager

model_manager = ModelManager()
# DataProcessor is automatically initialized and used
```

### With Feature Engineering
Works in conjunction with EMRFeatureProcessor for comprehensive feature engineering.

### With Configuration System
Uses `config/settings.py` for:
- Feature configuration
- Categorical column definitions
- Excluded features list
- Data leakage feature identification

## 🧪 Testing & Quality Assurance

### Unit Tests
The module includes comprehensive unit tests covering:
- Data validation scenarios
- Missing value handling
- Categorical encoding edge cases
- Feature selection accuracy

### Integration Tests
- End-to-end processing pipeline
- Model integration testing
- API endpoint validation

## 📋 Best Practices

### 1. Always Fit Before Transform
```python
# Correct usage
processor.fit(training_data)
X_processed = processor.transform(new_data)

# Incorrect usage
X_processed = processor.transform(new_data)  # Will raise error
```

### 2. Validate Input Data
```python
validation_result = processor.validate_input_data(df)
if not validation_result['valid']:
    # Handle validation errors
    pass
```

### 3. Monitor Processing Statistics
```python
stats = processor.get_processing_stats()
# Log or monitor these statistics
```

### 4. Handle Unseen Categories
The processor automatically handles unseen categories, but monitor warnings in logs.

## 🚨 Common Issues & Solutions

### Issue: "DataProcessor must be fitted before transform"
**Solution**: Always call `fit()` before `transform()` or load a pre-fitted processor.

### Issue: Missing required features
**Solution**: Ensure all features from the feature dictionary are present in input data.

### Issue: High missing values warning
**Solution**: Review data quality and consider data collection improvements.

### Issue: Unseen categories warning
**Solution**: Monitor logs and consider updating training data to include new categories.

## 📚 Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Preprocessing utilities
- `pathlib` - Path handling
- `pickle` - Serialization
- `logging` - Logging functionality

## 🔄 Version History

- **v2.1.0** - Enhanced error handling and validation
- **v2.0.0** - Production-ready implementation
- **v1.0.0** - Initial implementation

## 📞 Support

For issues or questions regarding the data processing module:
1. Check the logs for detailed error messages
2. Validate input data using `validate_input_data()`
3. Review processing statistics using `get_processing_stats()`
4. Consult the main system documentation

---

*This module is part of the Pfizer EMR Alert System and is designed for production use in healthcare environments.*
