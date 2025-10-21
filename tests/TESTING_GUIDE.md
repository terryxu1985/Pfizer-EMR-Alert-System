# Pfizer EMR Alert System - Testing Guide

## Overview

The `tests` directory contains a comprehensive test suite for validating the EMR Alert System's API endpoints and prediction logic. The test framework is designed to ensure that the system correctly identifies patients who should receive Disease X treatment alerts based on clinical criteria and AI predictions.

## System Testing Architecture

```
tests/
├── test_api_endpoint.py     # Main test runner and orchestration
├── input/                    # Test case definitions (JSON)
│   ├── test_elderly_multiple_risk_factors.json
│   ├── test_immunocompromised_patient.json
│   ├── test_obese_copd_patient.json
│   ├── test_age_under_12_child.json
│   ├── test_contraindication_patient.json
│   ├── test_healthy_young_adult.json
│   ├── test_high_risk_senior_patient.json
│   └── test_late_symptoms_patient.json
└── outputs/                  # Test results and reports (JSON)
    ├── *_result.json         # Individual test results
    └── test_summary.json     # Comprehensive test summary
```

## Key Features

### 🧪 Comprehensive Test Coverage
- **Positive Test Cases**: Patients who should trigger alerts (high-risk, multiple conditions)
- **Negative Test Cases**: Patients who should not trigger alerts (ineligible, low-risk)
- **Edge Cases**: Boundary conditions for age, timing, and risk thresholds
- **Clinical Validation**: Ensures adherence to clinical guidelines and business rules

### 🔧 Flexible Test Execution
- **Direct Mode**: Tests model logic directly without requiring API server
- **HTTP Mode**: Tests full API stack including network, routing, and server
- **Batch Execution**: Run all tests sequentially with comprehensive reporting
- **Single Test Mode**: Execute individual test cases for debugging

### 📊 Detailed Result Reporting
- **Individual Results**: Per-test JSON files with complete input/output
- **Summary Reports**: Aggregate statistics across all test cases
- **Performance Metrics**: Response time tracking and throughput analysis
- **Validation Status**: Clear pass/fail indicators with detailed reasons

### 🎯 Test Expectations Framework
- **Pre-defined Expectations**: Each test case has documented expected outcomes
- **Automated Validation**: Compare actual results against expected behavior
- **Reason Documentation**: Clear explanations for why each test should pass/fail

## Test Case Anatomy

### JSON Test File Structure

Each test case is defined as a JSON file in the `tests/input/` directory:

```json
{
  "description": "Human-readable description of the test scenario",
  "input": {
    "patient_id": 2001,
    "birth_year": 1940,
    "gender": "M",
    "diagnosis_date": "2025-10-21T00:00:00",
    "transactions": [
      {
        "txn_dt": "2025-10-19T00:00:00",
        "physician_id": 102,
        "txn_location_type": "OFFICE",
        "insurance_type": "MEDICARE",
        "txn_type": "SYMPTOMS",
        "txn_desc": "FEVER"
      }
    ],
    "physician_info": {
      "physician_id": 102,
      "state": "NY",
      "physician_type": "Internal Medicine",
      "gender": "F",
      "birth_year": 1975
    }
  }
}
```

### Test Components

1. **Description**: Clear, concise description of what the test validates
2. **Patient Demographics**: Age (via birth_year), gender, diagnosis date
3. **Transactions**: EMR transaction history including:
   - **Symptoms**: FEVER, COUGH, DIFFICULTY_BREATHING, etc.
   - **Conditions**: Pre-existing conditions and comorbidities
   - **Medications**: Current medications and treatments
   - **Contraindications**: Known drug interactions or allergies
4. **Physician Information**: Doctor credentials, specialty, experience

### Test Categories

#### Positive Test Cases (Should Alert)

These cases represent patients who meet clinical criteria and should receive treatment alerts:

1. **test_elderly_multiple_risk_factors.json**
   - 85-year-old with diabetes, heart disease, obesity, COPD
   - Multiple symptoms within 5-day window
   - Expected: Alert recommended

2. **test_immunocompromised_patient.json**
   - 72-year-old immunocompromised patient
   - Diabetes and heart disease
   - Symptoms within 1 day
   - Expected: Alert recommended

3. **test_obese_copd_patient.json**
   - 65-year-old with obesity, COPD, diabetes, CVD
   - Respiratory symptoms
   - Expected: Alert recommended

#### Negative Test Cases (Should Not Alert)

These cases represent patients who do not meet criteria and should not receive alerts:

1. **test_age_under_12_child.json**
   - Patient younger than 12 years old
   - Age eligibility criterion not met
   - Expected: No alert

2. **test_healthy_young_adult.json**
   - No significant risk factors
   - Not in high-risk population
   - Expected: No alert

3. **test_late_symptoms_patient.json**
   - Symptoms beyond 5-day window
   - Treatment timing criterion not met
   - Expected: No alert

4. **test_contraindication_patient.json**
   - Severe contraindication present
   - Safety criterion not met
   - Expected: No alert

5. **test_high_risk_senior_patient.json**
   - Clinical criteria met but AI probability below threshold
   - AI confidence not sufficient (0.669 < 0.7)
   - Expected: No alert

## Running Tests

### Prerequisites

Ensure the system is properly set up:

```bash
# Install dependencies
pip install -r config/requirements.txt

# Verify models are available
ls backend/ml_models/models/
```

### Test Execution Modes

#### 1. Direct Mode (No Server Required)

Direct mode calls the model manager directly, bypassing the API layer. This is faster and ideal for development:

```bash
# From project root
python tests/test_api_endpoint.py --mode direct

# Or use default (direct mode)
python tests/test_api_endpoint.py
```

**Advantages:**
- ✅ No need to start API server
- ✅ Faster execution (no HTTP overhead)
- ✅ Direct access to Python exceptions and stack traces
- ✅ Ideal for development and debugging

#### 2. HTTP Mode (Full Stack Testing)

HTTP mode tests the complete API stack, including FastAPI routing and middleware:

```bash
# First, start the API server in a separate terminal
python run_api_only.py

# Then run tests in HTTP mode
python tests/test_api_endpoint.py --mode http --api-url http://localhost:8000
```

**Advantages:**
- ✅ Tests complete API stack
- ✅ Validates HTTP request/response handling
- ✅ Tests middleware and error handling
- ✅ Simulates real-world usage

### Understanding Test Output

#### Console Output

The test runner provides detailed console output for each test:

```
================================================================================
Test case: elderly_multiple_risk_factors
================================================================================
Description: 85-year-old patient with multiple risk factors...
Expected: Should alert
Reason: 85-year-old with multiple high-risk conditions
Input file: tests/input/test_elderly_multiple_risk_factors.json
--------------------------------------------------------------------------------
Response time: 0.152s
AI prediction: prediction=1, probability=0.892
Clinical eligibility: meets_criteria=True
  - Age: 85 years ✅
  - Within 5 days: True ✅
  - Symptom days: 2 days
  - High risk: True ✅
  - Risk factors: ['DIABETES', 'HEART_DISEASE', 'OBESITY', 'COPD']
  - No severe contraindication: True ✅
  - Contraindication level: None
Alert recommended: True
✅ Result saved: tests/outputs/elderly_multiple_risk_factors_result.json
✅ Test passed
```

#### Test Summary

After all tests complete, a summary is displayed:

```
================================================================================
Test Summary
================================================================================
Total test files: 8
Executed: 8
Skipped: 0
Passed: 8
Failed: 0
Errors: 0
Summary report: tests/outputs/test_summary.json
================================================================================
```

### Test Results Files

#### Individual Test Results

Each test generates a detailed JSON result file in `tests/outputs/`:

```json
{
  "test_name": "elderly_multiple_risk_factors",
  "test_file": "test_elderly_multiple_risk_factors.json",
  "description": "85-year-old patient with multiple risk factors...",
  "expected_alert": true,
  "expected_reason": "85-year-old with multiple high-risk conditions",
  "timestamp": "2025-10-21T14:30:45.123456",
  "response_time_seconds": 0.152,
  "api_mode": "direct_call",
  "input": { ... },
  "output": {
    "prediction": 1,
    "probability": 0.892,
    "alert_recommended": true,
    "clinical_eligibility": {
      "meets_criteria": true,
      "patient_age": 85,
      "within_5day_window": true,
      "is_high_risk": true,
      "no_severe_contraindication": true,
      "risk_factors_found": ["DIABETES", "HEART_DISEASE", "OBESITY", "COPD"]
    }
  },
  "test_passed": true
}
```

#### Summary Report

The `test_summary.json` file contains aggregate statistics:

```json
{
  "total_tests": 8,
  "executed": 8,
  "skipped": 0,
  "passed": 8,
  "failed": 0,
  "errors": 0,
  "timestamp": "2025-10-21T14:30:50.123456",
  "api_mode": "direct_call",
  "test_results": [ ... ]
}
```

## Creating New Test Cases

### Step 1: Define Test File

Create a new JSON file in `tests/input/` following the naming convention `test_<scenario_name>.json`:

```json
{
  "description": "Your test scenario description",
  "input": {
    "patient_id": 3001,
    "birth_year": 1960,
    "gender": "F",
    "diagnosis_date": "2025-10-21T00:00:00",
    "transactions": [
      {
        "txn_dt": "2025-10-19T00:00:00",
        "physician_id": 101,
        "txn_location_type": "OFFICE",
        "insurance_type": "PRIVATE",
        "txn_type": "SYMPTOMS",
        "txn_desc": "FEVER"
      },
      {
        "txn_dt": "2025-10-21T00:00:00",
        "physician_id": 101,
        "txn_location_type": "OFFICE",
        "insurance_type": "PRIVATE",
        "txn_type": "CONDITIONS",
        "txn_desc": "DISEASE_X"
      }
    ],
    "physician_info": {
      "physician_id": 101,
      "state": "CA",
      "physician_type": "Family Medicine",
      "gender": "M",
      "birth_year": 1980
    }
  }
}
```

### Step 2: Define Test Expectations

Add the test expectations to `TEST_EXPECTATIONS` dictionary in `test_api_endpoint.py`:

```python
TEST_EXPECTATIONS = {
    # ... existing tests ...
    
    "test_your_scenario.json": {
        "should_alert": True,  # or False
        "reason": "Detailed reason why this test should pass/fail"
    }
}
```

### Step 3: Run and Validate

Execute the test suite to validate your new test case:

```bash
python tests/test_api_endpoint.py
```

## Test Data Requirements

### Required Fields

Every test case must include:

1. **Patient Demographics**
   - `patient_id`: Unique identifier
   - `birth_year`: For age calculation
   - `gender`: M, F, or OTHER
   - `diagnosis_date`: Disease X diagnosis date (ISO format)

2. **EMR Transactions**
   - At least one DISEASE_X condition transaction
   - Relevant symptoms and conditions
   - Proper date formatting (ISO 8601)

3. **Physician Information**
   - `physician_id`: Unique identifier
   - `state`: Two-letter state code
   - `physician_type`: Medical specialty
   - `gender`: M or F
   - `birth_year`: For experience calculation

### Valid Transaction Types

#### Transaction Types (`txn_type`)
- `SYMPTOMS`: Patient symptoms
- `CONDITIONS`: Medical conditions and diagnoses
- `MEDICATIONS`: Current and past medications
- `CONTRAINDICATIONS`: Known drug interactions or allergies

#### Common Symptom Descriptions (`txn_desc` for SYMPTOMS)
- `FEVER`
- `COUGH`
- `DIFFICULTY_BREATHING`
- `FATIGUE`
- `BODY_ACHES`
- `HEADACHE`
- `SORE_THROAT`
- `LOSS_OF_TASTE_OR_SMELL`

#### Common Condition Descriptions (`txn_desc` for CONDITIONS)
- `DISEASE_X`: Primary diagnosis (required)
- `DIABETES`: Diabetes mellitus
- `HEART_DISEASE`: Cardiovascular disease
- `OBESITY`: Obesity (BMI ≥ 30)
- `COPD`: Chronic obstructive pulmonary disease
- `ASTHMA`: Asthma
- `KIDNEY_DISEASE`: Chronic kidney disease
- `IMMUNOCOMPROMISED`: Weakened immune system
- `CANCER`: Active cancer diagnosis

#### Location Types (`txn_location_type`)
- `OFFICE`: Doctor's office visit
- `HOSPITAL`: Hospital setting
- `ER`: Emergency room
- `TELEHEALTH`: Remote consultation
- `URGENT_CARE`: Urgent care facility

#### Insurance Types (`insurance_type`)
- `MEDICARE`: Medicare insurance
- `MEDICAID`: Medicaid insurance
- `PRIVATE`: Private insurance
- `SELF_PAY`: Self-pay/uninsured

## Clinical Criteria Validation

The system validates the following clinical criteria for each patient:

### 1. Age Eligibility
- **Rule**: Patient must be ≥ 12 years old
- **Validation**: Calculated from `birth_year` and `diagnosis_date`
- **Result**: `patient_age` in clinical eligibility

### 2. Treatment Timing Window
- **Rule**: Symptoms must be within 5 days of diagnosis
- **Validation**: Checks time between first symptom and diagnosis
- **Result**: `within_5day_window` boolean
- **Detail**: `symptom_to_diagnosis_days` count

### 3. High-Risk Population
- **Rule**: Patient must have at least one high-risk condition
- **High-Risk Conditions**:
  - Age ≥ 65 years
  - Diabetes
  - Heart disease
  - Obesity
  - COPD
  - Asthma
  - Kidney disease
  - Immunocompromised status
  - Active cancer
- **Result**: `is_high_risk` boolean
- **Detail**: `risk_factors_found` list

### 4. Contraindication Check
- **Rule**: No severe contraindications present
- **Severe Contraindications**: Known allergies or drug interactions
- **Result**: `no_severe_contraindication` boolean
- **Detail**: `contraindication_level` severity

### 5. AI Prediction Threshold
- **Rule**: AI probability must be ≥ 0.7 (70%)
- **Validation**: Model prediction confidence score
- **Result**: `probability` value and `prediction` (0 or 1)

### Final Alert Decision

An alert is recommended **only when ALL criteria are met**:
```
alert_recommended = 
    (patient_age >= 12) AND
    (within_5day_window) AND
    (is_high_risk) AND
    (no_severe_contraindication) AND
    (AI_probability >= 0.7)
```

## Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error**: `Model files not found in backend/ml_models/models/`

**Solution**:
```bash
# Verify model files exist
ls -la backend/ml_models/models/

# If missing, train models first
cd scripts/model_training
python train.py
```

#### 2. Test Expectations Not Defined

**Warning**: `⚠️ Skipping test_xxx.json - no expectations defined`

**Solution**: Add test expectations to `TEST_EXPECTATIONS` dictionary in `test_api_endpoint.py`

#### 3. API Connection Error (HTTP Mode)

**Error**: `Cannot connect to API server http://localhost:8000`

**Solution**:
```bash
# Start API server in separate terminal
python run_api_only.py

# Wait for "Application startup complete" message
# Then run tests
python tests/test_api_endpoint.py --mode http
```

#### 4. Test Failed - Unexpected Result

**Issue**: Test passes/fails contrary to expectations

**Debug Steps**:
1. Check individual result file in `tests/outputs/`
2. Review clinical eligibility details
3. Verify AI probability value
4. Check feature engineering output
5. Review model prediction logic

Example debugging:
```bash
# Run single test in direct mode for detailed output
python tests/test_api_endpoint.py --mode direct

# Check result file
cat tests/outputs/your_test_result.json | python -m json.tool

# Review logs
tail -f logs/emr_alert_system.log
```

#### 5. Import Errors

**Error**: `ModuleNotFoundError: No module named 'backend'`

**Solution**:
```bash
# Ensure you're in project root
pwd  # Should show: .../Pfizer-EMR Alert System

# Install dependencies
pip install -r config/requirements.txt

# Run from project root
python tests/test_api_endpoint.py
```

### Performance Issues

If tests are running slowly:

1. **Use Direct Mode**: Bypass HTTP overhead
   ```bash
   python tests/test_api_endpoint.py --mode direct
   ```

2. **Check Model Cache**: Ensure models are cached properly
   ```bash
   ls -lh data/storage/system/model_cache.json
   ```

3. **Review Logs**: Check for repeated model loading
   ```bash
   grep "Loading model" logs/emr_alert_system.log
   ```

## Best Practices

### 1. Test Design

- **Clear Scenarios**: Each test should validate one specific scenario
- **Descriptive Names**: Use meaningful file names that indicate the test purpose
- **Comprehensive Coverage**: Include both positive and negative test cases
- **Edge Cases**: Test boundary conditions and unusual inputs
- **Documentation**: Include detailed descriptions in JSON files

### 2. Test Maintenance

- **Regular Execution**: Run tests after every code change
- **Review Results**: Analyze failed tests promptly
- **Update Expectations**: Keep expectations aligned with business rules
- **Version Control**: Track test files in git
- **Result Archival**: Keep historical results for regression analysis

### 3. Development Workflow

```bash
# 1. Make code changes
vim backend/api/model_manager.py

# 2. Run tests in direct mode (fast)
python tests/test_api_endpoint.py --mode direct

# 3. Fix any failures
# ... debug and fix ...

# 4. Run full HTTP stack test
python run_api_only.py &  # Start server
python tests/test_api_endpoint.py --mode http

# 5. Review results
cat tests/outputs/test_summary.json

# 6. Commit changes
git add .
git commit -m "Fix: Updated model prediction logic"
```

### 4. Continuous Integration

For CI/CD pipelines:

```bash
#!/bin/bash
# ci_test.sh

set -e  # Exit on any error

echo "Installing dependencies..."
pip install -r config/requirements.txt

echo "Running test suite..."
python tests/test_api_endpoint.py --mode direct

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
    exit 0
else
    echo "❌ Tests failed"
    cat tests/outputs/test_summary.json
    exit 1
fi
```

## Integration with System

### Test Suite in Development Workflow

The test suite integrates with the overall system architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     Development Workflow                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Code Changes                                            │
│     └─> backend/api/                                        │
│     └─> backend/feature_engineering/                        │
│     └─> scripts/model_training/                            │
│                                                              │
│  2. Run Tests (Direct Mode)                                 │
│     └─> python tests/test_api_endpoint.py                  │
│     └─> Fast validation, no server required                │
│                                                              │
│  3. Review Results                                          │
│     └─> tests/outputs/test_summary.json                    │
│     └─> Individual test result files                        │
│                                                              │
│  4. Debug if Needed                                         │
│     └─> logs/emr_alert_system.log                          │
│     └─> Individual result JSON files                        │
│                                                              │
│  5. Full Stack Test (HTTP Mode)                            │
│     └─> Start: python run_api_only.py                      │
│     └─> Test: python tests/test_api_endpoint.py --mode http│
│                                                              │
│  6. Deploy                                                  │
│     └─> Production deployment with confidence              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Related Documentation

For comprehensive system understanding, refer to:

- **Backend Guide**: `backend/BACKEND_GUIDE.md`
  - API architecture and endpoints
  - Model management and prediction flow
  
- **Data Processing Guide**: `backend/data_processing/DATA_PROCESSING_GUIDE.md`
  - Feature preprocessing and validation
  - Data quality requirements
  
- **Feature Engineering Guide**: `backend/feature_engineering/FEATURE_ENGINEERING_GUIDE.md`
  - EMR transaction to feature conversion
  - Domain-specific logic implementation
  
- **Models Guide**: `backend/ml_models/MODELS_GUIDE.md`
  - Model architecture and performance
  - Version management and updates
  
- **Quick Start**: `QUICK_START.md`
  - System setup and first steps
  - Running the complete system

## Advanced Testing

### Custom Test Scenarios

For complex testing scenarios, you can extend the test framework:

```python
# custom_test.py
from tests.test_api_endpoint import APIEndpointTestRunner

# Create custom runner
runner = APIEndpointTestRunner(use_direct_call=True)

# Load and modify test case
test_data = runner.load_test_case("test_elderly_multiple_risk_factors.json")

# Customize test data
test_data['input']['birth_year'] = 1950  # Change age
test_data['input']['transactions'].append({
    "txn_dt": "2025-08-01T00:00:00",
    "physician_id": 102,
    "txn_location_type": "OFFICE",
    "insurance_type": "MEDICARE",
    "txn_type": "CONDITIONS",
    "txn_desc": "ASTHMA"
})

# Run custom test
result = runner.call_api_direct(test_data)
print(f"Alert recommended: {result['alert_recommended']}")
print(f"Probability: {result['probability']:.3f}")
```

### Batch Processing Tests

Test batch prediction endpoints:

```python
import requests
import json

# Prepare batch of test cases
batch_input = []
for test_file in ["test_elderly_multiple_risk_factors.json", 
                  "test_immunocompromised_patient.json",
                  "test_obese_copd_patient.json"]:
    with open(f"tests/input/{test_file}") as f:
        test_data = json.load(f)
        batch_input.append(test_data['input'])

# Call batch endpoint
response = requests.post(
    "http://localhost:8000/api/v1/predict/batch",
    json={"patients": batch_input}
)

results = response.json()
print(f"Processed {len(results['results'])} patients")
for i, result in enumerate(results['results']):
    print(f"Patient {i+1}: Alert={result['alert_recommended']}")
```

### Performance Testing

Measure system performance under load:

```python
import time
from concurrent.futures import ThreadPoolExecutor

def run_test_concurrent(test_file):
    runner = APIEndpointTestRunner(use_direct_call=False)
    test_data = runner.load_test_case(test_file)
    start = time.time()
    result = runner.call_api_http(test_data)
    duration = time.time() - start
    return duration

# Run 100 concurrent requests
test_file = "test_elderly_multiple_risk_factors.json"
with ThreadPoolExecutor(max_workers=10) as executor:
    durations = list(executor.map(
        run_test_concurrent, 
        [test_file] * 100
    ))

print(f"Average response time: {sum(durations)/len(durations):.3f}s")
print(f"Min: {min(durations):.3f}s, Max: {max(durations):.3f}s")
```

## Summary

The test suite provides comprehensive validation of the EMR Alert System's prediction logic and API functionality. By following this guide, developers can:

- ✅ Understand test architecture and organization
- ✅ Run tests in multiple modes (direct and HTTP)
- ✅ Create new test cases for additional scenarios
- ✅ Interpret test results and debug failures
- ✅ Integrate testing into development workflow
- ✅ Ensure system quality and reliability

For questions or issues, refer to the related guides or review the system logs in the `logs/` directory.

---

**Document Version**: 1.0  
**Last Updated**: October 21, 2025  
**Maintained By**: EMR Alert System Development Team

