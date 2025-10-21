#!/usr/bin/env python3
"""
Legacy script for checking model features - now uses ModelValidator
This script is maintained for backward compatibility.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.model_validation.model_validator import ModelValidator

def main():
    """Check model features using the new ModelValidator"""
    print("Checking model features...")
    
    # Initialize validator
    validator = ModelValidator()
    
    # Test data columns (from original script)
    test_data_columns = [
        "patient_age", "patient_gender", "risk_immuno", "risk_cvd",
        "risk_diabetes", "risk_obesity", "risk_num", "risk_age_flag",
        "phys_total_dx", "phys_experience_level", "physician_state",
        "physician_type", "sym_count_5d", "dx_season", "location_type",
        "insurance_type_at_dx", "symptom_to_diagnosis_days",
        "diagnosis_within_5days_flag", "prior_contra_lvl"
    ]
    
    # Generate validation report
    report = validator.generate_validation_report(test_columns=test_data_columns)
    
    # Print detailed feature information
    print("\n" + "="*50)
    print("FEATURE COMPARISON")
    print("="*50)
    
    fv = report['feature_validation']
    print(f"Model features ({fv['model_feature_count']}):")
    for i, feature in enumerate(fv['model_features'], 1):
        print(f"  {i:2d}. {feature}")
    
    print(f"\nTest data features ({fv['test_feature_count']}):")
    for i, feature in enumerate(fv['test_features'], 1):
        print(f"  {i:2d}. {feature}")
    
    if fv['missing_features']:
        print(f"\n❌ Missing from test data ({len(fv['missing_features'])}):")
        for feature in fv['missing_features']:
            print(f"  - {feature}")
    
    if fv['extra_features']:
        print(f"\n⚠️  Extra in test data ({len(fv['extra_features'])}):")
        for feature in fv['extra_features']:
            print(f"  - {feature}")
    
    if fv['is_consistent']:
        print("\n✅ Feature consistency: PASSED")
    else:
        print("\n❌ Feature consistency: FAILED")
    
    return 0 if fv['is_consistent'] else 1

if __name__ == "__main__":
    sys.exit(main())
