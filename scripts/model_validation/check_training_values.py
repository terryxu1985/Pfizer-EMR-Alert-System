#!/usr/bin/env python3
"""
Legacy script for checking training values - now uses ModelValidator
This script is maintained for backward compatibility.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.model_validation.model_validator import ModelValidator

def main():
    """Check training values using the new ModelValidator"""
    print("Checking training values...")
    
    # Initialize validator
    validator = ModelValidator()
    
    # Get training values
    training_values = validator.validate_training_values()
    
    print("\n" + "="*50)
    print("LABEL ENCODERS CLASSES")
    print("="*50)
    
    for col, values in training_values.items():
        print(f"\n{col}:")
        print(f"  Total classes: {len(values)}")
        if len(values) <= 20:  # Show all values if not too many
            for i, value in enumerate(values, 1):
                print(f"  {i:2d}. {value}")
        else:  # Show first few and last few
            for i, value in enumerate(values[:10], 1):
                print(f"  {i:2d}. {value}")
            print(f"  ... ({len(values) - 20} more values) ...")
            for i, value in enumerate(values[-10:], len(values) - 9):
                print(f"  {i:2d}. {value}")
    
    print(f"\n✅ Found {len(training_values)} categorical columns with label encoders")
    return 0

if __name__ == "__main__":
    sys.exit(main())
