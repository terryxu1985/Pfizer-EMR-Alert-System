#!/usr/bin/env python
"""
Check Model-Data Compatibility

Validates that model requirements match the current data version.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class CompatibilityChecker:
    """Check compatibility between models and data"""
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.model_dir = base_dir / "backend" / "ml_models" / "models"
    
    def load_data_manifest(self) -> Dict[str, Any]:
        """Load data manifest"""
        manifest_file = self.data_dir / ".metadata" / "DATA_VERSIONS.json"
        if not manifest_file.exists():
            return None
        
        with open(manifest_file, 'r') as f:
            return json.load(f)
    
    def load_model_metadata(self) -> Dict[str, Any]:
        """Load model metadata"""
        metadata_file = self.model_dir / "model_metadata.pkl"
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'rb') as f:
            return pickle.load(f)
    
    def get_data_features(self) -> Dict[str, Any]:
        """Get feature information from data manifest"""
        manifest = self.load_data_manifest()
        if not manifest:
            return None
        
        # Extract feature information from model_ready dataset
        model_ready_files = manifest.get("data_pipeline", {}).get("model_ready", {})
        
        for file_path, file_info in model_ready_files.items():
            if "dataset" in file_path.lower() and "csv" in file_path:
                return {
                    "feature_count": file_info.get("feature_count"),
                    "target_distribution": file_info.get("target_distribution"),
                    "total_samples": file_info.get("total_samples")
                }
        
        return None
    
    def check_compatibility(self) -> Dict[str, Any]:
        """Check model-data compatibility"""
        result = {
            "compatible": True,
            "data_version": None,
            "model_version": None,
            "issues": []
        }
        
        # Load data manifest
        data_manifest = self.load_data_manifest()
        if not data_manifest:
            result["compatible"] = False
            result["issues"].append("Data manifest not found")
            return result
        
        result["data_version"] = data_manifest.get("version")
        
        # Load model metadata
        model_metadata = self.load_model_metadata()
        if not model_metadata:
            result["compatible"] = False
            result["issues"].append("Model metadata not found")
            return result
        
        result["model_version"] = model_metadata.get("version", "unknown")
        
        # Get data features
        data_features = self.get_data_features()
        model_features = model_metadata.get("feature_count")
        
        # Check feature count
        if data_features and model_features:
            data_feature_count = data_features.get("feature_count")
            if data_feature_count != model_features:
                result["compatible"] = False
                result["issues"].append(
                    f"Feature count mismatch: data has {data_feature_count}, "
                    f"model expects {model_features}"
                )
        
        # Check categorical columns
        data_requirements = data_manifest.get("model_compatibility", {}).get("data_requirements", {})
        model_categorical = model_metadata.get("categorical_columns", [])
        data_categorical = data_requirements.get("categorical_columns", [])
        
        if model_categorical and data_categorical:
            if set(model_categorical) != set(data_categorical):
                result["compatible"] = False
                result["issues"].append(
                    f"Categorical column mismatch: "
                    f"model expects {set(model_categorical)}, "
                    f"data has {set(data_categorical)}"
                )
        
        return result
    
    def print_report(self, result: Dict[str, Any]):
        """Print compatibility report"""
        print("\n" + "=" * 80)
        print("Model-Data Compatibility Report")
        print("=" * 80)
        
        if result["data_version"]:
            print(f"\nData Version: {result['data_version']}")
        if result["model_version"]:
            print(f"Model Version: {result['model_version']}")
        
        if result["compatible"]:
            print("\n✅ Compatibility check passed!")
            print("   Model and data are compatible.")
        else:
            print("\n❌ Compatibility check failed!")
            print("\nIssues found:")
            for issue in result["issues"]:
                print(f"  - {issue}")
        
        return result["compatible"]


def main():
    """Main execution"""
    print("Checking model-data compatibility...")
    
    checker = CompatibilityChecker()
    result = checker.check_compatibility()
    is_compatible = checker.print_report(result)
    
    exit(0 if is_compatible else 1)


if __name__ == "__main__":
    main()
