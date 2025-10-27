#!/usr/bin/env python
"""
Create Data Manifest for Version Control

Generates DATA_VERSIONS.json with metadata about all data files,
including MD5 checksums, row counts, and compatibility information.
"""

import os
import json
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from version_manager import VersionManager, VersionChangeType


class DataManifestCreator:
    """Create data version manifest with metadata"""
    
    def __init__(self, base_dir: Path = None, auto_version: bool = True):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.auto_version = auto_version
        self.version_manager = VersionManager(base_dir) if auto_version else None
        
    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file metadata including MD5 and statistics"""
        info = {
            "file_size_bytes": file_path.stat().st_size,
            "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "md5": self.calculate_md5(file_path)
        }
        
        # Try to get row counts for CSV/Excel files
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, nrows=0)  # Just read header
                info["columns"] = len(df.columns)
                # Count rows efficiently
                with open(file_path, 'r') as f:
                    info["rows"] = sum(1 for _ in f) - 1  # Subtract header
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=0)
                info["columns"] = len(df.columns)
                # For Excel, read full file (smaller files only)
                if file_path.stat().st_size < 10 * 1024 * 1024:  # < 10MB
                    df = pd.read_excel(file_path)
                    info["rows"] = len(df)
                else:
                    info["rows"] = "unknown (large file)"
        except Exception as e:
            info["rows"] = f"error: {str(e)}"
            info["columns"] = "unknown"
        
        return info
    
    def scan_data_directory(self) -> Dict[str, Any]:
        """Scan data directory and collect metadata"""
        manifest = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat() + "Z",
            "data_pipeline": {
                "raw": {},
                "processed": {},
                "model_ready": {}
            },
            "statistics": {},
            "model_compatibility": {}
        }
        
        # Scan raw directory
        raw_dir = self.data_dir / "raw"
        if raw_dir.exists():
            for file_path in raw_dir.glob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    relative_path = file_path.relative_to(self.data_dir)
                    manifest["data_pipeline"]["raw"][relative_path.as_posix()] = \
                        self.get_file_info(file_path)
        
        # Scan processed directory
        processed_dir = self.data_dir / "processed"
        if processed_dir.exists():
            for file_path in processed_dir.glob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    relative_path = file_path.relative_to(self.data_dir)
                    file_info = self.get_file_info(file_path)
                    # Add source information
                    file_info["source"] = self._infer_source(file_path)
                    manifest["data_pipeline"]["processed"][relative_path.as_posix()] = file_info
        
        # Scan model_ready directory
        model_ready_dir = self.data_dir / "model_ready"
        if model_ready_dir.exists():
            for file_path in model_ready_dir.glob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    relative_path = file_path.relative_to(self.data_dir)
                    file_info = self.get_file_info(file_path)
                    if 'dataset' in file_path.name.lower():
                        file_info.update(self._get_dataset_metadata(file_path))
                    manifest["data_pipeline"]["model_ready"][relative_path.as_posix()] = file_info
        
        # Calculate statistics
        manifest["statistics"] = self._calculate_statistics(manifest)
        
        # Add model compatibility information
        manifest["model_compatibility"] = self._get_model_compatibility()
        
        return manifest
    
    def _infer_source(self, file_path: Path) -> str:
        """Infer source file for processed data"""
        if 'patient' in file_path.name.lower():
            return "dim_patient.xlsx"
        elif 'physician' in file_path.name.lower():
            return "dim_physician.xlsx"
        elif 'txn' in file_path.name.lower() or 'transaction' in file_path.name.lower():
            return "fact_txn.xlsx"
        return "unknown"
    
    def _get_dataset_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get additional metadata for model-ready datasets"""
        metadata = {}
        try:
            df = pd.read_csv(file_path)
            
            # Get feature information
            if 'TARGET' in df.columns:
                target_counts = df['TARGET'].value_counts().to_dict()
                metadata["target_distribution"] = {
                    str(k): int(v) for k, v in target_counts.items()
                }
                metadata["class_imbalance_ratio"] = round(
                    target_counts.get(0, 0) / target_counts.get(1, 1), 2
                )
            
            # Get feature counts
            metadata["feature_count"] = len(df.columns) - 1 if 'TARGET' in df.columns else len(df.columns)
            metadata["total_samples"] = len(df)
            
            # Get missing data statistics
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)
            metadata["missing_data_pct"] = {
                col: float(pct) for col, pct in missing_pct.items() 
                if pct > 0
            }
            
        except Exception as e:
            metadata["error"] = str(e)
        
        return metadata
    
    def _calculate_statistics(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall statistics"""
        stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "file_count_by_type": {}
        }
        
        for stage in ["raw", "processed", "model_ready"]:
            if stage in manifest["data_pipeline"]:
                stage_files = manifest["data_pipeline"][stage]
                stats["total_files"] += len(stage_files)
                
                for file_info in stage_files.values():
                    size_mb = file_info.get("file_size_bytes", 0) / (1024 * 1024)
                    stats["total_size_mb"] += size_mb
                    
                    # Count by file extension
                    ext = file_info.get("extension", "unknown")
                    stats["file_count_by_type"][ext] = stats["file_count_by_type"].get(ext, 0) + 1
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        
        return stats
    
    def _get_model_compatibility(self) -> Dict[str, Any]:
        """Get model compatibility information"""
        # Check available models
        model_dir = self.base_dir / "backend" / "ml_models" / "models"
        
        compatibility = {
            "supported_models": [],
            "current_model_version": "2.1.0",
            "data_requirements": {
                "features": 33,
                "target_column": "TARGET",
                "categorical_columns": [
                    "PATIENT_GENDER", "PHYS_EXPERIENCE_LEVEL",
                    "PHYSICIAN_STATE", "PHYSICIAN_TYPE", "DX_SEASON",
                    "LOCATION_TYPE", "INSURANCE_TYPE_AT_DX"
                ]
            }
        }
        
        if model_dir.exists():
            for model_file in model_dir.glob("*.pkl"):
                if "xgboost" in model_file.name:
                    compatibility["supported_models"].append(model_file.name)
        
        return compatibility
    
    def create_manifest(self, output_file: Path = None, 
                       force_change_type: Optional[VersionChangeType] = None) -> Dict[str, Any]:
        """Create data manifest with optional automatic version increment"""
        if output_file is None:
            output_file = self.base_dir / "data" / ".metadata" / "DATA_VERSIONS.json"
        
        manifest = self.scan_data_directory()
        
        # Handle automatic versioning
        if self.auto_version and self.version_manager:
            try:
                old_version, new_version, changes = self.version_manager.auto_increment_version(
                    manifest, force_change_type
                )
                
                print(f"🔄 Version automatically incremented:")
                print(f"   {old_version} → {new_version}")
                print(f"   Change type: {changes.get('change_type', 'unknown')}")
                
                if changes.get('new_files'):
                    print(f"   New files: {len(changes['new_files'])}")
                if changes.get('modified_files'):
                    print(f"   Modified files: {len(changes['modified_files'])}")
                if changes.get('deleted_files'):
                    print(f"   Deleted files: {len(changes['deleted_files'])}")
                
            except Exception as e:
                print(f"⚠️  Auto-versioning failed: {e}")
                print("   Proceeding with manual version...")
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Created data manifest: {output_file}")
        print(f"   Version: {manifest['version']}")
        print(f"   Total files: {manifest['statistics']['total_files']}")
        print(f"   Total size: {manifest['statistics']['total_size_mb']} MB")
        
        return manifest


def main():
    """Main execution with command line support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create data version manifest with automatic versioning')
    parser.add_argument('--no-auto-version', action='store_true', 
                       help='Disable automatic version increment')
    parser.add_argument('--force-change-type', choices=['major', 'minor', 'patch'],
                       help='Force a specific change type for version increment')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Creating Data Version Manifest")
    if not args.no_auto_version:
        print("🔄 Automatic versioning enabled")
    print("=" * 80)
    
    # Determine change type
    force_change_type = None
    if args.force_change_type:
        force_change_type = VersionChangeType(args.force_change_type)
        print(f"🔧 Forcing change type: {force_change_type.value}")
    
    creator = DataManifestCreator(auto_version=not args.no_auto_version)
    manifest = creator.create_manifest(force_change_type=force_change_type)
    
    print("\n✅ Data manifest created successfully!")
    print(f"   Location: data/.metadata/DATA_VERSIONS.json")
    print(f"   MD5 checksums calculated for {manifest['statistics']['total_files']} files")
    
    # Show version history if available
    if creator.version_manager:
        version_info = creator.version_manager.get_version_info()
        if version_info.get('recent_changes'):
            print(f"\n📋 Recent version history:")
            for change in version_info['recent_changes'][-2:]:  # Show last 2 changes
                print(f"   {change['from_version']} → {change['to_version']} ({change['change_type']})")


if __name__ == "__main__":
    main()
