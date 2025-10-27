#!/usr/bin/env python
"""
Verify Data Integrity

Validates data files against their MD5 checksums and checks for consistency.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple


class DataIntegrityVerifier:
    """Verify data integrity against manifest"""
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.manifest_file = self.data_dir / ".metadata" / "DATA_VERSIONS.json"
    
    def load_manifest(self) -> Dict[str, Any]:
        """Load data manifest"""
        if not self.manifest_file.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {self.manifest_file}\n"
                "Please run scripts/data_versioning/create_data_manifest.py first"
            )
        
        with open(self.manifest_file, 'r') as f:
            return json.load(f)
    
    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except FileNotFoundError:
            return None
    
    def verify_file(self, relative_path: str, expected_md5: str) -> Tuple[bool, str]:
        """Verify a single file"""
        file_path = self.data_dir / relative_path
        
        if not file_path.exists():
            return False, f"File not found: {relative_path}"
        
        actual_md5 = self.calculate_md5(file_path)
        
        if actual_md5 is None:
            return False, f"Could not calculate MD5: {relative_path}"
        
        if actual_md5 != expected_md5:
            return False, f"MD5 mismatch: expected {expected_md5[:8]}..., got {actual_md5[:8]}..."
        
        return True, "OK"
    
    def verify_all_files(self) -> Dict[str, Any]:
        """Verify all files in manifest"""
        manifest = self.load_manifest()
        
        results = {
            "timestamp": datetime.now().isoformat() + "Z",
            "manifest_version": manifest.get("version"),
            "files_checked": 0,
            "files_ok": 0,
            "files_failed": 0,
            "failed_files": [],
            "details": {}
        }
        
        # Check all files in manifest
        for stage in ["raw", "processed", "model_ready"]:
            if stage not in manifest.get("data_pipeline", {}):
                continue
            
            for relative_path, file_info in manifest["data_pipeline"][stage].items():
                results["files_checked"] += 1
                
                expected_md5 = file_info.get("md5")
                if not expected_md5:
                    results["files_failed"] += 1
                    results["failed_files"].append({
                        "file": relative_path,
                        "reason": "No MD5 in manifest"
                    })
                    results["details"][relative_path] = "No MD5 in manifest"
                    continue
                
                is_valid, message = self.verify_file(relative_path, expected_md5)
                
                if is_valid:
                    results["files_ok"] += 1
                    results["details"][relative_path] = "✓ OK"
                else:
                    results["files_failed"] += 1
                    results["failed_files"].append({
                        "file": relative_path,
                        "reason": message
                    })
                    results["details"][relative_path] = f"✗ {message}"
        
        results["integrity_ok"] = results["files_failed"] == 0
        
        return results
    
    def print_report(self, results: Dict[str, Any]):
        """Print verification report"""
        print("\n" + "=" * 80)
        print("Data Integrity Verification Report")
        print("=" * 80)
        
        print(f"\nManifest Version: {results['manifest_version']}")
        print(f"Verification Timestamp: {results['timestamp']}")
        
        print(f"\nSummary:")
        print(f"  Files Checked: {results['files_checked']}")
        print(f"  Files OK: {results['files_ok']}")
        print(f"  Files Failed: {results['files_failed']}")
        
        if results['failed_files']:
            print(f"\n❌ Failed Files:")
            for failed in results['failed_files']:
                print(f"  - {failed['file']}")
                print(f"    Reason: {failed['reason']}")
        else:
            print("\n✅ All files verified successfully!")
        
        print("\nDetailed Results:")
        for file_path, status in results['details'].items():
            status_icon = "✓" if status == "✓ OK" else "✗"
            print(f"  {status_icon} {file_path}")
    
    def verify(self) -> bool:
        """Run verification and return success status"""
        results = self.verify_all_files()
        self.print_report(results)
        return results['integrity_ok']


def main():
    """Main execution"""
    print("Verifying data integrity...")
    
    verifier = DataIntegrityVerifier()
    
    try:
        is_valid = verifier.verify()
        
        if is_valid:
            print("\n✅ Data integrity verification passed!")
            exit(0)
        else:
            print("\n❌ Data integrity verification failed!")
            print("   Some files have been modified or are missing.")
            exit(1)
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
