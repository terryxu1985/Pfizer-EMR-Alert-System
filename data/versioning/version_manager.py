#!/usr/bin/env python
"""
Automatic Version Management for Data Versioning

Provides automatic version increment functionality with change detection
and semantic versioning support.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum


class VersionChangeType(Enum):
    """Types of version changes"""
    PATCH = "patch"  # Bug fixes, data cleaning
    MINOR = "minor"  # New records, backward compatible changes
    MAJOR = "major"  # Breaking changes, structure changes


class VersionManager:
    """Manages automatic version incrementing for data versioning"""
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.manifest_file = self.data_dir / ".metadata" / "DATA_VERSIONS.json"
        self.version_history_file = self.data_dir / ".metadata" / "VERSION_HISTORY.json"
    
    def load_manifest(self) -> Dict[str, Any]:
        """Load current data manifest"""
        if not self.manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_file}")
        
        with open(self.manifest_file, 'r') as f:
            return json.load(f)
    
    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save updated manifest"""
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def parse_version(self, version_str: str) -> Tuple[int, int, int]:
        """Parse semantic version string into major.minor.patch"""
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    
    def format_version(self, major: int, minor: int, patch: int) -> str:
        """Format version components into semantic version string"""
        return f"{major}.{minor}.{patch}"
    
    def increment_version(self, current_version: str, change_type: VersionChangeType) -> str:
        """Increment version based on change type"""
        major, minor, patch = self.parse_version(current_version)
        
        if change_type == VersionChangeType.MAJOR:
            return self.format_version(major + 1, 0, 0)
        elif change_type == VersionChangeType.MINOR:
            return self.format_version(major, minor + 1, 0)
        else:  # PATCH
            return self.format_version(major, minor, patch + 1)
    
    def detect_change_type(self, old_manifest: Dict[str, Any], new_manifest: Dict[str, Any]) -> VersionChangeType:
        """Detect the type of changes between manifests"""
        changes = self._analyze_changes(old_manifest, new_manifest)
        
        # Major changes: structure changes, new tables, deleted fields
        if changes['structure_changes'] or changes['deleted_files'] or changes['schema_changes']:
            return VersionChangeType.MAJOR
        
        # Minor changes: new records, new files, backward compatible changes
        if changes['new_files'] or changes['record_count_changes']:
            return VersionChangeType.MINOR
        
        # Patch changes: data cleaning, bug fixes, metadata updates
        return VersionChangeType.PATCH
    
    def _analyze_changes(self, old_manifest: Dict[str, Any], new_manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze changes between two manifests"""
        changes = {
            'new_files': [],
            'deleted_files': [],
            'modified_files': [],
            'structure_changes': False,
            'schema_changes': False,
            'record_count_changes': False,
            'file_size_changes': []
        }
        
        old_files = self._get_all_files(old_manifest)
        new_files = self._get_all_files(new_manifest)
        
        # Find new files
        changes['new_files'] = list(set(new_files.keys()) - set(old_files.keys()))
        
        # Find deleted files
        changes['deleted_files'] = list(set(old_files.keys()) - set(new_files.keys()))
        
        # Find modified files
        for file_path in set(old_files.keys()) & set(new_files.keys()):
            old_info = old_files[file_path]
            new_info = new_files[file_path]
            
            if old_info.get('md5') != new_info.get('md5'):
                changes['modified_files'].append(file_path)
                
                # Check for specific types of changes
                if old_info.get('columns') != new_info.get('columns'):
                    changes['schema_changes'] = True
                
                if old_info.get('rows') != new_info.get('rows'):
                    changes['record_count_changes'] = True
                
                if old_info.get('file_size_bytes') != new_info.get('file_size_bytes'):
                    changes['file_size_changes'].append({
                        'file': file_path,
                        'old_size': old_info.get('file_size_bytes'),
                        'new_size': new_info.get('file_size_bytes')
                    })
        
        # Check for structure changes in model compatibility
        old_compat = old_manifest.get('model_compatibility', {})
        new_compat = new_manifest.get('model_compatibility', {})
        
        if old_compat.get('data_requirements', {}).get('features') != new_compat.get('data_requirements', {}).get('features'):
            changes['structure_changes'] = True
        
        return changes
    
    def _get_all_files(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all files from manifest"""
        all_files = {}
        data_pipeline = manifest.get('data_pipeline', {})
        
        for stage in ['raw', 'processed', 'model_ready']:
            if stage in data_pipeline:
                all_files.update(data_pipeline[stage])
        
        return all_files
    
    def create_version_history_entry(self, old_version: str, new_version: str, 
                                   change_type: VersionChangeType, 
                                   changes: Dict[str, Any]) -> Dict[str, Any]:
        """Create a version history entry"""
        return {
            "from_version": old_version,
            "to_version": new_version,
            "change_type": change_type.value,
            "timestamp": datetime.now().isoformat() + "Z",
            "changes": changes,
            "summary": self._generate_change_summary(changes)
        }
    
    def _generate_change_summary(self, changes: Dict[str, Any]) -> str:
        """Generate a human-readable summary of changes"""
        summary_parts = []
        
        if changes['new_files']:
            summary_parts.append(f"Added {len(changes['new_files'])} new files")
        
        if changes['deleted_files']:
            summary_parts.append(f"Removed {len(changes['deleted_files'])} files")
        
        if changes['modified_files']:
            summary_parts.append(f"Modified {len(changes['modified_files'])} files")
        
        if changes['schema_changes']:
            summary_parts.append("Schema changes detected")
        
        if changes['record_count_changes']:
            summary_parts.append("Record count changes")
        
        return "; ".join(summary_parts) if summary_parts else "Minor updates"
    
    def load_version_history(self) -> List[Dict[str, Any]]:
        """Load version history"""
        if not self.version_history_file.exists():
            return []
        
        with open(self.version_history_file, 'r') as f:
            return json.load(f)
    
    def save_version_history(self, history: List[Dict[str, Any]]) -> None:
        """Save version history"""
        with open(self.version_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def auto_increment_version(self, new_manifest: Dict[str, Any], 
                             force_change_type: Optional[VersionChangeType] = None) -> Tuple[str, str, Dict[str, Any]]:
        """
        Automatically increment version based on changes detected
        
        Returns:
            Tuple of (old_version, new_version, changes_summary)
        """
        # Load current manifest
        try:
            old_manifest = self.load_manifest()
            old_version = old_manifest.get('version', '1.0.0')
        except FileNotFoundError:
            # First time - start with version 1.0.0
            old_version = '1.0.0'
            old_manifest = None
        
        # Determine change type
        if force_change_type:
            change_type = force_change_type
        elif old_manifest:
            change_type = self.detect_change_type(old_manifest, new_manifest)
        else:
            change_type = VersionChangeType.MAJOR  # Initial version
        
        # Increment version
        new_version = self.increment_version(old_version, change_type)
        
        # Analyze changes
        if old_manifest:
            changes = self._analyze_changes(old_manifest, new_manifest)
        else:
            changes = {'new_files': list(self._get_all_files(new_manifest).keys())}
        
        # Update manifest with new version
        new_manifest['version'] = new_version
        new_manifest['previous_version'] = old_version
        new_manifest['version_change_type'] = change_type.value
        new_manifest['version_timestamp'] = datetime.now().isoformat() + "Z"
        
        # Save updated manifest
        self.save_manifest(new_manifest)
        
        # Update version history
        if old_manifest:
            history = self.load_version_history()
            history_entry = self.create_version_history_entry(
                old_version, new_version, change_type, changes
            )
            history.append(history_entry)
            self.save_version_history(history)
        
        return old_version, new_version, changes
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get current version information"""
        try:
            manifest = self.load_manifest()
            history = self.load_version_history()
            
            return {
                "current_version": manifest.get('version'),
                "previous_version": manifest.get('previous_version'),
                "change_type": manifest.get('version_change_type'),
                "version_timestamp": manifest.get('version_timestamp'),
                "total_versions": len(history) + 1,
                "recent_changes": history[-3:] if history else []
            }
        except FileNotFoundError:
            return {"current_version": None, "error": "No manifest found"}


def main():
    """Main execution for testing"""
    print("=" * 80)
    print("Data Version Manager - Auto Increment Test")
    print("=" * 80)
    
    manager = VersionManager()
    
    # Test version parsing and incrementing
    test_versions = ["1.0.0", "1.2.3", "2.0.0"]
    
    print("\nTesting version increment logic:")
    for version in test_versions:
        print(f"\nCurrent version: {version}")
        for change_type in VersionChangeType:
            new_version = manager.increment_version(version, change_type)
            print(f"  {change_type.value:>6} -> {new_version}")
    
    # Test version info
    print("\nCurrent version information:")
    version_info = manager.get_version_info()
    for key, value in version_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
