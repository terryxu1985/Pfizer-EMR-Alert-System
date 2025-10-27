#!/usr/bin/env python
"""
Version History Management Tool

Provides commands to view, manage, and analyze data version history.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from version_manager import VersionManager, VersionChangeType


class VersionHistoryManager:
    """Manage and display version history"""
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        self.version_manager = VersionManager(base_dir)
    
    def show_current_version(self):
        """Display current version information"""
        version_info = self.version_manager.get_version_info()
        
        print("=" * 80)
        print("Current Data Version Information")
        print("=" * 80)
        
        if version_info.get('current_version'):
            print(f"Current Version: {version_info['current_version']}")
            print(f"Previous Version: {version_info.get('previous_version', 'N/A')}")
            print(f"Change Type: {version_info.get('change_type', 'N/A')}")
            print(f"Version Timestamp: {version_info.get('version_timestamp', 'N/A')}")
            print(f"Total Versions: {version_info.get('total_versions', 0)}")
        else:
            print("❌ No version information found")
            return
        
        # Show recent changes
        recent_changes = version_info.get('recent_changes', [])
        if recent_changes:
            print(f"\n📋 Recent Version History:")
            for change in recent_changes[-5:]:  # Show last 5 changes
                print(f"  {change['from_version']} → {change['to_version']} ({change['change_type']})")
                print(f"    {change['timestamp']}")
                print(f"    {change['summary']}")
                print()
    
    def show_full_history(self):
        """Display complete version history"""
        history = self.version_manager.load_version_history()
        
        print("=" * 80)
        print("Complete Data Version History")
        print("=" * 80)
        
        if not history:
            print("No version history found")
            return
        
        for i, change in enumerate(history, 1):
            print(f"{i:2d}. {change['from_version']} → {change['to_version']} ({change['change_type']})")
            print(f"    Timestamp: {change['timestamp']}")
            print(f"    Summary: {change['summary']}")
            
            # Show detailed changes
            changes = change.get('changes', {})
            if changes.get('new_files'):
                print(f"    New files: {', '.join(changes['new_files'][:3])}")
                if len(changes['new_files']) > 3:
                    print(f"    ... and {len(changes['new_files']) - 3} more")
            
            if changes.get('modified_files'):
                print(f"    Modified: {', '.join(changes['modified_files'][:3])}")
                if len(changes['modified_files']) > 3:
                    print(f"    ... and {len(changes['modified_files']) - 3} more")
            
            if changes.get('deleted_files'):
                print(f"    Deleted: {', '.join(changes['deleted_files'][:3])}")
                if len(changes['deleted_files']) > 3:
                    print(f"    ... and {len(changes['deleted_files']) - 3} more")
            
            print()
    
    def show_version_stats(self):
        """Display version statistics"""
        history = self.version_manager.load_version_history()
        version_info = self.version_manager.get_version_info()
        
        print("=" * 80)
        print("Version Statistics")
        print("=" * 80)
        
        if not history:
            print("No version history available for statistics")
            return
        
        # Count change types
        change_types = {}
        for change in history:
            change_type = change['change_type']
            change_types[change_type] = change_types.get(change_type, 0) + 1
        
        print(f"Total Versions: {len(history) + 1}")
        print(f"Current Version: {version_info.get('current_version', 'N/A')}")
        print("\nChange Type Distribution:")
        for change_type, count in change_types.items():
            print(f"  {change_type}: {count}")
        
        # Calculate time between versions
        if len(history) > 1:
            print(f"\nVersion Frequency:")
            time_diffs = []
            for i in range(1, len(history)):
                prev_time = datetime.fromisoformat(history[i-1]['timestamp'].replace('Z', '+00:00'))
                curr_time = datetime.fromisoformat(history[i]['timestamp'].replace('Z', '+00:00'))
                diff_days = (curr_time - prev_time).days
                time_diffs.append(diff_days)
            
            if time_diffs:
                avg_days = sum(time_diffs) / len(time_diffs)
                print(f"  Average time between versions: {avg_days:.1f} days")
                print(f"  Shortest interval: {min(time_diffs)} days")
                print(f"  Longest interval: {max(time_diffs)} days")
    
    def compare_versions(self, version1: str, version2: str):
        """Compare two specific versions"""
        print(f"Comparing versions {version1} and {version2}")
        print("=" * 80)
        
        # This would require loading historical manifests
        # For now, show a placeholder
        print("⚠️  Version comparison feature coming soon")
        print("   This will show detailed differences between versions")
    
    def export_history(self, output_file: str):
        """Export version history to file"""
        history = self.version_manager.load_version_history()
        
        export_data = {
            "export_timestamp": datetime.now().isoformat() + "Z",
            "total_versions": len(history) + 1,
            "version_history": history
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Version history exported to: {output_file}")
        print(f"   Exported {len(history)} version changes")


def main():
    """Main execution with command line interface"""
    parser = argparse.ArgumentParser(description='Manage data version history')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Current version command
    subparsers.add_parser('current', help='Show current version information')
    
    # History command
    subparsers.add_parser('history', help='Show complete version history')
    
    # Stats command
    subparsers.add_parser('stats', help='Show version statistics')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('version1', help='First version to compare')
    compare_parser.add_argument('version2', help='Second version to compare')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export version history')
    export_parser.add_argument('output_file', help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = VersionHistoryManager()
    
    if args.command == 'current':
        manager.show_current_version()
    elif args.command == 'history':
        manager.show_full_history()
    elif args.command == 'stats':
        manager.show_version_stats()
    elif args.command == 'compare':
        manager.compare_versions(args.version1, args.version2)
    elif args.command == 'export':
        manager.export_history(args.output_file)


if __name__ == "__main__":
    main()
