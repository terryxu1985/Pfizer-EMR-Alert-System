#!/usr/bin/env python
"""
Data Version Management Automation Script

Provides convenient commands for common data versioning tasks.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from version_manager import VersionManager, VersionChangeType


def run_command(cmd: list, description: str):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def update_data_version(change_type: str = None, force: bool = False):
    """Update data version with automatic change detection"""
    print("=" * 80)
    print("Data Version Update")
    print("=" * 80)
    
    # Step 1: Generate new manifest with auto-versioning
    cmd = ["python", "manifest.py"]
    if change_type:
        cmd.extend(["--force-change-type", change_type])
    
    if not run_command(cmd, "Generating data manifest with version increment"):
        return False
    
    # Step 2: Verify data integrity
    if not run_command(["python", "verify.py"], "Verifying data integrity"):
        return False
    
    # Step 3: Check model compatibility
    if not run_command(["python", "compatibility.py"], "Checking model compatibility"):
        return False
    
    print("\n✅ Data version update completed successfully!")
    return True


def show_version_status():
    """Show current version status"""
    print("=" * 80)
    print("Data Version Status")
    print("=" * 80)
    
    # Show current version info
    if not run_command(["python", "history.py", "current"], "Getting current version"):
        return False
    
    # Show recent history
    print("\n" + "=" * 40)
    if not run_command(["python", "history.py", "history"], "Getting version history"):
        return False
    
    return True


def create_git_tag():
    """Create Git tag for current version"""
    manager = VersionManager()
    version_info = manager.get_version_info()
    
    if not version_info.get('current_version'):
        print("❌ No current version found")
        return False
    
    current_version = version_info['current_version']
    tag_name = f"data-v{current_version}"
    
    print(f"🔄 Creating Git tag: {tag_name}")
    
    try:
        # Create annotated tag
        subprocess.run([
            "git", "tag", "-a", tag_name, 
            "-m", f"Data version {current_version}"
        ], check=True)
        
        print(f"✅ Git tag created: {tag_name}")
        print(f"   Use 'git push origin {tag_name}' to push the tag")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create Git tag: {e}")
        return False


def setup_git_hooks():
    """Setup Git hooks for automatic versioning"""
    print("=" * 80)
    print("Setting up Git Hooks")
    print("=" * 80)
    
    hooks_dir = Path(".git/hooks")
    if not hooks_dir.exists():
        print("❌ Not in a Git repository")
        return False
    
    # Pre-commit hook
    pre_commit_hook = hooks_dir / "pre-commit"
    hook_content = """#!/bin/bash
# Data integrity check before commit
echo "Checking data integrity..."
python data/versioning/verify.py
if [ $? -ne 0 ]; then
    echo "❌ Data integrity check failed!"
    echo "   Please run 'python data/versioning/manifest.py' to update manifest"
    exit 1
fi
echo "✅ Data integrity check passed"
"""
    
    try:
        with open(pre_commit_hook, 'w') as f:
            f.write(hook_content)
        
        # Make executable
        pre_commit_hook.chmod(0o755)
        
        print("✅ Pre-commit hook installed")
        print("   Data integrity will be checked before each commit")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup Git hooks: {e}")
        return False


def main():
    """Main execution with command line interface"""
    parser = argparse.ArgumentParser(description='Data version management automation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update data version')
    update_parser.add_argument('--change-type', choices=['major', 'minor', 'patch'],
                              help='Force specific change type')
    update_parser.add_argument('--force', action='store_true',
                              help='Force update even if no changes detected')
    
    # Status command
    subparsers.add_parser('status', help='Show version status')
    
    # Tag command
    subparsers.add_parser('tag', help='Create Git tag for current version')
    
    # Setup command
    subparsers.add_parser('setup', help='Setup Git hooks')
    
    # Quick update command
    subparsers.add_parser('quick', help='Quick version update (patch)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Change to versioning directory
    versioning_dir = Path(__file__).parent
    import os
    os.chdir(versioning_dir)
    
    if args.command == 'update':
        update_data_version(args.change_type, args.force)
    elif args.command == 'status':
        show_version_status()
    elif args.command == 'tag':
        create_git_tag()
    elif args.command == 'setup':
        setup_git_hooks()
    elif args.command == 'quick':
        update_data_version('patch')


if __name__ == "__main__":
    main()
