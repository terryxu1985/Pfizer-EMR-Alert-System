#!/usr/bin/env python3
"""
Pfizer EMR Alert System - Quick Start Script
This is a convenience script that calls the main system launcher
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Main function that delegates to the system launcher"""
    print("🏥 Pfizer EMR Alert System - Quick Start")
    print("=" * 50)
    print("🚀 Launching the complete EMR system...")
    print("=" * 50)
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    launcher_script = script_dir / "run_complete_system.py"
    
    if not launcher_script.exists():
        print(f"❌ System launcher not found: {launcher_script}")
        print("Please ensure run_complete_system.py is in the same directory.")
        return 1
    
    # Run the main launcher
    try:
        result = subprocess.run([sys.executable, str(launcher_script)], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running system launcher: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n🛑 Startup cancelled by user")
        return 0

if __name__ == "__main__":
    sys.exit(main())
