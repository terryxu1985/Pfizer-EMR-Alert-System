#!/usr/bin/env python3
"""
Enhanced EMR Alert System Startup Script
This script starts the complete system with enhanced doctor input functionality
"""
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def print_banner():
    print("=" * 80)
    print("🏥 Pfizer EMR Alert System - Enhanced Doctor Input Version")
    print("=" * 80)
    print("🚀 Starting AI-powered clinical decision support system...")
    print("📋 Features:")
    print("   • Multi-step doctor data input")
    print("   • Real-time AI analysis")
    print("   • Smart medication alerts")
    print("   • Enhanced UI/UX")
    print("=" * 80)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI and Uvicorn are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r config/requirements.txt")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("\n🔧 Starting API server...")
    try:
        # Start the API server
        api_process = subprocess.Popen([
            sys.executable, "scripts/startup/run_api_only.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        if api_process.poll() is None:
            print("✅ API server started successfully on http://localhost:8000")
            return api_process
        else:
            stdout, stderr = api_process.communicate()
            print(f"❌ Failed to start API server: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def start_ui_server():
    """Start the UI server"""
    print("\n🌐 Starting UI server...")
    try:
        # Start the UI server
        ui_process = subprocess.Popen([
            sys.executable, "frontend/server/emr_ui_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Check if server is running
        if ui_process.poll() is None:
            print("✅ UI server started successfully on http://localhost:8080")
            return ui_process
        else:
            stdout, stderr = ui_process.communicate()
            print(f"❌ Failed to start UI server: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting UI server: {e}")
        return None

def open_browser():
    """Open the application in the browser"""
    print("\n🌍 Opening application in browser...")
    try:
        webbrowser.open("http://localhost:8080")
        print("✅ Browser opened successfully")
    except Exception as e:
        print(f"⚠️ Could not open browser automatically: {e}")
        print("Please manually open: http://localhost:8080")

def main():
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        return 1
    
    # Start UI server
    ui_process = start_ui_server()
    if not ui_process:
        api_process.terminate()
        return 1
    
    # Open browser
    open_browser()
    
    print("\n" + "=" * 80)
    print("🎉 System is ready!")
    print("📱 Access the application at: http://localhost:8080")
    print("🔗 API documentation at: http://localhost:8000/docs")
    print("=" * 80)
    print("\n📋 How to use:")
    print("1. Click 'Add Patient' to enter new patient data")
    print("2. Fill out the multi-step form with patient information")
    print("3. Review AI analysis preview")
    print("4. Save patient and run AI analysis")
    print("5. Review medication alerts and recommendations")
    print("\n⚠️ Press Ctrl+C to stop the system")
    print("=" * 80)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down system...")
        
        # Terminate processes
        if api_process:
            api_process.terminate()
            print("✅ API server stopped")
        
        if ui_process:
            ui_process.terminate()
            print("✅ UI server stopped")
        
        print("👋 System shutdown complete")
        return 0

if __name__ == "__main__":
    sys.exit(main())
