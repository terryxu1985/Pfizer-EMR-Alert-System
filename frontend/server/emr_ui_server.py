#!/usr/bin/env python3
"""
EMR Alert System UI Server
Serves the HTML UI for the Pfizer EMR Alert System
"""
import http.server
import socketserver
import webbrowser
import sys
from pathlib import Path

class EMRUIHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for EMR UI server with CORS support"""
    
    def do_GET(self):
        """Override GET to serve index.html for root path"""
        if self.path == '/' or self.path == '':
            self.path = '/templates/index.html'
        elif self.path.startswith('/static/'):
            # Keep static paths as is
            pass
        elif self.path.startswith('/templates/'):
            # Keep template paths as is
            pass
        else:
            # For any other path, try to serve from templates first
            if not self.path.startswith('/templates/'):
                self.path = '/templates' + self.path
        
        return super().do_GET()
    
    def end_headers(self):
        # Add CORS headers for API communication
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Custom log format for better readability"""
        print(f"🌐 UI Server: {format % args}")

def wait_for_api_server(api_url="http://localhost:8000/health", timeout=30):
    """Wait for API server to be ready"""
    try:
        import requests
    except ImportError:
        print("⚠️ requests library not available, skipping API health check")
        return True
    
    import time
    
    print("⏳ Waiting for API server to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                print("✅ API server is ready")
                return True
        except:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"\n⚠️ API server not ready after {timeout} seconds, starting UI anyway")
    return False

def start_ui_server(port=8080):
    """Start the UI server with enhanced error handling"""
    try:
        # Wait for API server to be ready
        wait_for_api_server()
        
        # Change to the frontend directory containing templates and static files
        ui_dir = Path(__file__).parent.parent  # Go up one level to frontend/
        
        # Validate directory structure
        templates_dir = ui_dir / "templates"
        static_dir = ui_dir / "static"
        
        if not templates_dir.exists():
            print(f"❌ Templates directory not found: {templates_dir}")
            return False
            
        if not static_dir.exists():
            print(f"❌ Static directory not found: {static_dir}")
            return False
        
        # Change to the ui directory
        import os
        os.chdir(ui_dir)
        
        with socketserver.TCPServer(("", port), EMRUIHandler) as httpd:
            print(f"🚀 EMR UI Server starting on port {port}")
            print(f"📁 Serving from: {ui_dir}")
            print(f"🌍 Access the application at: http://localhost:{port}")
            print(f"🔗 API server at: http://localhost:8000")
            print("⚠️ Press Ctrl+C to stop the server")
            print("=" * 60)
            
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Please try a different port.")
            print("💡 You can specify a different port by modifying the script.")
            # Try alternative port
            return start_ui_server(port + 1)
        else:
            print(f"❌ Error starting UI server: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 UI Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function to start the UI server"""
    print("🏥 Pfizer EMR Alert System - UI Server")
    print("=" * 60)
    
    # Check if HTML file exists
    ui_dir = Path(__file__).parent.parent  # Go up one level to frontend/
    html_file = ui_dir / "templates" / "index.html"
    if not html_file.exists():
        print(f"❌ HTML file not found: {html_file}")
        print("Please ensure index.html is in the frontend/templates/ directory.")
        return 1
    
    # Start the server
    success = start_ui_server()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
