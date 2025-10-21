#!/bin/bash

# Pfizer EMR Alert System - Complete System Startup Script
# Starts both API and UI servers with enhanced doctor input functionality

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_PORT=8000
UI_PORT=8080
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Process IDs for cleanup
API_PID=""
UI_PID=""

# Functions
print_banner() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}🏥 Pfizer EMR Alert System - Complete System${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}🚀 Starting AI-powered clinical decision support system...${NC}"
    echo -e "${GREEN}📋 Features:${NC}"
    echo -e "   • Multi-step doctor data input"
    echo -e "   • Real-time AI analysis"
    echo -e "   • Smart medication alerts"
    echo -e "   • Enhanced UI/UX"
    echo -e "${BLUE}================================================${NC}"
}

check_dependencies() {
    echo -e "${BLUE}🔍 Checking dependencies...${NC}"
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is not installed${NC}"
        exit 1
    fi
    
    # Check if required Python packages are installed
    if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
        echo -e "${RED}❌ Required Python packages not found${NC}"
        echo -e "${YELLOW}Please install requirements: pip install -r config/requirements.txt${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Dependencies check passed${NC}"
}

check_files() {
    echo -e "${BLUE}🔍 Checking required files...${NC}"
    
    # Check if API file exists
    if [ ! -f "${PROJECT_ROOT}/backend/api/api.py" ]; then
        echo -e "${RED}❌ API file not found: backend/api/api.py${NC}"
        exit 1
    fi
    
    # Check if UI server exists
    if [ ! -f "${PROJECT_ROOT}/frontend/server/emr_ui_server.py" ]; then
        echo -e "${RED}❌ UI server not found: frontend/server/emr_ui_server.py${NC}"
        exit 1
    fi
    
    # Check if startup scripts exist
    if [ ! -f "${PROJECT_ROOT}/scripts/startup/run_api_only.py" ]; then
        echo -e "${RED}❌ API startup script not found: scripts/startup/run_api_only.py${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Required files found${NC}"
}

start_api_server() {
    echo -e "${BLUE}🔧 Starting API server...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    # Start the API server in background
    python3 scripts/startup/run_api_only.py &
    API_PID=$!
    
    # Wait a moment for server to start
    sleep 3
    
    # Check if server is running
    if kill -0 $API_PID 2>/dev/null; then
        echo -e "${GREEN}✅ API server started successfully on http://localhost:${API_PORT}${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to start API server${NC}"
        return 1
    fi
}

start_ui_server() {
    echo -e "${BLUE}🌐 Starting UI server...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    # Start the UI server in background
    python3 frontend/server/emr_ui_server.py &
    UI_PID=$!
    
    # Wait a moment for server to start
    sleep 2
    
    # Check if server is running
    if kill -0 $UI_PID 2>/dev/null; then
        echo -e "${GREEN}✅ UI server started successfully on http://localhost:${UI_PORT}${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to start UI server${NC}"
        return 1
    fi
}

open_browser() {
    echo -e "${BLUE}🌍 Opening application in browser...${NC}"
    
    # Try to open browser (works on macOS, Linux with GUI)
    if command -v open &> /dev/null; then
        # macOS
        open "http://localhost:${UI_PORT}" 2>/dev/null || true
    elif command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open "http://localhost:${UI_PORT}" 2>/dev/null || true
    else
        echo -e "${YELLOW}⚠️ Could not open browser automatically${NC}"
    fi
    
    echo -e "${GREEN}✅ Browser opened (if available)${NC}"
}

show_access_info() {
    echo -e "\n${GREEN}🎉 System is ready!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}📱 Access URLs:${NC}"
    echo -e "   • UI Server: http://localhost:${UI_PORT}"
    echo -e "   • API Server: http://localhost:${API_PORT}"
    echo -e "   • API Documentation: http://localhost:${API_PORT}/docs"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}📋 How to use:${NC}"
    echo -e "1. Click 'Add Patient' to enter new patient data"
    echo -e "2. Fill out the multi-step form with patient information"
    echo -e "3. Review AI analysis preview"
    echo -e "4. Save patient and run AI analysis"
    echo -e "5. Review medication alerts and recommendations"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}⚠️ Press Ctrl+C to stop the system${NC}"
    echo -e "${BLUE}================================================${NC}"
}

cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down system...${NC}"
    
    # Terminate API server
    if [ ! -z "$API_PID" ] && kill -0 $API_PID 2>/dev/null; then
        kill $API_PID 2>/dev/null || true
        echo -e "${GREEN}✅ API server stopped${NC}"
    fi
    
    # Terminate UI server
    if [ ! -z "$UI_PID" ] && kill -0 $UI_PID 2>/dev/null; then
        kill $UI_PID 2>/dev/null || true
        echo -e "${GREEN}✅ UI server stopped${NC}"
    fi
    
    # Wait a moment for cleanup
    sleep 1
    
    echo -e "${GREEN}👋 System shutdown complete${NC}"
}

# Set up signal handlers for cleanup
trap cleanup EXIT INT TERM

# Main execution
main() {
    print_banner
    check_dependencies
    check_files
    
    # Start API server
    if ! start_api_server; then
        exit 1
    fi
    
    # Start UI server
    if ! start_ui_server; then
        cleanup
        exit 1
    fi
    
    # Open browser
    open_browser
    
    # Show access information
    show_access_info
    
    # Keep the script running
    while true; do
        sleep 1
    done
}

# Run main function
main "$@"
