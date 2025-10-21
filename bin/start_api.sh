#!/bin/bash

# Pfizer EMR Alert System - API Server Startup Script
# Starts only the FastAPI backend service

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_PORT=8000
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Functions
print_banner() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}🏥 Pfizer EMR Alert System - API Server${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}🚀 Starting FastAPI backend service...${NC}"
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
    
    # Check if run_api_only.py exists
    if [ ! -f "${PROJECT_ROOT}/scripts/startup/run_api_only.py" ]; then
        echo -e "${RED}❌ API startup script not found: scripts/startup/run_api_only.py${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Required files found${NC}"
}

start_api_server() {
    echo -e "${BLUE}🚀 Starting API server...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    # Start the API server
    echo -e "${YELLOW}Starting FastAPI server on port ${API_PORT}...${NC}"
    python3 scripts/startup/run_api_only.py
}

show_access_info() {
    echo -e "\n${GREEN}🎉 API Server is ready!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}📡 API Server: http://localhost:${API_PORT}${NC}"
    echo -e "${YELLOW}📚 API Documentation: http://localhost:${API_PORT}/docs${NC}"
    echo -e "${YELLOW}🔍 Health Check: http://localhost:${API_PORT}/health${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}📋 Available Endpoints:${NC}"
    echo -e "   • GET  /health - Health check"
    echo -e "   • GET  / - Root endpoint"
    echo -e "   • POST /predict - AI prediction"
    echo -e "   • GET  /docs - Interactive API docs"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}⚠️ Press Ctrl+C to stop the server${NC}"
    echo -e "${BLUE}================================================${NC}"
}

# Main execution
main() {
    print_banner
    check_dependencies
    check_files
    
    # Show access info before starting
    show_access_info
    
    # Start the API server
    start_api_server
}

# Run main function
main "$@"
