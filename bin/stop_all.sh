#!/bin/bash

# Pfizer EMR Alert System - Stop All Services Script
# Stops all running services (API, UI, Docker containers)

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

# Functions
print_banner() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}🏥 Pfizer EMR Alert System - Stop All Services${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}🛑 Stopping all running services...${NC}"
    echo -e "${BLUE}================================================${NC}"
}

stop_local_services() {
    echo -e "${BLUE}🔍 Checking for local running services...${NC}"
    
    # Stop processes running on API port
    API_PIDS=$(lsof -ti:${API_PORT} 2>/dev/null || true)
    if [ ! -z "$API_PIDS" ]; then
        echo -e "${YELLOW}🛑 Stopping API server on port ${API_PORT}...${NC}"
        echo "$API_PIDS" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        # Force kill if still running
        echo "$API_PIDS" | xargs kill -KILL 2>/dev/null || true
        echo -e "${GREEN}✅ API server stopped${NC}"
    else
        echo -e "${GREEN}✅ No API server running on port ${API_PORT}${NC}"
    fi
    
    # Stop processes running on UI port
    UI_PIDS=$(lsof -ti:${UI_PORT} 2>/dev/null || true)
    if [ ! -z "$UI_PIDS" ]; then
        echo -e "${YELLOW}🛑 Stopping UI server on port ${UI_PORT}...${NC}"
        echo "$UI_PIDS" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        # Force kill if still running
        echo "$UI_PIDS" | xargs kill -KILL 2>/dev/null || true
        echo -e "${GREEN}✅ UI server stopped${NC}"
    else
        echo -e "${GREEN}✅ No UI server running on port ${UI_PORT}${NC}"
    fi
}

stop_docker_containers() {
    echo -e "${BLUE}🐳 Checking for Docker containers...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    # Check if docker-compose.yml exists
    if [ -f "docker-compose.yml" ]; then
        echo -e "${YELLOW}🛑 Stopping Docker containers...${NC}"
        
        # Stop all containers defined in docker-compose.yml
        docker-compose down 2>/dev/null || true
        
        # Stop any remaining containers with project name
        docker ps -q --filter "name=pfizer-emr" | xargs -r docker stop 2>/dev/null || true
        
        echo -e "${GREEN}✅ Docker containers stopped${NC}"
    else
        echo -e "${GREEN}✅ No docker-compose.yml found${NC}"
    fi
}

stop_python_processes() {
    echo -e "${BLUE}🐍 Checking for Python processes...${NC}"
    
    # Find Python processes related to the EMR system
    PYTHON_PIDS=$(ps aux | grep -E "(run_api_only|run_complete_system|emr_ui_server)" | grep -v grep | awk '{print $2}' || true)
    
    if [ ! -z "$PYTHON_PIDS" ]; then
        echo -e "${YELLOW}🛑 Stopping Python EMR processes...${NC}"
        echo "$PYTHON_PIDS" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        # Force kill if still running
        echo "$PYTHON_PIDS" | xargs kill -KILL 2>/dev/null || true
        echo -e "${GREEN}✅ Python EMR processes stopped${NC}"
    else
        echo -e "${GREEN}✅ No Python EMR processes found${NC}"
    fi
}

cleanup_ports() {
    echo -e "${BLUE}🧹 Cleaning up ports...${NC}"
    
    # Kill any remaining processes on our ports
    for port in ${API_PORT} ${UI_PORT}; do
        PIDS=$(lsof -ti:${port} 2>/dev/null || true)
        if [ ! -z "$PIDS" ]; then
            echo -e "${YELLOW}🛑 Force killing processes on port ${port}...${NC}"
            echo "$PIDS" | xargs kill -KILL 2>/dev/null || true
        fi
    done
    
    echo -e "${GREEN}✅ Port cleanup completed${NC}"
}

show_status() {
    echo -e "${BLUE}📊 Final status check...${NC}"
    
    # Check API port
    if lsof -ti:${API_PORT} >/dev/null 2>&1; then
        echo -e "${RED}❌ API port ${API_PORT} still in use${NC}"
    else
        echo -e "${GREEN}✅ API port ${API_PORT} is free${NC}"
    fi
    
    # Check UI port
    if lsof -ti:${UI_PORT} >/dev/null 2>&1; then
        echo -e "${RED}❌ UI port ${UI_PORT} still in use${NC}"
    else
        echo -e "${GREEN}✅ UI port ${UI_PORT} is free${NC}"
    fi
    
    # Check Docker containers
    cd "${PROJECT_ROOT}"
    if [ -f "docker-compose.yml" ]; then
        RUNNING_CONTAINERS=$(docker-compose ps -q 2>/dev/null | wc -l)
        if [ "$RUNNING_CONTAINERS" -gt 0 ]; then
            echo -e "${RED}❌ Some Docker containers still running${NC}"
            docker-compose ps
        else
            echo -e "${GREEN}✅ No Docker containers running${NC}"
        fi
    fi
}

show_summary() {
    echo -e "\n${GREEN}🎉 Stop operation completed!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}📋 Summary:${NC}"
    echo -e "   • Local API server stopped"
    echo -e "   • Local UI server stopped"
    echo -e "   • Docker containers stopped"
    echo -e "   • Python processes stopped"
    echo -e "   • Ports cleaned up"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}💡 To start services again:${NC}"
    echo -e "   • API only: ./bin/start_api.sh"
    echo -e "   • Complete system: ./bin/start_complete.sh"
    echo -e "   • Docker: ./bin/start_docker.sh [command]"
    echo -e "${BLUE}================================================${NC}"
}

# Main execution
main() {
    print_banner
    
    # Stop all types of services
    stop_local_services
    stop_docker_containers
    stop_python_processes
    cleanup_ports
    
    # Show final status
    show_status
    
    # Show summary
    show_summary
}

# Run main function
main "$@"
