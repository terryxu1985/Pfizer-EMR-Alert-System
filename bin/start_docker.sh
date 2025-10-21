#!/bin/bash

# Pfizer EMR Alert System - Docker Startup Script
# Enhanced containerization management script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="pfizer-emr-alert"
DOCKER_IMAGE="${PROJECT_NAME}:latest"
API_PORT=8000
UI_PORT=8080
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Functions
print_banner() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}🏥 Pfizer EMR Alert System - Docker Setup${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}🐳 Enhanced containerization management${NC}"
    echo -e "${GREEN}📋 Features:${NC}"
    echo -e "   • Multi-service architecture"
    echo -e "   • Health checks and monitoring"
    echo -e "   • Volume persistence"
    echo -e "   • Security hardening"
    echo -e "${BLUE}================================================${NC}"
}

print_usage() {
    echo -e "${YELLOW}Usage: $0 [COMMAND]${NC}"
    echo ""
    echo -e "${GREEN}Available Commands:${NC}"
    echo -e "  ${BLUE}build${NC}           - Build Docker image"
    echo -e "  ${BLUE}complete${NC}        - Run complete system (API + UI)"
    echo -e "  ${BLUE}api-only${NC}        - Run API service only"
    echo -e "  ${BLUE}ui-only${NC}         - Run UI service only (requires API)"
    echo -e "  ${BLUE}microservices${NC}   - Run as microservices (API + UI separate)"
    echo -e "  ${BLUE}logs${NC}            - Show container logs"
    echo -e "  ${BLUE}status${NC}          - Show container status"
    echo -e "  ${BLUE}test${NC}            - Test API endpoints"
    echo -e "  ${BLUE}stop${NC}            - Stop all containers"
    echo -e "  ${BLUE}cleanup${NC}         - Clean up Docker resources"
    echo -e "  ${BLUE}restart${NC}         - Restart containers"
    echo -e "  ${BLUE}shell${NC}           - Open shell in container"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  $0 build"
    echo -e "  $0 complete"
    echo -e "  $0 api-only"
    echo -e "  $0 microservices"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker and Docker Compose are available${NC}"
}

check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    # Check if required files exist
    if [ ! -f "Dockerfile" ]; then
        echo -e "${RED}❌ Dockerfile not found${NC}"
        exit 1
    fi
    
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}❌ docker-compose.yml not found${NC}"
        exit 1
    fi
    
    if [ ! -f "config/requirements_production.txt" ]; then
        echo -e "${RED}❌ Production requirements file not found${NC}"
        exit 1
    fi
    
    # Check if model files exist
    if [ ! -d "backend/ml_models/models" ]; then
        echo -e "${YELLOW}⚠️ Model files not found. Creating directory...${NC}"
        mkdir -p backend/ml_models/models
    fi
    
    # Check if data directories exist
    mkdir -p logs data/storage data/model_ready
    
    echo -e "${GREEN}✅ Prerequisites check completed${NC}"
}

build_image() {
    echo -e "${BLUE}🔨 Building Docker image...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    # Build with build args for better caching
    docker build \
        --tag ${DOCKER_IMAGE} \
        --target application \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Docker image built successfully${NC}"
    else
        echo -e "${RED}❌ Docker image build failed${NC}"
        exit 1
    fi
}

run_complete() {
    echo -e "${BLUE}🚀 Starting complete system (API + UI)...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    docker-compose --profile complete up -d
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Complete system started successfully${NC}"
        show_access_info
    else
        echo -e "${RED}❌ Failed to start complete system${NC}"
        exit 1
    fi
}

run_api_only() {
    echo -e "${BLUE}🚀 Starting API service only...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    docker-compose --profile api-only up -d pfizer-emr-api
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ API service started successfully${NC}"
        echo -e "${YELLOW}📡 API available at: http://localhost:${API_PORT}${NC}"
        echo -e "${YELLOW}📚 API docs at: http://localhost:${API_PORT}/docs${NC}"
    else
        echo -e "${RED}❌ Failed to start API service${NC}"
        exit 1
    fi
}

run_ui_only() {
    echo -e "${BLUE}🚀 Starting UI service only...${NC}"
    echo -e "${YELLOW}⚠️ Note: This requires the API service to be running${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    docker-compose --profile ui-only up -d
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ UI service started successfully${NC}"
        echo -e "${YELLOW}🌐 UI available at: http://localhost:${UI_PORT}${NC}"
    else
        echo -e "${RED}❌ Failed to start UI service${NC}"
        exit 1
    fi
}

run_microservices() {
    echo -e "${BLUE}🚀 Starting microservices architecture...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    docker-compose --profile microservices up -d
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Microservices started successfully${NC}"
        show_access_info
    else
        echo -e "${RED}❌ Failed to start microservices${NC}"
        exit 1
    fi
}

show_logs() {
    echo -e "${BLUE}📋 Showing container logs...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    docker-compose logs -f --tail=100
}

show_status() {
    echo -e "${BLUE}📊 Container status:${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    docker-compose ps
    
    echo -e "\n${BLUE}📈 Resource usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" 2>/dev/null || echo "No running containers"
}

test_api() {
    echo -e "${BLUE}🧪 Testing API endpoints...${NC}"
    
    # Wait for API to be ready
    echo -e "${YELLOW}⏳ Waiting for API to be ready...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:${API_PORT}/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API is ready${NC}"
            break
        fi
        sleep 2
        echo -n "."
    done
    
    echo -e "\n${BLUE}🔍 Testing endpoints:${NC}"
    
    # Test health endpoint
    echo -e "${YELLOW}Testing health endpoint...${NC}"
    curl -s http://localhost:${API_PORT}/health | jq . 2>/dev/null || curl -s http://localhost:${API_PORT}/health
    
    # Test root endpoint
    echo -e "\n${YELLOW}Testing root endpoint...${NC}"
    curl -s http://localhost:${API_PORT}/ | jq . 2>/dev/null || curl -s http://localhost:${API_PORT}/
    
    # Test prediction endpoint with sample data
    echo -e "\n${YELLOW}Testing prediction endpoint...${NC}"
    curl -s -X POST "http://localhost:${API_PORT}/predict" \
        -H "Content-Type: application/json" \
        -d '{
            "patient_id": 1,
            "age": 45,
            "gender": "M",
            "symptoms": ["fever", "cough"],
            "comorbidities": ["diabetes"],
            "physician_id": 1,
            "diagnosis_date": "2024-01-15"
        }' | jq . 2>/dev/null || echo "Prediction endpoint test completed"
    
    echo -e "\n${GREEN}✅ API testing completed${NC}"
}

stop_containers() {
    echo -e "${BLUE}🛑 Stopping all containers...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    docker-compose down
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ All containers stopped${NC}"
    else
        echo -e "${RED}❌ Failed to stop containers${NC}"
        exit 1
    fi
}

cleanup_docker() {
    echo -e "${BLUE}🧹 Cleaning up Docker resources...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    # Stop containers
    docker-compose down -v
    
    # Remove images
    docker rmi ${DOCKER_IMAGE} 2>/dev/null || true
    
    # Remove unused containers, networks, images
    docker system prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    echo -e "${GREEN}✅ Docker cleanup completed${NC}"
}

restart_containers() {
    echo -e "${BLUE}🔄 Restarting containers...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    docker-compose restart
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Containers restarted successfully${NC}"
    else
        echo -e "${RED}❌ Failed to restart containers${NC}"
        exit 1
    fi
}

open_shell() {
    echo -e "${BLUE}🐚 Opening shell in container...${NC}"
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    # Try to find running container
    CONTAINER_ID=$(docker-compose ps -q | head -1)
    
    if [ -z "$CONTAINER_ID" ]; then
        echo -e "${RED}❌ No running containers found${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Opening shell in container: ${CONTAINER_ID}${NC}"
    docker exec -it ${CONTAINER_ID} /bin/bash
}

show_access_info() {
    echo -e "\n${GREEN}🎉 System is ready!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}📱 Access URLs:${NC}"
    echo -e "   • UI Server: http://localhost:${UI_PORT}"
    echo -e "   • API Server: http://localhost:${API_PORT}"
    echo -e "   • API Documentation: http://localhost:${API_PORT}/docs"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}📋 Quick Commands:${NC}"
    echo -e "   • View logs: $0 logs"
    echo -e "   • Check status: $0 status"
    echo -e "   • Test API: $0 test"
    echo -e "   • Stop system: $0 stop"
    echo -e "${BLUE}================================================${NC}"
}

# Main script logic
main() {
    print_banner
    
    # Check if no arguments provided
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi
    
    # Check Docker installation
    check_docker
    
    # Check prerequisites
    check_prerequisites
    
    # Handle commands
    case "$1" in
        "build")
            build_image
            ;;
        "complete")
            build_image
            run_complete
            ;;
        "api-only")
            build_image
            run_api_only
            ;;
        "ui-only")
            build_image
            run_ui_only
            ;;
        "microservices")
            build_image
            run_microservices
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "test")
            test_api
            ;;
        "stop")
            stop_containers
            ;;
        "cleanup")
            cleanup_docker
            ;;
        "restart")
            restart_containers
            ;;
        "shell")
            open_shell
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
