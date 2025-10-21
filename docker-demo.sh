#!/bin/bash

# Pfizer EMR Alert System - Containerization Demo
# This script demonstrates the enhanced Docker containerization features

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_demo() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}🏥 Pfizer EMR Alert System - Containerization Demo${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}🐳 Enhanced Docker containerization demonstration${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_step() {
    echo -e "\n${YELLOW}📋 Step $1: $2${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"
}

print_command() {
    echo -e "${GREEN}💻 Command:${NC} $1"
}

print_result() {
    echo -e "${GREEN}✅ Result:${NC} $1"
}

# Demo functions
demo_build() {
    print_step "1" "Building Docker Image"
    print_command "./docker-setup.sh build"
    echo -e "${YELLOW}This will:${NC}"
    echo "  • Use multi-stage build for optimization"
    echo "  • Install dependencies securely"
    echo "  • Create non-root user for security"
    echo "  • Set up health checks"
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

demo_complete_system() {
    print_step "2" "Running Complete System"
    print_command "./docker-setup.sh complete"
    echo -e "${YELLOW}This will:${NC}"
    echo "  • Start both API and UI in single container"
    echo "  • Expose ports 8000 (API) and 8080 (UI)"
    echo "  • Mount volumes for persistence"
    echo "  • Enable health checks"
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

demo_microservices() {
    print_step "3" "Running Microservices Architecture"
    print_command "./docker-setup.sh microservices"
    echo -e "${YELLOW}This will:${NC}"
    echo "  • Start API and UI as separate containers"
    echo "  • Enable service discovery between containers"
    echo "  • Allow independent scaling"
    echo "  • Provide better isolation"
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

demo_health_monitoring() {
    print_step "4" "Health Monitoring"
    print_command "./docker-health-check.sh"
    echo -e "${YELLOW}This will:${NC}"
    echo "  • Check API health endpoint"
    echo "  • Verify UI service status"
    echo "  • Monitor container resources"
    echo "  • Scan logs for errors"
    echo "  • Test API endpoints"
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

demo_api_testing() {
    print_step "5" "API Testing"
    print_command "./docker-setup.sh test"
    echo -e "${YELLOW}This will:${NC}"
    echo "  • Test health endpoint"
    echo "  • Test root endpoint"
    echo "  • Test prediction endpoint with sample data"
    echo "  • Verify API functionality"
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

demo_management() {
    print_step "6" "Container Management"
    echo -e "${YELLOW}Available management commands:${NC}"
    echo ""
    echo -e "${GREEN}Status and Monitoring:${NC}"
    echo "  ./docker-setup.sh status     # Container status"
    echo "  ./docker-setup.sh logs       # View logs"
    echo "  ./docker-health-check.sh monitor  # Continuous monitoring"
    echo ""
    echo -e "${GREEN}Control Commands:${NC}"
    echo "  ./docker-setup.sh restart    # Restart containers"
    echo "  ./docker-setup.sh stop       # Stop containers"
    echo "  ./docker-setup.sh shell      # Open shell in container"
    echo ""
    echo -e "${GREEN}Cleanup Commands:${NC}"
    echo "  ./docker-setup.sh cleanup    # Clean up Docker resources"
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

demo_deployment_modes() {
    print_step "7" "Deployment Modes Comparison"
    echo -e "${YELLOW}Different deployment modes for different use cases:${NC}"
    echo ""
    echo -e "${GREEN}1. Complete System (Development/Testing):${NC}"
    echo "   ./docker-setup.sh complete"
    echo "   • Single container with API + UI"
    echo "   • Simple deployment"
    echo "   • Good for development"
    echo ""
    echo -e "${GREEN}2. API-Only (Backend Services):${NC}"
    echo "   ./docker-setup.sh api-only"
    echo "   • API service only"
    echo "   • Good for microservices integration"
    echo "   • Lightweight backend"
    echo ""
    echo -e "${GREEN}3. Microservices (Production):${NC}"
    echo "   ./docker-setup.sh microservices"
    echo "   • Separate API and UI containers"
    echo "   • Independent scaling"
    echo "   • Production-ready architecture"
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

show_access_info() {
    print_step "8" "Accessing the Application"
    echo -e "${YELLOW}Once containers are running, access:${NC}"
    echo ""
    echo -e "${GREEN}🌐 Web Interface:${NC}"
    echo "   http://localhost:8080"
    echo ""
    echo -e "${GREEN}📡 API Endpoints:${NC}"
    echo "   http://localhost:8000"
    echo "   http://localhost:8000/docs (API Documentation)"
    echo "   http://localhost:8000/health (Health Check)"
    echo ""
    echo -e "${GREEN}📋 Quick Test:${NC}"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

show_security_features() {
    print_step "9" "Security Features"
    echo -e "${YELLOW}Enhanced security features:${NC}"
    echo ""
    echo -e "${GREEN}🔒 Container Security:${NC}"
    echo "   • Non-root user execution"
    echo "   • Minimal base image (Python slim)"
    echo "   • No unnecessary packages"
    echo "   • Proper file permissions"
    echo ""
    echo -e "${GREEN}🌐 Network Security:${NC}"
    echo "   • Isolated Docker networks"
    echo "   • Service-to-service communication"
    echo "   • No external network exposure"
    echo ""
    echo -e "${GREEN}📁 Data Security:${NC}"
    echo "   • Read-only model mounts"
    echo "   • Volume persistence"
    echo "   • Secure data handling"
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

show_production_considerations() {
    print_step "10" "Production Deployment Considerations"
    echo -e "${YELLOW}For production deployment:${NC}"
    echo ""
    echo -e "${GREEN}🚀 Performance:${NC}"
    echo "   • Use microservices mode for scalability"
    echo "   • Configure resource limits"
    echo "   • Enable health checks"
    echo "   • Monitor resource usage"
    echo ""
    echo -e "${GREEN}🔧 Configuration:${NC}"
    echo "   • Set production environment variables"
    echo "   • Configure reverse proxy (nginx)"
    echo "   • Enable SSL/TLS"
    echo "   • Set up log aggregation"
    echo ""
    echo -e "${GREEN}📊 Monitoring:${NC}"
    echo "   • Use health check script"
    echo "   • Monitor container metrics"
    echo "   • Set up alerting"
    echo "   • Regular log analysis"
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

# Main demo function
main() {
    print_demo
    
    echo -e "${YELLOW}This demo will walk you through the enhanced Docker containerization features.${NC}"
    echo -e "${YELLOW}Press Enter after each step to continue.${NC}"
    echo ""
    echo -e "${BLUE}Press Enter to start the demo...${NC}"
    read
    
    demo_build
    demo_complete_system
    demo_microservices
    demo_health_monitoring
    demo_api_testing
    demo_management
    demo_deployment_modes
    show_access_info
    show_security_features
    show_production_considerations
    
    echo -e "\n${GREEN}🎉 Demo completed!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}📚 Next Steps:${NC}"
    echo "1. Try the commands shown in the demo"
    echo "2. Read the full documentation in DOCKER_README.md"
    echo "3. Experiment with different deployment modes"
    echo "4. Set up monitoring for your environment"
    echo ""
    echo -e "${GREEN}🚀 Ready to containerize your Pfizer EMR Alert System!${NC}"
    echo -e "${BLUE}================================================${NC}"
}

# Run demo
main "$@"
