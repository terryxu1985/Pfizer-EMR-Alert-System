#!/bin/bash

# Pfizer EMR Alert System - Docker Health Check Script
# Comprehensive health monitoring for containerized services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_PORT=8000
UI_PORT=8080
HEALTH_TIMEOUT=30
RETRY_INTERVAL=5

# Functions
print_status() {
    echo -e "${GREEN}[HEALTH]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if a service is responding
check_service() {
    local service_name=$1
    local url=$2
    local timeout=${3:-10}
    
    print_info "Checking $service_name at $url..."
    
    if curl -s --max-time $timeout "$url" > /dev/null 2>&1; then
        print_status "$service_name is responding ✓"
        return 0
    else
        print_error "$service_name is not responding ✗"
        return 1
    fi
}

# Check API health endpoint
check_api_health() {
    local api_url="http://localhost:$API_PORT/health"
    
    print_info "Checking API health endpoint..."
    
    local response=$(curl -s --max-time 10 "$api_url" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        # Parse JSON response to check status
        local status=$(echo "$response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        
        if [ "$status" = "healthy" ]; then
            print_status "API health check passed ✓"
            print_info "API Response: $response"
            return 0
        else
            print_error "API health check failed - Status: $status ✗"
            print_info "API Response: $response"
            return 1
        fi
    else
        print_error "API health endpoint not accessible ✗"
        return 1
    fi
}

# Check UI service
check_ui_service() {
    local ui_url="http://localhost:$UI_PORT"
    
    print_info "Checking UI service..."
    
    if curl -s --max-time 10 "$ui_url" > /dev/null 2>&1; then
        print_status "UI service is responding ✓"
        return 0
    else
        print_error "UI service is not responding ✗"
        return 1
    fi
}

# Check Docker containers
check_containers() {
    print_info "Checking Docker containers..."
    
    local containers=$(docker-compose ps --services --filter "status=running" 2>/dev/null)
    
    if [ -n "$containers" ]; then
        print_status "Running containers:"
        docker-compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"
        return 0
    else
        print_error "No running containers found ✗"
        return 1
    fi
}

# Check container resource usage
check_resources() {
    print_info "Checking container resource usage..."
    
    local containers=$(docker-compose ps -q 2>/dev/null)
    
    if [ -n "$containers" ]; then
        print_status "Resource usage:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" $containers
        return 0
    else
        print_warning "No containers to check resources"
        return 1
    fi
}

# Check logs for errors
check_logs() {
    print_info "Checking recent logs for errors..."
    
    local error_count=$(docker-compose logs --tail=50 2>/dev/null | grep -i "error\|exception\|failed" | wc -l)
    
    if [ "$error_count" -eq 0 ]; then
        print_status "No recent errors found in logs ✓"
        return 0
    else
        print_warning "Found $error_count potential errors in recent logs"
        print_info "Recent error logs:"
        docker-compose logs --tail=50 2>/dev/null | grep -i "error\|exception\|failed" | tail -5
        return 1
    fi
}

# Comprehensive health check
comprehensive_check() {
    print_info "Starting comprehensive health check..."
    echo "=========================================="
    
    local overall_status=0
    
    # Check containers
    if ! check_containers; then
        overall_status=1
    fi
    
    echo ""
    
    # Check API service
    if ! check_api_health; then
        overall_status=1
    fi
    
    echo ""
    
    # Check UI service (if running)
    if docker-compose ps --services --filter "status=running" | grep -q "ui"; then
        if ! check_ui_service; then
            overall_status=1
        fi
        echo ""
    fi
    
    # Check resources
    check_resources
    echo ""
    
    # Check logs
    check_logs
    echo ""
    
    # Overall status
    if [ $overall_status -eq 0 ]; then
        print_status "Overall health check: PASSED ✓"
    else
        print_error "Overall health check: FAILED ✗"
    fi
    
    return $overall_status
}

# Wait for services to be ready
wait_for_services() {
    print_info "Waiting for services to be ready..."
    
    local max_attempts=$((HEALTH_TIMEOUT / RETRY_INTERVAL))
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        print_info "Attempt $attempt/$max_attempts..."
        
        if check_api_health > /dev/null 2>&1; then
            print_status "Services are ready ✓"
            return 0
        fi
        
        sleep $RETRY_INTERVAL
        attempt=$((attempt + 1))
    done
    
    print_error "Services did not become ready within $HEALTH_TIMEOUT seconds ✗"
    return 1
}

# Monitor services continuously
monitor_services() {
    print_info "Starting continuous monitoring (Press Ctrl+C to stop)..."
    
    while true; do
        echo "=========================================="
        echo "$(date): Health Check"
        echo "=========================================="
        
        comprehensive_check
        
        echo ""
        print_info "Next check in 60 seconds..."
        sleep 60
    done
}

# Test specific endpoint
test_endpoint() {
    local endpoint=$1
    local method=${2:-GET}
    local data=${3:-""}
    
    print_info "Testing endpoint: $method $endpoint"
    
    local url="http://localhost:$API_PORT$endpoint"
    
    if [ "$method" = "POST" ] && [ -n "$data" ]; then
        curl -s -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "$data" | jq . 2>/dev/null || curl -s -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "$data"
    else
        curl -s "$url" | jq . 2>/dev/null || curl -s "$url"
    fi
    
    echo ""
}

# Main function
main() {
    case "${1:-comprehensive}" in
        "comprehensive")
            comprehensive_check
            ;;
        "wait")
            wait_for_services
            ;;
        "monitor")
            monitor_services
            ;;
        "api")
            check_api_health
            ;;
        "ui")
            check_ui_service
            ;;
        "containers")
            check_containers
            ;;
        "resources")
            check_resources
            ;;
        "logs")
            check_logs
            ;;
        "test")
            test_endpoint "${2:-/}" "${3:-GET}" "${4:-}"
            ;;
        *)
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  comprehensive  - Run comprehensive health check (default)"
            echo "  wait           - Wait for services to be ready"
            echo "  monitor        - Monitor services continuously"
            echo "  api            - Check API health only"
            echo "  ui             - Check UI service only"
            echo "  containers     - Check container status only"
            echo "  resources      - Check resource usage only"
            echo "  logs           - Check logs for errors only"
            echo "  test [endpoint] [method] [data] - Test specific endpoint"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 wait"
            echo "  $0 monitor"
            echo "  $0 test /predict POST '{\"patient_id\": 1}'"
            ;;
    esac
}

# Run main function
main "$@"
