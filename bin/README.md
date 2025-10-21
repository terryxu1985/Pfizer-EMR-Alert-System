# Pfizer EMR Alert System - Startup Scripts

This directory contains organized startup scripts for the Pfizer EMR Alert System.

## Available Scripts

### 🚀 `start_api.sh`
Starts only the FastAPI backend service.

**Usage:**
```bash
./bin/start_api.sh
```

**Features:**
- Starts FastAPI server on port 8000
- Health checks and dependency validation
- Interactive API documentation at http://localhost:8000/docs

### 🌐 `start_complete.sh`
Starts both API and UI servers with enhanced doctor input functionality.

**Usage:**
```bash
./bin/start_complete.sh
```

**Features:**
- Starts API server on port 8000
- Starts UI server on port 8080
- Automatic browser opening
- Multi-step doctor data input
- Real-time AI analysis
- Smart medication alerts

### 🐳 `start_docker.sh`
Enhanced Docker containerization management script.

**Usage:**
```bash
./bin/start_docker.sh [COMMAND]
```

**Available Commands:**
- `build` - Build Docker image
- `complete` - Run complete system (API + UI)
- `api-only` - Run API service only
- `ui-only` - Run UI service only (requires API)
- `microservices` - Run as microservices (API + UI separate)
- `logs` - Show container logs
- `status` - Show container status
- `test` - Test API endpoints
- `stop` - Stop all containers
- `cleanup` - Clean up Docker resources
- `restart` - Restart containers
- `shell` - Open shell in container

**Examples:**
```bash
./bin/start_docker.sh build
./bin/start_docker.sh complete
./bin/start_docker.sh api-only
./bin/start_docker.sh microservices
```

### 🛑 `stop_all.sh`
Stops all running services (API, UI, Docker containers).

**Usage:**
```bash
./bin/stop_all.sh
```

**Features:**
- Stops local API server (port 8000)
- Stops local UI server (port 8080)
- Stops Docker containers
- Kills Python EMR processes
- Cleans up ports
- Shows final status

## Quick Start Guide

### For Development (Local)
```bash
# Start complete system
./bin/start_complete.sh

# Or start API only
./bin/start_api.sh
```

### For Production (Docker)
```bash
# Build and start complete system
./bin/start_docker.sh complete

# Or start API only
./bin/start_docker.sh api-only
```

### Stop Everything
```bash
./bin/stop_all.sh
```

## Access URLs

- **UI Server**: http://localhost:8080
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Prerequisites

### For Local Development
- Python 3.8+
- Required packages: `pip install -r config/requirements.txt`

### For Docker
- Docker
- Docker Compose

## Troubleshooting

### Port Already in Use
If you get "port already in use" errors:
```bash
./bin/stop_all.sh
```

### Docker Issues
```bash
./bin/start_docker.sh cleanup
./bin/start_docker.sh build
```

### Check Service Status
```bash
# For Docker
./bin/start_docker.sh status

# For local services
lsof -i :8000  # API port
lsof -i :8080  # UI port
```

## Migration from Old Scripts

The old startup scripts have been reorganized into this `bin/` directory:

- `run_api_only.py` → `./bin/start_api.sh`
- `run_complete_system.py` → `./bin/start_complete.sh`
- `docker-setup.sh` → `./bin/start_docker.sh`
- New: `./bin/stop_all.sh`

The old scripts are still available but deprecated. Use the new `bin/` scripts for better organization and functionality.
