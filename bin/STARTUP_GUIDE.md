# Pfizer EMR Alert System - Startup Scripts Guide

This directory contains organized startup and management scripts for the Pfizer EMR Alert System (v2.1.0).

## 🚀 Quick Start

### Local Development (Recommended for Development)
```bash
# Start complete system (API + UI)
./bin/start_complete.sh

# Start API only
./bin/start_api.sh
```

### Docker Deployment (Recommended for Production)
```bash
# Build and start complete system
./bin/start_docker.sh build
./bin/start_docker.sh complete

# Or run API only
./bin/start_docker.sh api-only
```

### Stop All Services
```bash
./bin/stop_all.sh
```

## 📋 Available Scripts

### 1. `start_api.sh` - API Server Only

Starts only the FastAPI backend service for development and testing.

**Usage:**
```bash
./bin/start_api.sh
```

**Features:**
- ✅ FastAPI server on port 8000
- ✅ Interactive API documentation (Swagger UI)
- ✅ Health checks and dependency validation
- ✅ Real-time model inference
- ✅ RESTful API endpoints

**Access URLs:**
- API Server: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Alternative Docs: http://localhost:8000/redoc

**Available Endpoints:**
- `GET /health` - System health check
- `GET /` - Root endpoint with system info
- `POST /predict` - AI-powered patient treatment prediction
- `GET /docs` - Interactive API documentation
- `GET /version` - System version information

### 2. `start_complete.sh` - Complete System (API + UI)

Starts both the FastAPI backend and frontend UI servers with enhanced clinical decision support features.

**Usage:**
```bash
./bin/start_complete.sh
```

**Features:**
- ✅ FastAPI server on port 8000
- ✅ Flask-based UI server on port 8080
- ✅ Automatic browser opening (macOS/Linux)
- ✅ Multi-step doctor data input interface
- ✅ Real-time AI analysis and predictions
- ✅ Smart medication alerts
- ✅ Enhanced clinical decision support UI
- ✅ Patient management system
- ✅ Evidence-based recommendations

**Access URLs:**
- UI Server: http://localhost:8080
- API Server: http://localhost:8000
- API Documentation: http://localhost:8000/docs

**How to Use:**
1. Click 'Add Patient' button to enter new patient data
2. Fill out the multi-step form with patient information
3. Review AI analysis preview
4. Save patient and run AI analysis
5. Review medication alerts and recommendations
6. View detailed risk stratification

### 3. `start_docker.sh` - Docker Containerization

Enhanced Docker containerization management script with version control, multi-service architecture, and production-ready features.

**Usage:**
```bash
./bin/start_docker.sh [COMMAND]
```

**Available Commands:**

| Command | Description |
|---------|-------------|
| `build` | Build Docker image with versioning (latest + versioned tags) |
| `complete` | Run complete system (API + UI in one container) |
| `api-only` | Run API service only |
| `ui-only` | Run UI service only (requires external API) |
| `microservices` | Run as microservices (API + UI in separate containers) |
| `logs` | Show container logs (real-time) |
| `status` | Show container status and health |
| `test` | Test API endpoints from within container |
| `stop` | Stop all running containers |
| `cleanup` | Clean up Docker resources (containers, images, volumes) |
| `restart` | Restart all containers |
| `shell` | Open interactive shell in running container |
| `version` | Show Docker image version information |

**Version Control Features:**
- ✅ Automatic versioning based on current app version (2.1.0)
- ✅ Creates both `latest` and `versioned` image tags
- ✅ Embeds build metadata (build date, Git commit, version) in image
- ✅ OCI-compliant image labels for better tracking
- ✅ Environment variables with version info accessible in containers

**Image Tags:**
- `pfizer-emr-alert:latest` - Latest development image
- `pfizer-emr-alert:2.1.0` - Version-tagged production image

**Examples:**
```bash
# Build image with versioning
./bin/start_docker.sh build

# Run complete system
./bin/start_docker.sh complete

# Run API only
./bin/start_docker.sh api-only

# Run as microservices (separate containers)
./bin/start_docker.sh microservices

# Check logs
./bin/start_docker.sh logs

# Check version information
./bin/start_docker.sh version

# Clean up everything
./bin/start_docker.sh cleanup
```

**Docker Architecture:**
- **Monolithic Mode**: Single container running both API and UI
- **Microservices Mode**: Separate containers for API and UI
- **Production Mode**: Optimized for performance with health checks

### 4. `stop_all.sh` - Stop All Services

Stops all running services including API, UI, Docker containers, and related Python processes.

**Usage:**
```bash
./bin/stop_all.sh
```

**Features:**
- ✅ Stops local API server (port 8000)
- ✅ Stops local UI server (port 8080)
- ✅ Stops all Docker containers
- ✅ Kills related Python EMR processes
- ✅ Cleans up ports
- ✅ Shows final status report
- ✅ Graceful shutdown with proper cleanup

## 🏗️ System Architecture

### Local Development Mode
```
┌─────────────────────────────────────────┐
│  Development Environment (Local)        │
├─────────────────────────────────────────┤
│                                          │
│  ┌─────────────┐      ┌─────────────┐  │
│  │   Port 8000 │      │   Port 8080 │  │
│  │   FastAPI   │◄─────┤   Flask UI  │  │
│  │   Backend   │      │   Frontend  │  │
│  └─────────────┘      └─────────────┘  │
│         │                   │           │
│         ▼                   ▼           │
│  ┌──────────────────────────────────┐  │
│  │  ML Models (XGBoost)             │  │
│  │  Data Processing                 │  │
│  │  Alert System                    │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Docker Production Mode
```
┌──────────────────────────────────────────────┐
│  Docker Environment (Production)             │
├──────────────────────────────────────────────┤
│                                               │
│  ┌────────────────────────────────────────┐  │
│  │  pfizer-emr-alert:2.1.0 Container     │  │
│  ├────────────────────────────────────────┤  │
│  │  ┌─────────────┐   ┌─────────────┐   │  │
│  │  │ FastAPI     │   │ Flask UI    │   │  │
│  │  │ (Port 8000) │   │ (Port 8080) │   │  │
│  │  └─────────────┘   └─────────────┘   │  │
│  │         │                              │  │
│  │         ▼                              │  │
│  │  ┌──────────────────────────────┐    │  │
│  │  │ ML Models & Services         │    │  │
│  │  └──────────────────────────────┘    │  │
│  └────────────────────────────────────────┘  │
│                                               │
│  Volumes: /app/data, /app/logs               │
└──────────────────────────────────────────────┘
```

## 📦 Prerequisites

### For Local Development
- **Python**: 3.8 or higher
- **Required Packages**: Install using `pip install -r config/requirements.txt`
- **Operating System**: macOS, Linux, or Windows (WSL recommended)

### For Docker Deployment
- **Docker**: 20.10 or higher
- **Docker Compose**: 1.29 or higher
- **Operating System**: macOS, Linux, or Windows (Docker Desktop)

## 🔧 Installation

### Local Setup
```bash
# Clone the repository
git clone <repository-url>
cd "Pfizer-EMR Alert System"

# Install dependencies
pip install -r config/requirements.txt

# Verify installation
python --version  # Should be 3.8+
pip list | grep fastapi  # Should show FastAPI installed
```

### Docker Setup
```bash
# No additional installation needed
# Just ensure Docker is running
docker --version
docker-compose --version
```

## 🔍 Troubleshooting

### Port Already in Use

**Problem:** Error message indicates port 8000 or 8080 is already in use.

**Solution:**
```bash
# Stop all services
./bin/stop_all.sh

# Or manually find and kill process
lsof -ti:8000 | xargs kill -9  # API port
lsof -ti:8080 | xargs kill -9  # UI port
```

### Dependencies Not Found

**Problem:** Import errors or missing packages.

**Solution:**
```bash
# Reinstall dependencies
pip install -r config/requirements.txt --upgrade

# Verify specific packages
python -c "import fastapi, uvicorn, flask"
```

### Docker Issues

**Problem:** Docker containers not starting or errors.

**Solution:**
```bash
# Complete cleanup and rebuild
./bin/start_docker.sh cleanup
./bin/start_docker.sh build
./bin/start_docker.sh complete
```

### Check Service Status

**For Docker:**
```bash
./bin/start_docker.sh status
```

**For Local Services:**
```bash
# Check API port
lsof -i :8000

# Check UI port
lsof -i :8080

# Check running processes
ps aux | grep -E "(fastapi|flask|uvicorn)"
```

### Model Not Found

**Problem:** Model file not found error.

**Solution:**
```bash
# Verify model files exist
ls -la backend/ml_models/models/

# Expected files:
# - xgboost_model.pkl
# - preprocessor.pkl
# - model_metadata.pkl
```

### Permission Denied

**Problem:** Script execution permission denied.

**Solution:**
```bash
# Add execute permissions
chmod +x bin/*.sh

# Or run with bash
bash bin/start_api.sh
```

## 📊 Performance & Monitoring

### Local Development
- API Response Time: < 100ms (typical)
- Model Inference: < 50ms (typical)
- Memory Usage: ~500MB - 1GB
- CPU Usage: Low (<10% typical)

### Docker Production
- Container Memory: ~1-2GB
- Image Size: ~2-3GB
- Startup Time: 5-10 seconds
- Health Checks: Every 30 seconds

## 🔐 Security Considerations

### Local Development
- API runs on localhost only
- No authentication required
- Development mode logging enabled

### Docker Production
- Network isolation by default
- No exposed admin interfaces
- Security-focused base image
- Minimal attack surface

## 📝 Logs

### Local Development
- API logs: `logs/emr_alert_system.log`
- API server logs: Console output
- Model selection logs: `logs/dynamic_model_selection.log`

### Docker Production
```bash
# View logs
./bin/start_docker.sh logs

# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs api
docker-compose logs ui
```

## 🧪 Testing

### API Testing
```bash
# Test with curl
curl http://localhost:8000/health

# Test Docker API
./bin/start_docker.sh test

# Or use the test script
python tests/test_api_endpoint.py
```

### Integration Testing
```bash
# Start complete system
./bin/start_complete.sh

# In another terminal, run tests
python tac.py  # Test All Components
```

## 🔄 Migration Guide

If you're migrating from older versions:

### From Python Scripts to Shell Scripts

**Old way:**
```bash
python scripts/startup/run_api_only.py
python scripts/startup/run_complete_system.py
```

**New way:**
```bash
./bin/start_api.sh
./bin/start_complete.sh
```

### From Docker Setup to Enhanced Docker

**Old way:**
```bash
docker-setup.sh
```

**New way:**
```bash
./bin/start_docker.sh [command]
```

## 📚 Additional Resources

- **Main README**: [README.md](../README.md)
- **API Documentation**: [backend/api/API_GUIDE.md](../backend/api/API_GUIDE.md)
- **Data Guide**: [data/DATA_GUIDE.md](../data/DATA_GUIDE.md)
- **Quick Start**: [QUICK_START.md](../QUICK_START.md)
- **Version Info**: System version 2.1.0

## 🆘 Support

For issues or questions:
1. Check this guide first
2. Review the troubleshooting section
3. Check logs for error messages
4. Refer to the main README.md
5. Check GitHub issues

## 📄 License

Proprietary - Pfizer EMR Alert System

---

**Last Updated**: October 2025  
**Version**: 2.1.0  
**Author**: Pfizer EMR Alert System Development Team

