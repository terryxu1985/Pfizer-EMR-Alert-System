# Version Control and Containerization Strategy

This document describes the comprehensive version control and containerization strategy for the Pfizer EMR Alert System.

## Overview

The system implements multi-layered versioning strategies for:
- **Data Versioning** - Tracking data integrity and compatibility
- **Model Versioning** - Managing ML model versions and deployment
- **Application Versioning** - Software version tracking
- **Docker Image Versioning** - Container version management

---

## 1. Data Versioning

**Location**: `data/versioning/`

### Features
- Semantic versioning (MAJOR.MINOR.PATCH)
- Automatic change detection and version increment
- MD5 checksums for data integrity verification
- Minimal metadata overhead (< 0.03% of data size)
- Model-data compatibility tracking

### Usage

```bash
# Generate version manifest with auto-increment
python data/versioning/manifest.py

# Verify data integrity
python data/versioning/verify.py

# View version history
python data/versioning/history.py history

# Automated version management
python data/versioning/auto_version.py update
```

### Version Rules
- **MAJOR**: Breaking changes (schema changes, new tables, deleted fields)
- **MINOR**: Compatible changes (new records, new files)
- **PATCH**: Data cleaning, bug fixes, metadata updates

**Current Version**: `1.0.0`

---

## 2. Model Versioning

**Location**: `backend/ml_models/models/`

### Naming Convention
```
{model_name}_v{version}_{timestamp}.pkl
```

**Examples**:
- `xgboost_v2.1.0_20251025_230426.pkl`
- `preprocessor_v2.1.0_20251025_230426.pkl`

### Features
- Timestamp-based versioning for traceability
- Automatic model metadata tracking
- Hot reload capability without service restart
- Model compatibility validation

**Current Version**: `2.1.0`

---

## 3. Application Versioning

**Version**: `2.2.0`

The application version is tracked in:
- `README.md` - Version badge and documentation
- Git tags - Version control integration
- Docker labels - Container metadata

### Version History
- `2.2.0` - Current production version (November 2025)
- Project-wide update and documentation improvements
- `2.1.0` - Previous production version (October 2025)
- Includes data versioning, enhanced monitoring, and improved containerization

---

## 4. Docker Image Versioning

**Location**: `Dockerfile`, `bin/start_docker.sh`

### Image Tagging Strategy

Each build creates two tags:
1. **Latest Tag**: `pfizer-emr-alert:latest` - For development
2. **Versioned Tag**: `pfizer-emr-alert:2.2.0` - For production

### Build Metadata

Embedded in Docker environment variables:
- `APP_VERSION` - Application version (2.2.0)
- `BUILD_DATE` - ISO 8601 timestamp
- `VCS_REF` - Git commit hash
- `BUILD_TYPE` - Build type (development/production)

### OCI Labels

Standard container labels for metadata:
```dockerfile
LABEL version="2.2.0"
LABEL maintainer="Pfizer EMR Development Team"
LABEL org.opencontainers.image.title="Pfizer EMR Alert System"
LABEL org.opencontainers.image.version="2.2.0"
LABEL org.opencontainers.image.description="Enterprise AI-Powered Clinical Decision Support System"
LABEL org.opencontainers.image.vendor="Pfizer"
LABEL com.pfizer.app.version="2.2.0"
LABEL com.pfizer.build.type="production"
```

### Usage

```bash
# Build with versioning
./bin/start_docker.sh build

# Check version information
./bin/start_docker.sh version

# List all versions
docker images pfizer-emr-alert

# Inspect metadata
docker inspect pfizer-emr-alert:2.2.0

# View labels
docker inspect pfizer-emr-alert:2.2.0 --format '{{json .Config.Labels}}' | jq '.'

# Deploy specific version
docker run -d --name pfizer-emr pfizer-emr-alert:2.2.0

# Deploy latest (development)
docker run -d --name pfizer-emr-dev pfizer-emr-alert:latest
```

---

## 5. Primitive Strategy

### Development Workflow

1. **Data Changes**
   ```bash
   # Modify data files
   # Run auto-versioning
   python data/versioning/auto_version.py quick
   ```

2. **Model Training**
   ```bash
   # Train new model (creates versioned files)
   python scripts/model_training/train_production_model.py
   ```

3. **Docker Build**
   ```bash
   # Update VERSION in start_docker.sh if needed
   ./bin/start_docker.sh build
   ```

4. **Testing**
   ```bash
   # Test with versioned image
   ./bin/start_docker.sh complete
   ./bin/start_docker.sh test
   ```

5. **Deployment**
   ```bash
   # Use specific version for production
   docker run pfizer-emr-alert:2.2.0
   ```

### Production Deployment

**Best Practices**:
1. Use versioned tags in production (not `latest`)
2. Maintain deployment logs with version information
3. Test versioned images before deployment
4. Keep older versions for rollback capability
5. Document breaking changes in model or data versions

**Example Production Deploy**:
```bash
# Pull specific version
docker pull registry/pfizer-emr-alert:2.2.0

# Run production container
docker run -d \
  --name pfizer-emr-prod \
  -p 8000:8000 \
  -p 8080:8080 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data/storage:/app/data/storage \
  pfizer-emr-alert:2.2.0
```

### Rollback Strategy

If issues occur:
```bash
# Stop current version
docker stop pfizer-emr-prod
docker rm pfizer-emr-prod

# Run previous version
docker run -d --name pfizer-emr-prod pfizer-emr-alert:2.0.0
```

---

## 6. Version Compatibility Matrix

| Component | Current Version | Compatible Versions |
|-----------|----------------|---------------------|
| Application | 2.2.0 | 2.x |
| Models | 2.2.0 | Requires 38 features |
| Data | 1.0.0 | Compatible with models v2.2.0+ |
| Docker Images | 2.2.0 | Container images |

---

## 7. Troubleshooting

### Check Versions
```bash
# Application version
cat README.md | grep "Version:"

# Data version
cat data/.metadata/DATA_VERSIONS.json | grep version

# Model version
ls -la backend/ml_models/models/

# Docker image version
./bin/start_docker.sh version
```

### Version Mismatch Issues

**Problem**: Model and data versions incompatible

**Solution**:
```bash
# Check compatibility
python data/versioning/compatibility.py

# Retrain model if needed
python scripts/model_training/train_production_model.py
```

**Problem**: Docker image won't start

**Solution**:
```bash
# Rebuild with correct version
./bin/start_docker.sh build

# Check logs
./bin/start_docker.sh logs
```

---

## 8. Future Enhancements

- [ ] Automated version bumping in CI/CD
- [ ] Version release notes generation
- [ ] Automated compatibility testing
- [ ] Version archive system for old images
- [ ] Integration with container registry (e.g., Docker Hub, ECR)
- [ ] Semantic versioning automation from git commits

---

## Summary

The version control strategy provides:
- ✅ **Traceability** - Every component is versioned
- ✅ **Reproducibility** - Specific versions can be deployed
- ✅ **Rollback** - Previous versions are maintained
- ✅ **Compatibility** - Version compatibility is tracked
- ✅ **Metadata** - Build information embedded in images
- ✅ **Standards** - OCI-compliant labels

All version information is accessible at runtime through environment variables and labels, enabling comprehensive monitoring and auditing.

---

**Last Updated**: November 2025  
**Current System Version**: 2.2.0  
**Maintainer**: Pfizer EMR Development Team
