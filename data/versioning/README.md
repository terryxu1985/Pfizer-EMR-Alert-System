# Data Versioning Module

A lightweight data version control system for the Pfizer EMR Alert System that tracks data integrity, versions, and model-data compatibility using metadata.

## 📋 Overview

This module provides data version management capabilities without modifying or duplicating data files. Instead, it maintains a metadata manifest with MD5 checksums, statistical information, and compatibility details.

### Key Features

- **📝 Metadata Manifest**: MD5 checksums, file statistics, and version tracking
- **🔍 Integrity Verification**: Automated validation of data file integrity
- **🔗 Compatibility Checking**: Ensures model versions match data versions
- **🚀 Zero Overhead**: Only 3.9KB metadata for 15MB of data (< 0.03% overhead)
- **⚡ Fast Verification**: Complete integrity check in < 1 second

## 🚀 Quick Start

### 1. Generate Data Manifest (with Auto-Versioning)

Create a version manifest for all data files with automatic version increment:

```bash
# Automatic version increment (recommended)
python data/versioning/manifest.py

# Force specific change type
python data/versioning/manifest.py --force-change-type minor

# Disable auto-versioning (manual control)
python data/versioning/manifest.py --no-auto-version
```

**Output**: `data/.metadata/DATA_VERSIONS.json`

Contains:
- MD5 checksums for all files
- File statistics (size, rows, columns)
- Data lineage (source relationships)
- Model compatibility information
- **Automatic version increment** based on detected changes

### 2. Verify Data Integrity

Verify all data files match their checksums:

```bash
python data/versioning/verify.py
```

**Checks**: All files in manifest are present and unchanged

### 4. Manage Version History

View and manage version history:

```bash
# Show current version info
python data/versioning/history.py current

# Show complete version history
python data/versioning/history.py history

# Show version statistics
python data/versioning/history.py stats

# Export version history
python data/versioning/history.py export version_history.json
```

### 5. Automated Version Management

Use the automation script for common tasks:

```bash
# Quick version update (patch increment)
python data/versioning/auto_version.py quick

# Full version update with verification
python data/versioning/auto_version.py update

# Force specific change type
python data/versioning/auto_version.py update --change-type minor

# Show version status
python data/versioning/auto_version.py status

# Create Git tag for current version
python data/versioning/auto_version.py tag

# Setup Git hooks for automatic integrity checks
python data/versioning/auto_version.py setup
```

## 📊 Manifest Structure

### DATA_VERSIONS.json

```json
{
  "version": "1.0.0",
  "created_at": "2025-10-25T23:39:50Z",
  "data_pipeline": {
    "raw": {
      "raw/dim_patient.xlsx": {
        "md5": "...",
        "rows": 4020,
        "columns": 3
      }
    },
    "processed": { ... },
    "model_ready": {
      "model_ready/model_ready_dataset.csv": {
        "md5": "...",
        "rows": 3612,
        "columns": 39,
        "feature_count": 38,
        "target_distribution": {"0": 631, "1": 2981}
      }
    }
  },
  "statistics": {
    "total_files": 11,
    "total_size_mb": 14.91
  },
  "model_compatibility": {
    "supported_models": [...],
    "data_requirements": { ... }
  }
}
```

## 🔧 Usage Scenarios

### Scenario 1: Update Data and Create New Version (Automated)

After modifying data files:

```bash
# Option A: Fully automated (recommended)
python data/versioning/auto_version.py quick

# Option B: Manual control with auto-detection
python data/versioning/manifest.py

# Option C: Force specific change type
python data/versioning/manifest.py --force-change-type minor

# The system will automatically:
# 1. Detect changes and increment version
# 2. Generate new manifest
# 3. Verify integrity
# 4. Check compatibility
```

**Manual Git workflow** (if not using automation):
```bash
# 1. Generate new manifest (auto-versioning)
python data/versioning/manifest.py

# 2. Verify integrity
python data/versioning/verify.py

# 3. Commit to Git
git add data/.metadata/DATA_VERSIONS.json
git commit -m "Data version $(python -c "import json; print(json.load(open('data/.metadata/DATA_VERSIONS.json'))['version'])")"

# 4. Create Git tag
python data/versioning/auto_version.py tag
```

### Scenario 2: Suspect Data Corruption

Verify if files have been modified:

```bash
# Run integrity check
python data/versioning/verify.py

# Output shows which files fail MD5 validation
# Locates exact files that have changed
```

### Scenario 3: Pre-Deployment Validation

Before deploying to production:

```bash
# Option A: Automated validation
python data/versioning/auto_version.py update

# Option B: Manual validation
python data/versioning/verify.py
python data/versioning/compatibility.py

# Option C: Check version status
python data/versioning/auto_version.py status
```

### Scenario 4: View Version History

```bash
# Show current version and recent changes
python data/versioning/history.py current

# Show complete version history
python data/versioning/history.py history

# Show version statistics
python data/versioning/history.py stats
```

## 📝 Version Management

### Automatic Version Detection

The system automatically detects change types and increments versions:

- **MAJOR**: Breaking changes (structure changes, new tables, deleted fields, schema changes)
- **MINOR**: Compatible changes (new records, new files, backward compatible changes)
- **PATCH**: Data cleaning, bug fixes, metadata updates

### Change Detection Logic

```python
# Automatic change detection
if structure_changes or deleted_files or schema_changes:
    change_type = "major"
elif new_files or record_count_changes:
    change_type = "minor"
else:
    change_type = "patch"
```

### Examples

```
v1.0.0  - Initial dataset
v1.0.1  - Fixed date format errors (patch)
v1.1.0  - Added 1000 new patient records (minor)
v1.1.1  - Data cleaning updates (patch)
v2.0.0  - Added new feature columns (major - breaking change)
```

### Manual Override

You can force specific change types:

```bash
# Force major version increment
python data/versioning/manifest.py --force-change-type major

# Force minor version increment  
python data/versioning/manifest.py --force-change-type minor

# Force patch version increment
python data/versioning/manifest.py --force-change-type patch
```

## ⚙️ Integration

### CI/CD Integration

Add to `.github/workflows/ci.yml`:

```yaml
jobs:
  verify-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Verify data integrity
        run: python data/versioning/verify.py
      - name: Check compatibility
        run: python data/versioning/compatibility.py
```

### Git Hooks

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python data/versioning/verify.py
if [ $? -ne 0 ]; then
    echo "Data integrity check failed!"
    exit 1
fi
```

## 🔍 Troubleshooting

### MD5 Mismatch

**Problem**: Files don't match their checksums

**Possible causes**:
- Files modified outside workflow
- File corruption
- Operating system issues

**Solution**:
```bash
# Identify which files failed
python data/versioning/verify.py

# Regenerate manifest if intentional changes
python data/versioning/manifest.py
```

### Compatibility Check Failure

**Problem**: Model and data versions incompatible

**Possible causes**:
- Feature count mismatch
- Column name changes
- Version mismatch

**Solution**:
```bash
# Check model version
cat backend/ml_models/models/model_metadata.pkl

# Check data version
cat data/.metadata/DATA_VERSIONS.json | grep version

# Retrain model or use compatible data version
```

### Manifest Not Found

**Problem**: `data/.metadata/DATA_VERSIONS.json` missing

**Solution**:
```bash
# Generate initial manifest
python data/versioning/manifest.py
```

## 📈 Future Enhancements

- [x] **Automatic version increment** ✅ Implemented
- [x] **Version history tracking** ✅ Implemented  
- [x] **Change detection logic** ✅ Implemented
- [x] **Automation scripts** ✅ Implemented
- [ ] Version comparison and diff reports
- [ ] Rollback capability
- [ ] DVC integration for large files
- [ ] Version archive system

## 🏗️ Architecture

```
data/
├── .metadata/                  # Metadata storage
│   ├── DATA_VERSIONS.json     # Version manifest
│   └── VERSION_HISTORY.json   # Version history log
├── versioning/                # Tools
│   ├── manifest.py            # Generate manifest (with auto-versioning)
│   ├── verify.py              # Integrity verification
│   ├── compatibility.py       # Compatibility checking
│   ├── version_manager.py     # Core version management
│   ├── history.py             # Version history management
│   ├── auto_version.py         # Automation scripts
│   └── README.md              # This file
└── [data directories]/        # Actual data files
```

## 💡 Design Principles

1. **Zero Overhead**: Minimal metadata footprint (< 0.03% of data size)
2. **Non-Invasive**: No changes to existing data files or structure
3. **Fast Performance**: Sub-second verification for entire dataset
4. **Clear Separation**: Metadata separate from operational data
5. **Easy Integration**: Simple scripts that can be automated

## 📊 Current Status

- **Data Version**: 1.0.0 (with automatic versioning enabled)
- **Total Files**: 11 data files
- **Total Size**: 14.91 MB
- **Metadata Size**: 3.9 KB (0.026% overhead)
- **Compatible Models**: XGBoost v2.1.0
- **Auto-Versioning**: ✅ Enabled
- **Version History**: ✅ Tracked

## 🤝 Contributing

When modifying data versioning tools:

1. Maintain backward compatibility
2. Update version numbers appropriately
3. Add tests for new features
4. Update documentation

## 📞 Support

For issues or questions:

- Check this README for common solutions
- Review `data/DATA_GUIDE.md` for data pipeline documentation
- Examine logs in `logs/` directory
- Contact the development team

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-25  
**Status**: ✅ Production Ready
