# Configuration Directory

This directory contains the unified configuration system for the Pfizer EMR Alert System model training.

## 📁 Files

### Core Configuration Files
- **`environment_config.yaml`** - Configuration values (edit this for most changes)
- **`optimized_config.py`** - Configuration structure and logic
- **`config_manager.py`** - High-level configuration management API

### Documentation
- **`CONFIG_GUIDE.md`** - Complete configuration system guide (START HERE!)
- **`CONFIGURATION_UNIFICATION_SUMMARY.md`** - Summary of unification work
- **`README.md`** - This file

### Tools
- **`validate_config_consistency.py`** - Validates configuration consistency

### Legacy (Deprecated)
- **`base_config.py`** - Old configuration system (being phased out)

## 🚀 Quick Start

```python
# Recommended: Use ConfigManager
from scripts.model_training.config.config_manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.get_production_config()

# Access configuration
print(f"Model: {config.model_type}")
hyperparams = config.get_model_hyperparameters()
```

## 📚 Documentation

**New to the configuration system?** Read the **[CONFIG_GUIDE.md](./CONFIG_GUIDE.md)** first!

It covers:
- Architecture and design
- How to use the configuration
- How to modify configuration
- Best practices
- Troubleshooting

## ✅ Validation

Always validate after making changes:

```bash
python3 scripts/model_training/config/validate_config_consistency.py
```

## 🔑 Key Concepts

### Two-File Architecture
- **`.yaml`** stores VALUES (easy to edit)
- **`.py`** provides STRUCTURE (defines data model)

### Configuration Priority
1. Environment-specific YAML overrides (highest)
2. Base YAML configuration
3. Python defaults (fallback only)

### Model Neutrality
Configuration does NOT hardcode which model is "best". The `model_selection.py` system determines the optimal model based on actual performance.

## 📝 Common Tasks

### Change a hyperparameter
→ Edit `environment_config.yaml`

### Add a new parameter
→ Edit both `.py` and `.yaml`, then validate

### Switch environments
```python
config = config_manager.get_development_config()  # dev
config = config_manager.get_production_config()   # prod
```

### Check consistency
```bash
python3 validate_config_consistency.py
```

## 🆘 Need Help?

1. Read **[CONFIG_GUIDE.md](./CONFIG_GUIDE.md)**
2. Check validation output for specific errors
3. Review **[CONFIGURATION_UNIFICATION_SUMMARY.md](./CONFIGURATION_UNIFICATION_SUMMARY.md)** for design decisions

---

**Last Updated**: 2025-10-21  
**Status**: ✅ Unified and validated

