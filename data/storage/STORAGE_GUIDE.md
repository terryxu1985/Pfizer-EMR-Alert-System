# EMR Alert System - Data Storage Structure

This directory contains persistent data storage for the EMR Alert System.

## Directory Structure

```
data/storage/
├── patients/                    # Patient-related persistent data
│   ├── doctor_input_patients.json    # Patients created by doctors via UI
│   ├── ai_analysis_results.json      # AI analysis results cache
│   └── patient_sessions.json         # Patient session data
├── system/                      # System configuration and state
│   ├── model_cache.json             # Model loading cache
│   ├── system_state.json            # System runtime state
│   └── user_preferences.json        # User interface preferences
└── logs/                        # Application logs (if needed)
    ├── patient_creation.log
    ├── ai_analysis.log
    └── system_errors.log
```

## File Naming Conventions

### Patient Data Files
- **Format**: `{data_source}_{data_type}.json`
- **Examples**:
  - `doctor_input_patients.json` - Patients created by doctors
  - `emr_imported_patients.json` - Patients imported from EMR systems
  - `ai_analysis_results.json` - Cached AI analysis results

### System Data Files
- **Format**: `{component}_{purpose}.json`
- **Examples**:
  - `model_cache.json` - Model loading and caching data
  - `system_state.json` - Current system state
  - `user_preferences.json` - User interface preferences

## Data File Structure

### doctor_input_patients.json
```json
{
  "metadata": {
    "version": "1.0",
    "created_at": "2024-01-01T00:00:00Z",
    "last_updated": "2024-01-01T00:00:00Z",
    "total_patients": 0,
    "data_source": "Doctor Input UI"
  },
  "patients": [
    {
      "id": 1,
      "name": "Patient Name",
      "age": 45,
      "gender": "Male",
      "diagnosisDate": "2024-01-15",
      "hasDiseaseX": true,
      "riskLevel": "Medium Risk",
      "hasAlert": true,
      "comorbidities": ["Diabetes"],
      "symptoms": ["Fever", "Cough"],
      "physician": {
        "id": 12345,
        "specialty": "Internal Medicine",
        "experience": "10 years"
      },
      "dataSource": "Doctor Input",
      "createdAt": "2024-01-01T00:00:00Z",
      "lastUpdated": "2024-01-01T00:00:00Z"
    }
  ]
}
```

## Benefits of This Structure

1. **Clear Separation**: Patient data, system data, and logs are clearly separated
2. **Scalability**: Easy to add new data types without cluttering the main data directory
3. **Maintainability**: Each file has a specific purpose and clear naming convention
4. **Backup Friendly**: Easy to backup specific data types
5. **Version Control**: Can track changes to specific data types independently

## Migration Notes

- Old `persistent_patients.json` has been moved to `patients/doctor_input_patients.json`
- The new structure maintains backward compatibility
- All existing functionality continues to work with the new file locations
