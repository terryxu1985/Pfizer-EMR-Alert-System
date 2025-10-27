# Startup Scripts

This directory contains Python startup scripts for the Pfizer EMR Alert System.

## Available Scripts

### `run_api_only.py`
Starts only the FastAPI backend service.

**Features:**
- FastAPI server on port 8000
- Hot reload for development
- Health check endpoint
- API documentation at `/docs`

**Usage:**
```bash
python scripts/startup/run_api_only.py
```

### `run_complete_system.py`
Starts both API and UI servers with enhanced doctor input functionality.

**Features:**
- API server on port 8000
- UI server on port 8080
- Automatic browser opening
- Process management and cleanup
- Multi-step doctor data input

**Usage:**
```bash
python scripts/startup/run_complete_system.py
```

### `run_quick_start.py`
Convenience script that delegates to the complete system launcher.

**Usage:**
```bash
python scripts/startup/run_quick_start.py
```

## Integration with bin/ Scripts

These Python scripts are used by the shell scripts in the `bin/` directory:

- `bin/start_api.sh` → `scripts/startup/run_api_only.py`
- `bin/start_complete.sh` → `scripts/startup/run_api_only.py` + `frontend/server/emr_ui_server.py`

## Docker Integration

The Docker setup also uses these scripts:
- `docker-compose.yml` references `scripts/startup/run_api_only.py`

## Migration Notes

These scripts were moved from the project root directory to `scripts/startup/` for better organization:

**Before:**
```
run_api_only.py
run_complete_system.py
run_quick_start.py
```

**After:**
```
scripts/startup/
├── run_api_only.py
├── run_complete_system.py
└── run_quick_start.py
```

## Development

When modifying these scripts, ensure:
1. They work from the project root directory
2. They maintain compatibility with the bin/ scripts
3. They work in Docker containers
4. They handle errors gracefully
