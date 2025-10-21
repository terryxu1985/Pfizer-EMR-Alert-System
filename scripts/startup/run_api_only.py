#!/usr/bin/env python3
"""
Start Pfizer EMR Alert System API service Developed by Dr. Terry Xu
"""
import sys
import uvicorn
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.api.api import app

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
