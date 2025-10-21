"""
Debug the health check endpoint
"""
import sys
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.api.model_manager import ModelManager
from backend.api.api_models import HealthResponse

def test_health_check():
    """Test health check logic"""
    try:
        print("Testing health check logic...")
        manager = ModelManager()
        manager.load_model()
        
        print("Model loaded successfully")
        
        # Test get_model_info
        model_info = manager.get_model_info()
        print(f"Model info: {model_info}")
        
        # Test health check response
        health_response = HealthResponse(
            status="healthy" if manager.is_loaded else "unhealthy",
            model_loaded=manager.is_loaded,
            model_info=model_info,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        print(f"Health response: {health_response.dict()}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_health_check()
