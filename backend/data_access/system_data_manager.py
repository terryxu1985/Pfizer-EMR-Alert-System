"""
System Data Manager for EMR Alert System
Manages system-level persistent data including configuration, state, and user preferences
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class SystemDataManager:
    """Manages system-level persistent data"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data" / "storage" / "system"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # System data files
        self.model_cache_file = self.data_dir / "model_cache.json"
        self.system_state_file = self.data_dir / "system_state.json"
        self.user_preferences_file = self.data_dir / "user_preferences.json"
    
    def load_model_cache(self) -> Dict[str, Any]:
        """Load model cache data"""
        try:
            if self.model_cache_file.exists():
                with open(self.model_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return self._get_default_model_cache()
        except Exception as e:
            print(f"❌ Error loading model cache: {str(e)}")
            return self._get_default_model_cache()
    
    def save_model_cache(self, data: Dict[str, Any]) -> bool:
        """Save model cache data"""
        try:
            data['metadata']['last_updated'] = datetime.now().isoformat()
            with open(self.model_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Error saving model cache: {str(e)}")
            return False
    
    def load_system_state(self) -> Dict[str, Any]:
        """Load system state data"""
        try:
            if self.system_state_file.exists():
                with open(self.system_state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return self._get_default_system_state()
        except Exception as e:
            print(f"❌ Error loading system state: {str(e)}")
            return self._get_default_system_state()
    
    def save_system_state(self, data: Dict[str, Any]) -> bool:
        """Save system state data"""
        try:
            data['metadata']['last_updated'] = datetime.now().isoformat()
            with open(self.system_state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Error saving system state: {str(e)}")
            return False
    
    def load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences data"""
        try:
            if self.user_preferences_file.exists():
                with open(self.user_preferences_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return self._get_default_user_preferences()
        except Exception as e:
            print(f"❌ Error loading user preferences: {str(e)}")
            return self._get_default_user_preferences()
    
    def save_user_preferences(self, data: Dict[str, Any]) -> bool:
        """Save user preferences data"""
        try:
            data['metadata']['last_updated'] = datetime.now().isoformat()
            with open(self.user_preferences_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Error saving user preferences: {str(e)}")
            return False
    
    def update_model_status(self, is_loaded: bool, model_path: str = None, error: str = None) -> bool:
        """Update model loading status"""
        try:
            cache_data = self.load_model_cache()
            cache_data['model_info']['is_loaded'] = is_loaded
            cache_data['model_info']['last_loaded'] = datetime.now().isoformat() if is_loaded else None
            cache_data['model_info']['load_attempts'] += 1
            
            if model_path:
                cache_data['model_info']['current_model_path'] = model_path
            
            if error:
                cache_data['model_info']['last_error'] = error
            else:
                cache_data['model_info']['last_error'] = None
            
            return self.save_model_cache(cache_data)
        except Exception as e:
            print(f"❌ Error updating model status: {str(e)}")
            return False
    
    def update_system_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Update system performance metrics"""
        try:
            state_data = self.load_system_state()
            state_data['performance_metrics'].update(metrics)
            return self.save_system_state(state_data)
        except Exception as e:
            print(f"❌ Error updating system metrics: {str(e)}")
            return False
    
    def _get_default_model_cache(self) -> Dict[str, Any]:
        """Get default model cache structure"""
        return {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "file_type": "model_cache",
                "description": "Model loading and caching information"
            },
            "model_info": {
                "current_model_path": None,
                "model_type": "XGBoost",
                "model_version": "1.0.0",
                "is_loaded": False,
                "last_loaded": None,
                "load_attempts": 0,
                "last_error": None
            },
            "feature_info": {
                "feature_count": 0,
                "categorical_features": [],
                "numerical_features": []
            },
            "performance_metrics": {
                "last_training_accuracy": 0.0,
                "last_validation_accuracy": 0.0,
                "last_test_accuracy": 0.0,
                "feature_importance": {}
            }
        }
    
    def _get_default_system_state(self) -> Dict[str, Any]:
        """Get default system state structure"""
        return {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "file_type": "system_state",
                "description": "Current system runtime state and status"
            },
            "system_status": {
                "api_status": "unknown",
                "model_status": "unknown",
                "database_status": "unknown",
                "last_health_check": None,
                "uptime_seconds": 0,
                "startup_time": datetime.now().isoformat()
            },
            "performance_metrics": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time_ms": 0,
                "peak_memory_usage_mb": 0,
                "current_memory_usage_mb": 0
            },
            "feature_flags": {
                "enable_ai_analysis": True,
                "enable_patient_creation": True,
                "enable_batch_processing": True,
                "enable_caching": True,
                "debug_mode": False,
                "maintenance_mode": False
            },
            "data_sources": {
                "real_emr_data_available": False,
                "generated_data_available": True,
                "sample_data_available": True,
                "last_data_refresh": None
            },
            "errors": {
                "recent_errors": [],
                "error_count_24h": 0,
                "last_error": None
            }
        }
    
    def _get_default_user_preferences(self) -> Dict[str, Any]:
        """Get default user preferences structure"""
        return {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "file_type": "user_preferences",
                "description": "User interface preferences and settings"
            },
            "ui_preferences": {
                "theme": "light",
                "language": "en",
                "timezone": "UTC",
                "date_format": "YYYY-MM-DD",
                "time_format": "24h"
            },
            "display_settings": {
                "patients_per_page": 20,
                "show_advanced_fields": False,
                "auto_refresh_interval": 30,
                "show_debug_info": False,
                "compact_view": False
            },
            "alert_settings": {
                "enable_sound_alerts": True,
                "enable_visual_alerts": True,
                "alert_threshold": 0.7,
                "auto_dismiss_alerts": False,
                "alert_duration_seconds": 10
            },
            "data_preferences": {
                "default_risk_level_filter": "all",
                "default_physician_filter": "all",
                "default_date_range_days": 30,
                "show_sample_data": True,
                "show_generated_data": True
            },
            "export_settings": {
                "default_export_format": "json",
                "include_metadata": True,
                "include_timestamps": True,
                "compress_exports": False
            },
            "recent_activity": {
                "last_viewed_patient": None,
                "last_analysis_date": None,
                "favorite_physicians": [],
                "recent_searches": []
            }
        }

# Global instance
system_data_manager = SystemDataManager()
