/**
 * EMR Alert System - Configuration
 * Centralized configuration management
 */

const CONFIG = {
    // API Configuration
    API: {
        BASE_URL: 'http://localhost:8000',
        TIMEOUT: 30000,
        RETRY_COUNT: 3,
        RETRY_DELAY: 1000
    },
    
    // UI Configuration
    UI: {
        HEALTH_CHECK_INTERVAL: 30000,
        ANIMATION_DURATION: 300,
        DEBOUNCE_DELAY: 500
    },
    
    // Feature Flags
    FEATURES: {
        ENABLE_ANALYTICS: false,
        ENABLE_OFFLINE_MODE: false,
        ENABLE_PUSH_NOTIFICATIONS: false
    },
    
    // Form Configuration
    FORM: {
        MAX_STEPS: 4,
        AUTO_SAVE_INTERVAL: 30000,
        VALIDATION_DEBOUNCE: 300
    },
    
    // Error Messages
    MESSAGES: {
        ERRORS: {
            NETWORK_ERROR: 'Network connection failed. Please check your internet connection.',
            API_ERROR: 'Server error occurred. Please try again later.',
            VALIDATION_ERROR: 'Please fill in all required fields correctly.',
            UNAUTHORIZED: 'You are not authorized to perform this action.',
            NOT_FOUND: 'The requested resource was not found.',
            TIMEOUT: 'Request timed out. Please try again.'
        },
        SUCCESS: {
            PATIENT_SAVED: 'Patient information saved successfully!',
            ANALYSIS_COMPLETE: 'AI analysis completed successfully!',
            DATA_REFRESHED: 'Data refreshed successfully!'
        },
        INFO: {
            LOADING: 'Loading...',
            PROCESSING: 'Processing your request...',
            SAVING: 'Saving changes...'
        }
    },
    
    // API Endpoints
    ENDPOINTS: {
        HEALTH: '/health',
        PATIENTS: '/patients',
        PATIENT_BY_ID: '/patients/{id}',
        PREDICT: '/predict',
        MODEL_INFO: '/model/info',
        MODEL_FEATURES: '/model/features'
    },
    
    // Local Storage Keys
    STORAGE: {
        CURRENT_PATIENT: 'emr_current_patient',
        FORM_DATA: 'emr_form_data',
        USER_PREFERENCES: 'emr_user_preferences',
        CACHE_TIMESTAMP: 'emr_cache_timestamp'
    }
};

// Environment-specific overrides
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    // Development environment
    CONFIG.API.BASE_URL = 'http://localhost:8000';
    CONFIG.FEATURES.ENABLE_ANALYTICS = false;
} else {
    // Production environment
    CONFIG.API.BASE_URL = window.location.origin.replace(':8080', ':8000');
    CONFIG.FEATURES.ENABLE_ANALYTICS = true;
}

// Export configuration
window.CONFIG = CONFIG;
