# Backend Monitoring System Guide

## Overview

The backend monitoring system provides comprehensive real-time monitoring capabilities for the Pfizer EMR Alert System. It includes performance monitoring, data drift detection, metrics collection, and automated alerting to ensure the system operates reliably and maintains high performance standards.

## Architecture

The monitoring system consists of four main components:

1. **Alert System** (`alert_system.py`) - Automated alerting and notification management
2. **Drift Detector** (`drift_detector.py`) - Data drift detection and analysis
3. **Metrics Collector** (`metrics_collector.py`) - Real-time metrics collection and aggregation
4. **Performance Monitor** (`performance_monitor.py`) - Model performance tracking and drift detection

## Components

### 1. Alert System (`alert_system.py`)

The Alert System provides comprehensive alerting capabilities with multiple notification channels and intelligent alert management.

#### Key Features:
- **Multiple Alert Channels**: Log, Console, Email, Webhook, Slack
- **Alert Types**: Performance drift, data drift, model errors, system errors, threshold exceeded
- **Severity Levels**: Low, Medium, High, Critical
- **Rate Limiting**: Prevents alert spam with configurable cooldown periods
- **Alert Management**: Acknowledge, resolve, and track alert lifecycle

#### Usage Example:
```python
from backend.monitoring.alert_system import AlertSystem, AlertRule, AlertType, AlertChannel

# Initialize alert system
alert_system = AlertSystem()

# Add custom alert rule
rule = AlertRule(
    name="Custom Performance Alert",
    alert_type=AlertType.PERFORMANCE_DRIFT,
    condition="decline_percentage > 0.08",
    threshold=0.08,
    severity="high",
    channels=[AlertChannel.LOG, AlertChannel.EMAIL],
    cooldown_minutes=30
)
alert_system.add_alert_rule(rule)

# Process performance drift alert
drift_data = {
    'metric': 'pr_auc',
    'decline_percentage': 0.12,
    'baseline_value': 0.881,
    'current_value': 0.775
}
alert_system.process_performance_drift_alert(drift_data)
```

#### Configuration:
```python
config = {
    'alert_system_enabled': True,
    'max_alerts_per_hour': 10,
    'alert_retention_days': 30,
    'email': {
        'enabled': True,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'alerts@company.com',
        'password': 'password',
        'from_email': 'alerts@pfizer-emr.com',
        'to_emails': ['admin@company.com']
    },
    'slack': {
        'enabled': True,
        'webhook_url': 'https://hooks.slack.com/services/...',
        'channel': '#alerts'
    }
}
```

### 2. Drift Detector (`drift_detector.py`)

The Drift Detector identifies data drift in input features using statistical tests and distribution comparisons.

#### Key Features:
- **Statistical Tests**: Kolmogorov-Smirnov test, Chi-square test, Wasserstein distance
- **Feature Types**: Supports both numerical and categorical features
- **Multivariate Analysis**: PCA-based multivariate drift detection
- **Severity Classification**: Automatic severity assessment (low, medium, high, critical)
- **Reference Data Management**: Load, update, and manage reference datasets

#### Usage Example:
```python
from backend.monitoring.drift_detector import DataDriftDetector
import pandas as pd

# Initialize drift detector with reference data
detector = DataDriftDetector(reference_data_path="data/model_ready_dataset.csv")

# Detect drift in new data
new_data = pd.read_csv("data/new_batch.csv")
drift_results = detector.detect_drift(new_data)

# Process results
for result in drift_results:
    if result.drift_detected:
        print(f"Drift detected in {result.feature_name}: {result.message}")
        print(f"Severity: {result.severity}, Score: {result.drift_score:.4f}")
```

#### Statistical Tests:
- **Kolmogorov-Smirnov Test**: Detects distribution changes in numerical features
- **Chi-square Test**: Detects changes in categorical feature distributions
- **Wasserstein Distance**: Measures distribution similarity
- **Multivariate PCA**: Detects complex multivariate drift patterns

### 3. Metrics Collector (`metrics_collector.py`)

The Metrics Collector provides real-time metrics collection, aggregation, and storage capabilities.

#### Key Features:
- **Real-time Collection**: In-memory buffer for immediate access
- **Database Storage**: SQLite-based persistent storage
- **Aggregation**: Multiple time windows (1m, 5m, 15m, 1h)
- **Background Processing**: Automated aggregation in background threads
- **Metric Types**: Prediction metrics, error metrics, performance metrics, drift metrics

#### Usage Example:
```python
from backend.monitoring.metrics_collector import MetricsCollector

# Initialize metrics collector
collector = MetricsCollector(db_path="logs/metrics.db")

# Record prediction metrics
prediction_data = {
    "model_version": "v2.1.0",
    "response_time_ms": 45.2,
    "probability": 0.85,
    "prediction": 1,
    "actual_label": 1
}
collector.record_prediction_metrics(prediction_data)

# Record error metrics
error_data = {
    "error_type": "validation_error",
    "model_version": "v2.1.0",
    "error_count": 1,
    "prediction_count": 100
}
collector.record_error_metrics(error_data)

# Get metrics summary
summary = collector.get_metrics_summary(hours=24)
print(f"Total metrics: {summary['total_metrics']}")
```

#### Aggregation Intervals:
- **1m**: 1-minute aggregations for real-time monitoring
- **5m**: 5-minute aggregations for short-term trends
- **15m**: 15-minute aggregations for medium-term analysis
- **1h**: 1-hour aggregations for long-term reporting

### 4. Performance Monitor (`performance_monitor.py`)

The Performance Monitor tracks model performance metrics and detects performance drift in real-time.

#### Key Features:
- **Real-time Tracking**: Continuous monitoring of prediction performance
- **Drift Detection**: Automatic detection of performance degradation
- **Baseline Management**: Configurable baseline metrics for comparison
- **Alert Generation**: Automatic alerts when performance thresholds are exceeded
- **Historical Analysis**: Long-term performance trend analysis

#### Usage Example:
```python
from backend.monitoring.performance_monitor import PerformanceMonitor

# Initialize performance monitor
monitor = PerformanceMonitor(db_path="logs/performance.db")

# Record prediction
monitor.record_prediction(
    prediction=1,
    probability=0.85,
    actual_label=1,
    response_time_ms=45.2,
    model_version="v2.1.0"
)

# Record error
monitor.record_error(
    error_type="prediction_error",
    error_message="Invalid input format",
    model_version="v2.1.0"
)

# Get performance summary
summary = monitor.get_performance_summary(hours=24)
print(f"Total predictions: {summary['total_predictions']}")
print(f"Average response time: {summary['avg_response_time_ms']:.2f}ms")
```

#### Performance Metrics:
- **PR-AUC**: Precision-Recall Area Under Curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Correct predictions / Total predictions

## Integration

### API Integration

The monitoring system integrates with the API server to automatically collect metrics:

```python
# In api.py
from backend.monitoring.metrics_collector import MetricsCollector
from backend.monitoring.performance_monitor import PerformanceMonitor
from backend.monitoring.alert_system import AlertSystem

# Initialize monitoring components
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor()
alert_system = AlertSystem()

# In prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    try:
        # Make prediction
        prediction, probability = model.predict(features)
        
        # Record metrics
        response_time = (time.time() - start_time) * 1000
        prediction_data = {
            "model_version": model.version,
            "response_time_ms": response_time,
            "probability": probability,
            "prediction": prediction
        }
        
        metrics_collector.record_prediction_metrics(prediction_data)
        performance_monitor.record_prediction(
            prediction=prediction,
            probability=probability,
            response_time_ms=response_time,
            model_version=model.version
        )
        
        return {"prediction": prediction, "probability": probability}
        
    except Exception as e:
        # Record error
        error_data = {
            "error_type": "prediction_error",
            "model_version": model.version,
            "error_count": 1
        }
        metrics_collector.record_error_metrics(error_data)
        performance_monitor.record_error(
            error_type="prediction_error",
            error_message=str(e),
            model_version=model.version
        )
        raise
```

### Data Processing Integration

```python
# In data_processor.py
from backend.monitoring.drift_detector import DataDriftDetector

# Initialize drift detector
drift_detector = DataDriftDetector(reference_data_path="data/model_ready_dataset.csv")

def process_batch(data_batch):
    # Detect drift
    drift_results = drift_detector.detect_drift(data_batch)
    
    # Process drift alerts
    for result in drift_results:
        if result.drift_detected:
            alert_system.process_data_drift_alert(result.to_dict())
    
    # Continue with normal processing
    return processed_data
```

## Configuration

### Environment Variables

```bash
# Monitoring Configuration
MONITORING_ENABLED=true
METRICS_RETENTION_DAYS=30
DRIFT_CHECK_INTERVAL_MINUTES=15
ALERT_COOLDOWN_MINUTES=60

# Email Configuration
EMAIL_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=alerts@company.com
EMAIL_PASSWORD=password
FROM_EMAIL=alerts@pfizer-emr.com
TO_EMAILS=admin@company.com,ops@company.com

# Slack Configuration
SLACK_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_CHANNEL=#alerts
```

### Configuration File

```yaml
# monitoring_config.yaml
monitoring:
  enabled: true
  metrics_retention_days: 30
  drift_check_interval_minutes: 15
  alert_cooldown_minutes: 60

alert_system:
  enabled: true
  max_alerts_per_hour: 10
  alert_retention_days: 30
  
  email:
    enabled: true
    smtp_server: smtp.gmail.com
    smtp_port: 587
    username: alerts@company.com
    password: password
    from_email: alerts@pfizer-emr.com
    to_emails:
      - admin@company.com
      - ops@company.com
  
  slack:
    enabled: true
    webhook_url: https://hooks.slack.com/services/...
    channel: "#alerts"

drift_detector:
  enabled: true
  min_samples_for_drift_check: 100
  max_features_to_check: 50
  statistical_tests:
    - ks_test
    - chi2_test
    - wasserstein_distance
  categorical_threshold: 0.05
  drift_check_interval_hours: 6

performance_monitor:
  enabled: true
  min_predictions_for_drift_check: 100
  drift_thresholds:
    pr_auc: 0.05
    precision: 0.03
    recall: 0.03
    f1_score: 0.03
    accuracy: 0.05
  alert_thresholds:
    low: 0.02
    medium: 0.05
    high: 0.10
    critical: 0.15
```

## Monitoring Dashboard

### Key Metrics to Monitor

1. **System Health**:
   - API response times
   - Error rates
   - Request throughput
   - Database connection status

2. **Model Performance**:
   - Prediction accuracy
   - Model confidence scores
   - Performance drift indicators
   - Feature importance stability

3. **Data Quality**:
   - Data drift scores
   - Missing value rates
   - Feature distribution changes
   - Input validation failures

4. **Operational Metrics**:
   - Prediction volume
   - System resource usage
   - Alert frequency
   - Maintenance windows

### Alert Thresholds

| Metric | Low | Medium | High | Critical |
|--------|-----|--------|------|----------|
| Performance Decline | 2% | 5% | 10% | 15% |
| Data Drift Score | 0.05 | 0.10 | 0.20 | 0.30 |
| Error Rate | 1% | 3% | 5% | 10% |
| Response Time | +20% | +50% | +100% | +200% |

## Troubleshooting

### Common Issues

1. **High Alert Volume**:
   - Check cooldown periods
   - Adjust alert thresholds
   - Review alert rules

2. **Missing Metrics**:
   - Verify database connectivity
   - Check buffer configuration
   - Review collection intervals

3. **Drift Detection Issues**:
   - Ensure reference data is loaded
   - Check sample size requirements
   - Verify feature compatibility

4. **Performance Degradation**:
   - Review baseline metrics
   - Check for data quality issues
   - Analyze recent model changes

### Debug Commands

```python
# Check monitoring status
print(alert_system.get_status())
print(drift_detector.get_status())
print(metrics_collector.get_status())
print(performance_monitor.get_current_status())

# Get recent alerts
recent_alerts = alert_system.get_alert_history(hours=24)
print(f"Recent alerts: {len(recent_alerts)}")

# Get metrics summary
summary = metrics_collector.get_metrics_summary(hours=24)
print(f"Metrics collected: {summary['total_metrics']}")

# Get performance summary
perf_summary = performance_monitor.get_performance_summary(hours=24)
print(f"Total predictions: {perf_summary['total_predictions']}")
```

## Best Practices

### 1. Alert Management
- Set appropriate cooldown periods to prevent alert fatigue
- Use severity levels to prioritize responses
- Implement alert acknowledgment workflows
- Regular review and tuning of alert thresholds

### 2. Metrics Collection
- Collect metrics at appropriate granularity
- Use tags for metric categorization
- Implement data retention policies
- Monitor collection performance impact

### 3. Drift Detection
- Maintain up-to-date reference data
- Use multiple statistical tests for validation
- Set appropriate detection thresholds
- Regular review of drift patterns

### 4. Performance Monitoring
- Establish clear baseline metrics
- Monitor both individual and aggregate performance
- Implement automated remediation where possible
- Regular performance trend analysis

## Maintenance

### Regular Tasks

1. **Daily**:
   - Review alert logs
   - Check system health metrics
   - Monitor performance trends

2. **Weekly**:
   - Analyze drift detection results
   - Review alert thresholds
   - Clean up old data

3. **Monthly**:
   - Update reference data
   - Review and tune monitoring configuration
   - Analyze long-term trends

### Data Cleanup

```python
# Clean up old data
alert_system.cleanup_old_alerts(retention_days=30)
metrics_collector.cleanup_old_metrics(retention_days=30)
performance_monitor.cleanup_old_data(retention_days=30)
```

## API Endpoints

The monitoring system provides several API endpoints for external access:

- `GET /monitoring/status` - Get overall monitoring status
- `GET /monitoring/alerts` - Get recent alerts
- `GET /monitoring/metrics` - Get metrics summary
- `GET /monitoring/performance` - Get performance summary
- `GET /monitoring/drift` - Get drift detection results
- `POST /monitoring/alerts/{alert_id}/acknowledge` - Acknowledge alert
- `POST /monitoring/alerts/{alert_id}/resolve` - Resolve alert

## Security Considerations

1. **Access Control**: Implement proper authentication for monitoring endpoints
2. **Data Privacy**: Ensure sensitive data is not exposed in alerts
3. **Network Security**: Use secure connections for external notifications
4. **Audit Logging**: Log all monitoring activities for compliance

## Future Enhancements

1. **Machine Learning Integration**: Use ML models for anomaly detection
2. **Advanced Visualization**: Real-time dashboards and charts
3. **Predictive Alerting**: Proactive alerts based on trend analysis
4. **Integration APIs**: Connect with external monitoring systems
5. **Automated Remediation**: Self-healing capabilities for common issues

---

This monitoring system provides comprehensive coverage of the Pfizer EMR Alert System's operational health, ensuring reliable performance and early detection of issues that could impact patient care.
