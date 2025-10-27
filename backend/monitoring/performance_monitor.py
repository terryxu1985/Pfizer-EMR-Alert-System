"""
Performance Monitor for Real-time Model Performance Tracking

This module provides continuous monitoring of model performance metrics,
detects performance drift, and triggers automated alerts when thresholds are exceeded.
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: str
    pr_auc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    prediction_count: int
    error_count: int
    avg_response_time_ms: float
    model_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DriftAlert:
    """Drift detection alert"""
    timestamp: str
    metric: str
    baseline_value: float
    current_value: float
    decline_percentage: float
    alert_level: AlertLevel
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PerformanceMonitor:
    """
    Real-time performance monitoring and drift detection system
    """
    
    def __init__(self, db_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitor
        
        Args:
            db_path: Path to SQLite database for storing metrics
            config: Configuration dictionary with thresholds and settings
        """
        self.db_path = db_path or "logs/performance_monitor.db"
        self.config = config or self._get_default_config()
        
        # Performance baselines (from current model)
        self.baseline_metrics = {
            'pr_auc': 0.881,
            'precision': 0.844,
            'recall': 0.849,
            'f1_score': 0.846,
            'accuracy': 0.746
        }
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'pr_auc': 0.05,      # 5% decline threshold
            'precision': 0.03,   # 3% decline threshold
            'recall': 0.03,      # 3% decline threshold
            'f1_score': 0.03,   # 3% decline threshold
            'accuracy': 0.05     # 5% decline threshold
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            AlertLevel.LOW: 0.02,      # 2% decline
            AlertLevel.MEDIUM: 0.05,   # 5% decline
            AlertLevel.HIGH: 0.10,    # 10% decline
            AlertLevel.CRITICAL: 0.15 # 15% decline
        }
        
        # Initialize database
        self._init_database()
        
        # Performance tracking
        self.current_metrics = {}
        self.prediction_history = []
        self.error_history = []
        self.response_times = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("Performance Monitor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'monitoring_enabled': True,
            'metrics_retention_days': 30,
            'drift_check_interval_minutes': 15,
            'alert_cooldown_minutes': 60,
            'min_predictions_for_drift_check': 100
        }
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        pr_auc REAL,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        accuracy REAL,
                        prediction_count INTEGER,
                        error_count INTEGER,
                        avg_response_time_ms REAL,
                        model_version TEXT
                    )
                ''')
                
                # Create drift alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS drift_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric TEXT NOT NULL,
                        baseline_value REAL,
                        current_value REAL,
                        decline_percentage REAL,
                        alert_level TEXT,
                        message TEXT
                    )
                ''')
                
                # Create prediction history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS prediction_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        prediction REAL,
                        actual_label INTEGER,
                        probability REAL,
                        response_time_ms REAL,
                        model_version TEXT
                    )
                ''')
                
                conn.commit()
                logger.info(f"Performance monitoring database initialized: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize performance monitoring database: {e}")
            raise
    
    def record_prediction(self, prediction: int, probability: float, 
                         actual_label: Optional[int] = None, 
                         response_time_ms: float = 0.0,
                         model_version: str = "unknown") -> None:
        """
        Record a single prediction for performance tracking
        
        Args:
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            actual_label: Actual label if available (for accuracy calculation)
            response_time_ms: Response time in milliseconds
            model_version: Model version used for prediction
        """
        with self._lock:
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            # Store prediction data
            prediction_data = {
                'timestamp': timestamp,
                'prediction': prediction,
                'probability': probability,
                'actual_label': actual_label,
                'response_time_ms': response_time_ms,
                'model_version': model_version
            }
            
            self.prediction_history.append(prediction_data)
            self.response_times.append(response_time_ms)
            
            # Store in database
            self._store_prediction(prediction_data)
            
            # Check if we have enough data for drift detection
            if len(self.prediction_history) >= self.config['min_predictions_for_drift_check']:
                self._check_performance_drift()
    
    def record_error(self, error_type: str, error_message: str, 
                    model_version: str = "unknown") -> None:
        """
        Record an error for monitoring
        
        Args:
            error_type: Type of error (prediction_error, validation_error, etc.)
            error_message: Error message
            model_version: Model version when error occurred
        """
        with self._lock:
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            error_data = {
                'timestamp': timestamp,
                'error_type': error_type,
                'error_message': error_message,
                'model_version': model_version
            }
            
            self.error_history.append(error_data)
            logger.warning(f"Performance monitor recorded error: {error_type} - {error_message}")
    
    def _store_prediction(self, prediction_data: Dict[str, Any]) -> None:
        """Store prediction data in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO prediction_history 
                    (timestamp, prediction, actual_label, probability, response_time_ms, model_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_data['timestamp'],
                    prediction_data['prediction'],
                    prediction_data['actual_label'],
                    prediction_data['probability'],
                    prediction_data['response_time_ms'],
                    prediction_data['model_version']
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store prediction data: {e}")
    
    def calculate_current_metrics(self) -> PerformanceMetrics:
        """
        Calculate current performance metrics from recent predictions
        
        Returns:
            PerformanceMetrics object with current performance data
        """
        with self._lock:
            if not self.prediction_history:
                return None
            
            # Get recent predictions (last hour)
            recent_cutoff = datetime.utcnow() - timedelta(hours=1)
            recent_predictions = [
                p for p in self.prediction_history
                if datetime.fromisoformat(p['timestamp'].replace('Z', '')) > recent_cutoff
            ]
            
            if not recent_predictions:
                return None
            
            # Calculate metrics
            predictions = [p['prediction'] for p in recent_predictions]
            probabilities = [p['probability'] for p in recent_predictions]
            actual_labels = [p['actual_label'] for p in recent_predictions if p['actual_label'] is not None]
            response_times = [p['response_time_ms'] for p in recent_predictions]
            
            # Basic metrics
            prediction_count = len(recent_predictions)
            error_count = len(self.error_history)
            avg_response_time = np.mean(response_times) if response_times else 0.0
            
            # Calculate accuracy if we have actual labels
            accuracy = None
            if actual_labels and len(actual_labels) > 0:
                predicted_labels = [p['prediction'] for p in recent_predictions if p['actual_label'] is not None]
                accuracy = np.mean([p == a for p, a in zip(predicted_labels, actual_labels)])
            
            # For now, use baseline values for metrics that require ground truth
            # In production, these would be calculated from actual labels
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow().isoformat() + "Z",
                pr_auc=self.baseline_metrics['pr_auc'],  # Would need ground truth
                precision=self.baseline_metrics['precision'],  # Would need ground truth
                recall=self.baseline_metrics['recall'],  # Would need ground truth
                f1_score=self.baseline_metrics['f1_score'],  # Would need ground truth
                accuracy=accuracy or self.baseline_metrics['accuracy'],
                prediction_count=prediction_count,
                error_count=error_count,
                avg_response_time_ms=avg_response_time,
                model_version=recent_predictions[0]['model_version'] if recent_predictions else "unknown"
            )
            
            return metrics
    
    def _check_performance_drift(self) -> List[DriftAlert]:
        """
        Check for performance drift and generate alerts
        
        Returns:
            List of drift alerts
        """
        current_metrics = self.calculate_current_metrics()
        if not current_metrics:
            return []
        
        alerts = []
        
        # Check each metric for drift
        for metric_name in self.drift_thresholds.keys():
            baseline_value = self.baseline_metrics[metric_name]
            current_value = getattr(current_metrics, metric_name)
            
            if current_value is None:
                continue
            
            # Calculate decline percentage
            decline_percentage = (baseline_value - current_value) / baseline_value
            
            # Check if drift exceeds threshold
            threshold = self.drift_thresholds[metric_name]
            if decline_percentage > threshold:
                # Determine alert level
                alert_level = self._determine_alert_level(decline_percentage)
                
                alert = DriftAlert(
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    metric=metric_name,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    decline_percentage=decline_percentage,
                    alert_level=alert_level,
                    message=f"Performance drift detected in {metric_name}: "
                           f"{decline_percentage:.1%} decline from baseline"
                )
                
                alerts.append(alert)
                self._store_drift_alert(alert)
                
                logger.warning(f"Performance drift alert: {alert.message}")
        
        return alerts
    
    def _determine_alert_level(self, decline_percentage: float) -> AlertLevel:
        """Determine alert level based on decline percentage"""
        if decline_percentage >= self.alert_thresholds[AlertLevel.CRITICAL]:
            return AlertLevel.CRITICAL
        elif decline_percentage >= self.alert_thresholds[AlertLevel.HIGH]:
            return AlertLevel.HIGH
        elif decline_percentage >= self.alert_thresholds[AlertLevel.MEDIUM]:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
    
    def _store_drift_alert(self, alert: DriftAlert) -> None:
        """Store drift alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO drift_alerts 
                    (timestamp, metric, baseline_value, current_value, decline_percentage, alert_level, message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.timestamp,
                    alert.metric,
                    alert.baseline_value,
                    alert.current_value,
                    alert.decline_percentage,
                    alert.alert_level.value,
                    alert.message
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store drift alert: {e}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for the specified time period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Performance summary dictionary
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent metrics
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                cursor.execute('''
                    SELECT * FROM performance_metrics 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', (cutoff_time.isoformat() + "Z",))
                
                metrics_rows = cursor.fetchall()
                
                # Get recent alerts
                cursor.execute('''
                    SELECT * FROM drift_alerts 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', (cutoff_time.isoformat() + "Z",))
                
                alerts_rows = cursor.fetchall()
                
                # Get prediction statistics
                cursor.execute('''
                    SELECT COUNT(*) as total_predictions,
                           AVG(response_time_ms) as avg_response_time,
                           COUNT(CASE WHEN actual_label IS NOT NULL THEN 1 END) as labeled_predictions
                    FROM prediction_history 
                    WHERE timestamp > ?
                ''', (cutoff_time.isoformat() + "Z",))
                
                stats_row = cursor.fetchone()
                
                return {
                    'time_period_hours': hours,
                    'total_predictions': stats_row[0] if stats_row else 0,
                    'avg_response_time_ms': stats_row[1] if stats_row and stats_row[1] else 0,
                    'labeled_predictions': stats_row[2] if stats_row else 0,
                    'recent_metrics': metrics_rows,
                    'recent_alerts': alerts_rows,
                    'current_baseline': self.baseline_metrics,
                    'drift_thresholds': self.drift_thresholds
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def get_drift_alerts(self, hours: int = 24, alert_level: Optional[AlertLevel] = None) -> List[Dict[str, Any]]:
        """
        Get drift alerts for the specified time period
        
        Args:
            hours: Number of hours to look back
            alert_level: Filter by alert level (optional)
            
        Returns:
            List of drift alert dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                
                if alert_level:
                    cursor.execute('''
                        SELECT * FROM drift_alerts 
                        WHERE timestamp > ? AND alert_level = ?
                        ORDER BY timestamp DESC
                    ''', (cutoff_time.isoformat() + "Z", alert_level.value))
                else:
                    cursor.execute('''
                        SELECT * FROM drift_alerts 
                        WHERE timestamp > ?
                        ORDER BY timestamp DESC
                    ''', (cutoff_time.isoformat() + "Z",))
                
                alerts_rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                alerts = []
                for row in alerts_rows:
                    alerts.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'metric': row[2],
                        'baseline_value': row[3],
                        'current_value': row[4],
                        'decline_percentage': row[5],
                        'alert_level': row[6],
                        'message': row[7]
                    })
                
                return alerts
                
        except Exception as e:
            logger.error(f"Failed to get drift alerts: {e}")
            return []
    
    def update_baseline_metrics(self, new_baseline: Dict[str, float]) -> None:
        """
        Update baseline metrics (e.g., after model retraining)
        
        Args:
            new_baseline: New baseline metrics dictionary
        """
        with self._lock:
            self.baseline_metrics.update(new_baseline)
            logger.info(f"Updated baseline metrics: {new_baseline}")
    
    def cleanup_old_data(self, retention_days: Optional[int] = None) -> None:
        """
        Clean up old performance data
        
        Args:
            retention_days: Number of days to retain data (uses config default if None)
        """
        retention_days = retention_days or self.config['metrics_retention_days']
        cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean up old performance metrics
                cursor.execute('DELETE FROM performance_metrics WHERE timestamp < ?', 
                             (cutoff_time.isoformat() + "Z",))
                metrics_deleted = cursor.rowcount
                
                # Clean up old drift alerts
                cursor.execute('DELETE FROM drift_alerts WHERE timestamp < ?', 
                             (cutoff_time.isoformat() + "Z",))
                alerts_deleted = cursor.rowcount
                
                # Clean up old prediction history
                cursor.execute('DELETE FROM prediction_history WHERE timestamp < ?', 
                             (cutoff_time.isoformat() + "Z",))
                predictions_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up old data: {metrics_deleted} metrics, "
                           f"{alerts_deleted} alerts, {predictions_deleted} predictions")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status
        
        Returns:
            Status dictionary
        """
        current_metrics = self.calculate_current_metrics()
        recent_alerts = self.get_drift_alerts(hours=1)
        
        return {
            'monitoring_enabled': self.config['monitoring_enabled'],
            'current_metrics': current_metrics.to_dict() if current_metrics else None,
            'baseline_metrics': self.baseline_metrics,
            'recent_alerts_count': len(recent_alerts),
            'recent_alerts': recent_alerts[:5],  # Last 5 alerts
            'prediction_history_size': len(self.prediction_history),
            'error_history_size': len(self.error_history),
            'database_path': self.db_path
        }
