"""
Metrics Collector for Performance Monitoring

This module provides comprehensive metrics collection capabilities,
including real-time metrics aggregation, storage, and retrieval.
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class MetricValue:
    """Single metric value with metadata"""
    timestamp: str
    metric_name: str
    value: Union[float, int, str]
    tags: Dict[str, str]
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MetricAggregation:
    """Aggregated metric data"""
    metric_name: str
    time_window: str
    count: int
    sum_value: float
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    p50_value: float
    p95_value: float
    p99_value: float
    tags: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetricsCollector:
    """
    Comprehensive metrics collection and aggregation system
    """
    
    def __init__(self, db_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metrics collector
        
        Args:
            db_path: Path to SQLite database for storing metrics
            config: Configuration dictionary
        """
        self.db_path = db_path or "logs/metrics_collector.db"
        self.config = config or self._get_default_config()
        
        # In-memory metrics storage for real-time access
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=self.config['buffer_size']))
        self.metrics_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Background aggregation thread
        self._aggregation_thread = None
        self._stop_aggregation = threading.Event()
        
        # Start background aggregation if enabled
        if self.config['background_aggregation_enabled']:
            self._start_aggregation_thread()
        
        logger.info("Metrics Collector initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'metrics_collection_enabled': True,
            'buffer_size': 10000,
            'aggregation_intervals': ['1m', '5m', '15m', '1h'],
            'background_aggregation_enabled': True,
            'aggregation_thread_interval': 60,  # seconds
            'metrics_retention_days': 30,
            'batch_insert_size': 1000,
            'real_time_metrics': [
                'prediction_count',
                'prediction_latency_ms',
                'error_count',
                'accuracy_score',
                'model_confidence'
            ]
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metrics storage"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create raw metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS raw_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL,
                        tags TEXT,
                        source TEXT
                    )
                ''')
                
                # Create aggregated metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS aggregated_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        time_window TEXT NOT NULL,
                        count INTEGER,
                        sum_value REAL,
                        min_value REAL,
                        max_value REAL,
                        mean_value REAL,
                        std_value REAL,
                        p50_value REAL,
                        p95_value REAL,
                        p99_value REAL,
                        tags TEXT
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_metrics_timestamp ON raw_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_metrics_name ON raw_metrics(metric_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_agg_metrics_timestamp ON aggregated_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_agg_metrics_name_window ON aggregated_metrics(metric_name, time_window)')
                
                conn.commit()
                logger.info(f"Metrics database initialized: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
            raise
    
    def record_metric(self, metric_name: str, value: Union[float, int, str], 
                     tags: Optional[Dict[str, str]] = None, 
                     source: str = "system") -> None:
        """
        Record a single metric value
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
            source: Source of the metric
        """
        if not self.config['metrics_collection_enabled']:
            return
        
        timestamp = datetime.utcnow().isoformat() + "Z"
        tags = tags or {}
        
        metric_value = MetricValue(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value,
            tags=tags,
            source=source
        )
        
        with self.metrics_lock:
            self.metrics_buffer[metric_name].append(metric_value)
        
        # Store in database if it's a real-time metric
        if metric_name in self.config['real_time_metrics']:
            self._store_metric(metric_value)
    
    def record_prediction_metrics(self, prediction_data: Dict[str, Any]) -> None:
        """
        Record prediction-related metrics
        
        Args:
            prediction_data: Dictionary containing prediction information
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Record prediction count
        self.record_metric("prediction_count", 1, {
            "model_version": prediction_data.get("model_version", "unknown"),
            "prediction_type": "single"
        })
        
        # Record prediction latency
        if "response_time_ms" in prediction_data:
            self.record_metric("prediction_latency_ms", prediction_data["response_time_ms"], {
                "model_version": prediction_data.get("model_version", "unknown")
            })
        
        # Record prediction confidence
        if "probability" in prediction_data:
            self.record_metric("model_confidence", prediction_data["probability"], {
                "model_version": prediction_data.get("model_version", "unknown"),
                "prediction": str(prediction_data.get("prediction", "unknown"))
            })
        
        # Record accuracy if actual label is available
        if "actual_label" in prediction_data and prediction_data["actual_label"] is not None:
            predicted = prediction_data.get("prediction")
            actual = prediction_data["actual_label"]
            accuracy = 1.0 if predicted == actual else 0.0
            
            self.record_metric("accuracy_score", accuracy, {
                "model_version": prediction_data.get("model_version", "unknown")
            })
    
    def record_error_metrics(self, error_data: Dict[str, Any]) -> None:
        """
        Record error-related metrics
        
        Args:
            error_data: Dictionary containing error information
        """
        # Record error count
        self.record_metric("error_count", 1, {
            "error_type": error_data.get("error_type", "unknown"),
            "model_version": error_data.get("model_version", "unknown")
        })
        
        # Record error rate (if we have prediction count)
        if "prediction_count" in error_data:
            error_rate = error_data.get("error_count", 0) / error_data["prediction_count"]
            self.record_metric("error_rate", error_rate, {
                "error_type": error_data.get("error_type", "unknown"),
                "model_version": error_data.get("model_version", "unknown")
            })
    
    def record_performance_metrics(self, performance_data: Dict[str, Any]) -> None:
        """
        Record performance-related metrics
        
        Args:
            performance_data: Dictionary containing performance information
        """
        metrics_to_record = [
            "pr_auc", "precision", "recall", "f1_score", "accuracy",
            "roc_auc", "specificity", "sensitivity"
        ]
        
        for metric in metrics_to_record:
            if metric in performance_data:
                self.record_metric(metric, performance_data[metric], {
                    "model_version": performance_data.get("model_version", "unknown"),
                    "evaluation_type": performance_data.get("evaluation_type", "unknown")
                })
    
    def record_drift_metrics(self, drift_data: Dict[str, Any]) -> None:
        """
        Record drift detection metrics
        
        Args:
            drift_data: Dictionary containing drift detection information
        """
        if "drift_score" in drift_data:
            self.record_metric("drift_score", drift_data["drift_score"], {
                "feature_name": drift_data.get("feature_name", "unknown"),
                "drift_type": drift_data.get("drift_type", "unknown"),
                "severity": drift_data.get("severity", "unknown")
            })
        
        if "p_value" in drift_data:
            self.record_metric("drift_p_value", drift_data["p_value"], {
                "feature_name": drift_data.get("feature_name", "unknown"),
                "test_method": drift_data.get("test_method", "unknown")
            })
    
    def _store_metric(self, metric_value: MetricValue) -> None:
        """Store metric value in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO raw_metrics (timestamp, metric_name, value, tags, source)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    metric_value.timestamp,
                    metric_value.metric_name,
                    metric_value.value,
                    json.dumps(metric_value.tags),
                    metric_value.source
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
    
    def _start_aggregation_thread(self) -> None:
        """Start background aggregation thread"""
        self._aggregation_thread = threading.Thread(
            target=self._aggregation_worker,
            daemon=True
        )
        self._aggregation_thread.start()
        logger.info("Background aggregation thread started")
    
    def _aggregation_worker(self) -> None:
        """Background worker for metrics aggregation"""
        while not self._stop_aggregation.is_set():
            try:
                self._aggregate_metrics()
                time.sleep(self.config['aggregation_thread_interval'])
            except Exception as e:
                logger.error(f"Error in aggregation worker: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _aggregate_metrics(self) -> None:
        """Aggregate metrics for all configured intervals"""
        try:
            for interval in self.config['aggregation_intervals']:
                self._aggregate_for_interval(interval)
        except Exception as e:
            logger.error(f"Error in metrics aggregation: {e}")
    
    def _aggregate_for_interval(self, interval: str) -> None:
        """
        Aggregate metrics for a specific time interval
        
        Args:
            interval: Time interval (e.g., '1m', '5m', '1h')
        """
        try:
            # Calculate time window
            interval_minutes = self._parse_interval(interval)
            window_start = datetime.utcnow() - timedelta(minutes=interval_minutes)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get metrics for the time window
                cursor.execute('''
                    SELECT metric_name, value, tags
                    FROM raw_metrics
                    WHERE timestamp > ?
                    ORDER BY metric_name, timestamp
                ''', (window_start.isoformat() + "Z",))
                
                metrics_rows = cursor.fetchall()
                
                if not metrics_rows:
                    return
                
                # Group metrics by name and tags
                grouped_metrics = defaultdict(list)
                for metric_name, value, tags_json in metrics_rows:
                    tags = json.loads(tags_json) if tags_json else {}
                    key = (metric_name, json.dumps(tags, sort_keys=True))
                    grouped_metrics[key].append(value)
                
                # Calculate aggregations
                aggregations = []
                for (metric_name, tags_json), values in grouped_metrics.items():
                    if not values:
                        continue
                    
                    # Convert to numeric values
                    numeric_values = []
                    for value in values:
                        try:
                            numeric_values.append(float(value))
                        except (ValueError, TypeError):
                            continue
                    
                    if not numeric_values:
                        continue
                    
                    # Calculate statistics
                    aggregation = MetricAggregation(
                        metric_name=metric_name,
                        time_window=interval,
                        count=len(numeric_values),
                        sum_value=sum(numeric_values),
                        min_value=min(numeric_values),
                        max_value=max(numeric_values),
                        mean_value=np.mean(numeric_values),
                        std_value=np.std(numeric_values),
                        p50_value=np.percentile(numeric_values, 50),
                        p95_value=np.percentile(numeric_values, 95),
                        p99_value=np.percentile(numeric_values, 99),
                        tags=json.loads(tags_json) if tags_json else {}
                    )
                    
                    aggregations.append(aggregation)
                
                # Store aggregations
                self._store_aggregations(aggregations, window_start)
                
        except Exception as e:
            logger.error(f"Error aggregating metrics for interval {interval}: {e}")
    
    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to minutes"""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 24 * 60
        else:
            return int(interval)
    
    def _store_aggregations(self, aggregations: List[MetricAggregation], timestamp: datetime) -> None:
        """Store aggregated metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for aggregation in aggregations:
                    cursor.execute('''
                        INSERT INTO aggregated_metrics 
                        (timestamp, metric_name, time_window, count, sum_value, min_value, max_value, 
                         mean_value, std_value, p50_value, p95_value, p99_value, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp.isoformat() + "Z",
                        aggregation.metric_name,
                        aggregation.time_window,
                        aggregation.count,
                        aggregation.sum_value,
                        aggregation.min_value,
                        aggregation.max_value,
                        aggregation.mean_value,
                        aggregation.std_value,
                        aggregation.p50_value,
                        aggregation.p95_value,
                        aggregation.p99_value,
                        json.dumps(aggregation.tags)
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store aggregations: {e}")
    
    def get_metric_values(self, metric_name: str, hours: int = 24, 
                         tags: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Get metric values for a specific metric
        
        Args:
            metric_name: Name of the metric
            hours: Number of hours to look back
            tags: Optional tags to filter by
            
        Returns:
            List of metric value dictionaries
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if tags:
                    # Filter by tags
                    tag_conditions = []
                    tag_values = []
                    for key, value in tags.items():
                        tag_conditions.append("json_extract(tags, '$.{}') = ?".format(key))
                        tag_values.append(value)
                    
                    where_clause = " AND ".join(tag_conditions)
                    query = f'''
                        SELECT timestamp, metric_name, value, tags, source
                        FROM raw_metrics
                        WHERE metric_name = ? AND timestamp > ? AND {where_clause}
                        ORDER BY timestamp DESC
                    '''
                    params = [metric_name, cutoff_time.isoformat() + "Z"] + tag_values
                else:
                    query = '''
                        SELECT timestamp, metric_name, value, tags, source
                        FROM raw_metrics
                        WHERE metric_name = ? AND timestamp > ?
                        ORDER BY timestamp DESC
                    '''
                    params = [metric_name, cutoff_time.isoformat() + "Z"]
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [
                    {
                        'timestamp': row[0],
                        'metric_name': row[1],
                        'value': row[2],
                        'tags': json.loads(row[3]) if row[3] else {},
                        'source': row[4]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get metric values: {e}")
            return []
    
    def get_aggregated_metrics(self, metric_name: str, interval: str = "1h", 
                              hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get aggregated metrics for a specific metric and interval
        
        Args:
            metric_name: Name of the metric
            interval: Aggregation interval
            hours: Number of hours to look back
            
        Returns:
            List of aggregated metric dictionaries
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, metric_name, time_window, count, sum_value, min_value, max_value,
                           mean_value, std_value, p50_value, p95_value, p99_value, tags
                    FROM aggregated_metrics
                    WHERE metric_name = ? AND time_window = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                ''', (metric_name, interval, cutoff_time.isoformat() + "Z"))
                
                rows = cursor.fetchall()
                
                return [
                    {
                        'timestamp': row[0],
                        'metric_name': row[1],
                        'time_window': row[2],
                        'count': row[3],
                        'sum_value': row[4],
                        'min_value': row[5],
                        'max_value': row[6],
                        'mean_value': row[7],
                        'std_value': row[8],
                        'p50_value': row[9],
                        'p95_value': row[10],
                        'p99_value': row[11],
                        'tags': json.loads(row[12]) if row[12] else {}
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get aggregated metrics: {e}")
            return []
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of all metrics for specified time period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Metrics summary dictionary
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get metric counts
                cursor.execute('''
                    SELECT metric_name, COUNT(*) as count, AVG(value) as avg_value
                    FROM raw_metrics
                    WHERE timestamp > ?
                    GROUP BY metric_name
                    ORDER BY count DESC
                ''', (cutoff_time.isoformat() + "Z",))
                
                metric_counts = cursor.fetchall()
                
                # Get recent metrics from buffer
                recent_metrics = {}
                with self.metrics_lock:
                    for metric_name, values in self.metrics_buffer.items():
                        if values:
                            recent_values = [v.value for v in values if isinstance(v.value, (int, float))]
                            if recent_values:
                                recent_metrics[metric_name] = {
                                    'count': len(recent_values),
                                    'latest_value': recent_values[-1],
                                    'avg_value': np.mean(recent_values),
                                    'min_value': min(recent_values),
                                    'max_value': max(recent_values)
                                }
                
                return {
                    'time_period_hours': hours,
                    'total_metrics': len(metric_counts),
                    'metric_counts': [
                        {
                            'metric_name': row[0],
                            'count': row[1],
                            'avg_value': row[2]
                        }
                        for row in metric_counts
                    ],
                    'recent_metrics': recent_metrics,
                    'buffer_size': sum(len(values) for values in self.metrics_buffer.values())
                }
                
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}
    
    def cleanup_old_metrics(self, retention_days: Optional[int] = None) -> None:
        """
        Clean up old metrics data
        
        Args:
            retention_days: Number of days to retain data
        """
        retention_days = retention_days or self.config['metrics_retention_days']
        cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean up raw metrics
                cursor.execute('DELETE FROM raw_metrics WHERE timestamp < ?', 
                             (cutoff_time.isoformat() + "Z",))
                raw_deleted = cursor.rowcount
                
                # Clean up aggregated metrics
                cursor.execute('DELETE FROM aggregated_metrics WHERE timestamp < ?', 
                             (cutoff_time.isoformat() + "Z",))
                agg_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up old metrics: {raw_deleted} raw, {agg_deleted} aggregated")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
    
    def stop(self) -> None:
        """Stop the metrics collector"""
        if self._aggregation_thread:
            self._stop_aggregation.set()
            self._aggregation_thread.join(timeout=5)
            logger.info("Metrics collector stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get metrics collector status"""
        return {
            'metrics_collection_enabled': self.config['metrics_collection_enabled'],
            'background_aggregation_enabled': self.config['background_aggregation_enabled'],
            'buffer_size': sum(len(values) for values in self.metrics_buffer.values()),
            'database_path': self.db_path,
            'aggregation_intervals': self.config['aggregation_intervals'],
            'real_time_metrics': self.config['real_time_metrics']
        }
