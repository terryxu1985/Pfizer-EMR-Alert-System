"""
Alert System for Performance Monitoring and Drift Detection

This module provides automated alerting capabilities for performance degradation
and data drift detection, including multiple notification channels and alert management.
"""

import logging
import json
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertChannel(Enum):
    """Alert notification channels"""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    CONSOLE = "console"

class AlertType(Enum):
    """Types of alerts"""
    PERFORMANCE_DRIFT = "performance_drift"
    DATA_DRIFT = "data_drift"
    MODEL_ERROR = "model_error"
    SYSTEM_ERROR = "system_error"
    THRESHOLD_EXCEEDED = "threshold_exceeded"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: str
    alert_type: AlertType
    severity: str
    title: str
    message: str
    details: Dict[str, Any]
    channels: List[AlertChannel]
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    alert_type: AlertType
    condition: str
    threshold: float
    severity: str
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown_minutes: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AlertSystem:
    """
    Comprehensive alert system for performance monitoring and drift detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize alert system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.alert_rules = []
        self.alert_history = []
        self.active_alerts = {}
        self.notification_channels = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize notification channels
        self._init_notification_channels()
        
        # Load default alert rules
        self._load_default_rules()
        
        logger.info("Alert System initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'alert_system_enabled': True,
            'max_alerts_per_hour': 10,
            'alert_retention_days': 30,
            'email': {
                'enabled': False,
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_email': 'alerts@pfizer-emr.com',
                'to_emails': []
            },
            'webhook': {
                'enabled': False,
                'url': '',
                'headers': {}
            },
            'slack': {
                'enabled': False,
                'webhook_url': '',
                'channel': '#alerts'
            }
        }
    
    def _init_notification_channels(self) -> None:
        """Initialize notification channels"""
        # Log channel (always available)
        self.notification_channels[AlertChannel.LOG] = self._log_notification
        
        # Console channel (always available)
        self.notification_channels[AlertChannel.CONSOLE] = self._console_notification
        
        # Email channel
        if self.config['email']['enabled']:
            self.notification_channels[AlertChannel.EMAIL] = self._email_notification
        
        # Webhook channel
        if self.config['webhook']['enabled']:
            self.notification_channels[AlertChannel.WEBHOOK] = self._webhook_notification
        
        # Slack channel
        if self.config['slack']['enabled']:
            self.notification_channels[AlertChannel.SLACK] = self._slack_notification
    
    def _load_default_rules(self) -> None:
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                name="Performance Drift - High",
                alert_type=AlertType.PERFORMANCE_DRIFT,
                condition="decline_percentage > 0.10",
                threshold=0.10,
                severity="high",
                channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
                cooldown_minutes=60
            ),
            AlertRule(
                name="Performance Drift - Critical",
                alert_type=AlertType.PERFORMANCE_DRIFT,
                condition="decline_percentage > 0.15",
                threshold=0.15,
                severity="critical",
                channels=[AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.EMAIL],
                cooldown_minutes=30
            ),
            AlertRule(
                name="Data Drift - High",
                alert_type=AlertType.DATA_DRIFT,
                condition="drift_score > 0.2",
                threshold=0.2,
                severity="high",
                channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
                cooldown_minutes=120
            ),
            AlertRule(
                name="Data Drift - Critical",
                alert_type=AlertType.DATA_DRIFT,
                condition="drift_score > 0.3",
                threshold=0.3,
                severity="critical",
                channels=[AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.EMAIL],
                cooldown_minutes=60
            ),
            AlertRule(
                name="Model Error Rate",
                alert_type=AlertType.MODEL_ERROR,
                condition="error_rate > 0.05",
                threshold=0.05,
                severity="medium",
                channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
                cooldown_minutes=30
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """
        Add a new alert rule
        
        Args:
            rule: Alert rule to add
        """
        with self._lock:
            self.alert_rules.append(rule)
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule by name
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            for i, rule in enumerate(self.alert_rules):
                if rule.name == rule_name:
                    del self.alert_rules[i]
                    logger.info(f"Removed alert rule: {rule_name}")
                    return True
            return False
    
    def process_performance_drift_alert(self, drift_alert: Dict[str, Any]) -> None:
        """
        Process performance drift alert
        
        Args:
            drift_alert: Performance drift alert data
        """
        if not self.config['alert_system_enabled']:
            return
        
        # Check if alert should be triggered
        applicable_rules = [
            rule for rule in self.alert_rules
            if (rule.alert_type == AlertType.PERFORMANCE_DRIFT and 
                rule.enabled and
                drift_alert.get('decline_percentage', 0) >= rule.threshold)
        ]
        
        if not applicable_rules:
            return
        
        # Create alert
        alert = Alert(
            id=f"perf_drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            alert_type=AlertType.PERFORMANCE_DRIFT,
            severity=max(rule.severity for rule in applicable_rules),
            title=f"Performance Drift Detected - {drift_alert.get('metric', 'Unknown')}",
            message=f"Performance decline detected: {drift_alert.get('decline_percentage', 0):.1%} "
                   f"in {drift_alert.get('metric', 'Unknown')} metric",
            details=drift_alert,
            channels=list(set(channel for rule in applicable_rules for channel in rule.channels))
        )
        
        self._process_alert(alert, applicable_rules)
    
    def process_data_drift_alert(self, drift_result: Dict[str, Any]) -> None:
        """
        Process data drift alert
        
        Args:
            drift_result: Data drift detection result
        """
        if not self.config['alert_system_enabled']:
            return
        
        # Check if alert should be triggered
        applicable_rules = [
            rule for rule in self.alert_rules
            if (rule.alert_type == AlertType.DATA_DRIFT and 
                rule.enabled and
                drift_result.get('drift_score', 0) >= rule.threshold)
        ]
        
        if not applicable_rules:
            return
        
        # Create alert
        alert = Alert(
            id=f"data_drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            alert_type=AlertType.DATA_DRIFT,
            severity=max(rule.severity for rule in applicable_rules),
            title=f"Data Drift Detected - {drift_result.get('feature_name', 'Unknown')}",
            message=f"Data drift detected in feature {drift_result.get('feature_name', 'Unknown')}: "
                   f"drift score {drift_result.get('drift_score', 0):.4f}",
            details=drift_result,
            channels=list(set(channel for rule in applicable_rules for channel in rule.channels))
        )
        
        self._process_alert(alert, applicable_rules)
    
    def process_model_error_alert(self, error_data: Dict[str, Any]) -> None:
        """
        Process model error alert
        
        Args:
            error_data: Model error data
        """
        if not self.config['alert_system_enabled']:
            return
        
        # Calculate error rate
        error_rate = error_data.get('error_count', 0) / max(error_data.get('prediction_count', 1), 1)
        
        # Check if alert should be triggered
        applicable_rules = [
            rule for rule in self.alert_rules
            if (rule.alert_type == AlertType.MODEL_ERROR and 
                rule.enabled and
                error_rate >= rule.threshold)
        ]
        
        if not applicable_rules:
            return
        
        # Create alert
        alert = Alert(
            id=f"model_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            alert_type=AlertType.MODEL_ERROR,
            severity=max(rule.severity for rule in applicable_rules),
            title="Model Error Rate Exceeded",
            message=f"Model error rate {error_rate:.1%} exceeds threshold",
            details=error_data,
            channels=list(set(channel for rule in applicable_rules for channel in rule.channels))
        )
        
        self._process_alert(alert, applicable_rules)
    
    def _process_alert(self, alert: Alert, applicable_rules: List[AlertRule]) -> None:
        """
        Process and send alert
        
        Args:
            alert: Alert to process
            applicable_rules: Rules that triggered the alert
        """
        with self._lock:
            # Check cooldown periods
            if self._is_in_cooldown(alert, applicable_rules):
                logger.debug(f"Alert {alert.id} suppressed due to cooldown")
                return
            
            # Check rate limiting
            if self._is_rate_limited():
                logger.warning("Alert suppressed due to rate limiting")
                return
            
            # Store alert
            self.alert_history.append(alert)
            self.active_alerts[alert.id] = alert
            
            # Send notifications
            self._send_notifications(alert)
            
            logger.info(f"Alert processed: {alert.title} (severity: {alert.severity})")
    
    def _is_in_cooldown(self, alert: Alert, applicable_rules: List[AlertRule]) -> bool:
        """Check if alert is in cooldown period"""
        for rule in applicable_rules:
            cooldown_cutoff = datetime.utcnow() - timedelta(minutes=rule.cooldown_minutes)
            
            # Check if similar alert was sent recently
            for historical_alert in reversed(self.alert_history):
                if (historical_alert.alert_type == alert.alert_type and
                    historical_alert.details.get('metric') == alert.details.get('metric') and
                    datetime.fromisoformat(historical_alert.timestamp.replace('Z', '')) > cooldown_cutoff):
                    return True
        
        return False
    
    def _is_rate_limited(self) -> bool:
        """Check if alert rate limit is exceeded"""
        hour_cutoff = datetime.utcnow() - timedelta(hours=1)
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert.timestamp.replace('Z', '')) > hour_cutoff
        ]
        
        return len(recent_alerts) >= self.config['max_alerts_per_hour']
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications through configured channels"""
        for channel in alert.channels:
            if channel in self.notification_channels:
                try:
                    self.notification_channels[channel](alert)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel.value}: {e}")
    
    def _log_notification(self, alert: Alert) -> None:
        """Send notification via logging"""
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT [{alert.severity.upper()}]: {alert.title} - {alert.message}")
    
    def _console_notification(self, alert: Alert) -> None:
        """Send notification to console"""
        severity_colors = {
            'low': '\033[94m',      # Blue
            'medium': '\033[93m',   # Yellow
            'high': '\033[91m',     # Red
            'critical': '\033[95m'  # Magenta
        }
        reset_color = '\033[0m'
        
        color = severity_colors.get(alert.severity, '')
        print(f"{color}[{alert.severity.upper()}] {alert.title}{reset_color}")
        print(f"  {alert.message}")
        print(f"  Time: {alert.timestamp}")
        print()
    
    def _email_notification(self, alert: Alert) -> None:
        """Send notification via email"""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = ', '.join(email_config['to_emails'])
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
            
            # Create email body
            body = f"""
Alert Details:
- Type: {alert.alert_type.value}
- Severity: {alert.severity}
- Time: {alert.timestamp}
- Message: {alert.message}

Details:
{json.dumps(alert.details, indent=2)}

This is an automated alert from the Pfizer EMR Alert System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _webhook_notification(self, alert: Alert) -> None:
        """Send notification via webhook"""
        try:
            webhook_config = self.config['webhook']
            
            payload = {
                'alert_id': alert.id,
                'timestamp': alert.timestamp,
                'type': alert.alert_type.value,
                'severity': alert.severity,
                'title': alert.title,
                'message': alert.message,
                'details': alert.details
            }
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook alert sent: {alert.title}")
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _slack_notification(self, alert: Alert) -> None:
        """Send notification via Slack"""
        try:
            slack_config = self.config['slack']
            
            # Determine color based on severity
            colors = {
                'low': '#36a64f',      # Green
                'medium': '#ff9800',   # Orange
                'high': '#f44336',     # Red
                'critical': '#9c27b0'  # Purple
            }
            
            payload = {
                'channel': slack_config['channel'],
                'attachments': [{
                    'color': colors.get(alert.severity, '#ff9800'),
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity, 'short': True},
                        {'title': 'Type', 'value': alert.alert_type.value, 'short': True},
                        {'title': 'Time', 'value': alert.timestamp, 'short': False}
                    ],
                    'footer': 'Pfizer EMR Alert System',
                    'ts': int(datetime.utcnow().timestamp())
                }]
            }
            
            response = requests.post(slack_config['webhook_url'], json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert.title}")
            else:
                logger.error(f"Slack alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if acknowledged, False if not found
        """
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if resolved, False if not found
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.utcnow().isoformat() + "Z"
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert_id}")
                return True
            return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        with self._lock:
            return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get alert history for specified time period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of alert dictionaries
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            recent_alerts = [
                alert for alert in self.alert_history
                if datetime.fromisoformat(alert.timestamp.replace('Z', '')) > cutoff_time
            ]
            
            return [alert.to_dict() for alert in recent_alerts]
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get alert statistics for specified time period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Alert statistics dictionary
        """
        recent_alerts = self.get_alert_history(hours)
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'severity_counts': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                'type_counts': {},
                'acknowledged_count': 0,
                'resolved_count': 0
            }
        
        # Count by severity
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for alert in recent_alerts:
            severity_counts[alert['severity']] += 1
        
        # Count by type
        type_counts = {}
        for alert in recent_alerts:
            alert_type = alert['alert_type']
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        # Count acknowledged and resolved
        acknowledged_count = sum(1 for alert in recent_alerts if alert['acknowledged'])
        resolved_count = sum(1 for alert in recent_alerts if alert['resolved'])
        
        return {
            'total_alerts': len(recent_alerts),
            'severity_counts': severity_counts,
            'type_counts': type_counts,
            'acknowledged_count': acknowledged_count,
            'resolved_count': resolved_count,
            'time_period_hours': hours
        }
    
    def cleanup_old_alerts(self, retention_days: Optional[int] = None) -> None:
        """
        Clean up old alerts
        
        Args:
            retention_days: Number of days to retain alerts
        """
        retention_days = retention_days or self.config['alert_retention_days']
        cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
        
        with self._lock:
            original_count = len(self.alert_history)
            self.alert_history = [
                alert for alert in self.alert_history
                if datetime.fromisoformat(alert.timestamp.replace('Z', '')) > cutoff_time
            ]
            
            removed_count = original_count - len(self.alert_history)
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old alerts")
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert system status"""
        return {
            'alert_system_enabled': self.config['alert_system_enabled'],
            'active_alerts_count': len(self.active_alerts),
            'total_rules': len(self.alert_rules),
            'enabled_rules': len([rule for rule in self.alert_rules if rule.enabled]),
            'notification_channels': list(self.notification_channels.keys()),
            'recent_alerts_count': len(self.get_alert_history(hours=1))
        }
