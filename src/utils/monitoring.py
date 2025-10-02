#!/usr/bin/env python3
"""
System Monitoring Utilities

System performance monitoring and metrics collection for the quantum network platform.
Tracks CPU, memory, network, and application-specific metrics.
"""

import asyncio
from typing import Dict, List, Optional, Any
import time
import json
from collections import deque
from dataclasses import dataclass


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_total: float
    network_bytes_sent: int
    network_bytes_recv: int
    disk_usage: float
    process_count: int
    thread_count: int


class SystemMonitor:
    """System performance monitoring and alerting."""
    
    def __init__(self, config):
        """Initialize system monitor.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.monitoring_enabled = config.get('performance.monitoring_enabled', True)
        self.metrics_interval = config.get('performance.metrics_interval', 30)
        self.memory_threshold = config.get('performance.memory_threshold', 0.8)
        self.cpu_threshold = config.get('performance.cpu_threshold', 0.9)
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=1000)
        self.alerts_history: List[Dict[str, Any]] = []
        
        self.running = False
    
    async def start(self):
        """Start system monitoring."""
        if not self.monitoring_enabled:
            print("System monitoring disabled")
            return
        
        self.running = True
        
        # Start monitoring tasks
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._alerting_loop())
        
        print("System Monitor started")
    
    async def stop(self):
        """Stop system monitoring."""
        self.running = False
        print("System Monitor stopped")
    
    async def _metrics_collection_loop(self):
        """Background task for metrics collection."""
        while self.running:
            try:
                # Collect current system metrics
                metrics = await self._collect_system_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                await asyncio.sleep(self.metrics_interval)
                
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.metrics_interval)
    
    async def _alerting_loop(self):
        """Background task for system alerting."""
        while self.running:
            try:
                if len(self.metrics_history) > 0:
                    latest_metrics = self.metrics_history[-1]
                    await self._check_thresholds(latest_metrics)
                
                await asyncio.sleep(60)  # Check thresholds every minute
                
            except Exception as e:
                print(f"Error in alerting loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics.
        
        Returns:
            Current system metrics
        """
        # Simulate system metrics collection
        # In a real implementation, this would use psutil or similar library
        
        current_time = time.time()
        
        # Simulate varying metrics
        import random
        cpu_usage = 0.1 + random.random() * 0.3  # 10-40% CPU
        memory_usage = 0.3 + random.random() * 0.2  # 30-50% memory
        
        return SystemMetrics(
            timestamp=current_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_total=16.0,  # 16GB
            network_bytes_sent=random.randint(1000, 10000),
            network_bytes_recv=random.randint(1000, 10000),
            disk_usage=0.45,  # 45% disk usage
            process_count=random.randint(150, 200),
            thread_count=random.randint(800, 1200)
        )
    
    async def _check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against configured thresholds.
        
        Args:
            metrics: Current system metrics
        """
        alerts = []
        
        # Check CPU threshold
        if metrics.cpu_usage > self.cpu_threshold:
            alerts.append({
                'type': 'cpu_high',
                'severity': 'warning',
                'message': f'High CPU usage: {metrics.cpu_usage:.1%}',
                'value': metrics.cpu_usage,
                'threshold': self.cpu_threshold
            })
        
        # Check memory threshold
        if metrics.memory_usage > self.memory_threshold:
            alerts.append({
                'type': 'memory_high',
                'severity': 'warning',
                'message': f'High memory usage: {metrics.memory_usage:.1%}',
                'value': metrics.memory_usage,
                'threshold': self.memory_threshold
            })
        
        # Process alerts
        for alert in alerts:
            await self._process_alert(alert, metrics)
    
    async def _process_alert(self, alert: Dict[str, Any], metrics: SystemMetrics):
        """Process system alert.
        
        Args:
            alert: Alert information
            metrics: Associated metrics
        """
        alert['timestamp'] = metrics.timestamp
        alert['metrics_snapshot'] = {
            'cpu': metrics.cpu_usage,
            'memory': metrics.memory_usage,
            'processes': metrics.process_count,
            'threads': metrics.thread_count
        }
        
        self.alerts_history.append(alert)
        
        # Log alert
        severity_upper = alert['severity'].upper()
        print(f"SYSTEM ALERT [{severity_upper}]: {alert['message']}")
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = time.time() - 86400
        self.alerts_history = [
            alert for alert in self.alerts_history
            if alert['timestamp'] >= cutoff_time
        ]
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent system metrics.
        
        Returns:
            Latest system metrics or None if not available
        """
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """Get metrics history for specified time period.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of historical metrics
        """
        if not self.metrics_history:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics.
        
        Returns:
            Performance summary data
        """
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        # Calculate averages over last hour
        recent_metrics = self.get_metrics_history(1)
        
        if not recent_metrics:
            latest = self.metrics_history[-1]
            return {
                'current_cpu': latest.cpu_usage,
                'current_memory': latest.memory_usage,
                'current_processes': latest.process_count,
                'monitoring_duration': 0
            }
        
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_processes = sum(m.process_count for m in recent_metrics) / len(recent_metrics)
        
        return {
            'average_cpu_1h': avg_cpu,
            'average_memory_1h': avg_memory,
            'average_processes_1h': avg_processes,
            'current_cpu': recent_metrics[-1].cpu_usage,
            'current_memory': recent_metrics[-1].memory_usage,
            'recent_alerts': len([
                alert for alert in self.alerts_history
                if time.time() - alert['timestamp'] < 3600
            ]),
            'monitoring_duration': len(recent_metrics) * self.metrics_interval / 60,  # minutes
            'thresholds': {
                'cpu': self.cpu_threshold,
                'memory': self.memory_threshold
            }
        }
    
    def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics to JSON file.
        
        Args:
            filepath: Output file path
            hours: Hours of data to export
        """
        metrics_data = self.get_metrics_history(hours)
        
        # Convert to serializable format
        export_data = {
            'export_timestamp': time.time(),
            'metrics_count': len(metrics_data),
            'time_range_hours': hours,
            'metrics': [
                {
                    'timestamp': m.timestamp,
                    'cpu_usage': m.cpu_usage,
                    'memory_usage': m.memory_usage,
                    'memory_total': m.memory_total,
                    'network_bytes_sent': m.network_bytes_sent,
                    'network_bytes_recv': m.network_bytes_recv,
                    'disk_usage': m.disk_usage,
                    'process_count': m.process_count,
                    'thread_count': m.thread_count
                }
                for m in metrics_data
            ],
            'alerts': self.alerts_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported {len(metrics_data)} metrics to {filepath}")
