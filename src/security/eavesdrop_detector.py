#!/usr/bin/env python3
"""
Eavesdropping Detection System

Quantum security monitoring and intrusion detection for quantum networks.
Detects eavesdropping attempts through quantum bit error rate analysis.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import statistics
from collections import deque, defaultdict


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Represents a security event or anomaly."""
    event_id: str
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source_node: str
    target_node: Optional[str]
    description: str
    metrics: Dict[str, float]


class EavesdropDetector:
    """Quantum eavesdropping detection and security monitoring system."""
    
    def __init__(self, config):
        """Initialize the eavesdropping detector.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.qber_threshold = config.get('security.eavesdrop_threshold', 0.11)
        self.monitoring_interval = config.get('security.monitoring_interval', 1.0)
        self.alert_threshold = config.get('security.alert_threshold', 0.05)
        
        # Monitoring data
        self.qber_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.security_events: List[SecurityEvent] = []
        self.active_channels: Dict[str, dict] = {}
        
        self.running = False
        self.event_counter = 0
    
    async def initialize(self):
        """Initialize the eavesdropping detector."""
        print("Initializing Eavesdropping Detector...")
        # Detector initialization logic
    
    async def start(self):
        """Start the eavesdropping detector."""
        self.running = True
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_qber())
        asyncio.create_task(self._analyze_patterns())
        
        print("Eavesdropping Detector started")
    
    async def shutdown(self):
        """Shutdown the eavesdropping detector."""
        self.running = False
        print("Eavesdropping Detector shutdown")
    
    async def register_channel(self, channel_id: str, node1: str, node2: str):
        """Register a quantum channel for monitoring.
        
        Args:
            channel_id: Unique channel identifier
            node1: First node ID
            node2: Second node ID
        """
        self.active_channels[channel_id] = {
            'nodes': (node1, node2),
            'start_time': time.time(),
            'packets_sent': 0,
            'errors_detected': 0
        }
        print(f"Registered channel {channel_id} for monitoring")
    
    async def report_qber(self, channel_id: str, error_rate: float):
        """Report quantum bit error rate for a channel.
        
        Args:
            channel_id: Channel identifier
            error_rate: Measured QBER (0.0 to 1.0)
        """
        if channel_id not in self.active_channels:
            return
        
        # Store QBER measurement
        self.qber_history[channel_id].append((time.time(), error_rate))
        
        # Check for eavesdropping
        if error_rate > self.qber_threshold:
            await self._trigger_security_event(
                event_type="high_qber",
                threat_level=ThreatLevel.HIGH,
                channel_id=channel_id,
                description=f"High QBER detected: {error_rate:.3f}",
                metrics={'qber': error_rate, 'threshold': self.qber_threshold}
            )
    
    async def report_packet_transmission(self, channel_id: str, success: bool):
        """Report packet transmission result.
        
        Args:
            channel_id: Channel identifier
            success: Whether transmission was successful
        """
        if channel_id not in self.active_channels:
            return
        
        channel = self.active_channels[channel_id]
        channel['packets_sent'] += 1
        
        if not success:
            channel['errors_detected'] += 1
            
            # Calculate current error rate
            error_rate = channel['errors_detected'] / channel['packets_sent']
            await self.report_qber(channel_id, error_rate)
    
    async def _monitor_qber(self):
        """Background task to monitor QBER across all channels."""
        while self.running:
            try:
                current_time = time.time()
                
                for channel_id, history in self.qber_history.items():
                    if len(history) < 10:  # Need minimum samples
                        continue
                    
                    # Analyze recent QBER trend
                    recent_measurements = [
                        rate for timestamp, rate in history
                        if current_time - timestamp < 60  # Last minute
                    ]
                    
                    if recent_measurements:
                        avg_qber = statistics.mean(recent_measurements)
                        qber_variance = statistics.variance(recent_measurements) if len(recent_measurements) > 1 else 0
                        
                        # Detect unusual patterns
                        if qber_variance > 0.01:  # High variance might indicate attack
                            await self._trigger_security_event(
                                event_type="qber_variance",
                                threat_level=ThreatLevel.MEDIUM,
                                channel_id=channel_id,
                                description=f"High QBER variance detected: {qber_variance:.4f}",
                                metrics={'variance': qber_variance, 'avg_qber': avg_qber}
                            )
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in QBER monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _analyze_patterns(self):
        """Background task to analyze security patterns."""
        while self.running:
            try:
                # Analyze channel patterns
                await self._analyze_channel_patterns()
                
                # Analyze temporal patterns
                await self._analyze_temporal_patterns()
                
                await asyncio.sleep(30)  # Analyze every 30 seconds
                
            except Exception as e:
                print(f"Error in pattern analysis: {e}")
                await asyncio.sleep(30)
    
    async def _analyze_channel_patterns(self):
        """Analyze patterns across multiple channels."""
        if len(self.active_channels) < 2:
            return
        
        # Look for correlated attacks across channels
        current_time = time.time()
        recent_events = [
            event for event in self.security_events
            if current_time - event.timestamp < 300  # Last 5 minutes
        ]
        
        if len(recent_events) > 5:  # Multiple events in short time
            await self._trigger_security_event(
                event_type="coordinated_attack",
                threat_level=ThreatLevel.CRITICAL,
                channel_id="multiple",
                description=f"Potential coordinated attack: {len(recent_events)} events in 5 minutes",
                metrics={'event_count': len(recent_events)}
            )
    
    async def _analyze_temporal_patterns(self):
        """Analyze temporal patterns in security events."""
        current_time = time.time()
        
        # Count events in different time windows
        windows = [60, 300, 900]  # 1 min, 5 min, 15 min
        
        for window in windows:
            recent_count = sum(
                1 for event in self.security_events
                if current_time - event.timestamp < window
            )
            
            # Define thresholds based on window size
            threshold = max(3, window // 60)
            
            if recent_count > threshold:
                await self._trigger_security_event(
                    event_type="high_event_rate",
                    threat_level=ThreatLevel.HIGH,
                    channel_id="system",
                    description=f"High security event rate: {recent_count} events in {window}s",
                    metrics={'event_count': recent_count, 'window': window}
                )
    
    async def _trigger_security_event(self, event_type: str, threat_level: ThreatLevel,
                                    channel_id: str, description: str, 
                                    metrics: Dict[str, float]):
        """Trigger a security event.
        
        Args:
            event_type: Type of security event
            threat_level: Severity level
            channel_id: Associated channel ID
            description: Event description
            metrics: Associated metrics
        """
        self.event_counter += 1
        event_id = f"sec_event_{self.event_counter}"
        
        # Extract node information
        source_node = None
        target_node = None
        if channel_id in self.active_channels:
            nodes = self.active_channels[channel_id]['nodes']
            source_node = nodes[0]
            target_node = nodes[1]
        
        event = SecurityEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source_node=source_node or "unknown",
            target_node=target_node,
            description=description,
            metrics=metrics
        )
        
        self.security_events.append(event)
        
        # Log event
        print(f"SECURITY EVENT [{threat_level.value.upper()}] {event_id}: {description}")
        
        # Trigger immediate response for critical events
        if threat_level == ThreatLevel.CRITICAL:
            await self._respond_to_critical_event(event)
    
    async def _respond_to_critical_event(self, event: SecurityEvent):
        """Respond to critical security events.
        
        Args:
            event: Critical security event
        """
        print(f"CRITICAL RESPONSE: Implementing security measures for {event.event_id}")
        
        # Could implement:
        # - Automatic channel isolation
        # - Key refresh
        # - Network reconfiguration
        # - Alert external systems
    
    async def get_security_status(self) -> dict:
        """Get current security status.
        
        Returns:
            Dictionary with security metrics and status
        """
        current_time = time.time()
        
        # Count recent events by threat level
        recent_events = [
            event for event in self.security_events
            if current_time - event.timestamp < 3600  # Last hour
        ]
        
        threat_counts = {
            level.value: sum(1 for event in recent_events if event.threat_level == level)
            for level in ThreatLevel
        }
        
        # Calculate average QBER
        avg_qber = 0.0
        if self.qber_history:
            all_recent_qber = []
            for history in self.qber_history.values():
                recent_qber = [
                    rate for timestamp, rate in history
                    if current_time - timestamp < 300  # Last 5 minutes
                ]
                all_recent_qber.extend(recent_qber)
            
            if all_recent_qber:
                avg_qber = statistics.mean(all_recent_qber)
        
        return {
            'running': self.running,
            'monitored_channels': len(self.active_channels),
            'total_events': len(self.security_events),
            'recent_events': len(recent_events),
            'threat_counts': threat_counts,
            'average_qber': avg_qber,
            'qber_threshold': self.qber_threshold
        }
