#!/usr/bin/env python3
"""
AI-Based Anomaly Detection

Machine learning algorithms for detecting anomalous behavior in quantum networks.
Implements various ML models for real-time network behavior analysis.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import time
import json


class AnomalyType(Enum):
    """Types of network anomalies."""
    TRAFFIC_SPIKE = "traffic_spike"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    UNUSUAL_ROUTING = "unusual_routing"
    PROTOCOL_VIOLATION = "protocol_violation"
    TIMING_ANOMALY = "timing_anomaly"


@dataclass
class NetworkMetrics:
    """Network performance metrics for analysis."""
    timestamp: float
    node_id: str
    packet_rate: float
    latency: float
    error_rate: float
    throughput: float
    cpu_usage: float
    memory_usage: float


@dataclass
class Anomaly:
    """Detected network anomaly."""
    anomaly_id: str
    timestamp: float
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    node_id: str
    description: str
    metrics: Dict[str, float]
    confidence: float


class AnomalyDetector:
    """AI-powered network anomaly detection system."""
    
    def __init__(self, config):
        """Initialize the anomaly detector.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.model_type = config.get('ai.model_type', 'lstm')
        self.training_window = config.get('ai.training_window', 1000)
        self.prediction_threshold = config.get('ai.prediction_threshold', 0.8)
        
        # Data storage
        self.metrics_history: Dict[str, deque] = {}
        self.detected_anomalies: List[Anomaly] = []
        
        # Model state
        self.models: Dict[str, Any] = {}  # Per-node models
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
        self.running = False
        self.anomaly_counter = 0
    
    async def initialize(self):
        """Initialize the anomaly detector."""
        print("Initializing AI Anomaly Detector...")
        
        # Initialize baseline statistics
        await self._initialize_baselines()
        
        print(f"Anomaly Detector initialized with {self.model_type} model")
    
    async def start(self):
        """Start the anomaly detector."""
        self.running = True
        
        # Start analysis tasks
        asyncio.create_task(self._continuous_analysis())
        asyncio.create_task(self._model_training_loop())
        
        print("AI Anomaly Detector started")
    
    async def shutdown(self):
        """Shutdown the anomaly detector."""
        self.running = False
        print("AI Anomaly Detector shutdown")
    
    async def ingest_metrics(self, metrics: NetworkMetrics):
        """Ingest network metrics for analysis.
        
        Args:
            metrics: Network performance metrics
        """
        node_id = metrics.node_id
        
        # Initialize history for new nodes
        if node_id not in self.metrics_history:
            self.metrics_history[node_id] = deque(maxlen=self.training_window)
            self.baseline_stats[node_id] = {}
        
        # Store metrics
        self.metrics_history[node_id].append(metrics)
        
        # Perform real-time analysis
        await self._analyze_metrics(metrics)
    
    async def _analyze_metrics(self, metrics: NetworkMetrics):
        """Analyze metrics for anomalies.
        
        Args:
            metrics: Current network metrics
        """
        node_id = metrics.node_id
        
        # Statistical anomaly detection
        anomalies = []
        
        # Check for statistical outliers
        stat_anomalies = await self._statistical_analysis(metrics)
        anomalies.extend(stat_anomalies)
        
        # Check for pattern-based anomalies
        pattern_anomalies = await self._pattern_analysis(metrics)
        anomalies.extend(pattern_anomalies)
        
        # ML-based detection (if model is trained)
        if node_id in self.models:
            ml_anomalies = await self._ml_analysis(metrics)
            anomalies.extend(ml_anomalies)
        
        # Process detected anomalies
        for anomaly in anomalies:
            await self._process_anomaly(anomaly)
    
    async def _statistical_analysis(self, metrics: NetworkMetrics) -> List[Anomaly]:
        """Perform statistical anomaly detection.
        
        Args:
            metrics: Current network metrics
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        node_id = metrics.node_id
        
        if node_id not in self.baseline_stats:
            return anomalies
        
        baseline = self.baseline_stats[node_id]
        
        # Check each metric against baseline
        metric_checks = [
            ('packet_rate', metrics.packet_rate, AnomalyType.TRAFFIC_SPIKE),
            ('latency', metrics.latency, AnomalyType.PERFORMANCE_DEGRADATION),
            ('error_rate', metrics.error_rate, AnomalyType.PROTOCOL_VIOLATION),
            ('cpu_usage', metrics.cpu_usage, AnomalyType.PERFORMANCE_DEGRADATION)
        ]
        
        for metric_name, value, anomaly_type in metric_checks:
            if metric_name in baseline:
                mean = baseline[metric_name + '_mean']
                std = baseline[metric_name + '_std']
                
                # Z-score based detection (3-sigma rule)
                if std > 0:
                    z_score = abs(value - mean) / std
                    if z_score > 3:  # 3-sigma threshold
                        self.anomaly_counter += 1
                        anomaly = Anomaly(
                            anomaly_id=f"stat_{self.anomaly_counter}",
                            timestamp=metrics.timestamp,
                            anomaly_type=anomaly_type,
                            severity=min(z_score / 5.0, 1.0),
                            node_id=node_id,
                            description=f"Statistical outlier in {metric_name}: {value:.3f} (z-score: {z_score:.2f})",
                            metrics={metric_name: value, 'z_score': z_score},
                            confidence=0.7
                        )
                        anomalies.append(anomaly)
        
        return anomalies
    
    async def _pattern_analysis(self, metrics: NetworkMetrics) -> List[Anomaly]:
        """Perform pattern-based anomaly detection.
        
        Args:
            metrics: Current network metrics
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        node_id = metrics.node_id
        
        if node_id not in self.metrics_history or len(self.metrics_history[node_id]) < 10:
            return anomalies
        
        history = list(self.metrics_history[node_id])
        recent_metrics = history[-10:]  # Last 10 measurements
        
        # Detect sudden changes in trends
        packet_rates = [m.packet_rate for m in recent_metrics]
        latencies = [m.latency for m in recent_metrics]
        
        # Detect rate spikes
        if len(packet_rates) >= 5:
            recent_avg = np.mean(packet_rates[-3:])
            historical_avg = np.mean(packet_rates[:-3])
            
            if recent_avg > historical_avg * 2.0:  # 100% increase
                self.anomaly_counter += 1
                anomaly = Anomaly(
                    anomaly_id=f"pattern_{self.anomaly_counter}",
                    timestamp=metrics.timestamp,
                    anomaly_type=AnomalyType.TRAFFIC_SPIKE,
                    severity=min((recent_avg / historical_avg - 1.0), 1.0),
                    node_id=node_id,
                    description=f"Traffic spike detected: {recent_avg:.1f} vs {historical_avg:.1f} packets/s",
                    metrics={'current_rate': recent_avg, 'baseline_rate': historical_avg},
                    confidence=0.8
                )
                anomalies.append(anomaly)
        
        # Detect latency degradation
        if len(latencies) >= 5:
            recent_avg = np.mean(latencies[-3:])
            historical_avg = np.mean(latencies[:-3])
            
            if recent_avg > historical_avg * 1.5:  # 50% increase in latency
                self.anomaly_counter += 1
                anomaly = Anomaly(
                    anomaly_id=f"latency_{self.anomaly_counter}",
                    timestamp=metrics.timestamp,
                    anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                    severity=min((recent_avg / historical_avg - 1.0), 1.0),
                    node_id=node_id,
                    description=f"Latency degradation: {recent_avg:.3f}s vs {historical_avg:.3f}s",
                    metrics={'current_latency': recent_avg, 'baseline_latency': historical_avg},
                    confidence=0.75
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _ml_analysis(self, metrics: NetworkMetrics) -> List[Anomaly]:
        """Perform ML-based anomaly detection using quantum models.
        
        Args:
            metrics: Current network metrics
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Try quantum ML prediction first
        try:
            prediction_score = await self._quantum_ml_prediction(metrics)
        except Exception:
            # Fallback to classical simulation
            prediction_score = await self._simulate_ml_prediction(metrics)
        
        if prediction_score > self.prediction_threshold:
            self.anomaly_counter += 1
            anomaly = Anomaly(
                anomaly_id=f"qml_{self.anomaly_counter}",
                timestamp=metrics.timestamp,
                anomaly_type=AnomalyType.UNUSUAL_ROUTING,
                severity=prediction_score,
                node_id=metrics.node_id,
                description=f"Quantum ML model detected anomaly (score: {prediction_score:.3f})",
                metrics={'qml_score': prediction_score},
                confidence=prediction_score
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    async def _quantum_ml_prediction(self, metrics: NetworkMetrics) -> float:
        """Use quantum machine learning for anomaly prediction.
        
        Args:
            metrics: Network metrics
            
        Returns:
            Quantum ML anomaly score
        """
        # Check if we have a trained quantum model
        model_id = f"quantum_anomaly_detector_{metrics.node_id}"
        
        # Convert metrics to feature vector
        features = np.array([
            metrics.packet_rate / 1000.0,  # Normalize
            metrics.latency * 1000.0,      # Convert to ms
            metrics.error_rate,
            metrics.throughput / 1000.0,
            metrics.cpu_usage,
            metrics.memory_usage
        ])
        
        # Pad or truncate to match expected input size
        expected_size = 8
        if len(features) < expected_size:
            features = np.pad(features, (0, expected_size - len(features)))
        else:
            features = features[:expected_size]
        
        # Quantum feature encoding
        quantum_features = self._encode_quantum_features(features)
        
        # Simple quantum-inspired prediction (since we don't have trained models yet)
        # In production, this would use actual trained quantum models
        quantum_score = np.sum(np.sin(quantum_features) ** 2)
        
        return min(quantum_score, 1.0)
    
    def _encode_quantum_features(self, features: np.ndarray) -> np.ndarray:
        """Encode classical features into quantum-inspired representation.
        
        Args:
            features: Classical feature vector
            
        Returns:
            Quantum-encoded features
        """
        # Amplitude encoding: map to [0, π]
        normalized_features = (features - features.min()) / (features.max() - features.min() + 1e-8)
        quantum_encoded = normalized_features * np.pi
        
        # Add quantum correlations
        correlations = np.zeros_like(quantum_encoded)
        for i in range(len(quantum_encoded) - 1):
            correlations[i] = np.cos(quantum_encoded[i]) * np.sin(quantum_encoded[i + 1])
        
        return quantum_encoded + 0.1 * correlations
    
    async def _simulate_ml_prediction(self, metrics: NetworkMetrics) -> float:
        """Simulate ML model prediction.
        
        Args:
            metrics: Network metrics
            
        Returns:
            Anomaly score (0.0 to 1.0)
        """
        # Simple heuristic simulation
        score = 0.0
        
        # High error rate contributes to anomaly score
        if metrics.error_rate > 0.1:
            score += 0.3
        
        # High latency contributes
        if metrics.latency > 0.1:
            score += 0.2
        
        # Resource usage patterns
        if metrics.cpu_usage > 0.8 or metrics.memory_usage > 0.9:
            score += 0.3
        
        return min(score, 1.0)
    
    async def _process_anomaly(self, anomaly: Anomaly):
        """Process detected anomaly.
        
        Args:
            anomaly: Detected anomaly
        """
        self.detected_anomalies.append(anomaly)
        
        # Log anomaly
        severity_label = "LOW" if anomaly.severity < 0.3 else "MED" if anomaly.severity < 0.7 else "HIGH"
        print(f"ANOMALY [{severity_label}] {anomaly.anomaly_id}: {anomaly.description}")
        
        # Trigger responses for high-severity anomalies
        if anomaly.severity > 0.8:
            await self._respond_to_anomaly(anomaly)
    
    async def _respond_to_anomaly(self, anomaly: Anomaly):
        """Respond to high-severity anomalies.
        
        Args:
            anomaly: High-severity anomaly
        """
        print(f"AUTO-RESPONSE: Taking action for anomaly {anomaly.anomaly_id}")
        
        # Could implement:
        # - Automatic traffic rerouting
        # - Resource scaling
        # - Security protocol activation
        # - Network reconfiguration
    
    async def _continuous_analysis(self):
        """Background task for continuous analysis."""
        while self.running:
            try:
                # Update baseline statistics
                await self._update_baselines()
                
                # Cleanup old anomalies
                await self._cleanup_old_anomalies()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                print(f"Error in continuous analysis: {e}")
                await asyncio.sleep(60)
    
    async def _model_training_loop(self):
        """Background task for model training."""
        while self.running:
            try:
                # Train/update models for nodes with sufficient data
                for node_id, history in self.metrics_history.items():
                    if len(history) >= self.training_window:
                        await self._train_node_model(node_id)
                
                await asyncio.sleep(3600)  # Retrain every hour
                
            except Exception as e:
                print(f"Error in model training: {e}")
                await asyncio.sleep(3600)
    
    async def _initialize_baselines(self):
        """Initialize baseline statistics."""
        # Placeholder for baseline initialization
        print("Initialized baseline statistics")
    
    async def _update_baselines(self):
        """Update baseline statistics for all nodes."""
        for node_id, history in self.metrics_history.items():
            if len(history) >= 10:
                metrics_list = list(history)[-100:]  # Last 100 measurements
                
                # Calculate statistics for each metric
                packet_rates = [m.packet_rate for m in metrics_list]
                latencies = [m.latency for m in metrics_list]
                error_rates = [m.error_rate for m in metrics_list]
                cpu_usage = [m.cpu_usage for m in metrics_list]
                
                self.baseline_stats[node_id] = {
                    'packet_rate_mean': np.mean(packet_rates),
                    'packet_rate_std': np.std(packet_rates),
                    'latency_mean': np.mean(latencies),
                    'latency_std': np.std(latencies),
                    'error_rate_mean': np.mean(error_rates),
                    'error_rate_std': np.std(error_rates),
                    'cpu_usage_mean': np.mean(cpu_usage),
                    'cpu_usage_std': np.std(cpu_usage)
                }
    
    async def _train_node_model(self, node_id: str):
        """Train ML model for specific node.
        
        Args:
            node_id: Node to train model for
        """
        print(f"Training model for node {node_id}")
        # Placeholder for actual model training
        self.models[node_id] = {"trained": True, "timestamp": time.time()}
    
    async def _cleanup_old_anomalies(self):
        """Remove old anomalies from memory."""
        current_time = time.time()
        # Keep anomalies from last 24 hours
        self.detected_anomalies = [
            anomaly for anomaly in self.detected_anomalies
            if current_time - anomaly.timestamp < 86400
        ]
    
    async def get_anomaly_report(self, hours: int = 24) -> dict:
        """Get anomaly detection report.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Anomaly report dictionary
        """
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        recent_anomalies = [
            anomaly for anomaly in self.detected_anomalies
            if anomaly.timestamp >= cutoff_time
        ]
        
        # Group by type and severity
        by_type = {}
        by_severity = {'low': 0, 'medium': 0, 'high': 0}
        
        for anomaly in recent_anomalies:
            # Count by type
            anomaly_type = anomaly.anomaly_type.value
            by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1
            
            # Count by severity
            if anomaly.severity < 0.3:
                by_severity['low'] += 1
            elif anomaly.severity < 0.7:
                by_severity['medium'] += 1
            else:
                by_severity['high'] += 1
        
        return {
            'total_anomalies': len(recent_anomalies),
            'by_type': by_type,
            'by_severity': by_severity,
            'monitored_nodes': len(self.metrics_history),
            'trained_models': len(self.models),
            'detection_methods': ['statistical', 'pattern', 'ml'],
            'time_window_hours': hours
        }
