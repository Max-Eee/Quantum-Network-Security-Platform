#!/usr/bin/env python3
"""
Quantum Network Visualization Dashboard

Interactive dashboard for monitoring quantum network topology, security events,
and performance metrics in real-time.
"""

import asyncio
from typing import Dict, List, Optional, Any
import json
import time
from pathlib import Path


class DashboardManager:
    """Manager for quantum network visualization dashboard."""
    
    def __init__(self, config, engine):
        """Initialize dashboard manager.
        
        Args:
            config: Configuration manager instance
            engine: Quantum network engine instance
        """
        self.config = config
        self.engine = engine
        self.host = config.get('dashboard.host', '0.0.0.0')
        self.port = config.get('dashboard.port', 3000)
        self.update_interval = config.get('dashboard.update_interval', 5.0)
        
        # Dashboard state
        self.running = False
        self.connected_clients = set()
        self.dashboard_data = {}
    
    async def initialize(self):
        """Initialize the dashboard system."""
        print(f"Initializing Dashboard on {self.host}:{self.port}")
        
        # Initialize dashboard data structure
        self.dashboard_data = {
            'network_topology': {'nodes': [], 'connections': []},
            'performance_metrics': {},
            'security_events': [],
            'kdc_status': {},
            'routing_info': {},
            'system_status': {}
        }
    
    async def start(self):
        """Start the dashboard server."""
        self.running = True
        
        # Start background data collection
        asyncio.create_task(self._data_collection_loop())
        
        print(f"Dashboard started at http://{self.host}:{self.port}")
    
    async def shutdown(self):
        """Shutdown the dashboard server."""
        self.running = False
        print("Dashboard shutdown")
    
    async def _data_collection_loop(self):
        """Background task to collect and update dashboard data."""
        while self.running:
            try:
                # Collect data from all system components
                await self._collect_network_topology()
                await self._collect_performance_metrics()
                await self._collect_security_events()
                await self._collect_kdc_status()
                await self._collect_routing_info()
                await self._collect_system_status()
                
                # Broadcast updates to connected clients
                await self._broadcast_updates()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in dashboard data collection: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _collect_network_topology(self):
        """Collect network topology data."""
        if not self.engine or not hasattr(self.engine, 'nodes'):
            return
        
        nodes = []
        connections = []
        
        # Convert nodes to dashboard format
        for node_id, node in self.engine.nodes.items():
            nodes.append({
                'id': node_id,
                'x': node.position[0],
                'y': node.position[1],
                'status': node.status.value,
                'capabilities': node.capabilities,
                'connections': len(node.connections)
            })
        
        # Convert connections to dashboard format
        processed_pairs = set()
        for node_id, node in self.engine.nodes.items():
            for connected_id in node.connections:
                pair = tuple(sorted([node_id, connected_id]))
                if pair not in processed_pairs:
                    connections.append({
                        'source': pair[0],
                        'target': pair[1],
                        'status': 'active',
                        'bandwidth': 1000,  # Placeholder
                        'latency': 0.001   # Placeholder
                    })
                    processed_pairs.add(pair)
        
        self.dashboard_data['network_topology'] = {
            'nodes': nodes,
            'connections': connections,
            'last_updated': time.time()
        }
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics."""
        metrics = {
            'total_nodes': len(self.engine.nodes) if self.engine else 0,
            'active_nodes': 0,
            'total_connections': 0,
            'average_latency': 0.001,  # Placeholder
            'network_throughput': 1000,  # Placeholder
            'cpu_usage': 0.25,  # Placeholder
            'memory_usage': 0.4,  # Placeholder
            'last_updated': time.time()
        }
        
        if self.engine and hasattr(self.engine, 'nodes'):
            metrics['active_nodes'] = sum(
                1 for node in self.engine.nodes.values()
                if hasattr(node, 'status') and node.status.value == 'active'
            )
            metrics['total_connections'] = sum(
                len(node.connections) for node in self.engine.nodes.values()
            ) // 2
        
        self.dashboard_data['performance_metrics'] = metrics
    
    async def _collect_security_events(self):
        """Collect security events and alerts."""
        events = []
        
        # Get security events from detector if available
        if (self.engine and hasattr(self.engine, 'detector') and 
            self.engine.detector and hasattr(self.engine.detector, 'security_events')):
            
            recent_events = [
                event for event in self.engine.detector.security_events
                if time.time() - event.timestamp < 3600  # Last hour
            ]
            
            for event in recent_events[-20:]:  # Last 20 events
                events.append({
                    'id': event.event_id,
                    'timestamp': event.timestamp,
                    'type': event.event_type,
                    'threat_level': event.threat_level.value,
                    'source': event.source_node,
                    'target': event.target_node,
                    'description': event.description,
                    'metrics': event.metrics
                })
        
        self.dashboard_data['security_events'] = {
            'events': events,
            'total_events': len(events),
            'last_updated': time.time()
        }
    
    async def _collect_kdc_status(self):
        """Collect KDC status and key information."""
        kdc_data = {
            'active_keys': 0,
            'key_pairs': 0,
            'protocols': [],
            'key_refresh_rate': 300,
            'average_key_lifetime': 600,
            'last_updated': time.time()
        }
        
        if (self.engine and hasattr(self.engine, 'kdc') and self.engine.kdc):
            status = await self.engine.kdc.get_status()
            kdc_data.update({
                'active_keys': status.get('active_keys', 0),
                'key_pairs': status.get('key_pairs', 0),
                'protocols': status.get('supported_protocols', [])
            })
        
        self.dashboard_data['kdc_status'] = kdc_data
    
    async def _collect_routing_info(self):
        """Collect routing information."""
        routing_data = {
            'total_routes': 0,
            'active_routes': 0,
            'routing_protocol': 'dijkstra',
            'average_path_length': 2.5,
            'route_efficiency': 0.85,
            'last_updated': time.time()
        }
        
        if (self.engine and hasattr(self.engine, 'router') and self.engine.router):
            status = await self.engine.router.get_status()
            routing_data.update({
                'total_routes': status.get('routes', 0),
                'routing_protocol': status.get('protocol', 'dijkstra')
            })
        
        self.dashboard_data['routing_info'] = routing_data
    
    async def _collect_system_status(self):
        """Collect overall system status."""
        system_data = {
            'uptime': time.time(),  # Placeholder
            'system_health': 'healthy',
            'component_status': {
                'engine': 'running' if self.engine and self.engine.running else 'stopped',
                'router': 'running',
                'kdc': 'running',
                'detector': 'running',
                'ai_detector': 'running'
            },
            'resource_usage': {
                'cpu': 0.25,
                'memory': 0.4,
                'network': 0.15,
                'storage': 0.3
            },
            'last_updated': time.time()
        }
        
        self.dashboard_data['system_status'] = system_data
    
    async def _broadcast_updates(self):
        """Broadcast data updates to connected clients."""
        if not self.connected_clients:
            return
        
        # Prepare update message
        update_message = {
            'type': 'dashboard_update',
            'timestamp': time.time(),
            'data': self.dashboard_data
        }
        
        # In a real implementation, this would send via WebSocket
        print(f"Broadcasting updates to {len(self.connected_clients)} clients")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data.
        
        Returns:
            Complete dashboard data dictionary
        """
        return self.dashboard_data.copy()
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get network summary for quick status check.
        
        Returns:
            Network summary data
        """
        topology = self.dashboard_data.get('network_topology', {})
        metrics = self.dashboard_data.get('performance_metrics', {})
        security = self.dashboard_data.get('security_events', {})
        
        return {
            'nodes': len(topology.get('nodes', [])),
            'connections': len(topology.get('connections', [])),
            'active_nodes': metrics.get('active_nodes', 0),
            'recent_security_events': len(security.get('events', [])),
            'system_health': self.dashboard_data.get('system_status', {}).get('system_health', 'unknown'),
            'last_updated': max(
                topology.get('last_updated', 0),
                metrics.get('last_updated', 0),
                security.get('last_updated', 0)
            )
        }
