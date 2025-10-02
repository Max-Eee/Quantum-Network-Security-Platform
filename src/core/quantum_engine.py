#!/usr/bin/env python3
"""
Quantum Network Engine

Core engine for quantum network simulation, routing, and key distribution.
This module orchestrates all quantum network operations including node management,
routing protocols, and quantum key distribution.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import uuid
from loguru import logger

try:
    from .config import ConfigManager
    from ..routing.quantum_router import QuantumRouter
    from ..kdc.key_distribution import KeyDistributionCenter
    from ..security.eavesdrop_detector import EavesdropDetector
    from ..ai.anomaly_detector import AnomalyDetector
    from ..quantum.quantum_engine import QuantumEngine
    from ..quantum.quantum_circuits import QuantumCircuitManager
    from ..quantum.quantum_ml import QuantumMachineLearning
except ImportError:
    # Fallback to absolute imports
    from core.config import ConfigManager
    from routing.quantum_router import QuantumRouter
    from kdc.key_distribution import KeyDistributionCenter
    from security.eavesdrop_detector import EavesdropDetector
    from ai.anomaly_detector import AnomalyDetector
    from quantum.quantum_engine import QuantumEngine
    from quantum.quantum_circuits import QuantumCircuitManager
    from quantum.quantum_ml import QuantumMachineLearning


class NodeStatus(Enum):
    """Quantum node status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class QuantumNode:
    """Represents a quantum network node."""
    node_id: str
    position: tuple[float, float]
    status: NodeStatus = NodeStatus.INACTIVE
    capabilities: List[str] = None
    connections: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.connections is None:
            self.connections = []


class QuantumNetworkEngine:
    """Core quantum network simulation and management engine."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the quantum network engine.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.nodes: Dict[str, QuantumNode] = {}
        self.router: Optional[QuantumRouter] = None
        self.kdc: Optional[KeyDistributionCenter] = None
        self.detector: Optional[EavesdropDetector] = None
        self.ai_detector: Optional[AnomalyDetector] = None
        self.quantum_engine: Optional[QuantumEngine] = None
        self.circuit_manager: Optional[QuantumCircuitManager] = None
        self.quantum_ml: Optional[QuantumMachineLearning] = None
        self.running = False
        
    async def initialize(self):
        """Initialize all engine components."""
        logger.info("Initializing Quantum Network Engine...")
        
        try:
            # Initialize router
            self.router = QuantumRouter(self.config)
            await self.router.initialize()
            
            # Initialize KDC
            self.kdc = KeyDistributionCenter(self.config)
            await self.kdc.initialize()
            
            # Initialize security detector
            self.detector = EavesdropDetector(self.config)
            await self.detector.initialize()
            
            # Initialize AI anomaly detector
            self.ai_detector = AnomalyDetector(self.config)
            await self.ai_detector.initialize()
            
            # Initialize TensorFlow Quantum components
            try:
                self.quantum_engine = QuantumEngine(self.config)
                self.circuit_manager = QuantumCircuitManager(self.config)
                self.quantum_ml = QuantumMachineLearning(self.config)
                logger.info("TensorFlow Quantum integration initialized")
            except ImportError as e:
                logger.warning(f"TensorFlow Quantum not available: {e}")
                # Quantum engine handled above
                self.circuit_manager = None
                self.quantum_ml = None
            
            # Create initial network topology
            await self._create_initial_topology()
            
            logger.info("Quantum Network Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise
    
    async def start(self):
        """Start the quantum network engine."""
        logger.info("Starting Quantum Network Engine...")
        
        try:
            # Start all components
            await self.router.start()
            await self.kdc.start()
            await self.detector.start()
            await self.ai_detector.start()
            
            # Start TensorFlow Quantum components
            if self.quantum_engine:
                logger.info("TensorFlow Quantum engine ready")
            
            self.running = True
            logger.info("Quantum Network Engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the quantum network engine."""
        logger.info("Shutting down Quantum Network Engine...")
        
        try:
            self.running = False
            
            # Shutdown components
            if self.ai_detector:
                await self.ai_detector.shutdown()
            if self.detector:
                await self.detector.shutdown()
            if self.kdc:
                await self.kdc.shutdown()
            if self.router:
                await self.router.shutdown()
            
            logger.info("Quantum Network Engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during engine shutdown: {e}")
    
    async def add_node(self, position: tuple[float, float], 
                      capabilities: List[str] = None) -> str:
        """Add a new quantum node to the network.
        
        Args:
            position: (x, y) coordinates of the node
            capabilities: List of node capabilities
            
        Returns:
            Node ID of the created node
        """
        node_id = str(uuid.uuid4())
        node = QuantumNode(
            node_id=node_id,
            position=position,
            status=NodeStatus.ACTIVE,
            capabilities=capabilities or ["qkd", "routing"]
        )
        
        self.nodes[node_id] = node
        
        # Register with router
        if self.router:
            await self.router.register_node(node)
        
        logger.info(f"Added quantum node {node_id} at position {position}")
        return node_id
    
    async def remove_node(self, node_id: str):
        """Remove a quantum node from the network.
        
        Args:
            node_id: ID of the node to remove
        """
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not found")
            return
        
        # Unregister from router
        if self.router:
            await self.router.unregister_node(node_id)
        
        # Remove from nodes
        del self.nodes[node_id]
        
        logger.info(f"Removed quantum node {node_id}")
    
    async def establish_connection(self, node1_id: str, node2_id: str):
        """Establish a quantum connection between two nodes.
        
        Args:
            node1_id: First node ID
            node2_id: Second node ID
        """
        if node1_id not in self.nodes or node2_id not in self.nodes:
            logger.error(f"One or both nodes not found: {node1_id}, {node2_id}")
            return
        
        # Add connections
        self.nodes[node1_id].connections.append(node2_id)
        self.nodes[node2_id].connections.append(node1_id)
        
        # Register with router
        if self.router:
            await self.router.add_connection(node1_id, node2_id)
        
        logger.info(f"Established connection between {node1_id} and {node2_id}")
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get current network status.
        
        Returns:
            Dictionary containing network status information
        """
        return {
            "running": self.running,
            "nodes": len(self.nodes),
            "active_nodes": len([n for n in self.nodes.values() 
                               if n.status == NodeStatus.ACTIVE]),
            "connections": sum(len(n.connections) for n in self.nodes.values()) // 2,
            "router_status": await self.router.get_status() if self.router else None,
            "kdc_status": await self.kdc.get_status() if self.kdc else None
        }
    
    async def _create_initial_topology(self):
        """Create initial network topology based on configuration."""
        topology_config = self.config.get('network.topology', {})
        node_count = topology_config.get('initial_nodes', 5)
        
        # Create initial nodes in a simple grid
        for i in range(node_count):
            x = (i % 3) * 100
            y = (i // 3) * 100
            await self.add_node((x, y))
        
        # Create some initial connections
        node_ids = list(self.nodes.keys())
        for i in range(len(node_ids) - 1):
            await self.establish_connection(node_ids[i], node_ids[i + 1])
        
        logger.info(f"Created initial topology with {node_count} nodes")
    
    async def create_quantum_circuit(self, circuit_id: str, num_qubits: int):
        """Create a quantum circuit using TensorFlow Quantum.
        
        Args:
            circuit_id: Circuit identifier
            num_qubits: Number of qubits
        """
        if not self.circuit_manager:
            logger.warning("TensorFlow Quantum not available - cannot create circuit")
            return None
        
        try:
            circuit = self.circuit_manager.create_circuit(circuit_id, num_qubits)
            logger.info(f"Created quantum circuit '{circuit_id}' with {num_qubits} qubits")
            return circuit
        except Exception as e:
            logger.error(f"Failed to create quantum circuit: {e}")
            return None
    
    async def create_quantum_ml_model(self, model_id: str, model_type: str = 'classifier'):
        """Create a quantum machine learning model.
        
        Args:
            model_id: Model identifier
            model_type: Type of model (classifier, autoencoder, etc.)
        """
        if not self.quantum_engine:
            logger.warning("TensorFlow Quantum not available - cannot create ML model")
            return None
        
        try:
            if model_type == 'classifier':
                model = self.quantum_engine.create_hybrid_model(model_id, input_dim=8, num_classes=3)
            elif model_type == 'autoencoder':
                model = self.quantum_engine.create_hybrid_model(model_id + "_autoencoder", input_dim=8, num_classes=8)
            elif model_type == 'rl_agent':
                model = self.quantum_engine.create_hybrid_model(model_id + "_rl", input_dim=8, num_classes=4)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            logger.info(f"Created quantum ML model '{model_id}' of type '{model_type}'")
            return model
        except Exception as e:
            logger.error(f"Failed to create quantum ML model: {e}")
            return None
