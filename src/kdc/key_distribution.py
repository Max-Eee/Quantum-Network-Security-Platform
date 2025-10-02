#!/usr/bin/env python3
"""
Key Distribution Center (KDC)

Quantum key distribution protocols and key management system.
Implements BB84, E91, and other QKD protocols for secure key exchange.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import secrets
import time
from collections import defaultdict


class QKDProtocol(Enum):
    """Supported quantum key distribution protocols."""
    BB84 = "bb84"
    E91 = "e91"
    SARG04 = "sarg04"
    COW = "cow"  # Coherent One-Way


@dataclass
class QuantumKey:
    """Represents a quantum cryptographic key."""
    key_id: str
    key_data: bytes
    node_pair: Tuple[str, str]
    protocol: QKDProtocol
    fidelity: float
    creation_time: float
    expiry_time: float
    usage_count: int = 0


class KeyDistributionCenter:
    """Centralized quantum key distribution and management system."""
    
    def __init__(self, config):
        """Initialize the Key Distribution Center.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.active_keys: Dict[str, QuantumKey] = {}
        self.key_pairs: Dict[Tuple[str, str], List[QuantumKey]] = defaultdict(list)
        self.key_length = config.get('kdc.key_length', 256)
        self.refresh_interval = config.get('kdc.refresh_interval', 300)
        # Handle case-insensitive protocol names
        protocol_names = config.get('kdc.protocols', ['bb84'])
        self.protocols = []
        for p in protocol_names:
            try:
                # Try lowercase first (standard enum values)
                self.protocols.append(QKDProtocol(p.lower()))
            except ValueError:
                # If that fails, try the original case
                try:
                    self.protocols.append(QKDProtocol(p))
                except ValueError:
                    print(f"Warning: Unknown QKD protocol '{p}', defaulting to BB84")
                    self.protocols.append(QKDProtocol.BB84)
        self.running = False
        
    async def initialize(self):
        """Initialize the KDC system."""
        print("Initializing Key Distribution Center...")
        # KDC initialization logic
    
    async def start(self):
        """Start the KDC system."""
        self.running = True
        # Start background key refresh task
        asyncio.create_task(self._key_refresh_loop())
        print("Key Distribution Center started")
    
    async def shutdown(self):
        """Shutdown the KDC system."""
        self.running = False
        print("Key Distribution Center shutdown")
    
    async def establish_key(self, node1: str, node2: str, 
                          protocol: QKDProtocol = None) -> Optional[QuantumKey]:
        """Establish a quantum key between two nodes.
        
        Args:
            node1: First node ID
            node2: Second node ID
            protocol: QKD protocol to use
            
        Returns:
            QuantumKey if successful, None otherwise
        """
        if protocol is None:
            protocol = self.protocols[0]
        
        # Simulate QKD protocol execution
        key_data = await self._execute_qkd_protocol(node1, node2, protocol)
        
        if key_data:
            key_id = self._generate_key_id()
            node_pair = tuple(sorted([node1, node2]))
            
            quantum_key = QuantumKey(
                key_id=key_id,
                key_data=key_data,
                node_pair=node_pair,
                protocol=protocol,
                fidelity=0.98,  # Simulated fidelity
                creation_time=time.time(),
                expiry_time=time.time() + self.refresh_interval
            )
            
            self.active_keys[key_id] = quantum_key
            self.key_pairs[node_pair].append(quantum_key)
            
            print(f"Established {protocol.value} key {key_id} for {node1}<->{node2}")
            return quantum_key
        
        return None
    
    async def get_key(self, node1: str, node2: str) -> Optional[QuantumKey]:
        """Get an active key for communication between two nodes.
        
        Args:
            node1: First node ID
            node2: Second node ID
            
        Returns:
            QuantumKey if available, None otherwise
        """
        node_pair = tuple(sorted([node1, node2]))
        
        # Find most recent valid key
        current_time = time.time()
        valid_keys = [
            key for key in self.key_pairs[node_pair]
            if key.expiry_time > current_time
        ]
        
        if valid_keys:
            # Return most recent key
            key = max(valid_keys, key=lambda k: k.creation_time)
            key.usage_count += 1
            return key
        
        # No valid key, establish new one
        return await self.establish_key(node1, node2)
    
    async def revoke_key(self, key_id: str):
        """Revoke a quantum key.
        
        Args:
            key_id: ID of key to revoke
        """
        if key_id in self.active_keys:
            key = self.active_keys[key_id]
            del self.active_keys[key_id]
            
            # Remove from pair list
            self.key_pairs[key.node_pair] = [
                k for k in self.key_pairs[key.node_pair]
                if k.key_id != key_id
            ]
            
            print(f"Revoked key {key_id}")
    
    async def _execute_qkd_protocol(self, node1: str, node2: str, 
                                   protocol: QKDProtocol) -> Optional[bytes]:
        """Execute a QKD protocol to generate shared key.
        
        Args:
            node1: First node ID
            node2: Second node ID
            protocol: QKD protocol to execute
            
        Returns:
            Generated key bytes or None if protocol failed
        """
        # Simulate protocol execution with error rates
        if protocol == QKDProtocol.BB84:
            return await self._simulate_bb84()
        elif protocol == QKDProtocol.E91:
            return await self._simulate_e91()
        elif protocol == QKDProtocol.SARG04:
            return await self._simulate_sarg04()
        
        return None
    
    async def _simulate_bb84(self) -> bytes:
        """Simulate BB84 protocol execution."""
        # Simulate BB84 key sifting and error correction
        await asyncio.sleep(0.1)  # Simulate protocol time
        
        # Generate random key
        key_bytes = secrets.token_bytes(self.key_length // 8)
        return key_bytes
    
    async def _simulate_e91(self) -> bytes:
        """Simulate E91 protocol execution."""
        await asyncio.sleep(0.15)  # E91 typically takes longer
        key_bytes = secrets.token_bytes(self.key_length // 8)
        return key_bytes
    
    async def _simulate_sarg04(self) -> bytes:
        """Simulate SARG04 protocol execution."""
        await asyncio.sleep(0.12)
        key_bytes = secrets.token_bytes(self.key_length // 8)
        return key_bytes
    
    def _generate_key_id(self) -> str:
        """Generate unique key identifier."""
        return f"qkey_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
    
    async def _key_refresh_loop(self):
        """Background task to refresh expired keys."""
        while self.running:
            try:
                current_time = time.time()
                expired_keys = [
                    key_id for key_id, key in self.active_keys.items()
                    if key.expiry_time <= current_time
                ]
                
                for key_id in expired_keys:
                    await self.revoke_key(key_id)
                
                if expired_keys:
                    print(f"Refreshed {len(expired_keys)} expired keys")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error in key refresh loop: {e}")
                await asyncio.sleep(60)
    
    async def get_status(self) -> dict:
        """Get KDC status information."""
        current_time = time.time()
        active_count = sum(
            1 for key in self.active_keys.values()
            if key.expiry_time > current_time
        )
        
        return {
            'running': self.running,
            'active_keys': active_count,
            'total_keys': len(self.active_keys),
            'key_pairs': len(self.key_pairs),
            'supported_protocols': [p.value for p in self.protocols]
        }
