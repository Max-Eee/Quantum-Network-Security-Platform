"""Quantum computing components using TensorFlow Quantum."""

from .quantum_circuits import QuantumCircuitManager, QuantumGate
from .quantum_ml import QuantumMachineLearning, QuantumNeuralNetwork
from .quantum_algorithms import QuantumAlgorithms, QuantumOptimizer
from .quantum_engine import QuantumEngine, QuantumCircuitBuilder
# Legacy compatibility
TFQEngine = QuantumEngine
QuantumLayer = QuantumCircuitBuilder

__all__ = [
    'QuantumCircuitManager', 'QuantumGate',
    'QuantumMachineLearning', 'QuantumNeuralNetwork',
    'QuantumAlgorithms', 'QuantumOptimizer',
    'TFQEngine', 'QuantumLayer'
]
