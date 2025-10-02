#!/usr/bin/env python3
"""
TensorFlow Quantum Demo Script

This script demonstrates the quantum computing capabilities of the platform
including quantum circuits, quantum machine learning, and hybrid algorithms.
"""

import asyncio
import sys
import os
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import ConfigManager
from quantum.quantum_engine import QuantumEngine
from quantum.quantum_circuits import QuantumCircuitManager
from quantum.quantum_ml import QuantumMachineLearning, QuantumNeuralNetwork


async def demo_quantum_circuits():
    """Demonstrate quantum circuit creation and execution."""
    print("\\n" + "="*60)
    print("QUANTUM CIRCUITS DEMONSTRATION")
    print("="*60)
    
    # Load configuration
    config = ConfigManager("../config/default.yaml").get_config()
    
    try:
        # Initialize circuit manager
        circuit_manager = QuantumCircuitManager(config)
        
        print("\\n1. Creating Bell State Circuit...")
        bell_circuit = circuit_manager.create_bell_state_circuit(0, 1)
        print(f"   ✓ Bell state circuit created with {len(bell_circuit.all_qubits())} qubits")
        
        print("\\n2. Creating QKD Circuit...")
        qkd_circuit = circuit_manager.create_qkd_circuit([0, 1], [0, 1])
        print(f"   ✓ QKD circuit created with {len(qkd_circuit.all_qubits())} qubits")
        
        print("\\n3. Creating Variational Circuit...")
        var_circuit, symbols = circuit_manager.create_variational_circuit(4, depth=2)
        print(f"   ✓ Variational circuit created with {len(symbols)} parameters")
        
        print("\\n4. Available Circuits:")
        circuits = circuit_manager.get_available_circuits()
        for circuit_id, info in circuits.items():
            print(f"   - {circuit_id}: {info['qubits']} qubits, {info['gates']} gates")
            
    except ImportError as e:
        print(f"   ⚠️  Cirq not available: {e}")
        print("   Please install: pip install cirq")


async def demo_quantum_ml():
    """Demonstrate quantum machine learning capabilities."""
    print("\\n" + "="*60)
    print("QUANTUM MACHINE LEARNING DEMONSTRATION")
    print("="*60)
    
    # Load configuration
    config = ConfigManager("../config/default.yaml").get_config()
    
    try:
        # Initialize quantum ML
        qml = QuantumMachineLearning(config)
        
        print("\\n1. Quantum Anomaly Detection...")
        # Generate sample network data
        normal_data = np.random.normal(0, 1, (100, 6))  # Normal network behavior
        anomaly_data = np.random.normal(2, 0.5, (10, 6))  # Anomalous behavior
        
        # Test anomaly detection
        normal_score = await qml.detect_anomaly(normal_data[0])
        anomaly_score = await qml.detect_anomaly(anomaly_data[0])
        
        print(f"   ✓ Normal data score: {normal_score:.4f}")
        print(f"   ✓ Anomaly data score: {anomaly_score:.4f}")
        
        print("\\n2. Network Optimization...")
        network_params = {
            'topology': 'mesh',
            'node_count': 10,
            'connection_density': 0.3,
            'latency_target': 50.0,
            'throughput_target': 1000.0
        }
        
        optimized_params = await qml.optimize_network_parameters(network_params)
        print("   ✓ Network optimization completed")
        print(f"   Original density: {network_params['connection_density']}")
        print(f"   Optimized density: {optimized_params.get('connection_density', 'N/A')}")
        
        print("\\n3. Quantum Neural Network...")
        qnn = QuantumNeuralNetwork(config)
        
        # Generate training data
        X_train = np.random.random((100, 4))
        y_train = np.random.randint(0, 2, 100)
        
        print("   Training quantum neural network...")
        history = await qnn.train(X_train, y_train, epochs=3)
        
        # Test prediction
        X_test = np.random.random((10, 4))
        predictions = await qnn.predict(X_test)
        
        print(f"   ✓ QNN trained on {len(X_train)} samples")
        print(f"   ✓ Predictions made for {len(X_test)} test samples")
        print(f"   ✓ Prediction accuracy: {np.mean(predictions > 0.5):.2f}")
        
    except ImportError as e:
        print(f"   ⚠️  TensorFlow Quantum not available: {e}")
        print("   Please install: pip install tensorflow-quantum")


async def demo_tfq_engine():
    """Demonstrate TensorFlow Quantum engine capabilities."""
    print("\\n" + "="*60)
    print("TENSORFLOW QUANTUM ENGINE DEMONSTRATION")
    print("="*60)
    
    # Load configuration
    config = ConfigManager("../config/default.yaml").get_config()
    
    try:
        # Initialize TFQ engine
        quantum_engine = QuantumEngine(config)
        
        print("\\n1. Creating Quantum Circuit...")
        circuit = quantum_engine.create_quantum_circuit(
            "demo_circuit", 
            num_qubits=4
        )
        print("   ✓ Quantum circuit created")
        
        print("\\n2. Creating Hybrid Model...")
        model = quantum_engine.create_hybrid_model(
            "demo_model",
            input_dim=8,
            num_classes=3
        )
        print("   ✓ Hybrid quantum-classical model created")
        
        print("\\n3. Circuit and Model Summary:")
        circuits = quantum_engine.get_available_circuits()
        models = quantum_engine.get_available_models()
        for model_id, model_info in models.items():
            print(f"   - {model_id}: {model_info['type']} ({model_info['status']})")
            
    except ImportError as e:
        print(f"   ⚠️  TensorFlow Quantum not available: {e}")
        print("   Please install: pip install tensorflow-quantum")


async def demo_quantum_algorithms():
    """Demonstrate quantum algorithm implementations."""
    print("\\n" + "="*60)
    print("QUANTUM ALGORITHMS DEMONSTRATION")
    print("="*60)
    
    # Load configuration
    config = ConfigManager("../config/default.yaml").get_config()
    
    try:
        from quantum.quantum_algorithms import QuantumAlgorithms, QuantumOptimizer
        
        print("\\n1. Quantum Optimizer...")
        optimizer = QuantumOptimizer(config)
        
        # Define a simple optimization problem
        def cost_function(params):
            return np.sum(params**2)
        
        initial_params = np.random.random(4)
        result = await optimizer.optimize(cost_function, initial_params)
        
        print(f"   ✓ Optimization completed")
        print(f"   ✓ Initial cost: {cost_function(initial_params):.4f}")
        print(f"   ✓ Final cost: {result['final_cost']:.4f}")
        
        print("\\n2. Quantum Algorithms...")
        qa = QuantumAlgorithms(config)
        
        # VQE demonstration
        print("   Running VQE...")
        vqe_result = await qa.run_vqe(num_qubits=4, max_iterations=10)
        print(f"   ✓ VQE energy: {vqe_result['energy']:.6f}")
        
        # QAOA demonstration
        print("   Running QAOA...")
        qaoa_result = await qa.run_qaoa(
            cost_hamiltonian=np.array([[1, 0], [0, -1]]),
            p=2
        )
        print(f"   ✓ QAOA cost: {qaoa_result['cost']:.6f}")
        
    except ImportError as e:
        print(f"   ⚠️  Required quantum libraries not available: {e}")


def print_header():
    """Print demo header."""
    print("="*60)
    print("QUANTUM NETWORK PLATFORM - TensorFlow Quantum Demo")
    print("="*60)
    print("This demo showcases the quantum computing capabilities")
    print("integrated into the quantum network security platform.")
    print()
    print("Features demonstrated:")
    print("• Quantum circuit creation and management")
    print("• Quantum machine learning models")
    print("• TensorFlow Quantum integration")
    print("• Quantum algorithms (VQE, QAOA)")
    print("• Hybrid quantum-classical models")


def print_footer():
    """Print demo footer."""
    print("\\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\\nTo use these features in your application:")
    print("1. Ensure TensorFlow Quantum is installed:")
    print("   pip install tensorflow-quantum")
    print("\\n2. Import the quantum modules:")
    print("   from src.quantum.quantum_engine import QuantumEngine")
    print("   from src.quantum.quantum_ml import QuantumMachineLearning")
    print("\\n3. Initialize and use in your code:")
    print("   tfq_engine = TFQEngine(config)")
    print("   model = await tfq_engine.create_quantum_classifier('my_model')")
    print("\\nFor more information, see the documentation at:")
    print("https://github.com/your-repo/quantum-network-platform")


async def main():
    """Run the complete TensorFlow Quantum demonstration."""
    print_header()
    
    try:
        # Run all demonstrations
        await demo_quantum_circuits()
        await demo_quantum_ml()
        await demo_tfq_engine()
        await demo_quantum_algorithms()
        
    except Exception as e:
        print(f"\\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print_footer()


if __name__ == "__main__":
    # Change to script directory for relative imports
    os.chdir(Path(__file__).parent)
    
    # Run the demo
    asyncio.run(main())