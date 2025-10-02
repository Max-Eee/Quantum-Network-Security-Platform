#!/usr/bin/env python3
"""
Quantum Network Platform Setup Script

Initial    # Check quantum packages
    print("\\nChecking quantum computing packages...")
    missing_quantum = []
    
    for package in quantum_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            missing_quantum.append(package)
            print(f"✗ {package} is missing")
    
    # Check optional packages
    print("\\nChecking optional quantum packages...")
    missing_optional = []
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️  {package} is missing (optional)")
    
    if missing_packages:
        print(f"\\nMissing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    if missing_quantum:
        print(f"\\nMissing quantum packages: {', '.join(missing_quantum)}")
        print("Core functionality will work, but quantum features may be limited.")
        print("Install with: pip install -r requirements.txt")
    
    if missing_optional:
        print(f"\\nMissing optional packages: {', '.join(missing_optional)}")
        print("These provide enhanced quantum ML capabilities.")
        print("Install with: pip install tensorflow-quantum tensorboard")
    
    return Truehe platform with default configuration and prepares the environment.
"""

import os
import sys
from pathlib import Path
import shutil
from typing import Optional


def create_directories():
    """Create necessary directories for the platform."""
    directories = [
        'logs',
        'data',
        'config/user',
        'web/static',
        'web/templates',
        'exports',
        'backups'
    ]
    
    base_path = Path(__file__).parent.parent
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


def setup_configuration():
    """Setup initial configuration files."""
    base_path = Path(__file__).parent.parent
    default_config = base_path / 'config' / 'default.yaml'
    user_config = base_path / 'config' / 'user' / 'config.yaml'
    
    if not user_config.exists() and default_config.exists():
        shutil.copy2(default_config, user_config)
        print(f"Created user configuration: {user_config}")
    else:
        print("User configuration already exists or default config not found")


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'numpy',
        'scipy',
        'asyncio',
        'pathlib'
    ]
    
    quantum_packages = [
        'tensorflow',
        'cirq',
        'qiskit'
    ]
    
    optional_packages = [
        'tensorflow_quantum',
        'tensorboard'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def create_sample_data():
    """Create sample data files for testing."""
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data'
    
    # Create sample network topology
    sample_topology = {
        'name': 'Sample Network',
        'nodes': [
            {'id': 'node1', 'position': [100, 100], 'capabilities': ['qkd', 'routing']},
            {'id': 'node2', 'position': [200, 100], 'capabilities': ['qkd', 'routing']},
            {'id': 'node3', 'position': [150, 200], 'capabilities': ['qkd', 'routing']}
        ],
        'connections': [
            {'source': 'node1', 'target': 'node2'},
            {'source': 'node2', 'target': 'node3'},
            {'source': 'node3', 'target': 'node1'}
        ]
    }
    
    import json
    topology_file = data_path / 'sample_topology.json'
    with open(topology_file, 'w') as f:
        json.dump(sample_topology, f, indent=2)
    
    print(f"Created sample topology: {topology_file}")


def setup_logging():
    """Setup logging configuration."""
    base_path = Path(__file__).parent.parent
    logs_path = base_path / 'logs'
    
    # Create log files
    log_files = [
        'quantum_network.log',
        'security.log',
        'performance.log',
        'errors.log'
    ]
    
    for log_file in log_files:
        log_path = logs_path / log_file
        if not log_path.exists():
            log_path.touch()
            print(f"Created log file: {log_path}")


def main():
    """Main setup function."""
    print("Quantum Network Platform Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Setup steps
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Setting up configuration...")
    setup_configuration()
    
    print("\n3. Checking dependencies...")
    if not check_dependencies():
        print("\nSetup incomplete. Please install missing dependencies.")
        return
    
    print("\n4. Creating sample data...")
    create_sample_data()
    
    print("\n5. Setting up logging...")
    setup_logging()
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Review configuration in config/user/config.yaml")
    print("2. Install additional dependencies: pip install -r requirements.txt")
    print("3. Run the platform: python src/main.py start")
    print("4. Access dashboard at: http://localhost:3000")
    print("5. Access API at: http://localhost:8080")


if __name__ == '__main__':
    main()
