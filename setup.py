#!/usr/bin/env python3
"""
Setup script for Quantum Network Platform package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="quantum-network-platform",
    version="1.0.0",
    author="Quantum Network Team",
    author_email="team@quantumnetwork.com",
    description="Comprehensive quantum network simulation and security platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantum-network/platform",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: System :: Networking",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "quantum": [
            "qiskit>=0.45.0",
            "qiskit-aer>=0.12.0",
            "cirq>=1.2.0",
            "pennylane>=0.32.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "tensorflow>=2.13.0",
            "scikit-learn>=1.3.0",
            "xgboost>=2.0.0",
        ],
        "viz": [
            "plotly>=5.15.0",
            "bokeh>=3.2.0",
            "dash>=2.14.0",
            "streamlit>=1.26.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "quantum-network=src.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "quantum", "network", "simulation", "cryptography", "security", 
        "qkd", "quantum key distribution", "machine learning", "anomaly detection"
    ],
    project_urls={
        "Bug Reports": "https://github.com/quantum-network/platform/issues",
        "Source": "https://github.com/quantum-network/platform",
        "Documentation": "https://quantum-network-platform.readthedocs.io/",
    },
)
