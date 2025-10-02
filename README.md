# <i>**`Quantum`** Network Security Platform</i>

A comprehensive **`quantum computing platform`** built with modern technologies for quantum network simulation, cryptography, security monitoring, and AI-powered anomaly detection.

<samp>

## 📥 Setup

For **development setup**, clone this repository and follow the installation instructions below. <br>
The platform supports both production deployment and development environments with comprehensive quantum computing simulation.
  
> [!IMPORTANT]
> **Default Configuration**: The platform runs with secure defaults and simulation mode enabled.
> 
> **Quantum Security**: This application features quantum key distribution, eavesdropping detection, and post-quantum cryptography for next-generation security.

## ✨ Features

- **`Quantum Computing Integration`**: Full TensorFlow Quantum and Cirq support with quantum circuit simulation
- **`Quantum Key Distribution`**: Secure key exchange using BB84, E91, and SARG04 protocols
- **`Advanced Security Monitoring`**: Real-time eavesdropping detection and quantum security validation
- **`AI-Powered Anomaly Detection`**: Machine learning algorithms for network behavior analysis
- **`Dynamic Quantum Routing`**: Intelligent routing algorithms optimized for quantum networks
- **`Real-time Visualization`**: Interactive dashboard for network topology and performance monitoring
- **`RESTful API Interface`**: Comprehensive API for network management and integration
- **`Scalable Architecture`**: Production-ready platform for enterprise quantum network deployment
- **`Multi-Protocol Support`**: Support for various quantum cryptographic protocols
- **`Performance Analytics`**: Real-time metrics and historical performance analysis
- **`Security Event Management`**: Automated threat detection and response system
- **`Containerized Deployment`**: Docker support for easy deployment and scaling

## ⚙️ Usage

### Getting Started

1. **Install dependencies** and set up the environment
2. **Configure the platform** using YAML configuration files
3. **Start the API server** for REST interface access
4. **Launch the dashboard** for real-time monitoring
5. **Run quantum simulations** to test network scenarios

### Key Operations

- **Start Platform**: Use `python run_platform.py` for interactive launcher
- **API Server**: Direct access via `python src/main.py start --port 8080`
- **Dashboard**: Launch monitoring interface with dashboard option
- **Quantum Simulation**: Run network simulations with custom parameters
- **Demo Mode**: Experience all features with `live_demo.py`

### API Endpoints

- **`GET /api/network/status`** - Get network topology and status
- **`GET /api/security/threats`** - List detected security threats
- **`POST /api/qkd/session`** - Start quantum key distribution session
- **`GET /api/quantum/circuits`** - List available quantum circuits
- **`GET /api/metrics/performance`** - Get real-time performance metrics
- **`WS /api/events`** - Real-time event stream

## ⬇️ Installation

### Prerequisites

- **Python 3.8+** (Python 3.13 supported with minimal requirements)
- **Git** for repository cloning
- **Docker** (optional, for containerized deployment)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Max-Eee/Quantum-Network-Security-Platform.git
   cd Quantum
   ```

2. **Create virtual environment**
   ```bash
   python -m venv quantum_env
   
   # Windows
   quantum_env\\Scripts\\activate
   
   # Linux/Mac
   source quantum_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Full installation with TensorFlow Quantum (Python 3.8-3.11)
   pip install -r requirements.txt
   
   # Minimal installation for Python 3.13
   pip install -r requirements-minimal.txt
   ```

4. **Run the platform**
   ```bash
   # Interactive launcher
   python run_platform.py
   
   # Direct API server
   python src/main.py start --port 8080
   
   # Full demo
   python live_demo.py
   ```

### Docker Deployment

1. **Build container**
   ```bash
   docker build -t quantum-platform .
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

## 🔧 Technical Stack

### Core Technologies
- **Quantum Computing**: TensorFlow Quantum, Cirq, Qiskit compatibility
- **Backend**: Python 3.8+, FastAPI, AsyncIO
- **Machine Learning**: TensorFlow, Scikit-learn, XGBoost, LightGBM
- **Cryptography**: Cryptography, PyCryptodome, PyNaCl
- **Network Simulation**: NetworkX, SimPy, Igraph

### Web & API
- **Web Framework**: FastAPI, Uvicorn
- **Real-time Communication**: WebSockets
- **Documentation**: Automatic OpenAPI/Swagger generation
- **Authentication**: JWT, bcrypt (when enabled)

### Data & Analytics
- **Data Processing**: Pandas, Polars, Dask
- **Visualization**: Plotly, Bokeh, Matplotlib
- **Monitoring**: Prometheus, Loguru, Structlog
- **Database**: SQLAlchemy, Redis (optional), MongoDB (optional)

## 📁 Project Structure

```
Quantum/
├── src/
│   ├── core/
│   │   ├── config.py              # Configuration management
│   │   └── quantum_engine.py      # Main quantum engine
│   ├── quantum/
│   │   ├── quantum_algorithms.py  # Quantum algorithms (VQE, QAOA)
│   │   ├── quantum_circuits.py    # Circuit management
│   │   ├── quantum_engine.py      # Quantum simulation engine
│   │   └── quantum_ml.py          # Quantum machine learning
│   ├── security/
│   │   └── eavesdrop_detector.py  # Security monitoring
│   ├── kdc/
│   │   └── key_distribution.py    # Quantum key distribution
│   ├── routing/
│   │   └── quantum_router.py      # Network routing algorithms
│   ├── ai/
│   │   └── anomaly_detector.py    # AI-powered anomaly detection
│   ├── api/
│   │   └── server.py              # FastAPI REST interface
│   ├── visualization/
│   │   └── dashboard.py           # Real-time dashboard
│   ├── utils/
│   │   └── monitoring.py          # System monitoring utilities
│   └── main.py                    # Main application entry point
├── config/
│   └── default.yaml               # Default configuration
├── docs/
│   └── index.html                 # Documentation homepage
├── scripts/
│   └── quantum_demo.py            # Demo scripts
├── requirements.txt               # Python dependencies
├── requirements-minimal.txt       # Minimal dependencies (Python 3.13)
├── run_platform.py               # Interactive platform launcher
├── live_demo.py                  # Comprehensive feature demonstration
├── docker-compose.yml            # Docker deployment
├── Dockerfile                    # Container configuration
└── setup.py                     # Package installation
```

## 🚀 Platform Features

### Quantum Computing Capabilities
- **Quantum Circuit Simulation**: Full support for quantum gates and measurements
- **Quantum Algorithms**: Implementation of VQE, QAOA, and other quantum algorithms
- **Quantum Machine Learning**: Hybrid quantum-classical ML models
- **Quantum Entanglement**: Bell states, GHZ states, and custom entangled systems

### Security Features
- **Quantum Key Distribution**: BB84, E91, SARG04 protocols
- **Eavesdropping Detection**: Real-time security monitoring and threat detection
- **Post-Quantum Cryptography**: Future-proof cryptographic implementations
- **Security Analytics**: Advanced threat analysis and response systems

### Network Management
- **Dynamic Topology**: Automatic network discovery and topology management
- **Intelligent Routing**: Quantum-optimized routing algorithms
- **Load Balancing**: Traffic distribution across quantum channels
- **Performance Optimization**: Real-time network performance tuning

### Monitoring & Analytics
- **Real-time Metrics**: Live performance and security monitoring
- **Historical Analytics**: Long-term trend analysis and reporting
- **Event Management**: Automated alert and response systems
- **Dashboard Interface**: Interactive web-based monitoring console

## 📊 API Documentation

When the server is running, comprehensive API documentation is available at:
- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`
- **OpenAPI Spec**: `http://localhost:8080/openapi.json`

## ⚡ Performance

- **Quantum Simulation**: Handles complex multi-qubit systems efficiently
- **Real-time Processing**: Low-latency event processing and response
- **Scalable Architecture**: Supports network scaling from small to enterprise-size
- **Memory Optimization**: Efficient memory usage for large quantum systems
- **Concurrent Processing**: Asynchronous operations for maximum throughput

## 🔒 Security Considerations

- **Quantum Security**: Implements quantum-safe cryptographic protocols
- **Data Protection**: Encrypted storage and secure communication channels
- **Access Control**: Authentication and authorization systems (configurable)
- **Audit Logging**: Comprehensive security event logging and monitoring
- **Threat Detection**: AI-powered anomaly detection and response

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
</samp>