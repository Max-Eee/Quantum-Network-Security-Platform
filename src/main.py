#!/usr/bin/env python3
"""
Quantum Network Security Platform
Main Application Entry Point

This module serves as the primary entry point for the quantum network simulation
and security platform, orchestrating all core components including routing,
KDC, security monitoring, AI anomaly detection, and visualization.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import click
from loguru import logger

# Add src to path for imports
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from core.quantum_engine import QuantumNetworkEngine
    from core.config import ConfigManager
    from api.server import create_app
    from visualization.dashboard import DashboardManager
    from utils.monitoring import SystemMonitor
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Trying alternative import paths...")
    # Try importing from src prefix
    from src.core.quantum_engine import QuantumNetworkEngine
    from src.core.config import ConfigManager
    from src.api.server import create_app
    from src.visualization.dashboard import DashboardManager
    from src.utils.monitoring import SystemMonitor


class QuantumNetworkPlatform:
    """Main platform orchestrator for the quantum network system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the quantum network platform.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.engine: Optional[QuantumNetworkEngine] = None
        self.dashboard: Optional[DashboardManager] = None
        self.monitor: Optional[SystemMonitor] = None
        self.running = False
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the platform."""
        log_level = self.config.get('logging.level', 'INFO')
        log_format = self.config.get(
            'logging.format',
            "<green>{time}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        
        logger.remove()
        logger.add(sys.stdout, format=log_format, level=log_level)
        
        if self.config.get('logging.file.enabled', False):
            log_file = self.config.get('logging.file.path', 'logs/quantum_network.log')
            logger.add(log_file, rotation="1 day", retention="1 month", level=log_level)
    
    async def initialize(self):
        """Initialize all platform components."""
        logger.info("Initializing Quantum Network Platform...")
        
        try:
            # Initialize core engine
            self.engine = QuantumNetworkEngine(self.config)
            await self.engine.initialize()
            
            # Initialize monitoring
            self.monitor = SystemMonitor(self.config)
            await self.monitor.start()
            
            # Initialize dashboard if enabled
            if self.config.get('dashboard.enabled', True):
                self.dashboard = DashboardManager(self.config, self.engine)
                await self.dashboard.initialize()
            
            logger.info("Platform initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize platform: {e}")
            raise
    
    async def start(self):
        """Start the quantum network platform."""
        if self.running:
            logger.warning("Platform is already running")
            return
        
        logger.info("Starting Quantum Network Platform...")
        
        try:
            # Start core engine
            await self.engine.start()
            
            # Start dashboard
            if self.dashboard:
                await self.dashboard.start()
            
            # Start API server
            app = create_app(self.engine, self.config)
            
            # Start uvicorn server in background task
            import uvicorn
            port = self.config.get('api.port', 8080)
            host = self.config.get('api.host', '127.0.0.1')
            
            logger.info(f"Starting API server on http://{host}:{port}")
            logger.info(f"API Documentation available at: http://{host}:{port}/docs")
            
            # Create uvicorn server configuration
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            # Start server as background task
            server_task = asyncio.create_task(server.serve())
            
            self.running = True
            logger.info("Quantum Network Platform started successfully")
            
            try:
                # Keep running until shutdown
                await self._run_forever()
            finally:
                # Shutdown server
                server.should_exit = True
                await server_task
            
        except Exception as e:
            logger.error(f"Failed to start platform: {e}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown the platform."""
        if not self.running:
            return
        
        logger.info("Shutting down Quantum Network Platform...")
        
        try:
            self.running = False
            
            # Shutdown components in reverse order
            if self.dashboard:
                await self.dashboard.shutdown()
            
            if self.monitor:
                await self.monitor.stop()
            
            if self.engine:
                await self.engine.shutdown()
            
            logger.info("Platform shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _run_forever(self):
        """Keep the platform running until shutdown signal."""
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            await self.shutdown()


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Quantum Network Security Platform CLI."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8080, help='Port to bind to')
@click.option('--workers', default=1, help='Number of worker processes')
@click.pass_context
def start(ctx, host, port, workers):
    """Start the quantum network platform."""
    config_path = ctx.obj.get('config')
    
    async def run_platform():
        platform = QuantumNetworkPlatform(config_path)
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(platform.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize and start
        await platform.initialize()
        await platform.start()
        
        # Initialize TensorFlow Quantum components
        if platform.engine.quantum_engine:
            logger.info("Setting up TensorFlow Quantum components...")
            
            # Create quantum circuits
            await platform.engine.create_quantum_circuit("bell_circuit", 2)
            await platform.engine.create_quantum_circuit("qkd_circuit", 4)
            
            # Create quantum ML models
            await platform.engine.create_quantum_ml_model("anomaly_classifier", "classifier")
            await platform.engine.create_quantum_ml_model("network_autoencoder", "autoencoder")
            
            logger.info("TensorFlow Quantum setup completed")
    
    # Run the platform
    asyncio.run(run_platform())


@cli.command()
@click.option('--topology', '-t', help='Network topology file')
@click.option('--nodes', '-n', default=10, help='Number of nodes')
@click.option('--duration', '-d', default=3600, help='Simulation duration (seconds)')
@click.pass_context
def simulate(ctx, topology, nodes, duration):
    """Run a quantum network simulation."""
    logger.info(f"Starting simulation with {nodes} nodes for {duration} seconds")
    
    # Implementation would go here
    # This is a placeholder for simulation-specific logic
    

@cli.command()
@click.option('--output', '-o', default='network_report.html', help='Output file')
@click.pass_context
def report(ctx, output):
    """Generate a network analysis report."""
    logger.info(f"Generating network report: {output}")
    
    # Implementation would go here
    # This is a placeholder for report generation logic


@cli.command()
def status():
    """Check platform status."""
    logger.info("Checking platform status...")
    
    # Implementation would go here
    # This is a placeholder for status check logic


if __name__ == "__main__":
    cli()