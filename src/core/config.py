#!/usr/bin/env python3
"""
Configuration Manager

Centralized configuration management for the quantum network platform.
Handles loading, validation, and access to configuration parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import json
from loguru import logger


class ConfigManager:
    """Centralized configuration management system."""
    
    DEFAULT_CONFIG = {
        "network": {
            "topology": {
                "initial_nodes": 5,
                "auto_connect": True,
                "connection_probability": 0.3
            },
            "simulation": {
                "time_step": 0.1,
                "max_simulation_time": 3600
            }
        },
        "kdc": {
            "key_length": 256,
            "refresh_interval": 300,
            "protocols": ["BB84", "E91", "SARG04"]
        },
        "security": {
            "eavesdrop_threshold": 0.11,
            "monitoring_interval": 1.0,
            "alert_threshold": 0.05
        },
        "ai": {
            "model_type": "lstm",
            "training_window": 1000,
            "prediction_threshold": 0.8
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8080,
            "cors_enabled": True
        },
        "dashboard": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 3000,
            "update_interval": 5.0
        },
        "logging": {
            "level": "INFO",
            "format": "<green>{time}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "file": {
                "enabled": False,
                "path": "logs/quantum_network.log"
            }
        },
        "performance": {
            "monitoring_enabled": True,
            "metrics_interval": 30,
            "memory_threshold": 0.8,
            "cpu_threshold": 0.9
        }
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config_data = self.DEFAULT_CONFIG.copy()
        
        # Load configuration from file if provided
        if self.config_path and self.config_path.exists():
            self._load_config()
        
        # Override with environment variables
        self._load_env_overrides()
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:  # Assume YAML
                    file_config = yaml.safe_load(f)
            
            # Deep merge with default config
            self._deep_merge(self.config_data, file_config)
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        env_prefix = "QUANTUM_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert QUANTUM_NETWORK_PORT to ["network", "port"]
                config_key = key[len(env_prefix):].lower().split('_')
                
                # Convert string value to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Set in config
                self._set_nested_value(self.config_data, config_key, converted_value)
                logger.debug(f"Set config from env: {'.'.join(config_key)} = {converted_value}")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to boolean
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        if value.lower() in ('false', '0', 'no', 'off'):
            return False
        
        # Try to parse as JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Return as string
        return value
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, data: Dict, keys: list, value: Any):
        """Set a nested value in dictionary using key path."""
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'network.topology.initial_nodes')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = self.config_data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        self._set_nested_value(self.config_data, keys, value)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary containing section configuration
        """
        return self.get(section, {})
    
    def save(self, output_path: Optional[Union[str, Path]] = None):
        """Save current configuration to file.
        
        Args:
            output_path: Output file path (defaults to original config_path)
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        if not save_path:
            raise ValueError("No output path specified and no original config path")
        
        # Create directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() == '.json':
                    json.dump(self.config_data, f, indent=2)
                else:  # Save as YAML
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def validate(self) -> bool:
        """Validate current configuration.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Validate required sections exist
            required_sections = ['network', 'kdc', 'security', 'ai']
            for section in required_sections:
                if section not in self.config_data:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate specific values
            if self.get('network.topology.initial_nodes', 0) <= 0:
                logger.error("network.topology.initial_nodes must be positive")
                return False
            
            if not (0 < self.get('security.eavesdrop_threshold', 0) < 1):
                logger.error("security.eavesdrop_threshold must be between 0 and 1")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.config_data, default_flow_style=False)
