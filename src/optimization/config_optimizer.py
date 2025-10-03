"""
Dynamic Configuration Manager for production optimization.

This module provides adaptive configuration management with hot-reload capabilities,
performance-based parameter tuning, and configuration history tracking.

Features:
- Dynamic configuration updates without restart
- Performance-based adaptive parameter adjustment
- Configuration validation and rollback
- File monitoring and hot-reload
- Configuration history and impact analysis
"""

import asyncio
import json
import os
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class DynamicConfig:
    """
    Dynamic configuration class with validation and serialization support.

    Manages all tunable parameters for the trading system with automatic
    validation and constraint enforcement.
    """

    # Database configuration
    database_pool_size: int = 20
    max_overflow: int = 30

    # Event system configuration
    event_queue_size: int = 10000
    max_concurrent_orders: int = 10

    # Monitoring configuration
    risk_check_interval: int = 30  # seconds
    metrics_interval: int = 10     # seconds

    # Cache configuration
    cache_ttl: int = 300           # seconds

    # System configuration
    hot_reload_enabled: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate all configuration parameters."""
        # Database validation
        if not (1 <= self.database_pool_size <= 200):
            raise ConfigValidationError("database_pool_size must be between 1 and 200")

        if not (1 <= self.max_overflow <= 500):
            raise ConfigValidationError("max_overflow must be between 1 and 500")

        # Event system validation
        if not (1000 <= self.event_queue_size <= 100000):
            raise ConfigValidationError("event_queue_size must be between 1000 and 100000")

        if not (1 <= self.max_concurrent_orders <= 1000):
            raise ConfigValidationError("max_concurrent_orders must be between 1 and 1000")

        # Monitoring validation
        if not (1 <= self.risk_check_interval <= 300):
            raise ConfigValidationError("risk_check_interval must be between 1 and 300")

        if not (1 <= self.metrics_interval <= 60):
            raise ConfigValidationError("metrics_interval must be between 1 and 60")

        # Cache validation
        if not (30 <= self.cache_ttl <= 3600):
            raise ConfigValidationError("cache_ttl must be between 30 and 3600")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DynamicConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class ConfigOptimizer:
    """
    Configuration optimizer with adaptive parameter tuning and hot-reload support.

    This class manages dynamic configuration updates, performance-based optimization,
    and provides configuration history and impact analysis.
    """

    def __init__(self, initial_config: Optional[DynamicConfig] = None):
        """
        Initialize ConfigOptimizer.

        Args:
            initial_config: Initial configuration. If None, uses default.
        """
        self.current_config = initial_config or DynamicConfig()
        self.config_file_path: Optional[str] = None
        self.is_monitoring: bool = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._config_history: List[Dict[str, Any]] = []
        self._previous_config: Optional[DynamicConfig] = None

        # Track configuration changes
        self._add_to_history(self.current_config.to_dict(), "Initial configuration")

    async def load_config_from_file(self, file_path: str) -> None:
        """
        Load configuration from JSON file.

        Args:
            file_path: Path to configuration file

        Raises:
            ConfigValidationError: If file cannot be parsed or configuration is invalid
        """
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)

            # Create new config and validate
            new_config = DynamicConfig.from_dict(config_data)

            # Store previous config for rollback
            self._previous_config = self.current_config
            self.current_config = new_config
            self.config_file_path = file_path

            self._add_to_history(config_data, f"Loaded from file: {file_path}")

            logger.info(f"Configuration loaded from {file_path}")

        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Failed to parse configuration file: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to load configuration: {e}")

    async def save_config_to_file(self, file_path: str) -> None:
        """
        Save current configuration to JSON file.

        Args:
            file_path: Path to save configuration
        """
        try:
            config_data = self.current_config.to_dict()

            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Configuration saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise

    async def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration dynamically.

        Args:
            config_updates: Dictionary of configuration updates

        Raises:
            ConfigValidationError: If updated configuration is invalid
        """
        # Store previous config for rollback
        self._previous_config = self.current_config

        # Create updated configuration
        current_dict = self.current_config.to_dict()
        current_dict.update(config_updates)

        try:
            # Validate new configuration
            new_config = DynamicConfig.from_dict(current_dict)
            self.current_config = new_config

            self._add_to_history(config_updates, "Dynamic update")

            logger.info(f"Configuration updated: {config_updates}")

        except ConfigValidationError:
            # Restore previous config on validation failure
            self.current_config = self._previous_config
            raise

    async def start_monitoring(self) -> None:
        """Start monitoring configuration file for changes."""
        if not self.config_file_path:
            logger.warning("No configuration file path set for monitoring")
            return

        if self.is_monitoring:
            logger.warning("Configuration monitoring already started")
            return

        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitor_config_file())

        logger.info("Configuration file monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop monitoring configuration file."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Configuration file monitoring stopped")

    async def _monitor_config_file(self) -> None:
        """Monitor configuration file for changes."""
        if not self.config_file_path:
            return

        last_modified = 0

        while self.is_monitoring:
            try:
                if os.path.exists(self.config_file_path):
                    current_modified = os.path.getmtime(self.config_file_path)

                    if current_modified > last_modified:
                        last_modified = current_modified
                        await self.load_config_from_file(self.config_file_path)
                        logger.info("Configuration reloaded due to file change")

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error monitoring configuration file: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    async def adapt_to_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Adapt configuration based on performance metrics.

        Args:
            metrics: Performance metrics including latency, CPU, memory, throughput
        """
        avg_latency = metrics.get('avg_latency_ms', 0)
        cpu_usage = metrics.get('cpu_usage_percent', 0)
        memory_usage = metrics.get('memory_usage_percent', 0)
        events_per_second = metrics.get('events_per_second', 0)

        config_updates = {}

        # Adapt database pool size based on latency and load
        current_pool_size = self.current_config.database_pool_size

        if avg_latency > 100 or cpu_usage > 80:
            # High load: increase pool size
            new_pool_size = min(current_pool_size + 10, 200)
            if new_pool_size != current_pool_size:
                config_updates['database_pool_size'] = new_pool_size

        elif avg_latency < 50 and cpu_usage < 40 and current_pool_size > 20:
            # Low load: decrease pool size
            new_pool_size = max(current_pool_size - 5, 20)
            if new_pool_size != current_pool_size:
                config_updates['database_pool_size'] = new_pool_size

        # Adapt concurrent orders based on throughput
        current_concurrent = self.current_config.max_concurrent_orders

        if events_per_second > 1000 and cpu_usage < 70:
            # High throughput with available resources: increase concurrency
            new_concurrent = min(current_concurrent + 10, 1000)
            if new_concurrent != current_concurrent:
                config_updates['max_concurrent_orders'] = new_concurrent

        elif events_per_second < 300 and current_concurrent > 10:
            # Low throughput: reduce concurrency
            new_concurrent = max(current_concurrent - 5, 10)
            if new_concurrent != current_concurrent:
                config_updates['max_concurrent_orders'] = new_concurrent

        # Apply updates if any
        if config_updates:
            await self.update_config(config_updates)
            logger.info(f"Configuration adapted based on metrics: {config_updates}")

    def get_optimization_recommendations(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Get configuration optimization recommendations based on metrics.

        Args:
            metrics: Current performance metrics

        Returns:
            Dictionary with suggested changes and impact assessment
        """
        recommendations = {
            'suggested_changes': {},
            'impact_assessment': {},
            'confidence_score': 0.0
        }

        avg_latency = metrics.get('avg_latency_ms', 0)
        cpu_usage = metrics.get('cpu_usage_percent', 0)
        memory_usage = metrics.get('memory_usage_percent', 0)

        # Database recommendations
        if avg_latency > 80:
            recommendations['suggested_changes']['database_pool_size'] = \
                min(self.current_config.database_pool_size + 20, 200)
            recommendations['impact_assessment']['database_pool_size'] = \
                'Increased pool size should reduce query latency'

        # Concurrency recommendations
        if cpu_usage < 50 and avg_latency < 60:
            recommendations['suggested_changes']['max_concurrent_orders'] = \
                min(self.current_config.max_concurrent_orders + 15, 1000)
            recommendations['impact_assessment']['max_concurrent_orders'] = \
                'Increased concurrency can improve throughput'

        # Calculate confidence score
        data_quality = 1.0  # Assume good data quality for now
        metric_consistency = 1.0  # Assume consistent metrics
        recommendations['confidence_score'] = (data_quality + metric_consistency) / 2

        return recommendations

    def calculate_change_impact(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate the impact of configuration changes.

        Args:
            old_config: Previous configuration
            new_config: New configuration

        Returns:
            Impact assessment with performance, resource, and risk metrics
        """
        impact = {
            'performance_impact': 0.0,
            'resource_impact': 0.0,
            'risk_level': 0.0
        }

        # Calculate resource impact
        pool_change = new_config.get('database_pool_size', 0) - old_config.get('database_pool_size', 0)
        concurrent_change = new_config.get('max_concurrent_orders', 0) - old_config.get('max_concurrent_orders', 0)

        impact['resource_impact'] = (pool_change + concurrent_change) / 100.0

        # Calculate performance impact (positive = better performance)
        if pool_change > 0 or concurrent_change > 0:
            impact['performance_impact'] = 0.1  # Positive impact expected
        else:
            impact['performance_impact'] = -0.05  # Slight negative impact

        # Calculate risk level
        total_change = abs(pool_change) + abs(concurrent_change)
        impact['risk_level'] = min(total_change / 200.0, 1.0)  # Normalize to 0-1

        return impact

    async def rollback_configuration(self) -> None:
        """Rollback to previous configuration."""
        if self._previous_config is None:
            logger.warning("No previous configuration available for rollback")
            return

        current_config = self.current_config
        self.current_config = self._previous_config
        self._previous_config = current_config

        self._add_to_history(self.current_config.to_dict(), "Configuration rollback")

        logger.info("Configuration rolled back to previous state")

    def get_configuration_history(self) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return self._config_history.copy()

    def get_configuration_metrics(self) -> Dict[str, Any]:
        """Get configuration-related metrics."""
        return {
            'current_config': self.current_config.to_dict(),
            'optimization_score': self._calculate_optimization_score(),
            'change_count': len(self._config_history),
            'last_update': self._config_history[-1]['timestamp'] if self._config_history else None
        }

    def _add_to_history(self, config: Dict[str, Any], reason: str) -> None:
        """Add configuration change to history."""
        self._config_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'config': config.copy(),
            'change_reason': reason
        })

        # Keep only last 100 entries
        if len(self._config_history) > 100:
            self._config_history = self._config_history[-100:]

    def _calculate_optimization_score(self) -> float:
        """Calculate optimization score based on current configuration."""
        # Simple optimization score based on configuration balance
        config = self.current_config

        # Score components (0-1 each)
        pool_score = min(config.database_pool_size / 100.0, 1.0)
        concurrent_score = min(config.max_concurrent_orders / 50.0, 1.0)
        cache_score = min(config.cache_ttl / 600.0, 1.0)

        # Average score
        return (pool_score + concurrent_score + cache_score) / 3.0