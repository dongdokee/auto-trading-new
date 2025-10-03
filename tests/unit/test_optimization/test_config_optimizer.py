"""
Tests for Dynamic Configuration Manager.

Following TDD methodology: Red -> Green -> Refactor
Tests for configuration optimization, hot-reload, and adaptive parameter adjustment.
"""

import pytest
import asyncio
import tempfile
import json
import os
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.optimization.config_optimizer import ConfigOptimizer, DynamicConfig, ConfigValidationError


class TestDynamicConfig:
    """Test suite for DynamicConfig class."""

    def test_should_initialize_with_default_values(self):
        """Test that DynamicConfig initializes with appropriate default values."""
        config = DynamicConfig()

        assert config.database_pool_size == 20
        assert config.max_overflow == 30
        assert config.event_queue_size == 10000
        assert config.max_concurrent_orders == 10
        assert config.risk_check_interval == 30
        assert config.metrics_interval == 10
        assert config.cache_ttl == 300
        assert config.hot_reload_enabled is True

    def test_should_initialize_with_custom_values(self):
        """Test that DynamicConfig accepts custom initialization values."""
        config = DynamicConfig(
            database_pool_size=50,
            max_concurrent_orders=100,
            hot_reload_enabled=False
        )

        assert config.database_pool_size == 50
        assert config.max_concurrent_orders == 100
        assert config.hot_reload_enabled is False
        assert config.max_overflow == 30  # Default value

    def test_should_validate_configuration_values(self):
        """Test that DynamicConfig validates configuration constraints."""
        # Test invalid pool size
        with pytest.raises(ConfigValidationError, match="database_pool_size must be between 1 and 200"):
            DynamicConfig(database_pool_size=0)

        with pytest.raises(ConfigValidationError, match="database_pool_size must be between 1 and 200"):
            DynamicConfig(database_pool_size=300)

        # Test invalid concurrent orders
        with pytest.raises(ConfigValidationError, match="max_concurrent_orders must be between 1 and 1000"):
            DynamicConfig(max_concurrent_orders=0)

        # Test invalid intervals
        with pytest.raises(ConfigValidationError, match="risk_check_interval must be between 1 and 300"):
            DynamicConfig(risk_check_interval=0)

    def test_should_convert_to_dictionary(self):
        """Test that DynamicConfig can be converted to dictionary."""
        config = DynamicConfig(database_pool_size=50, max_concurrent_orders=25)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['database_pool_size'] == 50
        assert config_dict['max_concurrent_orders'] == 25
        assert 'hot_reload_enabled' in config_dict

    def test_should_create_from_dictionary(self):
        """Test that DynamicConfig can be created from dictionary."""
        config_dict = {
            'database_pool_size': 75,
            'max_concurrent_orders': 50,
            'cache_ttl': 600
        }

        config = DynamicConfig.from_dict(config_dict)

        assert config.database_pool_size == 75
        assert config.max_concurrent_orders == 50
        assert config.cache_ttl == 600


class TestConfigOptimizer:
    """Test suite for ConfigOptimizer class."""

    @pytest.fixture
    def config_optimizer(self):
        """Create ConfigOptimizer instance for testing."""
        return ConfigOptimizer()

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'database_pool_size': 30,
                'max_concurrent_orders': 20,
                'cache_ttl': 400
            }
            json.dump(config_data, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_should_initialize_with_default_configuration(self, config_optimizer):
        """Test that ConfigOptimizer initializes with default configuration."""
        assert config_optimizer.current_config is not None
        assert isinstance(config_optimizer.current_config, DynamicConfig)
        assert config_optimizer.config_file_path is None
        assert config_optimizer.is_monitoring is False

    @pytest.mark.asyncio
    async def test_should_load_configuration_from_file(self, config_optimizer, temp_config_file):
        """Test that ConfigOptimizer can load configuration from file."""
        await config_optimizer.load_config_from_file(temp_config_file)

        assert config_optimizer.config_file_path == temp_config_file
        assert config_optimizer.current_config.database_pool_size == 30
        assert config_optimizer.current_config.max_concurrent_orders == 20
        assert config_optimizer.current_config.cache_ttl == 400

    @pytest.mark.asyncio
    async def test_should_handle_invalid_config_file(self, config_optimizer):
        """Test that ConfigOptimizer handles invalid configuration files gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            invalid_file = f.name

        try:
            with pytest.raises(ConfigValidationError, match="Failed to parse configuration file"):
                await config_optimizer.load_config_from_file(invalid_file)
        finally:
            os.unlink(invalid_file)

    @pytest.mark.asyncio
    async def test_should_save_configuration_to_file(self, config_optimizer, temp_config_file):
        """Test that ConfigOptimizer can save configuration to file."""
        # Modify configuration
        config_optimizer.current_config.database_pool_size = 60
        config_optimizer.current_config.max_concurrent_orders = 40

        await config_optimizer.save_config_to_file(temp_config_file)

        # Verify file contents
        with open(temp_config_file, 'r') as f:
            saved_config = json.load(f)

        assert saved_config['database_pool_size'] == 60
        assert saved_config['max_concurrent_orders'] == 40

    @pytest.mark.asyncio
    async def test_should_update_configuration_dynamically(self, config_optimizer):
        """Test that ConfigOptimizer can update configuration dynamically."""
        new_config = {
            'database_pool_size': 80,
            'max_concurrent_orders': 60
        }

        old_config = config_optimizer.current_config.to_dict()

        await config_optimizer.update_config(new_config)

        assert config_optimizer.current_config.database_pool_size == 80
        assert config_optimizer.current_config.max_concurrent_orders == 60
        # Unchanged values should remain
        assert config_optimizer.current_config.cache_ttl == old_config['cache_ttl']

    @pytest.mark.asyncio
    async def test_should_validate_configuration_before_update(self, config_optimizer):
        """Test that ConfigOptimizer validates configuration before updating."""
        invalid_config = {
            'database_pool_size': -10,  # Invalid value
            'max_concurrent_orders': 50
        }

        with pytest.raises(ConfigValidationError):
            await config_optimizer.update_config(invalid_config)

        # Original config should remain unchanged
        assert config_optimizer.current_config.database_pool_size == 20  # Default value

    @pytest.mark.asyncio
    async def test_should_start_file_monitoring(self, config_optimizer, temp_config_file):
        """Test that ConfigOptimizer can start monitoring configuration file."""
        await config_optimizer.load_config_from_file(temp_config_file)

        with patch('asyncio.create_task') as mock_create_task:
            await config_optimizer.start_monitoring()

            assert config_optimizer.is_monitoring is True
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_stop_file_monitoring(self, config_optimizer):
        """Test that ConfigOptimizer can stop monitoring configuration file."""
        config_optimizer.is_monitoring = True

        # Create a real task that we can cancel
        async def dummy_task():
            await asyncio.sleep(10)

        config_optimizer._monitoring_task = asyncio.create_task(dummy_task())

        await config_optimizer.stop_monitoring()

        assert config_optimizer.is_monitoring is False
        assert config_optimizer._monitoring_task.cancelled()

    @pytest.mark.asyncio
    async def test_should_adapt_configuration_based_on_metrics(self, config_optimizer):
        """Test that ConfigOptimizer can adapt configuration based on performance metrics."""
        # Simulate high load metrics
        metrics = {
            'avg_latency_ms': 150,  # High latency
            'cpu_usage_percent': 85,  # High CPU
            'memory_usage_percent': 75,
            'events_per_second': 1500  # High throughput
        }

        original_pool_size = config_optimizer.current_config.database_pool_size

        await config_optimizer.adapt_to_performance_metrics(metrics)

        # Should increase pool size due to high load
        assert config_optimizer.current_config.database_pool_size > original_pool_size

    @pytest.mark.asyncio
    async def test_should_reduce_resources_on_low_load(self, config_optimizer):
        """Test that ConfigOptimizer reduces resources during low load."""
        # First set high values
        config_optimizer.current_config.database_pool_size = 100
        config_optimizer.current_config.max_concurrent_orders = 80

        # Simulate low load metrics
        metrics = {
            'avg_latency_ms': 25,  # Low latency
            'cpu_usage_percent': 30,  # Low CPU
            'memory_usage_percent': 40,
            'events_per_second': 200  # Low throughput
        }

        await config_optimizer.adapt_to_performance_metrics(metrics)

        # Should reduce pool size due to low load
        assert config_optimizer.current_config.database_pool_size < 100

    @pytest.mark.asyncio
    async def test_should_not_exceed_configuration_limits(self, config_optimizer):
        """Test that ConfigOptimizer respects configuration limits during adaptation."""
        # Simulate extreme metrics
        metrics = {
            'avg_latency_ms': 1000,  # Very high latency
            'cpu_usage_percent': 95,  # Very high CPU
            'memory_usage_percent': 90,
            'events_per_second': 5000  # Very high throughput
        }

        await config_optimizer.adapt_to_performance_metrics(metrics)

        # Should not exceed maximum limits
        assert config_optimizer.current_config.database_pool_size <= 200
        assert config_optimizer.current_config.max_concurrent_orders <= 1000

    def test_should_get_configuration_recommendations(self, config_optimizer):
        """Test that ConfigOptimizer provides configuration recommendations."""
        current_metrics = {
            'avg_latency_ms': 100,
            'cpu_usage_percent': 70,
            'memory_usage_percent': 60,
            'events_per_second': 800
        }

        recommendations = config_optimizer.get_optimization_recommendations(current_metrics)

        assert isinstance(recommendations, dict)
        assert 'suggested_changes' in recommendations
        assert 'impact_assessment' in recommendations
        assert 'confidence_score' in recommendations

        # Should provide specific recommendations
        assert len(recommendations['suggested_changes']) > 0

    def test_should_calculate_configuration_impact(self, config_optimizer):
        """Test that ConfigOptimizer can calculate impact of configuration changes."""
        old_config = config_optimizer.current_config.to_dict()
        new_config = old_config.copy()
        new_config['database_pool_size'] = old_config['database_pool_size'] * 2

        impact = config_optimizer.calculate_change_impact(old_config, new_config)

        assert isinstance(impact, dict)
        assert 'performance_impact' in impact
        assert 'resource_impact' in impact
        assert 'risk_level' in impact

        # Should indicate increased resource usage
        assert impact['resource_impact'] > 0

    @pytest.mark.asyncio
    async def test_should_handle_configuration_rollback(self, config_optimizer):
        """Test that ConfigOptimizer can rollback configuration changes."""
        original_config = config_optimizer.current_config.to_dict()

        # Make changes
        await config_optimizer.update_config({
            'database_pool_size': 100,
            'max_concurrent_orders': 80
        })

        # Rollback
        await config_optimizer.rollback_configuration()

        current_config = config_optimizer.current_config.to_dict()
        assert current_config['database_pool_size'] == original_config['database_pool_size']
        assert current_config['max_concurrent_orders'] == original_config['max_concurrent_orders']

    @pytest.mark.asyncio
    async def test_should_maintain_configuration_history(self, config_optimizer):
        """Test that ConfigOptimizer maintains history of configuration changes."""
        # Make several configuration changes
        await config_optimizer.update_config({'database_pool_size': 50})
        await config_optimizer.update_config({'max_concurrent_orders': 30})
        await config_optimizer.update_config({'cache_ttl': 600})

        history = config_optimizer.get_configuration_history()

        assert len(history) >= 3
        assert all('timestamp' in entry for entry in history)
        assert all('config' in entry for entry in history)
        assert all('change_reason' in entry for entry in history)

    def test_should_export_configuration_metrics(self, config_optimizer):
        """Test that ConfigOptimizer can export configuration metrics."""
        metrics = config_optimizer.get_configuration_metrics()

        assert isinstance(metrics, dict)
        assert 'current_config' in metrics
        assert 'optimization_score' in metrics
        assert 'change_count' in metrics
        assert 'last_update' in metrics

        # Optimization score should be between 0 and 1
        assert 0 <= metrics['optimization_score'] <= 1