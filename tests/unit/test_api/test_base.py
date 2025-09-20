# tests/unit/test_api/test_base.py
import pytest
from abc import ABC
from decimal import Decimal
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.execution.models import Order, OrderSide, OrderUrgency


class TestBaseExchangeClient:
    """Test abstract base exchange client interface"""

    def test_should_be_abstract_class(self):
        """BaseExchangeClient should be an abstract class"""
        from src.api.base import BaseExchangeClient
        assert issubclass(BaseExchangeClient, ABC)

    def test_should_have_required_abstract_methods(self):
        """BaseExchangeClient should define required abstract methods"""
        from src.api.base import BaseExchangeClient

        abstract_methods = BaseExchangeClient.__abstractmethods__
        required_methods = {
            'connect',
            'disconnect',
            'submit_order',
            'cancel_order',
            'get_order_status',
            'get_account_balance',
            'get_positions',
            'get_market_data',
            'subscribe_to_orderbook',
            'subscribe_to_trades'
        }

        assert required_methods.issubset(abstract_methods)

    def test_should_fail_instantiation_directly(self):
        """Cannot instantiate BaseExchangeClient directly"""
        from src.api.base import BaseExchangeClient

        with pytest.raises(TypeError):
            BaseExchangeClient()


class TestBaseExchangeClientConfig:
    """Test base exchange client configuration"""

    def test_should_validate_api_credentials(self):
        """Should validate API key and secret are provided"""
        from src.api.base import ExchangeConfig

        # Should fail with empty credentials
        with pytest.raises(ValueError, match="API key cannot be empty"):
            ExchangeConfig(
                api_key="",
                api_secret="test_secret",
                testnet=True
            )

        with pytest.raises(ValueError, match="API secret cannot be empty"):
            ExchangeConfig(
                api_key="test_key",
                api_secret="",
                testnet=True
            )

    def test_should_validate_api_credential_length(self):
        """Should validate minimum length for API credentials"""
        from src.api.base import ExchangeConfig

        with pytest.raises(ValueError, match="API key must be at least 8 characters"):
            ExchangeConfig(
                api_key="short",
                api_secret="valid_secret_123",
                testnet=True
            )

    def test_should_create_valid_config(self):
        """Should create valid configuration with proper credentials"""
        from src.api.base import ExchangeConfig

        config = ExchangeConfig(
            api_key="valid_api_key_123",
            api_secret="valid_api_secret_456",
            testnet=True,
            timeout=30,
            rate_limit_per_minute=1200
        )

        assert config.api_key == "valid_api_key_123"
        assert config.api_secret == "valid_api_secret_456"
        assert config.testnet is True
        assert config.timeout == 30
        assert config.rate_limit_per_minute == 1200


class TestOrderConversion:
    """Test order model conversion utilities"""

    def test_should_convert_internal_order_to_exchange_format(self):
        """Should convert internal Order to exchange-specific format"""
        from src.api.base import OrderConverter

        internal_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("1.5"),
            urgency=OrderUrgency.MEDIUM,
            price=Decimal("50000.0")
        )

        converter = OrderConverter()
        exchange_order = converter.to_exchange_format(internal_order)

        assert exchange_order['symbol'] == "BTCUSDT"
        assert exchange_order['side'] == "BUY"
        assert exchange_order['quantity'] == "1.5"
        assert exchange_order['price'] == "50000.0"
        assert 'type' in exchange_order

    def test_should_handle_market_order_conversion(self):
        """Should handle market orders without price"""
        from src.api.base import OrderConverter

        market_order = Order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            size=Decimal("10.0"),
            urgency=OrderUrgency.IMMEDIATE
        )

        converter = OrderConverter()
        exchange_order = converter.to_exchange_format(market_order)

        assert exchange_order['type'] == 'MARKET'
        assert 'price' not in exchange_order or exchange_order['price'] is None


class TestRateLimitManager:
    """Test rate limiting functionality"""

    def test_should_track_request_rate(self):
        """Should track request rate within time window"""
        from src.api.base import RateLimitManager

        manager = RateLimitManager(requests_per_minute=60)

        # Should allow requests within limit
        assert manager.can_make_request() is True
        manager.record_request()

        assert manager.get_remaining_requests() == 59

    def test_should_block_when_rate_limit_exceeded(self):
        """Should block requests when rate limit is exceeded"""
        from src.api.base import RateLimitManager

        manager = RateLimitManager(requests_per_minute=2)

        # Use up the rate limit
        assert manager.can_make_request() is True
        manager.record_request()
        assert manager.can_make_request() is True
        manager.record_request()

        # Should now be blocked
        assert manager.can_make_request() is False

    @pytest.mark.asyncio
    async def test_should_wait_for_rate_limit_reset(self):
        """Should properly wait for rate limit reset"""
        from src.api.base import RateLimitManager

        manager = RateLimitManager(requests_per_minute=60, window_seconds=1)

        # Fill rate limit
        for _ in range(60):
            manager.record_request()

        # Should be blocked
        assert manager.can_make_request() is False

        # Wait for reset and check
        await manager.wait_for_reset()
        assert manager.can_make_request() is True


class TestConnectionManager:
    """Test connection management functionality"""

    @pytest.mark.asyncio
    async def test_should_manage_connection_state(self):
        """Should properly manage connection state"""
        from src.api.base import ConnectionManager

        manager = ConnectionManager()

        assert manager.is_connected() is False

        await manager.connect()
        assert manager.is_connected() is True

        await manager.disconnect()
        assert manager.is_connected() is False

    @pytest.mark.asyncio
    async def test_should_handle_connection_errors(self):
        """Should handle connection errors gracefully"""
        from src.api.base import ConnectionManager, ConnectionError

        manager = ConnectionManager()

        # Mock a connection failure
        manager._connection_factory = AsyncMock(side_effect=Exception("Network error"))

        with pytest.raises(ConnectionError):
            await manager.connect()

        assert manager.is_connected() is False