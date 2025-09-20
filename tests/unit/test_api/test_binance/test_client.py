# tests/unit/test_api/test_binance/test_client.py
import pytest
import json
import hmac
import hashlib
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientSession, ClientResponse

from src.execution.models import Order, OrderSide, OrderUrgency
from src.api.base import ExchangeConfig


class TestBinanceClient:
    """Test Binance REST API client"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return ExchangeConfig(
            api_key="test_api_key_123456",
            api_secret="test_api_secret_123456",
            testnet=True,
            timeout=30,
            rate_limit_per_minute=1200
        )

    @pytest.fixture
    def client(self, config):
        """Test client instance"""
        from src.api.binance.client import BinanceClient
        return BinanceClient(config)

    def test_should_inherit_from_base_client(self, client):
        """BinanceClient should inherit from BaseExchangeClient"""
        from src.api.base import BaseExchangeClient
        assert isinstance(client, BaseExchangeClient)

    def test_should_initialize_with_correct_base_urls(self, client):
        """Should set correct base URLs for testnet and mainnet"""
        assert client.testnet_base_url == "https://testnet.binancefuture.com"
        assert client.mainnet_base_url == "https://fapi.binance.com"

        # Should use testnet URL when testnet=True
        assert client.base_url == client.testnet_base_url

    def test_should_generate_signature(self, client):
        """Should generate correct HMAC-SHA256 signature"""
        query_string = "symbol=BTCUSDT&side=BUY&type=MARKET&quantity=1"
        signature = client._generate_signature(query_string)

        # Verify signature format
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex digest length

        # Verify signature calculation
        expected = hmac.new(
            client.config.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        assert signature == expected

    def test_should_add_timestamp_to_request(self, client):
        """Should add timestamp to authenticated requests"""
        params = {"symbol": "BTCUSDT"}
        signed_params = client._add_signature(params)

        assert "timestamp" in signed_params
        assert "signature" in signed_params
        assert isinstance(signed_params["timestamp"], int)

        # Timestamp should be recent (within 5 seconds)
        current_time = int(time.time() * 1000)
        assert abs(signed_params["timestamp"] - current_time) < 5000

    @pytest.mark.asyncio
    async def test_should_connect_successfully(self, client):
        """Should establish connection successfully"""
        with patch.object(client, '_test_connectivity') as mock_test:
            mock_test.return_value = True
            await client.connect()
            assert client._connected is True
            mock_test.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_handle_connection_failure(self, client):
        """Should handle connection failures gracefully"""
        from src.api.base import ConnectionError

        with patch.object(client, '_test_connectivity') as mock_test:
            mock_test.side_effect = Exception("Network error")

            with pytest.raises(ConnectionError):
                await client.connect()

            assert client._connected is False

    @pytest.mark.asyncio
    async def test_should_submit_market_order(self, client):
        """Should submit market order successfully"""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("1.0"),
            urgency=OrderUrgency.IMMEDIATE
        )

        mock_response = {
            "orderId": 123456,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "executedQty": "1.0",
            "avgPrice": "50000.0"
        }

        with patch.object(client, '_make_authenticated_request') as mock_request:
            mock_request.return_value = mock_response

            result = await client.submit_order(order)

            assert result["orderId"] == 123456
            assert result["symbol"] == "BTCUSDT"
            assert result["status"] == "FILLED"

            # Verify the request was made with correct parameters
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"  # method
            assert call_args[0][1] == "/fapi/v1/order"  # endpoint

    @pytest.mark.asyncio
    async def test_should_submit_limit_order(self, client):
        """Should submit limit order with price"""
        order = Order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            size=Decimal("10.0"),
            urgency=OrderUrgency.MEDIUM,
            price=Decimal("3000.0")
        )

        mock_response = {
            "orderId": 789012,
            "symbol": "ETHUSDT",
            "status": "NEW",
            "type": "LIMIT"
        }

        with patch.object(client, '_make_authenticated_request') as mock_request:
            mock_request.return_value = mock_response

            result = await client.submit_order(order)

            assert result["orderId"] == 789012
            assert result["type"] == "LIMIT"

            # Verify limit order parameters
            call_args = mock_request.call_args[1]["params"]
            assert call_args["type"] == "LIMIT"
            assert call_args["price"] == "3000.0"
            assert call_args["timeInForce"] == "GTC"

    @pytest.mark.asyncio
    async def test_should_cancel_order(self, client):
        """Should cancel order successfully"""
        order_id = "123456"

        mock_response = {
            "orderId": 123456,
            "symbol": "BTCUSDT",
            "status": "CANCELED"
        }

        with patch.object(client, '_make_authenticated_request') as mock_request:
            mock_request.return_value = mock_response

            result = await client.cancel_order(order_id)

            assert result is True
            mock_request.assert_called_once_with(
                "DELETE",
                "/fapi/v1/order",
                params={"orderId": order_id}
            )

    @pytest.mark.asyncio
    async def test_should_get_order_status(self, client):
        """Should retrieve order status"""
        order_id = "123456"

        mock_response = {
            "orderId": 123456,
            "symbol": "BTCUSDT",
            "status": "PARTIALLY_FILLED",
            "executedQty": "0.5",
            "origQty": "1.0"
        }

        with patch.object(client, '_make_authenticated_request') as mock_request:
            mock_request.return_value = mock_response

            result = await client.get_order_status(order_id)

            assert result["orderId"] == 123456
            assert result["status"] == "PARTIALLY_FILLED"
            assert result["executedQty"] == "0.5"

    @pytest.mark.asyncio
    async def test_should_get_account_balance(self, client):
        """Should retrieve account balance"""
        mock_response = {
            "totalWalletBalance": "10000.00",
            "availableBalance": "8500.00",
            "assets": [
                {"asset": "USDT", "free": "8500.00", "locked": "1500.00"}
            ]
        }

        with patch.object(client, '_make_authenticated_request') as mock_request:
            mock_request.return_value = mock_response

            result = await client.get_account_balance()

            assert "USDT" in result
            assert result["USDT"] == Decimal("8500.00")

    @pytest.mark.asyncio
    async def test_should_get_positions(self, client):
        """Should retrieve current positions"""
        mock_response = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "1.5",
                "entryPrice": "50000.0",
                "unrealizedPnl": "500.0",
                "positionSide": "LONG"
            }
        ]

        with patch.object(client, '_make_authenticated_request') as mock_request:
            mock_request.return_value = mock_response

            result = await client.get_positions()

            assert len(result) == 1
            assert result[0]["symbol"] == "BTCUSDT"
            assert result[0]["positionAmt"] == "1.5"

    @pytest.mark.asyncio
    async def test_should_get_market_data(self, client):
        """Should retrieve market data for symbol"""
        symbol = "BTCUSDT"
        mock_response = {
            "symbol": "BTCUSDT",
            "price": "50000.0",
            "bidPrice": "49999.0",
            "askPrice": "50001.0",
            "volume": "1000.0"
        }

        with patch.object(client, '_make_public_request') as mock_request:
            mock_request.return_value = mock_response

            result = await client.get_market_data(symbol)

            assert result["symbol"] == "BTCUSDT"
            assert result["price"] == "50000.0"
            mock_request.assert_called_once_with(
                "GET",
                "/fapi/v1/ticker/24hr",
                params={"symbol": symbol}
            )

    @pytest.mark.asyncio
    async def test_should_handle_api_errors(self, client):
        """Should handle API errors properly"""
        from src.api.binance.exceptions import BinanceAPIError

        with patch.object(client, '_make_authenticated_request') as mock_request:
            mock_request.side_effect = BinanceAPIError("Insufficient balance", -2019)

            order = Order("BTCUSDT", OrderSide.BUY, Decimal("100.0"))

            with pytest.raises(BinanceAPIError) as exc_info:
                await client.submit_order(order)

            assert exc_info.value.code == -2019
            assert "Insufficient balance" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_should_respect_rate_limits(self, client):
        """Should respect rate limiting"""
        # Mock rate limiter to be at limit
        client.rate_limiter.tokens = 0

        with patch.object(client.rate_limiter, 'wait_for_reset') as mock_wait:
            mock_wait.return_value = None

            with patch.object(client, '_make_authenticated_request') as mock_request:
                mock_request.return_value = {"orderId": 123}

                order = Order("BTCUSDT", OrderSide.BUY, Decimal("1.0"))
                await client.submit_order(order)

                # Should have waited for rate limit reset
                mock_wait.assert_called_once()