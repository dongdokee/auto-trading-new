# tests/unit/test_api/test_binance/test_websocket.py
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from websockets.exceptions import ConnectionClosed, InvalidURI

from src.api.base import ExchangeConfig


class TestBinanceWebSocket:
    """Test Binance WebSocket manager"""

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
    def websocket_manager(self, config):
        """Test WebSocket manager instance"""
        from src.api.binance.websocket import BinanceWebSocket
        return BinanceWebSocket(config)

    def test_should_initialize_with_correct_urls(self, websocket_manager):
        """Should set correct WebSocket URLs for testnet and mainnet"""
        assert websocket_manager.testnet_ws_url == "wss://stream.binancefuture.com"
        assert websocket_manager.mainnet_ws_url == "wss://fstream.binance.com"

        # Should use testnet URL when testnet=True
        assert websocket_manager.base_ws_url == websocket_manager.testnet_ws_url

    @pytest.mark.asyncio
    async def test_should_connect_to_websocket(self, websocket_manager):
        """Should establish WebSocket connection"""
        mock_websocket = AsyncMock()
        mock_websocket.ping.return_value = AsyncMock()

        async def mock_connect_func(*args, **kwargs):
            return mock_websocket

        with patch('src.api.binance.websocket.websockets.connect', side_effect=mock_connect_func) as mock_connect, \
             patch('src.api.binance.websocket.asyncio.create_task') as mock_create_task:
            mock_create_task.return_value = AsyncMock()

            await websocket_manager.connect()

            assert websocket_manager.is_connected() is True
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_disconnect_from_websocket(self, websocket_manager):
        """Should close WebSocket connection properly"""
        mock_websocket = AsyncMock()
        websocket_manager.websocket = mock_websocket
        websocket_manager._connected = True

        await websocket_manager.disconnect()

        assert websocket_manager.is_connected() is False
        assert websocket_manager.websocket is None
        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_handle_connection_failure(self, websocket_manager):
        """Should handle WebSocket connection failures"""
        from src.api.binance.exceptions import BinanceConnectionError

        with patch('src.api.binance.websocket.websockets.connect') as mock_connect:
            mock_connect.side_effect = ConnectionClosed(None, None)

            with pytest.raises(BinanceConnectionError):
                await websocket_manager.connect()

            assert websocket_manager.is_connected() is False

    @pytest.mark.asyncio
    async def test_should_subscribe_to_orderbook(self, websocket_manager):
        """Should subscribe to orderbook depth stream"""
        mock_websocket = AsyncMock()
        websocket_manager.websocket = mock_websocket
        websocket_manager._connected = True

        callback = AsyncMock()
        await websocket_manager.subscribe_orderbook("BTCUSDT", callback)

        # Should send subscription message
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])

        assert sent_message["method"] == "SUBSCRIBE"
        assert "btcusdt@depth20@100ms" in sent_message["params"]

    @pytest.mark.asyncio
    async def test_should_subscribe_to_trades(self, websocket_manager):
        """Should subscribe to trade stream"""
        mock_websocket = AsyncMock()
        websocket_manager.websocket = mock_websocket
        websocket_manager._connected = True

        callback = AsyncMock()
        await websocket_manager.subscribe_trades("ETHUSDT", callback)

        # Should send subscription message
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])

        assert sent_message["method"] == "SUBSCRIBE"
        assert "ethusdt@aggTrade" in sent_message["params"]

    @pytest.mark.asyncio
    async def test_should_process_orderbook_message(self, websocket_manager):
        """Should process orderbook depth messages correctly"""
        callback = AsyncMock()
        websocket_manager.subscriptions["btcusdt@depth20@100ms"] = callback

        orderbook_message = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1634567890123,
                "s": "BTCUSDT",
                "U": 157,
                "u": 160,
                "b": [["50000.00", "1.5"], ["49999.00", "2.0"]],
                "a": [["50001.00", "1.2"], ["50002.00", "0.8"]]
            }
        }

        await websocket_manager._process_message(json.dumps(orderbook_message))

        # Should call the callback with processed data
        callback.assert_called_once()
        called_data = callback.call_args[0][0]
        assert called_data["symbol"] == "BTCUSDT"
        assert "bids" in called_data
        assert "asks" in called_data

    @pytest.mark.asyncio
    async def test_should_process_trade_message(self, websocket_manager):
        """Should process trade messages correctly"""
        callback = AsyncMock()
        websocket_manager.subscriptions["ethusdt@aggTrade"] = callback

        trade_message = {
            "stream": "ethusdt@aggTrade",
            "data": {
                "e": "aggTrade",
                "E": 1634567890123,
                "s": "ETHUSDT",
                "a": 12345,
                "p": "3000.00",
                "q": "10.5",
                "f": 100,
                "l": 105,
                "T": 1634567890000,
                "m": False
            }
        }

        await websocket_manager._process_message(json.dumps(trade_message))

        # Should call the callback with processed data
        callback.assert_called_once()
        called_data = callback.call_args[0][0]
        assert called_data["symbol"] == "ETHUSDT"
        assert called_data["price"] == "3000.00"
        assert called_data["quantity"] == "10.5"

    @pytest.mark.asyncio
    async def test_should_auto_reconnect_on_connection_loss(self, websocket_manager):
        """Should automatically reconnect when connection is lost"""
        websocket_manager._connected = True
        websocket_manager._auto_reconnect = True

        mock_websocket = AsyncMock()
        mock_websocket.recv.side_effect = [
            ConnectionClosed(None, None),  # Simulate connection loss
            '{"test": "message"}'  # After reconnection
        ]

        with patch.object(websocket_manager, 'connect') as mock_connect:
            mock_connect.return_value = None

            with patch.object(websocket_manager, '_listen_loop') as mock_listen:
                await websocket_manager._handle_connection_loss()

                # Should attempt to reconnect
                mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_send_ping_pong_heartbeat(self, websocket_manager):
        """Should handle ping/pong heartbeat messages"""
        mock_websocket = AsyncMock()
        websocket_manager.websocket = mock_websocket
        websocket_manager._connected = True

        # Simulate ping message
        await websocket_manager._send_ping()

        mock_websocket.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_handle_subscription_errors(self, websocket_manager):
        """Should handle subscription errors gracefully"""
        from src.api.binance.exceptions import BinanceConnectionError

        mock_websocket = AsyncMock()
        mock_websocket.send.side_effect = ConnectionClosed(None, None)
        websocket_manager.websocket = mock_websocket
        websocket_manager._connected = True

        callback = AsyncMock()

        with pytest.raises(BinanceConnectionError):
            await websocket_manager.subscribe_orderbook("BTCUSDT", callback)

    @pytest.mark.asyncio
    async def test_should_unsubscribe_from_stream(self, websocket_manager):
        """Should unsubscribe from streams properly"""
        mock_websocket = AsyncMock()
        websocket_manager.websocket = mock_websocket
        websocket_manager._connected = True

        # Add a subscription
        stream_name = "btcusdt@depth20@100ms"
        websocket_manager.subscriptions[stream_name] = AsyncMock()

        await websocket_manager.unsubscribe(stream_name)

        # Should send unsubscribe message
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])

        assert sent_message["method"] == "UNSUBSCRIBE"
        assert stream_name in sent_message["params"]
        assert stream_name not in websocket_manager.subscriptions

    @pytest.mark.asyncio
    async def test_should_handle_invalid_json_messages(self, websocket_manager):
        """Should handle invalid JSON messages gracefully"""
        # Should not raise exception for invalid JSON
        await websocket_manager._process_message("invalid json")

        # Should not raise exception for malformed messages
        await websocket_manager._process_message('{"incomplete": ')

    def test_should_generate_stream_names_correctly(self, websocket_manager):
        """Should generate correct stream names for different types"""
        # Orderbook stream
        orderbook_stream = websocket_manager._get_orderbook_stream("BTCUSDT")
        assert orderbook_stream == "btcusdt@depth20@100ms"

        # Trade stream
        trade_stream = websocket_manager._get_trade_stream("ETHUSDT")
        assert trade_stream == "ethusdt@aggTrade"

        # Mark price stream
        markprice_stream = websocket_manager._get_markprice_stream("ADAUSDT")
        assert markprice_stream == "adausdt@markPrice@1s"