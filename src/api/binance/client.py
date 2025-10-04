# src/api/binance/client.py
import hmac
import hashlib
import time
import json
from decimal import Decimal
from typing import Dict, Any, List, Optional, Callable
from urllib.parse import urlencode

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from src.api.base import BaseExchangeClient, ExchangeConfig, RateLimitManager, ConnectionError
from src.execution.models import Order, OrderSide, OrderUrgency
from src.core.patterns import LoggerFactory
from .exceptions import BinanceAPIError, BinanceConnectionError, BinanceRateLimitError, BinanceOrderError


class BinanceClient(BaseExchangeClient):
    """Binance Futures REST API client"""

    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.name = "BinanceClient"
        self.logger = LoggerFactory.get_api_logger("binance")

        # API endpoints
        self.testnet_base_url = "https://testnet.binancefuture.com"
        self.mainnet_base_url = "https://fapi.binance.com"
        self.base_url = self.testnet_base_url if config.testnet else self.mainnet_base_url

        # Rate limiting - handle both config models
        rate_limit = getattr(config, 'rate_limit_requests', None) or getattr(config, 'rate_limit_per_minute', 1200)
        self.rate_limiter = RateLimitManager(rate_limit)

        # HTTP session
        self.session: Optional[ClientSession] = None

    async def _create_exchange_connection(self) -> ClientSession:
        """Create HTTP session for Binance API"""
        # Create HTTP session with threaded resolver to avoid aiodns issues on Windows
        timeout = ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
        session = ClientSession(timeout=timeout, connector=connector)
        self.session = session
        return session

    async def _close_exchange_connection(self, connection: ClientSession) -> None:
        """Close HTTP session"""
        if connection:
            await connection.close()
        self.session = None

    async def _test_connection(self, connection: ClientSession) -> bool:
        """Test if Binance API is reachable"""
        try:
            return await self._test_connectivity()
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    async def submit_order(self, order: Order) -> Dict[str, Any]:
        """Submit an order to Binance"""
        if not self.rate_limiter.can_make_request():
            await self.rate_limiter.wait_for_reset()

        # Convert order to Binance format
        params = self._order_to_binance_params(order)

        try:
            result = await self._make_authenticated_request("POST", "/fapi/v1/order", params=params)
            self.rate_limiter.record_request()
            return result
        except (BinanceAPIError, BinanceRateLimitError) as e:
            # Re-raise Binance-specific errors as-is
            raise e
        except Exception as e:
            raise BinanceOrderError(f"Failed to submit order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            params = {"orderId": order_id}
            result = await self._make_authenticated_request("DELETE", "/fapi/v1/order", params=params)
            return result.get("status") == "CANCELED"
        except Exception as e:
            raise BinanceOrderError(f"Failed to cancel order: {e}")

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get the status of an order"""
        try:
            params = {"orderId": order_id}
            return await self._make_authenticated_request("GET", "/fapi/v1/order", params=params)
        except Exception as e:
            raise BinanceAPIError(f"Failed to get order status: {e}")

    async def get_account_balance(self) -> Dict[str, Decimal]:
        """Get account balance information"""
        try:
            response = await self._make_authenticated_request("GET", "/fapi/v2/account")
            balances = {}

            for asset in response.get("assets", []):
                symbol = asset["asset"]
                free_balance = Decimal(asset["free"])
                balances[symbol] = free_balance

            return balances
        except Exception as e:
            raise BinanceAPIError(f"Failed to get account balance: {e}")

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            return await self._make_authenticated_request("GET", "/fapi/v2/positionRisk")
        except Exception as e:
            raise BinanceAPIError(f"Failed to get positions: {e}")

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data for a symbol"""
        try:
            params = {"symbol": symbol}
            return await self._make_public_request("GET", "/fapi/v1/ticker/24hr", params=params)
        except Exception as e:
            raise BinanceAPIError(f"Failed to get market data: {e}")

    async def subscribe_to_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to orderbook updates (WebSocket implementation needed)"""
        raise NotImplementedError("WebSocket subscriptions not yet implemented")

    async def subscribe_to_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to trade updates (WebSocket implementation needed)"""
        raise NotImplementedError("WebSocket subscriptions not yet implemented")

    # Private helper methods

    async def _test_connectivity(self) -> bool:
        """Test API connectivity"""
        try:
            response = await self._make_public_request("GET", "/fapi/v1/ping")
            return response == {}
        except Exception:
            return False

    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC-SHA256 signature for authenticated requests"""
        return hmac.new(
            self.config.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()

    def _add_signature(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add timestamp and signature to request parameters"""
        # Add timestamp
        params["timestamp"] = int(time.time() * 1000)

        # Create query string
        query_string = urlencode(sorted(params.items()))

        # Add signature
        params["signature"] = self._generate_signature(query_string)

        return params

    def _order_to_binance_params(self, order: Order) -> Dict[str, Any]:
        """Convert internal Order to Binance API parameters"""
        params = {
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": str(order.size)
        }

        # Determine order type based on urgency and price
        if order.urgency == OrderUrgency.IMMEDIATE or order.price is None:
            params["type"] = "MARKET"
        else:
            params["type"] = "LIMIT"
            params["price"] = str(order.price)
            params["timeInForce"] = "GTC"  # Good Till Canceled

        return params

    async def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to Binance API"""
        if not self.session:
            raise BinanceConnectionError("Not connected to Binance API")

        if params is None:
            params = {}

        # Add signature
        signed_params = self._add_signature(params.copy())

        # Prepare headers
        headers = {
            "X-MBX-APIKEY": self.config.api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                async with self.session.get(url, params=signed_params, headers=headers) as response:
                    return await self._handle_response(response)
            elif method == "POST":
                async with self.session.post(url, data=signed_params, headers=headers) as response:
                    return await self._handle_response(response)
            elif method == "DELETE":
                async with self.session.delete(url, params=signed_params, headers=headers) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

        except aiohttp.ClientError as e:
            raise BinanceConnectionError(f"HTTP request failed: {e}")

    async def _make_public_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make public request to Binance API"""
        if not self.session:
            raise BinanceConnectionError("Not connected to Binance API")

        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                async with self.session.get(url, params=params) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method for public request: {method}")

        except aiohttp.ClientError as e:
            raise BinanceConnectionError(f"HTTP request failed: {e}")

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle HTTP response and parse JSON"""
        content = await response.text()

        if response.status == 200:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                raise BinanceAPIError("Invalid JSON response")

        # Handle errors
        try:
            error_data = json.loads(content)
            error_code = error_data.get("code", response.status)
            error_msg = error_data.get("msg", "Unknown error")
        except json.JSONDecodeError:
            error_code = response.status
            error_msg = content

        if response.status == 429:
            raise BinanceRateLimitError(error_msg, error_code)
        else:
            raise BinanceAPIError(error_msg, error_code)