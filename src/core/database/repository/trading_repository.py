"""
Specialized repository implementations for trading entities.
Extends BaseRepository with domain-specific query methods.
"""

from typing import List, Optional
from datetime import datetime
from decimal import Decimal

from .base import BaseRepository, RepositoryError
from ..schemas.trading_schemas import Position, Trade, Order, MarketData


class PositionRepository(BaseRepository[Position]):
    """Repository for Position entities with trading-specific queries"""

    def __init__(self, session):
        """Initialize position repository"""
        super().__init__(Position, session)

    async def get_open_positions_by_symbol(self, symbol: str) -> List[Position]:
        """
        Get all open positions for a specific symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')

        Returns:
            List of open positions for the symbol
        """
        try:
            positions = await self.find_by(symbol=symbol, status='OPEN')

            self.logger.info(f"Found {len(positions)} open positions for {symbol}")
            return positions

        except Exception as e:
            error_msg = f"Failed to get open positions for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_total_exposure_by_symbol(self, symbol: str) -> Decimal:
        """
        Calculate total exposure (sum of sizes) for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Total exposure as Decimal
        """
        try:
            open_positions = await self.get_open_positions_by_symbol(symbol)

            total_exposure = Decimal('0.0')
            for position in open_positions:
                if position.status == 'OPEN':
                    total_exposure += position.size

            self.logger.debug(f"Total exposure for {symbol}: {total_exposure}")
            return total_exposure

        except Exception as e:
            error_msg = f"Failed to calculate total exposure for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_positions_by_status(self, status: str) -> List[Position]:
        """
        Get positions by status.

        Args:
            status: Position status ('OPEN', 'CLOSED', 'PENDING', etc.)

        Returns:
            List of positions with the specified status
        """
        try:
            positions = await self.find_by(status=status)

            self.logger.debug(f"Found {len(positions)} positions with status: {status}")
            return positions

        except Exception as e:
            error_msg = f"Failed to get positions with status {status}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_positions_by_side(self, side: str, symbol: Optional[str] = None) -> List[Position]:
        """
        Get positions by side (LONG/SHORT).

        Args:
            side: Position side ('LONG' or 'SHORT')
            symbol: Optional symbol filter

        Returns:
            List of positions with the specified side
        """
        try:
            criteria = {'side': side}
            if symbol:
                criteria['symbol'] = symbol

            positions = await self.find_by(**criteria)

            self.logger.debug(f"Found {len(positions)} {side} positions" +
                            (f" for {symbol}" if symbol else ""))
            return positions

        except Exception as e:
            error_msg = f"Failed to get {side} positions: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_leveraged_positions(self, min_leverage: Decimal = Decimal('1.0')) -> List[Position]:
        """
        Get positions with leverage above threshold.

        Args:
            min_leverage: Minimum leverage threshold

        Returns:
            List of leveraged positions
        """
        try:
            all_positions = await self.list_all()

            # Filter by leverage (mock implementation for testing)
            leveraged_positions = [
                pos for pos in all_positions
                if pos.leverage and pos.leverage >= min_leverage
            ]

            self.logger.debug(f"Found {len(leveraged_positions)} positions with leverage >= {min_leverage}")
            return leveraged_positions

        except Exception as e:
            error_msg = f"Failed to get leveraged positions: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e


class TradeRepository(BaseRepository[Trade]):
    """Repository for Trade entities with trade analysis queries"""

    def __init__(self, session):
        """Initialize trade repository"""
        super().__init__(Trade, session)

    async def get_trades_by_date_range(self, start_date: datetime,
                                     end_date: datetime) -> List[Trade]:
        """
        Get trades within date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of trades within the date range
        """
        try:
            all_trades = await self.list_all()

            # Filter by date range (mock implementation for testing)
            filtered_trades = [
                trade for trade in all_trades
                if hasattr(trade, 'execution_time') and trade.execution_time and
                start_date <= trade.execution_time <= end_date
            ]

            self.logger.info(f"Found {len(filtered_trades)} trades between {start_date} and {end_date}")
            return filtered_trades

        except Exception as e:
            error_msg = f"Failed to get trades by date range: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_total_pnl_by_symbol(self, symbol: str) -> Decimal:
        """
        Calculate total realized PnL for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Total realized PnL as Decimal
        """
        try:
            symbol_trades = await self.find_by(symbol=symbol)

            total_pnl = Decimal('0.0')
            for trade in symbol_trades:
                if hasattr(trade, 'realized_pnl') and trade.realized_pnl:
                    total_pnl += trade.realized_pnl

            self.logger.debug(f"Total PnL for {symbol}: {total_pnl}")
            return total_pnl

        except Exception as e:
            error_msg = f"Failed to calculate total PnL for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_trades_by_side(self, side: str, symbol: Optional[str] = None) -> List[Trade]:
        """
        Get trades by side (BUY/SELL).

        Args:
            side: Trade side ('BUY' or 'SELL')
            symbol: Optional symbol filter

        Returns:
            List of trades with the specified side
        """
        try:
            criteria = {'side': side}
            if symbol:
                criteria['symbol'] = symbol

            trades = await self.find_by(**criteria)

            self.logger.debug(f"Found {len(trades)} {side} trades" +
                            (f" for {symbol}" if symbol else ""))
            return trades

        except Exception as e:
            error_msg = f"Failed to get {side} trades: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_total_fees_by_symbol(self, symbol: str) -> Decimal:
        """
        Calculate total fees paid for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Total fees as Decimal
        """
        try:
            symbol_trades = await self.find_by(symbol=symbol)

            total_fees = Decimal('0.0')
            for trade in symbol_trades:
                if hasattr(trade, 'fee') and trade.fee:
                    total_fees += trade.fee

            self.logger.debug(f"Total fees for {symbol}: {total_fees}")
            return total_fees

        except Exception as e:
            error_msg = f"Failed to calculate total fees for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e


class OrderRepository(BaseRepository[Order]):
    """Repository for Order entities with order management queries"""

    def __init__(self, session):
        """Initialize order repository"""
        super().__init__(Order, session)

    async def get_active_orders(self) -> List[Order]:
        """
        Get all active orders (NEW, PENDING, PARTIALLY_FILLED).

        Returns:
            List of active orders
        """
        try:
            all_orders = await self.list_all()

            # Filter for active statuses
            active_statuses = {'NEW', 'PENDING', 'PARTIALLY_FILLED'}
            active_orders = [
                order for order in all_orders
                if hasattr(order, 'status') and order.status in active_statuses
            ]

            self.logger.debug(f"Found {len(active_orders)} active orders")
            return active_orders

        except Exception as e:
            error_msg = f"Failed to get active orders: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_orders_by_status(self, status: str) -> List[Order]:
        """
        Get orders by status.

        Args:
            status: Order status

        Returns:
            List of orders with the specified status
        """
        try:
            orders = await self.find_by(status=status)

            self.logger.debug(f"Found {len(orders)} orders with status: {status}")
            return orders

        except Exception as e:
            error_msg = f"Failed to get orders with status {status}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_orders_by_type(self, order_type: str) -> List[Order]:
        """
        Get orders by type.

        Args:
            order_type: Order type ('MARKET', 'LIMIT', 'STOP', etc.)

        Returns:
            List of orders with the specified type
        """
        try:
            orders = await self.find_by(type=order_type)

            self.logger.debug(f"Found {len(orders)} {order_type} orders")
            return orders

        except Exception as e:
            error_msg = f"Failed to get {order_type} orders: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e


class MarketDataRepository(BaseRepository[MarketData]):
    """Repository for MarketData entities with market analysis queries"""

    def __init__(self, session):
        """Initialize market data repository"""
        super().__init__(MarketData, session)

    async def get_latest_price(self, symbol: str) -> Optional[MarketData]:
        """
        Get latest price data for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest market data entry or None
        """
        try:
            symbol_data = await self.find_by(symbol=symbol)

            if not symbol_data:
                self.logger.debug(f"No market data found for {symbol}")
                return None

            # Get latest entry (mock implementation)
            latest_data = max(symbol_data, key=lambda x: x.timestamp if hasattr(x, 'timestamp') and x.timestamp else datetime.min)

            self.logger.debug(f"Retrieved latest price for {symbol}: {getattr(latest_data, 'price', 'N/A')}")
            return latest_data

        except Exception as e:
            error_msg = f"Failed to get latest price for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_price_history(self, symbol: str, start_date: datetime,
                              end_date: datetime) -> List[MarketData]:
        """
        Get price history for symbol within date range.

        Args:
            symbol: Trading symbol
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of market data entries within the date range
        """
        try:
            symbol_data = await self.find_by(symbol=symbol)

            # Filter by date range (mock implementation for testing)
            filtered_data = [
                data for data in symbol_data
                if hasattr(data, 'timestamp') and data.timestamp and
                start_date <= data.timestamp <= end_date
            ]

            self.logger.debug(f"Retrieved {len(filtered_data)} price records for {symbol}")
            return filtered_data

        except Exception as e:
            error_msg = f"Failed to get price history for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_ohlc_data(self, symbol: str, timeframe: str = '1h') -> List[MarketData]:
        """
        Get OHLC data for symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe ('1m', '5m', '1h', '1d', etc.)

        Returns:
            List of OHLC market data
        """
        try:
            criteria = {'symbol': symbol}
            if hasattr(MarketData, 'timeframe'):
                criteria['timeframe'] = timeframe

            ohlc_data = await self.find_by(**criteria)

            self.logger.debug(f"Retrieved {len(ohlc_data)} OHLC records for {symbol} ({timeframe})")
            return ohlc_data

        except Exception as e:
            error_msg = f"Failed to get OHLC data for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e