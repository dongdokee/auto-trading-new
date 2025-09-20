# src/integration/state/positions.py
"""
Position Tracker

Real-time position tracking and P&L calculation for the trading system.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
from dataclasses import dataclass, asdict


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: str  # LONG, SHORT
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    leverage: Decimal
    margin: Decimal
    liquidation_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    entry_time: datetime
    last_update: datetime
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None


class PositionTracker:
    """
    Real-time position tracking and management

    Features:
    - Position lifecycle tracking
    - Real-time P&L calculation
    - Risk metrics calculation
    - Position aggregation
    """

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.logger = logging.getLogger("position_tracker")

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = Decimal('0')
        self.largest_win = Decimal('0')
        self.largest_loss = Decimal('0')

    async def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update position data"""
        try:
            # Convert string values to appropriate types
            processed_data = self._process_position_data(position_data)

            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]

                # Update current price and recalculate PnL
                if 'current_price' in processed_data:
                    position.current_price = processed_data['current_price']
                    position.unrealized_pnl = self._calculate_unrealized_pnl(position)

                # Update other fields
                for field, value in processed_data.items():
                    if hasattr(position, field):
                        setattr(position, field, value)

                position.last_update = datetime.now()

            else:
                # Create new position
                position = Position(
                    symbol=symbol,
                    side=processed_data.get('side', 'LONG'),
                    size=processed_data.get('size', Decimal('0')),
                    entry_price=processed_data.get('entry_price', Decimal('0')),
                    current_price=processed_data.get('current_price', processed_data.get('entry_price', Decimal('0'))),
                    leverage=processed_data.get('leverage', Decimal('1')),
                    margin=processed_data.get('margin', Decimal('0')),
                    liquidation_price=processed_data.get('liquidation_price', Decimal('0')),
                    unrealized_pnl=processed_data.get('unrealized_pnl', Decimal('0')),
                    realized_pnl=processed_data.get('realized_pnl', Decimal('0')),
                    entry_time=processed_data.get('entry_time', datetime.now()),
                    last_update=datetime.now(),
                    stop_loss=processed_data.get('stop_loss'),
                    take_profit=processed_data.get('take_profit')
                )

                # Calculate unrealized PnL
                position.unrealized_pnl = self._calculate_unrealized_pnl(position)

                self.positions[symbol] = position

            self.logger.debug(f"Updated position for {symbol}")

        except Exception as e:
            self.logger.error(f"Error updating position for {symbol}: {e}")

    async def close_position(self, symbol: str, close_price: Decimal, reason: str = "MANUAL"):
        """Close a position"""
        if symbol not in self.positions:
            self.logger.warning(f"Cannot close position for {symbol}: Position not found")
            return

        position = self.positions[symbol]

        # Calculate final PnL
        position.current_price = close_price
        final_pnl = self._calculate_unrealized_pnl(position)
        position.realized_pnl = final_pnl
        position.unrealized_pnl = Decimal('0')

        # Update performance metrics
        self.total_trades += 1
        self.total_pnl += final_pnl

        if final_pnl > 0:
            self.winning_trades += 1
            if final_pnl > self.largest_win:
                self.largest_win = final_pnl
        else:
            self.losing_trades += 1
            if final_pnl < self.largest_loss:
                self.largest_loss = final_pnl

        # Move to closed positions
        self.closed_positions.append(position)

        # Remove from active positions
        del self.positions[symbol]

        self.logger.info(f"Closed position for {symbol}, PnL: {final_pnl}, Reason: {reason}")

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get specific position"""
        if symbol in self.positions:
            return self._position_to_dict(self.positions[symbol])
        return None

    async def get_all_positions(self) -> Dict[str, Any]:
        """Get all active positions"""
        positions_dict = {}
        for symbol, position in self.positions.items():
            positions_dict[symbol] = self._position_to_dict(position)
        return positions_dict

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary metrics"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_margin_used = sum(pos.margin for pos in self.positions.values())

        position_count = len(self.positions)
        long_positions = sum(1 for pos in self.positions.values() if pos.side == 'LONG')
        short_positions = position_count - long_positions

        return {
            'total_positions': position_count,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'total_unrealized_pnl': str(total_unrealized_pnl),
            'total_margin_used': str(total_margin_used),
            'total_realized_pnl': str(self.total_pnl)
        }

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk metrics from positions"""
        if not self.positions:
            return {
                'total_exposure': '0',
                'max_position_risk': '0',
                'portfolio_leverage': '1',
                'concentration_risk': '0'
            }

        total_notional = Decimal('0')
        max_position_notional = Decimal('0')
        total_margin = Decimal('0')

        for position in self.positions.values():
            position_notional = position.size * position.current_price / position.leverage
            total_notional += position_notional
            total_margin += position.margin

            if position_notional > max_position_notional:
                max_position_notional = position_notional

        # Calculate concentration risk (largest position / total notional)
        concentration_risk = float(max_position_notional / total_notional) if total_notional > 0 else 0.0

        # Calculate portfolio leverage (total notional / total margin)
        portfolio_leverage = float(total_notional / total_margin) if total_margin > 0 else 1.0

        return {
            'total_exposure': str(total_notional),
            'max_position_risk': str(max_position_notional),
            'portfolio_leverage': str(portfolio_leverage),
            'concentration_risk': str(concentration_risk)
        }

    async def update_market_prices(self, price_updates: Dict[str, Decimal]):
        """Update market prices for all positions"""
        for symbol, new_price in price_updates.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = new_price
                position.unrealized_pnl = self._calculate_unrealized_pnl(position)
                position.last_update = datetime.now()

        self.logger.debug(f"Updated market prices for {len(price_updates)} symbols")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get trading performance metrics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

        avg_win = (self.largest_win / self.winning_trades) if self.winning_trades > 0 else Decimal('0')
        avg_loss = (self.largest_loss / self.losing_trades) if self.losing_trades > 0 else Decimal('0')

        profit_factor = float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0

        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': win_rate,
            'total_pnl': str(self.total_pnl),
            'largest_win': str(self.largest_win),
            'largest_loss': str(self.largest_loss),
            'profit_factor': profit_factor,
            'avg_win': str(avg_win),
            'avg_loss': str(avg_loss)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get position tracker metrics"""
        return {
            'active_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'total_trades': self.total_trades,
            'total_pnl': str(self.total_pnl)
        }

    def _calculate_unrealized_pnl(self, position: Position) -> Decimal:
        """Calculate unrealized PnL for a position"""
        if position.side == 'LONG':
            pnl = (position.current_price - position.entry_price) * position.size
        else:  # SHORT
            pnl = (position.entry_price - position.current_price) * position.size

        return pnl

    def _process_position_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and convert position data types"""
        processed = {}

        for key, value in data.items():
            if key in ['size', 'entry_price', 'current_price', 'leverage', 'margin',
                      'liquidation_price', 'unrealized_pnl', 'realized_pnl',
                      'stop_loss', 'take_profit']:
                if value is not None:
                    processed[key] = Decimal(str(value))
            elif key in ['entry_time', 'last_update']:
                if isinstance(value, str):
                    processed[key] = datetime.fromisoformat(value)
                else:
                    processed[key] = value
            else:
                processed[key] = value

        return processed

    def _position_to_dict(self, position: Position) -> Dict[str, Any]:
        """Convert position to dictionary"""
        pos_dict = asdict(position)

        # Convert Decimal and datetime objects to strings
        for key, value in pos_dict.items():
            if isinstance(value, Decimal):
                pos_dict[key] = str(value)
            elif isinstance(value, datetime):
                pos_dict[key] = value.isoformat()

        return pos_dict