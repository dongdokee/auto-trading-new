"""
Trading-related database schema models for the AutoTrading system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Literal
from enum import Enum

from .base import BaseModel, TimestampMixin, AuditMixin, SoftDeleteMixin


# Enums for type safety
class PositionSide(str, Enum):
    """Position side enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(str, Enum):
    """Position status enumeration"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"


class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradeType(str, Enum):
    """Trade type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    LIQUIDATION = "LIQUIDATION"


@dataclass
class Position:
    """
    Position model representing an open trading position.
    """
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    status: str
    leverage: Decimal = field(default=Decimal('1.0'))
    margin_usdt: Decimal = field(default=Decimal('0.0'))
    liquidation_price: Optional[Decimal] = field(default=None)
    unrealized_pnl: Decimal = field(default=Decimal('0.0'))
    realized_pnl: Decimal = field(default=Decimal('0.0'))
    stop_loss_price: Optional[Decimal] = field(default=None)
    take_profit_price: Optional[Decimal] = field(default=None)
    trailing_stop_distance: Optional[Decimal] = field(default=None)
    exchange: str = field(default='BINANCE')
    strategy_name: Optional[str] = field(default=None)
    notes: Optional[str] = field(default=None)

    def __post_init__(self):
        # Initialize mixins
        BaseModel.__init__(self)
        TimestampMixin.__init__(self)
        AuditMixin.__init__(self)
        self._validate_position()

    def _validate_position(self):
        """Validate position fields"""
        if self.side not in [PositionSide.LONG, PositionSide.SHORT]:
            raise ValueError(f"Invalid position side: {self.side}. Must be LONG or SHORT")

        if self.size <= 0:
            raise ValueError(f"Position size must be positive: {self.size}")

        if self.entry_price <= 0:
            raise ValueError(f"Entry price must be positive: {self.entry_price}")

        if self.status not in [status.value for status in PositionStatus]:
            raise ValueError(f"Invalid position status: {self.status}")


@dataclass
class Trade:
    """
    Trade model representing individual trade executions.
    """
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    position_id: Optional[int] = field(default=None)
    fee: Decimal = field(default=Decimal('0.0'))
    fee_currency: str = field(default='USDT')
    execution_time: datetime = field(default_factory=datetime.utcnow)
    trade_type: str = field(default='MARKET')
    exchange_trade_id: Optional[str] = field(default=None)
    exchange: str = field(default='BINANCE')
    is_maker: bool = field(default=False)
    commission_asset: Optional[str] = field(default=None)
    realized_pnl: Decimal = field(default=Decimal('0.0'))

    def __post_init__(self):
        # Initialize mixins
        BaseModel.__init__(self)
        TimestampMixin.__init__(self)
        self._validate_trade()

    def _validate_trade(self):
        """Validate trade fields"""
        if self.quantity <= 0:
            raise ValueError(f"Trade quantity must be positive: {self.quantity}")

        if self.price <= 0:
            raise ValueError(f"Trade price must be positive: {self.price}")


@dataclass
class Order:
    """
    Order model representing trading orders.
    """
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    price: Optional[Decimal] = field(default=None)
    status: str = field(default='PENDING')
    time_in_force: str = field(default='GTC')
    filled_quantity: Decimal = field(default=Decimal('0.0'))
    average_price: Optional[Decimal] = field(default=None)
    stop_price: Optional[Decimal] = field(default=None)
    exchange_order_id: Optional[str] = field(default=None)
    exchange: str = field(default='BINANCE')
    strategy_name: Optional[str] = field(default=None)
    parent_order_id: Optional[int] = field(default=None)
    reduce_only: bool = field(default=False)
    post_only: bool = field(default=False)

    _valid_status_transitions = {
        'PENDING': ['PARTIALLY_FILLED', 'FILLED', 'CANCELLED', 'REJECTED'],
        'PARTIALLY_FILLED': ['FILLED', 'CANCELLED'],
        'FILLED': [],
        'CANCELLED': [],
        'REJECTED': []
    }

    def __post_init__(self):
        # Initialize mixins
        BaseModel.__init__(self)
        TimestampMixin.__init__(self)
        AuditMixin.__init__(self)
        self._validate_order()

    def _validate_order(self):
        """Validate order fields"""
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive: {self.quantity}")

        if self.side not in [side.value for side in OrderSide]:
            raise ValueError(f"Invalid order side: {self.side}")

        if self.order_type not in [otype.value for otype in OrderType]:
            raise ValueError(f"Invalid order type: {self.order_type}")

    def update_status(self, new_status: str):
        """Update order status with validation"""
        if new_status not in self._valid_status_transitions.get(self.status, []):
            raise ValueError(f"Invalid status transition from {self.status} to {new_status}")

        self.status = new_status
        # Update timestamp manually since we're using composition not inheritance
        self.updated_at = datetime.utcnow()

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining unfilled quantity"""
        return self.quantity - self.filled_quantity


@dataclass
class MarketData:
    """
    Market data model for OHLCV and market microstructure data.
    """
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    quote_volume: Decimal = field(default=Decimal('0.0'))
    trades_count: int = field(default=0)
    timeframe: str = field(default='1m')
    taker_buy_base_volume: Optional[Decimal] = None
    taker_buy_quote_volume: Optional[Decimal] = None
    exchange: str = field(default='BINANCE')
    is_closed: bool = field(default=True)

    def __post_init__(self):
        # Initialize mixins
        BaseModel.__init__(self)
        TimestampMixin.__init__(self)
        self._validate_market_data()

    def _validate_market_data(self):
        """Validate market data fields"""
        if not (self.low_price <= self.open_price <= self.high_price and
                self.low_price <= self.close_price <= self.high_price):
            raise ValueError("Invalid OHLC price relationships")

        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")


@dataclass
class Portfolio:
    """
    Portfolio model tracking overall account equity and metrics.
    """
    total_equity_usdt: Decimal
    available_balance_usdt: Decimal
    used_margin_usdt: Decimal = field(default=Decimal('0.0'))
    unrealized_pnl_usdt: Decimal = field(default=Decimal('0.0'))
    realized_pnl_usdt: Decimal = field(default=Decimal('0.0'))
    total_positions: int = field(default=0)
    leverage_ratio: Decimal = field(default=Decimal('1.0'))
    maintenance_margin_usdt: Decimal = field(default=Decimal('0.0'))
    margin_ratio: Optional[Decimal] = None
    account_type: str = field(default='FUTURES')
    exchange: str = field(default='BINANCE')

    def __post_init__(self):
        # Initialize mixins
        BaseModel.__init__(self)
        TimestampMixin.__init__(self)
        AuditMixin.__init__(self)
        self._validate_portfolio()

    def _validate_portfolio(self):
        """Validate portfolio fields"""
        if self.total_equity_usdt < 0:
            raise ValueError("Total equity cannot be negative")

        if self.available_balance_usdt < 0:
            raise ValueError("Available balance cannot be negative")

    @property
    def equity_utilization_pct(self) -> Decimal:
        """Calculate equity utilization percentage"""
        if self.total_equity_usdt > 0:
            return (self.used_margin_usdt / self.total_equity_usdt) * 100
        return Decimal('0.0')


@dataclass
class RiskMetrics:
    """
    Risk metrics model for portfolio risk monitoring.
    """
    portfolio_id: int
    var_daily_usdt: Decimal
    var_utilization_pct: Decimal = field(default=Decimal('0.0'))
    current_drawdown_pct: Decimal = field(default=Decimal('0.0'))
    max_drawdown_pct: Decimal = field(default=Decimal('0.0'))
    high_water_mark: Decimal = field(default=Decimal('0.0'))
    consecutive_loss_days: int = field(default=0)
    consecutive_win_days: int = field(default=0)
    sharpe_ratio: Optional[Decimal] = None
    sortino_ratio: Optional[Decimal] = None
    calmar_ratio: Optional[Decimal] = None
    kelly_fraction: Optional[Decimal] = None
    beta_to_market: Optional[Decimal] = None
    tracking_error: Optional[Decimal] = None

    def __post_init__(self):
        # Initialize mixins
        BaseModel.__init__(self)
        TimestampMixin.__init__(self)
        self._validate_risk_metrics()

    def _validate_risk_metrics(self):
        """Validate risk metrics"""
        if self.var_daily_usdt < 0:
            raise ValueError("VaR cannot be negative")


@dataclass
class StrategyPerformance:
    """
    Strategy performance tracking model.
    """
    strategy_name: str
    total_trades: int = field(default=0)
    winning_trades: int = field(default=0)
    losing_trades: int = field(default=0)
    win_rate_pct: Decimal = field(default=Decimal('0.0'))
    avg_win_usdt: Decimal = field(default=Decimal('0.0'))
    avg_loss_usdt: Decimal = field(default=Decimal('0.0'))
    profit_factor: Decimal = field(default=Decimal('0.0'))
    max_consecutive_wins: int = field(default=0)
    max_consecutive_losses: int = field(default=0)
    total_pnl_usdt: Decimal = field(default=Decimal('0.0'))
    sharpe_ratio: Optional[Decimal] = None
    max_drawdown_pct: Optional[Decimal] = None
    calmar_ratio: Optional[Decimal] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    def __post_init__(self):
        # Initialize mixins
        BaseModel.__init__(self)
        TimestampMixin.__init__(self)
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate derived metrics only if they're at default values"""
        # Only calculate win rate if it's at default value
        if self.total_trades > 0 and self.win_rate_pct == Decimal('0.0'):
            win_rate = (Decimal(str(self.winning_trades)) /
                       Decimal(str(self.total_trades))) * 100
            self.win_rate_pct = win_rate.quantize(Decimal('0.01'))

        # Only calculate profit factor if it's at default value
        if (self.losing_trades > 0 and self.avg_loss_usdt != 0 and
            self.profit_factor == Decimal('0.0')):
            total_wins = self.winning_trades * self.avg_win_usdt
            total_losses = abs(self.losing_trades * self.avg_loss_usdt)
            if total_losses > 0:
                profit_factor = total_wins / total_losses
                self.profit_factor = profit_factor.quantize(Decimal('0.01'))