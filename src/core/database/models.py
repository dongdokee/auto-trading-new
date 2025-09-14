"""
SQLAlchemy models for database migrations.
These models correspond to the dataclass schemas but are designed for Alembic migrations.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text, ForeignKey,
    Enum as SQLEnum, Index, UniqueConstraint
)
from sqlalchemy.types import DECIMAL
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()


# Enums for database constraints
class PositionStatus(enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"


class PositionSide(enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(enum.Enum):
    NEW = "NEW"
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderType(enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradeSide(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


# Base mixin for common fields
class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class AuditMixin:
    created_by = Column(String(100), nullable=True)
    updated_by = Column(String(100), nullable=True)
    version = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)


# Trading Models
class Position(Base, TimestampMixin, AuditMixin):
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(SQLEnum(PositionSide), nullable=False)
    size = Column(DECIMAL(precision=20, scale=8), nullable=False)
    entry_price = Column(DECIMAL(precision=20, scale=8), nullable=False)
    exit_price = Column(DECIMAL(precision=20, scale=8), nullable=True)
    leverage = Column(DECIMAL(precision=10, scale=2), nullable=False)
    status = Column(SQLEnum(PositionStatus), nullable=False)
    unrealized_pnl = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    realized_pnl = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    margin_used = Column(DECIMAL(precision=20, scale=8), nullable=True)
    liquidation_price = Column(DECIMAL(precision=20, scale=8), nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    trades = relationship("Trade", back_populates="position")
    orders = relationship("Order", back_populates="position")

    # Indexes for performance
    __table_args__ = (
        Index('idx_positions_symbol_status', 'symbol', 'status'),
        Index('idx_positions_status_created', 'status', 'created_at'),
        Index('idx_positions_uuid', 'uuid'),
    )


class Trade(Base, TimestampMixin, AuditMixin):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False)
    position_id = Column(Integer, ForeignKey('positions.id'), nullable=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=True)
    symbol = Column(String(20), nullable=False)
    side = Column(SQLEnum(TradeSide), nullable=False)
    quantity = Column(DECIMAL(precision=20, scale=8), nullable=False)
    price = Column(DECIMAL(precision=20, scale=8), nullable=False)
    fee = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    fee_currency = Column(String(10), nullable=True)
    realized_pnl = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    execution_time = Column(DateTime(timezone=True), nullable=True)
    exchange_trade_id = Column(String(100), nullable=True)
    exchange_order_id = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    position = relationship("Position", back_populates="trades")
    order = relationship("Order", back_populates="trades")

    # Indexes
    __table_args__ = (
        Index('idx_trades_symbol_execution_time', 'symbol', 'execution_time'),
        Index('idx_trades_position_id', 'position_id'),
        Index('idx_trades_exchange_trade_id', 'exchange_trade_id'),
        UniqueConstraint('exchange_trade_id', name='uq_trades_exchange_trade_id'),
    )


class Order(Base, TimestampMixin, AuditMixin):
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False)
    position_id = Column(Integer, ForeignKey('positions.id'), nullable=True)
    symbol = Column(String(20), nullable=False)
    side = Column(SQLEnum(OrderSide), nullable=False)
    type = Column(SQLEnum(OrderType), nullable=False)
    status = Column(SQLEnum(OrderStatus), nullable=False)
    quantity = Column(DECIMAL(precision=20, scale=8), nullable=False)
    filled_quantity = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    price = Column(DECIMAL(precision=20, scale=8), nullable=True)  # Null for market orders
    stop_price = Column(DECIMAL(precision=20, scale=8), nullable=True)
    time_in_force = Column(String(10), default='GTC', nullable=False)  # GTC, IOC, FOK
    exchange_order_id = Column(String(100), nullable=True)
    client_order_id = Column(String(100), nullable=True)
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    position = relationship("Position", back_populates="orders")
    trades = relationship("Trade", back_populates="order")

    # Indexes
    __table_args__ = (
        Index('idx_orders_status_symbol', 'status', 'symbol'),
        Index('idx_orders_exchange_order_id', 'exchange_order_id'),
        Index('idx_orders_client_order_id', 'client_order_id'),
        UniqueConstraint('exchange_order_id', name='uq_orders_exchange_order_id'),
    )


class MarketData(Base, TimestampMixin):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 1h, 1d, etc.
    open_price = Column(DECIMAL(precision=20, scale=8), nullable=False)
    high_price = Column(DECIMAL(precision=20, scale=8), nullable=False)
    low_price = Column(DECIMAL(precision=20, scale=8), nullable=False)
    close_price = Column(DECIMAL(precision=20, scale=8), nullable=False)
    volume = Column(DECIMAL(precision=20, scale=8), nullable=False)
    quote_volume = Column(DECIMAL(precision=20, scale=8), nullable=True)
    number_of_trades = Column(Integer, nullable=True)
    taker_buy_base_volume = Column(DECIMAL(precision=20, scale=8), nullable=True)
    taker_buy_quote_volume = Column(DECIMAL(precision=20, scale=8), nullable=True)

    # Technical indicators (optional, can be calculated on-demand)
    sma_20 = Column(DECIMAL(precision=20, scale=8), nullable=True)
    sma_50 = Column(DECIMAL(precision=20, scale=8), nullable=True)
    ema_12 = Column(DECIMAL(precision=20, scale=8), nullable=True)
    ema_26 = Column(DECIMAL(precision=20, scale=8), nullable=True)
    rsi_14 = Column(DECIMAL(precision=5, scale=2), nullable=True)
    atr_14 = Column(DECIMAL(precision=20, scale=8), nullable=True)

    # Indexes for time-series queries
    __table_args__ = (
        Index('idx_market_data_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        Index('idx_market_data_timestamp', 'timestamp'),
        UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uq_market_data_symbol_timeframe_timestamp'),
    )


# Portfolio and Risk Models
class Portfolio(Base, TimestampMixin, AuditMixin):
    __tablename__ = 'portfolios'

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    initial_capital = Column(DECIMAL(precision=20, scale=8), nullable=False)
    current_capital = Column(DECIMAL(precision=20, scale=8), nullable=False)
    available_balance = Column(DECIMAL(precision=20, scale=8), nullable=False)
    used_margin = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    unrealized_pnl = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    realized_pnl = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    high_water_mark = Column(DECIMAL(precision=20, scale=8), nullable=False)
    max_drawdown = Column(DECIMAL(precision=5, scale=4), default=0, nullable=False)
    is_paper_trading = Column(Boolean, default=True, nullable=False)

    # Risk metrics
    current_var = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    max_var_limit = Column(DECIMAL(precision=20, scale=8), nullable=True)
    max_leverage = Column(DECIMAL(precision=10, scale=2), default=1, nullable=False)

    # Relationships
    risk_metrics = relationship("RiskMetrics", back_populates="portfolio")
    strategy_performances = relationship("StrategyPerformance", back_populates="portfolio")


class RiskMetrics(Base, TimestampMixin):
    __tablename__ = 'risk_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # Risk measurements
    portfolio_value = Column(DECIMAL(precision=20, scale=8), nullable=False)
    var_1d = Column(DECIMAL(precision=20, scale=8), nullable=True)  # 1-day Value at Risk
    var_1w = Column(DECIMAL(precision=20, scale=8), nullable=True)  # 1-week Value at Risk
    expected_shortfall = Column(DECIMAL(precision=20, scale=8), nullable=True)
    max_drawdown = Column(DECIMAL(precision=5, scale=4), nullable=True)
    current_drawdown = Column(DECIMAL(precision=5, scale=4), default=0, nullable=False)
    volatility = Column(DECIMAL(precision=5, scale=4), nullable=True)
    sharpe_ratio = Column(DECIMAL(precision=8, scale=4), nullable=True)
    sortino_ratio = Column(DECIMAL(precision=8, scale=4), nullable=True)

    # Position concentration
    largest_position_pct = Column(DECIMAL(precision=5, scale=4), nullable=True)
    total_leverage = Column(DECIMAL(precision=10, scale=2), nullable=True)
    correlation_risk = Column(DECIMAL(precision=5, scale=4), nullable=True)

    # Relationship
    portfolio = relationship("Portfolio", back_populates="risk_metrics")

    # Indexes
    __table_args__ = (
        Index('idx_risk_metrics_portfolio_timestamp', 'portfolio_id', 'timestamp'),
    )


class StrategyPerformance(Base, TimestampMixin):
    __tablename__ = 'strategy_performances'

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # Performance metrics
    allocated_capital = Column(DECIMAL(precision=20, scale=8), nullable=False)
    current_value = Column(DECIMAL(precision=20, scale=8), nullable=False)
    unrealized_pnl = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    realized_pnl = Column(DECIMAL(precision=20, scale=8), default=0, nullable=False)
    total_return = Column(DECIMAL(precision=8, scale=6), nullable=True)  # As percentage
    sharpe_ratio = Column(DECIMAL(precision=8, scale=4), nullable=True)
    max_drawdown = Column(DECIMAL(precision=5, scale=4), nullable=True)
    win_rate = Column(DECIMAL(precision=5, scale=4), nullable=True)
    profit_factor = Column(DECIMAL(precision=8, scale=4), nullable=True)

    # Trade statistics
    total_trades = Column(Integer, default=0, nullable=False)
    winning_trades = Column(Integer, default=0, nullable=False)
    losing_trades = Column(Integer, default=0, nullable=False)
    largest_win = Column(DECIMAL(precision=20, scale=8), nullable=True)
    largest_loss = Column(DECIMAL(precision=20, scale=8), nullable=True)

    # Relationship
    portfolio = relationship("Portfolio", back_populates="strategy_performances")

    # Indexes
    __table_args__ = (
        Index('idx_strategy_performance_portfolio_strategy', 'portfolio_id', 'strategy_name'),
        Index('idx_strategy_performance_timestamp', 'timestamp'),
    )