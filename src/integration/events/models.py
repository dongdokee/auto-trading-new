# src/integration/events/models.py
"""
Event Models

Defines all event types used in the system integration layer.
Provides a type-safe, structured approach to inter-component communication.
"""

from abc import ABC
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid


class EventPriority(Enum):
    """Event priority levels for queue ordering"""
    CRITICAL = 1    # System failures, risk breaches
    HIGH = 2        # Order execution, stop losses
    NORMAL = 3      # Strategy signals, market data
    LOW = 4         # Analytics, reporting


class EventType(Enum):
    """Event type classification"""
    MARKET_DATA = "market_data"
    STRATEGY_SIGNAL = "strategy_signal"
    PORTFOLIO = "portfolio"
    ORDER = "order"
    EXECUTION = "execution"
    RISK = "risk"
    SYSTEM = "system"


@dataclass
class BaseEvent(ABC):
    """Base event class for all system events"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: EventType = field(init=False)
    priority: EventPriority = EventPriority.NORMAL
    source_component: str = ""
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate event after initialization"""
        if not self.source_component:
            raise ValueError("source_component is required")


@dataclass
class MarketDataEvent(BaseEvent):
    """Market data update event"""

    event_type: EventType = field(default=EventType.MARKET_DATA, init=False)

    symbol: str = ""
    price: Decimal = Decimal('0')
    volume: Decimal = Decimal('0')
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    orderbook: Optional[Dict[str, Any]] = None
    trade_data: Optional[Dict[str, Any]] = None
    mark_price: Optional[Decimal] = None
    funding_rate: Optional[Decimal] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.price <= 0:
            raise ValueError("price must be positive")


@dataclass
class StrategySignalEvent(BaseEvent):
    """Strategy signal generation event"""

    event_type: EventType = field(default=EventType.STRATEGY_SIGNAL, init=False)

    strategy_name: str = ""
    symbol: str = ""
    action: str = ""  # BUY, SELL, HOLD, CLOSE
    strength: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0  # 0.0 to 1.0
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    regime_info: Optional[Dict[str, Any]] = None
    strategy_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.strategy_name:
            raise ValueError("strategy_name is required")
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.action not in ['BUY', 'SELL', 'HOLD', 'CLOSE']:
            raise ValueError("action must be BUY, SELL, HOLD, or CLOSE")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class PortfolioEvent(BaseEvent):
    """Portfolio optimization and rebalancing event"""

    event_type: EventType = field(default=EventType.PORTFOLIO, init=False)

    action: str = ""  # OPTIMIZE, REBALANCE, UPDATE_ALLOCATION
    current_weights: Dict[str, float] = field(default_factory=dict)
    target_weights: Dict[str, float] = field(default_factory=dict)
    rebalance_orders: List[Dict[str, Any]] = field(default_factory=list)
    optimization_result: Optional[Dict[str, Any]] = None
    performance_attribution: Optional[Dict[str, Any]] = None
    correlation_matrix: Optional[Dict[str, Any]] = None
    transaction_costs: Optional[Dict[str, Decimal]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.action not in ['OPTIMIZE', 'REBALANCE', 'UPDATE_ALLOCATION']:
            raise ValueError("action must be OPTIMIZE, REBALANCE, or UPDATE_ALLOCATION")


@dataclass
class OrderEvent(BaseEvent):
    """Order creation and management event"""

    event_type: EventType = field(default=EventType.ORDER, init=False)
    priority: EventPriority = EventPriority.HIGH

    action: str = ""  # CREATE, CANCEL, MODIFY, STATUS_UPDATE
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""  # BUY, SELL
    size: Decimal = Decimal('0')
    order_type: str = "MARKET"  # MARKET, LIMIT, STOP_MARKET, STOP_LIMIT
    price: Optional[Decimal] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    urgency: str = "MEDIUM"  # IMMEDIATE, HIGH, MEDIUM, LOW
    execution_strategy: str = "AGGRESSIVE"  # AGGRESSIVE, PASSIVE, TWAP, ADAPTIVE
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    source_signal: Optional[str] = None  # Reference to strategy signal

    def __post_init__(self):
        super().__post_init__()
        if self.action not in ['CREATE', 'CANCEL', 'MODIFY', 'STATUS_UPDATE']:
            raise ValueError("action must be CREATE, CANCEL, MODIFY, or STATUS_UPDATE")
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.action == 'CREATE' and self.size <= 0:
            raise ValueError("size must be positive for CREATE action")


@dataclass
class ExecutionEvent(BaseEvent):
    """Order execution result event"""

    event_type: EventType = field(default=EventType.EXECUTION, init=False)
    priority: EventPriority = EventPriority.HIGH

    order_id: str = ""
    symbol: str = ""
    side: str = ""
    executed_qty: Decimal = Decimal('0')
    avg_price: Decimal = Decimal('0')
    commission: Decimal = Decimal('0')
    status: str = ""  # FILLED, PARTIALLY_FILLED, CANCELLED, REJECTED
    exchange_order_id: Optional[str] = None
    execution_time: Optional[datetime] = None
    slippage_bps: Optional[float] = None
    execution_cost: Optional[Decimal] = None
    market_impact: Optional[float] = None
    execution_strategy: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.order_id:
            raise ValueError("order_id is required")
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.executed_qty < 0:
            raise ValueError("executed_qty cannot be negative")


@dataclass
class RiskEvent(BaseEvent):
    """Risk monitoring and control event"""

    event_type: EventType = field(default=EventType.RISK, init=False)

    risk_type: str = ""  # VAR_BREACH, DRAWDOWN_BREACH, LEVERAGE_BREACH, POSITION_LIMIT
    severity: str = "INFO"  # INFO, WARNING, CRITICAL
    symbol: Optional[str] = None
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    recommended_action: Optional[str] = None
    position_impact: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.risk_type:
            raise ValueError("risk_type is required")
        if self.severity not in ['INFO', 'WARNING', 'CRITICAL']:
            raise ValueError("severity must be INFO, WARNING, or CRITICAL")
        if self.severity == 'CRITICAL':
            self.priority = EventPriority.CRITICAL


@dataclass
class SystemEvent(BaseEvent):
    """System status and control event"""

    event_type: EventType = field(default=EventType.SYSTEM, init=False)

    system_action: str = ""  # START, STOP, PAUSE, RESUME, HEALTH_CHECK, ERROR
    component: Optional[str] = None
    status: str = "RUNNING"  # RUNNING, STOPPED, PAUSED, ERROR, INITIALIZING
    message: str = ""
    error_details: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.system_action:
            raise ValueError("system_action is required")
        if self.system_action == 'ERROR':
            self.priority = EventPriority.CRITICAL
            self.severity = 'CRITICAL'


# Type aliases for convenience
EventUnion = Union[
    MarketDataEvent, StrategySignalEvent, PortfolioEvent,
    OrderEvent, ExecutionEvent, RiskEvent, SystemEvent
]

EventDict = Dict[str, Any]