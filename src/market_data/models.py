# src/market_data/models.py

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime
from enum import Enum


class TickType(Enum):
    TRADE = "TRADE"
    QUOTE = "QUOTE"
    ORDER = "ORDER"
    CANCEL = "CANCEL"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class BookShape(Enum):
    FLAT = "FLAT"
    BID_HEAVY = "BID_HEAVY"
    ASK_HEAVY = "ASK_HEAVY"


@dataclass
class OrderLevel:
    """Single order book level"""
    price: Decimal
    size: Decimal


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    symbol: str
    timestamp: datetime
    event_time: int
    first_update_id: int
    final_update_id: int
    bids: List[OrderLevel]
    asks: List[OrderLevel]

    def __post_init__(self):
        # Convert to OrderLevel objects if they aren't already
        if self.bids and not isinstance(self.bids[0], OrderLevel):
            self.bids = [OrderLevel(Decimal(str(price)), Decimal(str(size)))
                        for price, size in self.bids]
        if self.asks and not isinstance(self.asks[0], OrderLevel):
            self.asks = [OrderLevel(Decimal(str(price)), Decimal(str(size)))
                        for price, size in self.asks]


@dataclass
class TickData:
    """Individual tick data point"""
    symbol: str
    timestamp: datetime
    tick_type: TickType
    price: Optional[Decimal] = None
    size: Optional[Decimal] = None
    side: Optional[OrderSide] = None
    trade_id: Optional[int] = None
    order_id: Optional[str] = None
    event_time: Optional[int] = None
    is_buyer_maker: Optional[bool] = None


@dataclass
class MarketMetrics:
    """Comprehensive market condition metrics"""
    symbol: str
    timestamp: datetime

    # Spread metrics
    best_bid: Decimal
    best_ask: Decimal
    mid_price: Decimal
    spread: Decimal
    spread_bps: float

    # Liquidity metrics
    bid_volume_5: Decimal
    ask_volume_5: Decimal
    top_5_liquidity: Decimal
    imbalance: float
    liquidity_score: float

    # Price impact
    price_impact_function: Optional[Any] = None
    effective_spread: Optional[float] = None

    # Book shape
    book_shape: BookShape = BookShape.FLAT
    bid_slope: Optional[float] = None
    ask_slope: Optional[float] = None

    # Large orders
    large_orders: List[Dict] = None

    def __post_init__(self):
        if self.large_orders is None:
            self.large_orders = []


@dataclass
class LiquidityProfile:
    """Time-based liquidity profile"""
    symbol: str
    hour: int
    day_of_week: int
    expected_spread: float
    expected_depth: Decimal
    depth_std: float
    confidence: float
    sample_size: int
    historical_data: List[Dict] = None

    def __post_init__(self):
        if self.historical_data is None:
            self.historical_data = []


@dataclass
class MarketImpactEstimate:
    """Market impact estimation result"""
    symbol: str
    order_size: Decimal
    side: OrderSide
    temporary_impact: float
    permanent_impact: float
    total_impact: float

    # Impact breakdown
    spread_component: float
    size_component: float
    volatility_component: float
    timing_component: float
    permanent_drift: float

    # Model metadata
    model_confidence: float
    calibration_time: datetime
    features_used: List[str] = None

    def __post_init__(self):
        if self.features_used is None:
            self.features_used = []


@dataclass
class MicrostructurePatterns:
    """Detected microstructure patterns"""
    symbol: str
    timestamp: datetime

    # Pattern flags
    quote_stuffing: bool = False
    layering: bool = False
    momentum_ignition: bool = False
    ping_pong: bool = False

    # Pattern metrics
    quote_rate: float = 0.0
    order_cancel_ratio: float = 0.0
    price_momentum: float = 0.0
    trade_flow_imbalance: float = 0.0
    vpin_score: float = 0.5

    # Additional context
    pattern_confidence: float = 0.0
    alert_level: str = "NONE"  # NONE, LOW, MEDIUM, HIGH
    description: str = ""


@dataclass
class ExecutionWindow:
    """Optimal execution time window"""
    hour: int
    avg_spread: float
    avg_depth: Decimal
    estimated_cost: float
    samples: int
    confidence: float = 0.0

    def cost_score(self) -> float:
        """Combined cost score considering spread and liquidity"""
        liquidity_penalty = 1.0 / max(float(self.avg_depth), 1e-10)
        return self.avg_spread + liquidity_penalty * 1000


@dataclass
class AggregatedMarketData:
    """Aggregated market data for analysis"""
    symbol: str
    timestamp: datetime

    # Latest metrics
    current_metrics: MarketMetrics
    liquidity_profile: LiquidityProfile
    impact_estimate: Optional[MarketImpactEstimate] = None
    patterns: Optional[MicrostructurePatterns] = None

    # Historical data
    orderbook_history: List[OrderBookSnapshot] = None
    tick_history: List[TickData] = None
    metrics_history: List[MarketMetrics] = None

    # Cache metadata
    cache_timestamp: datetime = None
    cache_ttl: int = 60  # seconds

    def __post_init__(self):
        if self.orderbook_history is None:
            self.orderbook_history = []
        if self.tick_history is None:
            self.tick_history = []
        if self.metrics_history is None:
            self.metrics_history = []
        if self.cache_timestamp is None:
            self.cache_timestamp = datetime.utcnow()

    def is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if self.cache_timestamp is None:
            return False

        age = (datetime.utcnow() - self.cache_timestamp).total_seconds()
        return age < self.cache_ttl


# Type aliases for better readability
PriceLevel = Tuple[Decimal, Decimal]  # (price, size)
ImpactFunction = Any  # Callable[[Decimal, str], float]
CallbackFunction = Any  # Callable[[Any], None]