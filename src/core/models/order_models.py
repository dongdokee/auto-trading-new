# src/core/models/order_models.py

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class OrderSide(str, Enum):
    """주문 방향"""
    BUY = "BUY"
    SELL = "SELL"


class OrderUrgency(str, Enum):
    """주문 긴급도"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    IMMEDIATE = "IMMEDIATE"


class OrderStatus(str, Enum):
    """주문 상태"""
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """주문 데이터 클래스"""
    symbol: str
    side: OrderSide
    size: Decimal
    urgency: OrderUrgency = OrderUrgency.MEDIUM
    price: Optional[Decimal] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """생성 후 검증"""
        # Symbol validation
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if not self.symbol.isupper():
            raise ValueError("Symbol must be uppercase")

        # Size validation
        if self.size <= 0:
            raise ValueError("Size must be positive")

        # Price validation
        if self.price is not None and self.price <= 0:
            raise ValueError("Price must be positive")

    @property
    def notional_value(self) -> Optional[Decimal]:
        """명목 가치 계산"""
        if self.price is None:
            return None
        return self.size * self.price