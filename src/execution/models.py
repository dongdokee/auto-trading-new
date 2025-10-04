# src/execution/models.py

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import Optional, List, Dict, Any

# Re-export core models to maintain backward compatibility
from src.core.models import Order, OrderSide, OrderUrgency, OrderStatus


@dataclass
class ExecutionResult:
    """주문 실행 결과 데이터 클래스"""
    order_id: str
    strategy: str
    total_filled: Decimal
    avg_price: Decimal
    total_cost: Decimal = Decimal("0")
    slices: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: datetime = field(default_factory=datetime.now)
    original_size: Optional[Decimal] = None

    def __post_init__(self):
        """생성 후 검증"""
        if self.total_filled < 0:
            raise ValueError("Total filled cannot be negative")

        if self.avg_price < 0:
            raise ValueError("Average price cannot be negative")

    @property
    def total_value(self) -> Decimal:
        """총 거래 가치"""
        return self.total_filled * self.avg_price

    @property
    def fill_rate(self) -> Optional[Decimal]:
        """체결률 계산"""
        if self.original_size is None or self.original_size == 0:
            return None
        return self.total_filled / self.original_size