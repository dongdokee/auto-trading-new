# src/execution/slippage_controller.py
import asyncio
import math
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict
from src.execution.models import Order, OrderSide


@dataclass
class SlippageMetrics:
    """Slippage measurement data"""
    order_id: str
    symbol: str
    slippage_bps: Decimal
    cost_impact: Decimal
    timestamp: datetime
    slippage_type: Optional[str] = None


@dataclass
class SlippageAlert:
    """Slippage alert information"""
    symbol: str
    slippage_bps: Decimal
    severity: str
    timestamp: datetime
    order_id: str
    message: str


class SlippageController:
    """Real-time slippage monitoring and control"""

    def __init__(self):
        self.max_slippage_bps: int = 50  # 0.5% default limit
        self.alert_threshold_bps: int = 25  # 0.25% alert threshold
        self.measurement_window: int = 300  # 5 minutes

        self.slippage_history: List[SlippageMetrics] = []
        self.active_alerts: List[SlippageAlert] = []

        self.is_monitoring: bool = False
        self.monitoring_callback: Optional[Callable] = None
        self._lock = asyncio.Lock()

    def calculate_slippage(self, benchmark_price: Decimal, execution_price: Decimal, side: OrderSide) -> Decimal:
        """Calculate slippage in basis points"""
        if side == OrderSide.BUY:
            # For buy orders: positive slippage = paying more than benchmark
            slippage = (execution_price - benchmark_price) / benchmark_price * 10000
        else:
            # For sell orders: positive slippage = receiving less than benchmark
            slippage = (benchmark_price - execution_price) / benchmark_price * 10000

        return slippage

    async def record_slippage(self, order: Order, benchmark_price: Decimal,
                            execution_price: Decimal, filled_qty: Decimal):
        """Record slippage measurement"""
        async with self._lock:
            # Calculate slippage
            slippage_bps = self.calculate_slippage(benchmark_price, execution_price, order.side)

            # Calculate cost impact
            cost_impact = filled_qty * abs(execution_price - benchmark_price)

            # Create metrics
            metrics = SlippageMetrics(
                order_id=order.order_id,
                symbol=order.symbol,
                slippage_bps=slippage_bps,
                cost_impact=cost_impact,
                timestamp=datetime.now()
            )

            # Add to history
            self.slippage_history.append(metrics)

            # Check for alerts
            await self._check_slippage_alerts(metrics, order)

            # Notify monitoring callback if active
            if self.is_monitoring and self.monitoring_callback:
                await self.monitoring_callback(metrics)

    async def _check_slippage_alerts(self, metrics: SlippageMetrics, order: Order):
        """Check if slippage exceeds alert thresholds"""
        if metrics.slippage_bps <= self.alert_threshold_bps:
            return

        # Determine severity
        if metrics.slippage_bps >= 1000:  # 10%
            severity = "CRITICAL"
        elif metrics.slippage_bps >= 25:  # 0.25%
            severity = "HIGH"
        else:
            severity = "MEDIUM"

        # Create alert
        alert = SlippageAlert(
            symbol=metrics.symbol,
            slippage_bps=metrics.slippage_bps,
            severity=severity,
            timestamp=metrics.timestamp,
            order_id=metrics.order_id,
            message=f"High slippage detected: {metrics.slippage_bps}bps on {metrics.symbol}"
        )

        self.active_alerts.append(alert)

    async def check_slippage_limit(self, order: Order, benchmark_price: Decimal,
                                 proposed_price: Decimal) -> bool:
        """Check if proposed execution would violate slippage limits"""
        slippage_bps = self.calculate_slippage(benchmark_price, proposed_price, order.side)
        return slippage_bps <= self.max_slippage_bps

    def get_slippage_statistics(self) -> Dict[str, Any]:
        """Get aggregate slippage statistics"""
        if not self.slippage_history:
            return {
                'total_orders': 0,
                'avg_slippage_bps': Decimal('0'),
                'max_slippage_bps': Decimal('0'),
                'total_cost_impact': Decimal('0'),
                'symbols': {}
            }

        total_orders = len(self.slippage_history)
        avg_slippage = sum(m.slippage_bps for m in self.slippage_history) / total_orders
        max_slippage = max(m.slippage_bps for m in self.slippage_history)
        total_cost = sum(m.cost_impact for m in self.slippage_history)

        # Group by symbol
        symbols = defaultdict(list)
        for metrics in self.slippage_history:
            symbols[metrics.symbol].append(metrics)

        return {
            'total_orders': total_orders,
            'avg_slippage_bps': avg_slippage,
            'max_slippage_bps': max_slippage,
            'total_cost_impact': total_cost,
            'symbols': dict(symbols)
        }

    def get_symbol_slippage_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get slippage statistics for specific symbol"""
        symbol_metrics = [m for m in self.slippage_history if m.symbol == symbol]

        if not symbol_metrics:
            return {
                'symbol': symbol,
                'order_count': 0,
                'avg_slippage_bps': Decimal('0'),
                'total_cost_impact': Decimal('0')
            }

        order_count = len(symbol_metrics)
        avg_slippage = sum(m.slippage_bps for m in symbol_metrics) / order_count
        total_cost = sum(m.cost_impact for m in symbol_metrics)

        return {
            'symbol': symbol,
            'order_count': order_count,
            'avg_slippage_bps': avg_slippage,
            'total_cost_impact': total_cost
        }

    async def start_monitoring(self):
        """Start real-time slippage monitoring"""
        self.is_monitoring = True

    async def stop_monitoring(self):
        """Stop real-time slippage monitoring"""
        self.is_monitoring = False

    def cleanup_old_history(self, max_age_hours: int = 24):
        """Remove old slippage history"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        self.slippage_history = [
            m for m in self.slippage_history
            if m.timestamp > cutoff_time
        ]

    def calculate_implementation_shortfall(self, decision_price: Decimal,
                                         execution_price: Decimal,
                                         order_size: Decimal,
                                         side: OrderSide) -> Decimal:
        """Calculate implementation shortfall"""
        if side == OrderSide.BUY:
            shortfall = (execution_price - decision_price) * order_size
        else:
            shortfall = (decision_price - execution_price) * order_size

        return shortfall

    def estimate_market_impact(self, order_size: Decimal, avg_daily_volume: int,
                             volatility: Decimal) -> Decimal:
        """Estimate market impact using square-root model"""
        participation_rate = float(order_size) / avg_daily_volume

        # Square-root market impact model
        # Impact = alpha * volatility * sqrt(participation_rate)
        alpha = 0.1  # Market impact coefficient
        impact = alpha * float(volatility) * math.sqrt(participation_rate)

        # Convert to basis points
        return Decimal(str(impact * 10000))

    def get_slippage_attribution(self) -> Dict[str, Any]:
        """Analyze slippage attribution by type"""
        attribution = defaultdict(Decimal)

        for metrics in self.slippage_history:
            slippage_type = getattr(metrics, 'slippage_type', 'unknown')
            attribution[slippage_type] += metrics.slippage_bps

        total_slippage = sum(attribution.values())

        result = dict(attribution)
        result['total_slippage_bps'] = total_slippage

        return result

    def validate_benchmark_price(self, benchmark_price: Decimal,
                                current_market_price: Decimal,
                                max_deviation_bps: int = 100) -> bool:
        """Validate benchmark price quality"""
        deviation_bps = abs(benchmark_price - current_market_price) / current_market_price * 10000
        return deviation_bps <= max_deviation_bps

    def generate_slippage_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive slippage report"""
        # Filter metrics by time range
        filtered_metrics = [
            m for m in self.slippage_history
            if start_time <= m.timestamp <= end_time
        ]

        if not filtered_metrics:
            return {
                'summary': {'total_orders': 0},
                'by_symbol': {},
                'alerts': []
            }

        # Summary statistics
        total_orders = len(filtered_metrics)
        avg_slippage = sum(m.slippage_bps for m in filtered_metrics) / total_orders
        total_cost = sum(m.cost_impact for m in filtered_metrics)

        # Group by symbol
        by_symbol = defaultdict(list)
        for metrics in filtered_metrics:
            by_symbol[metrics.symbol].append(metrics)

        symbol_stats = {}
        for symbol, metrics_list in by_symbol.items():
            symbol_stats[symbol] = {
                'order_count': len(metrics_list),
                'avg_slippage_bps': sum(m.slippage_bps for m in metrics_list) / len(metrics_list),
                'total_cost_impact': sum(m.cost_impact for m in metrics_list)
            }

        # Alerts in time range
        filtered_alerts = [
            alert for alert in self.active_alerts
            if start_time <= alert.timestamp <= end_time
        ]

        return {
            'summary': {
                'total_orders': total_orders,
                'avg_slippage_bps': avg_slippage,
                'total_cost_impact': total_cost
            },
            'by_symbol': symbol_stats,
            'alerts': [
                {
                    'symbol': alert.symbol,
                    'slippage_bps': alert.slippage_bps,
                    'severity': alert.severity,
                    'timestamp': alert.timestamp,
                    'message': alert.message
                }
                for alert in filtered_alerts
            ]
        }