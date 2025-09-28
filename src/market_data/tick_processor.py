# src/market_data/tick_processor.py

import logging
import numpy as np
from typing import Dict, List, Optional, Deque
from decimal import Decimal
from datetime import datetime, timedelta
from collections import deque

from .models import TickData, TickType, OrderSide, MicrostructurePatterns


class TickDataAnalyzer:
    """Tick-level market data analysis and microstructure pattern detection"""

    def __init__(self, buffer_size: int = 1000):
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.buffer_size = buffer_size
        self.vpin_window = 50
        self.pattern_detection_window = 100

        # Data buffers
        self.tick_buffer: Deque[TickData] = deque(maxlen=buffer_size)
        self.trade_buffer: Deque[TickData] = deque(maxlen=buffer_size)
        self.quote_buffer: Deque[TickData] = deque(maxlen=buffer_size)

        # Real-time metrics
        self.trade_flow_imbalance = 0.0
        self.last_vpin_score = 0.5
        self.quote_rate_tracker = deque(maxlen=60)  # Track last 60 seconds

        # Pattern detection state
        self.recent_patterns = deque(maxlen=10)
        self.alert_state = "NONE"
        self.pattern_confidence = 0.0

        # Performance tracking
        self.processed_ticks = 0
        self.last_pattern_check = datetime.utcnow()

    def process_tick(self, tick: TickData) -> Optional[MicrostructurePatterns]:
        """
        Process a single tick and update internal state

        Args:
            tick: Individual tick data

        Returns:
            Optional[MicrostructurePatterns]: Detected patterns if any
        """
        self.tick_buffer.append(tick)
        self.processed_ticks += 1

        # Route tick to appropriate buffer
        if tick.tick_type == TickType.TRADE:
            self.trade_buffer.append(tick)
            self._update_trade_flow(tick)
        elif tick.tick_type == TickType.QUOTE:
            self.quote_buffer.append(tick)
            self._update_quote_rate(tick)

        # Perform pattern detection every 10 ticks
        if self.processed_ticks % 10 == 0:
            return self.detect_microstructure_patterns()

        return None

    def _update_trade_flow(self, trade_tick: TickData) -> None:
        """
        Update trade flow imbalance using tick rule

        Args:
            trade_tick: Trade tick data
        """
        if len(self.trade_buffer) < 2:
            return

        # Get previous trade
        prev_trade = list(self.trade_buffer)[-2]
        curr_trade = trade_tick

        # Determine trade direction using tick rule
        if curr_trade.price > prev_trade.price:
            direction = 1  # Buy pressure
        elif curr_trade.price < prev_trade.price:
            direction = -1  # Sell pressure
        else:
            # Use previous direction or side information if available
            if curr_trade.side == OrderSide.BUY:
                direction = 1
            elif curr_trade.side == OrderSide.SELL:
                direction = -1
            else:
                direction = 0  # Neutral

        # Update flow imbalance with exponential decay
        if curr_trade.size:
            flow_impact = direction * float(curr_trade.size)
            self.trade_flow_imbalance = (
                0.95 * self.trade_flow_imbalance + 0.05 * flow_impact
            )

    def _update_quote_rate(self, quote_tick: TickData) -> None:
        """
        Update quote rate tracking

        Args:
            quote_tick: Quote update tick
        """
        tick_time = quote_tick.timestamp
        self.quote_rate_tracker.append(tick_time)

        # Clean old quotes (older than 60 seconds)
        cutoff_time = tick_time - timedelta(seconds=60)
        while (self.quote_rate_tracker and
               self.quote_rate_tracker[0] < cutoff_time):
            self.quote_rate_tracker.popleft()

    def calculate_vpin(self, window: int = None) -> float:
        """
        Calculate Volume-synchronized Probability of Informed Trading (VPIN)

        Args:
            window: Window size for calculation

        Returns:
            float: VPIN score (0-1, higher indicates more informed trading)
        """
        if window is None:
            window = self.vpin_window

        if len(self.trade_buffer) < window:
            return self.last_vpin_score

        recent_trades = list(self.trade_buffer)[-window:]

        # Calculate buy and sell volumes
        buy_volume = Decimal('0')
        sell_volume = Decimal('0')

        for trade in recent_trades:
            if not trade.size:
                continue

            # Determine trade direction
            if trade.side == OrderSide.BUY:
                buy_volume += trade.size
            elif trade.side == OrderSide.SELL:
                sell_volume += trade.size
            else:
                # Use tick rule if side not available
                if len(recent_trades) >= 2:
                    idx = recent_trades.index(trade)
                    if idx > 0:
                        prev_trade = recent_trades[idx - 1]
                        if trade.price > prev_trade.price:
                            buy_volume += trade.size
                        elif trade.price < prev_trade.price:
                            sell_volume += trade.size
                        else:
                            # Split equally if no direction
                            buy_volume += trade.size / 2
                            sell_volume += trade.size / 2

        total_volume = buy_volume + sell_volume

        if total_volume > 0:
            vpin = float(abs(buy_volume - sell_volume) / total_volume)
        else:
            vpin = 0.5

        self.last_vpin_score = vpin
        return vpin

    def detect_microstructure_patterns(self) -> MicrostructurePatterns:
        """
        Detect various microstructure patterns

        Returns:
            MicrostructurePatterns: Comprehensive pattern analysis
        """
        current_time = datetime.utcnow()

        patterns = MicrostructurePatterns(
            symbol=self._get_symbol_from_buffer(),
            timestamp=current_time
        )

        if len(self.tick_buffer) < self.pattern_detection_window:
            return patterns

        # Update VPIN score
        patterns.vpin_score = self.calculate_vpin()
        patterns.trade_flow_imbalance = self.trade_flow_imbalance

        # Detect specific patterns
        patterns.quote_stuffing = self._detect_quote_stuffing()
        patterns.layering = self._detect_layering()
        patterns.momentum_ignition = self._detect_momentum_ignition()
        patterns.ping_pong = self._detect_ping_pong()

        # Calculate quote rate
        patterns.quote_rate = self._calculate_current_quote_rate()

        # Calculate order/cancel ratio
        patterns.order_cancel_ratio = self._calculate_order_cancel_ratio()

        # Calculate price momentum
        patterns.price_momentum = self._calculate_price_momentum()

        # Assess overall pattern confidence and alert level
        patterns.pattern_confidence, patterns.alert_level = self._assess_pattern_confidence(patterns)
        patterns.description = self._generate_pattern_description(patterns)

        # Store pattern for historical analysis
        self.recent_patterns.append(patterns)
        self.last_pattern_check = current_time

        return patterns

    def _detect_quote_stuffing(self) -> bool:
        """
        Detect quote stuffing pattern (excessive quote updates)

        Returns:
            bool: True if quote stuffing detected
        """
        quote_rate = self._calculate_current_quote_rate()

        # Quote stuffing threshold: >100 quotes per second
        return quote_rate > 100

    def _detect_layering(self) -> bool:
        """
        Detect layering pattern (placing large orders then canceling)

        Returns:
            bool: True if layering detected
        """
        if len(self.tick_buffer) < 20:
            return False

        # Look for pattern of orders followed by quick cancellations
        recent_ticks = list(self.tick_buffer)[-20:]

        order_events = [t for t in recent_ticks if t.tick_type in [TickType.ORDER, TickType.CANCEL]]

        if len(order_events) < 10:
            return False

        # Calculate order to cancel ratio
        order_count = sum(1 for t in order_events if t.tick_type == TickType.ORDER)
        cancel_count = sum(1 for t in order_events if t.tick_type == TickType.CANCEL)

        if order_count == 0:
            return False

        cancel_ratio = cancel_count / order_count

        # Layering indicator: high cancellation rate (>70%)
        return cancel_ratio > 0.7

    def _detect_momentum_ignition(self) -> bool:
        """
        Detect momentum ignition pattern (rapid price moves with volume)

        Returns:
            bool: True if momentum ignition detected
        """
        if len(self.trade_buffer) < 10:
            return False

        recent_trades = list(self.trade_buffer)[-10:]

        if len(recent_trades) < 5:
            return False

        # Check for consecutive trades in same direction
        prices = [float(t.price) for t in recent_trades if t.price is not None]

        if len(prices) < 5:
            return False

        # Calculate price momentum
        price_change = (prices[-1] - prices[0]) / prices[0]

        # Check for unidirectional movement
        consecutive_direction = 0
        max_consecutive = 0

        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                if consecutive_direction >= 0:
                    consecutive_direction += 1
                else:
                    consecutive_direction = 1
            elif prices[i] < prices[i-1]:
                if consecutive_direction <= 0:
                    consecutive_direction -= 1
                else:
                    consecutive_direction = -1
            else:
                consecutive_direction = 0

            max_consecutive = max(max_consecutive, abs(consecutive_direction))

        # Momentum ignition: >0.2% price move with >3 consecutive directional trades
        return abs(price_change) > 0.002 and max_consecutive >= 3

    def _detect_ping_pong(self) -> bool:
        """
        Detect ping-pong trading pattern (rapid back-and-forth trades)

        Returns:
            bool: True if ping-pong detected
        """
        if len(self.trade_buffer) < 6:
            return False

        recent_trades = list(self.trade_buffer)[-6:]

        # Look for alternating buy/sell pattern
        sides = []
        for trade in recent_trades:
            if trade.side:
                sides.append(trade.side)
            elif len(recent_trades) >= 2:
                # Use tick rule
                idx = recent_trades.index(trade)
                if idx > 0:
                    prev_trade = recent_trades[idx - 1]
                    if trade.price > prev_trade.price:
                        sides.append(OrderSide.BUY)
                    elif trade.price < prev_trade.price:
                        sides.append(OrderSide.SELL)

        if len(sides) < 4:
            return False

        # Check for alternating pattern
        alternating_count = 0
        for i in range(1, len(sides)):
            if sides[i] != sides[i-1]:
                alternating_count += 1

        # Ping-pong: >75% alternating trades
        return alternating_count / (len(sides) - 1) > 0.75

    def _calculate_current_quote_rate(self) -> float:
        """Calculate current quote update rate per second"""
        if len(self.quote_rate_tracker) < 2:
            return 0.0

        # Use the last quote timestamp as reference time instead of utcnow()
        # This ensures consistency with the timestamps we're tracking
        reference_time = self.quote_rate_tracker[-1]
        cutoff_time = reference_time - timedelta(seconds=60)

        # Clean old quotes first (older than 60 seconds from the last quote)
        while (self.quote_rate_tracker and
               self.quote_rate_tracker[0] < cutoff_time):
            self.quote_rate_tracker.popleft()

        # Calculate rate: quotes in last 60 seconds divided by time window
        if len(self.quote_rate_tracker) < 2:
            return 0.0

        time_span = (self.quote_rate_tracker[-1] - self.quote_rate_tracker[0]).total_seconds()
        if time_span <= 0:
            return float(len(self.quote_rate_tracker))  # If all quotes at same time

        return len(self.quote_rate_tracker) / max(time_span, 1.0)

    def _calculate_order_cancel_ratio(self) -> float:
        """Calculate ratio of cancelled orders to placed orders"""
        recent_ticks = list(self.tick_buffer)[-50:]

        orders = sum(1 for t in recent_ticks if t.tick_type == TickType.ORDER)
        cancels = sum(1 for t in recent_ticks if t.tick_type == TickType.CANCEL)

        if orders == 0:
            return 0.0

        return cancels / orders

    def _calculate_price_momentum(self) -> float:
        """Calculate recent price momentum"""
        if len(self.trade_buffer) < 5:
            return 0.0

        recent_trades = list(self.trade_buffer)[-5:]
        prices = [float(t.price) for t in recent_trades if t.price is not None]

        if len(prices) < 2:
            return 0.0

        return (prices[-1] - prices[0]) / prices[0]

    def _assess_pattern_confidence(self, patterns: MicrostructurePatterns) -> tuple:
        """
        Assess overall pattern confidence and alert level

        Args:
            patterns: Detected patterns

        Returns:
            tuple: (confidence, alert_level)
        """
        # Count detected patterns
        pattern_count = sum([
            patterns.quote_stuffing,
            patterns.layering,
            patterns.momentum_ignition,
            patterns.ping_pong
        ])

        # Base confidence on number of patterns and VPIN score
        base_confidence = pattern_count * 0.25
        vpin_confidence = abs(patterns.vpin_score - 0.5) * 2  # Distance from neutral

        overall_confidence = (base_confidence + vpin_confidence) / 2

        # Determine alert level
        if overall_confidence > 0.8 or pattern_count >= 3:
            alert_level = "HIGH"
        elif overall_confidence > 0.6 or pattern_count >= 2:
            alert_level = "MEDIUM"
        elif overall_confidence > 0.3 or pattern_count >= 1:
            alert_level = "LOW"
        else:
            alert_level = "NONE"

        return min(1.0, overall_confidence), alert_level

    def _generate_pattern_description(self, patterns: MicrostructurePatterns) -> str:
        """Generate human-readable pattern description"""
        detected_patterns = []

        if patterns.quote_stuffing:
            detected_patterns.append("quote stuffing")
        if patterns.layering:
            detected_patterns.append("layering")
        if patterns.momentum_ignition:
            detected_patterns.append("momentum ignition")
        if patterns.ping_pong:
            detected_patterns.append("ping-pong trading")

        if not detected_patterns:
            return "Normal market activity"

        pattern_str = ", ".join(detected_patterns)
        vpin_str = f"VPIN: {patterns.vpin_score:.3f}"

        return f"Detected: {pattern_str}. {vpin_str}"

    def _get_symbol_from_buffer(self) -> str:
        """Extract symbol from tick buffer"""
        if self.tick_buffer:
            return self.tick_buffer[-1].symbol
        return "UNKNOWN"

    def get_real_time_metrics(self) -> Dict:
        """Get current real-time market microstructure metrics"""
        return {
            'trade_flow_imbalance': self.trade_flow_imbalance,
            'vpin_score': self.last_vpin_score,
            'quote_rate': self._calculate_current_quote_rate(),
            'order_cancel_ratio': self._calculate_order_cancel_ratio(),
            'price_momentum': self._calculate_price_momentum(),
            'buffer_usage': {
                'tick_buffer': len(self.tick_buffer),
                'trade_buffer': len(self.trade_buffer),
                'quote_buffer': len(self.quote_buffer)
            },
            'processing_stats': {
                'processed_ticks': self.processed_ticks,
                'last_pattern_check': self.last_pattern_check
            }
        }

    def get_pattern_history(self, limit: int = 10) -> List[MicrostructurePatterns]:
        """Get recent pattern detection history"""
        return list(self.recent_patterns)[-limit:]

    def reset_buffers(self) -> None:
        """Reset all buffers and state"""
        self.tick_buffer.clear()
        self.trade_buffer.clear()
        self.quote_buffer.clear()
        self.quote_rate_tracker.clear()
        self.recent_patterns.clear()

        self.trade_flow_imbalance = 0.0
        self.last_vpin_score = 0.5
        self.processed_ticks = 0

        self.logger.info("Tick data analyzer buffers reset")