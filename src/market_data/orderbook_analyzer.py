# src/market_data/orderbook_analyzer.py

import logging
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from decimal import Decimal
from datetime import datetime

from .models import (
    OrderBookSnapshot, MarketMetrics, OrderLevel, BookShape,
    OrderSide, TickData
)


class OrderBookAnalyzer:
    """Real-time order book analysis and liquidity evaluation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._trade_history: List[Dict] = []
        self._trade_history_limit = 1000

    def analyze_orderbook(self, orderbook: OrderBookSnapshot) -> MarketMetrics:
        """
        Comprehensive order book microstructure analysis

        Args:
            orderbook: Order book snapshot to analyze

        Returns:
            MarketMetrics: Complete market condition analysis
        """
        if not orderbook.bids or not orderbook.asks:
            raise ValueError("Order book must contain both bids and asks")

        # Basic spread analysis
        best_bid = orderbook.bids[0].price
        best_ask = orderbook.asks[0].price
        mid_price = (best_bid + best_ask) / Decimal('2')
        spread = best_ask - best_bid
        spread_bps = float(spread / mid_price * Decimal('10000'))

        # Order book imbalance (top 5 levels)
        bid_volume_5 = sum(level.size for level in orderbook.bids[:5])
        ask_volume_5 = sum(level.size for level in orderbook.asks[:5])
        total_volume = bid_volume_5 + ask_volume_5

        if total_volume > 0:
            imbalance = float((bid_volume_5 - ask_volume_5) / total_volume)
        else:
            imbalance = 0.0

        # Liquidity score calculation
        liquidity_score = self._calculate_liquidity_score(
            orderbook.bids, orderbook.asks
        )

        # Price impact function
        price_impact_function = self._estimate_price_impact(
            orderbook.bids, orderbook.asks
        )

        # Effective spread (if trade history available)
        effective_spread = self._calculate_effective_spread()

        # Book shape analysis
        book_shape, bid_slope, ask_slope = self._analyze_book_shape(
            orderbook.bids, orderbook.asks
        )

        # Large order detection
        large_orders = self._detect_large_orders(
            orderbook.bids, orderbook.asks
        )

        return MarketMetrics(
            symbol=orderbook.symbol,
            timestamp=orderbook.timestamp,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            bid_volume_5=bid_volume_5,
            ask_volume_5=ask_volume_5,
            top_5_liquidity=total_volume,
            imbalance=imbalance,
            liquidity_score=liquidity_score,
            price_impact_function=price_impact_function,
            effective_spread=effective_spread,
            book_shape=book_shape,
            bid_slope=bid_slope,
            ask_slope=ask_slope,
            large_orders=large_orders
        )

    def _calculate_liquidity_score(self, bids: List[OrderLevel],
                                   asks: List[OrderLevel]) -> float:
        """
        Calculate liquidity score (0-1) based on depth and price uniformity

        Args:
            bids: Bid side order levels
            asks: Ask side order levels

        Returns:
            float: Liquidity score between 0 and 1
        """
        # Total liquidity in top 10 levels
        bid_liquidity = sum(level.size for level in bids[:10])
        ask_liquidity = sum(level.size for level in asks[:10])
        total_liquidity = bid_liquidity + ask_liquidity

        # Price uniformity check (consistent tick sizes)
        bid_price_diffs = []
        for i in range(min(9, len(bids) - 1)):
            diff = float(bids[i].price - bids[i + 1].price)
            bid_price_diffs.append(diff)

        if bid_price_diffs:
            price_uniformity = 1.0 / (1.0 + np.std(bid_price_diffs))
        else:
            price_uniformity = 0.0

        # Normalize liquidity component
        liquidity_component = min(1.0, float(total_liquidity) / 10000)

        # Combined score
        liquidity_score = liquidity_component * price_uniformity

        return float(liquidity_score)

    def _estimate_price_impact(self, bids: List[OrderLevel],
                              asks: List[OrderLevel]) -> Callable:
        """
        Create price impact function using square-root market impact model

        Args:
            bids: Bid side order levels
            asks: Ask side order levels

        Returns:
            Callable: Function that estimates price impact for given order size
        """

        def impact_function(size: Decimal, side: str = 'BUY') -> float:
            """
            Estimate price impact for order of given size

            Args:
                size: Order size
                side: 'BUY' or 'SELL'

            Returns:
                float: Expected price impact as percentage
            """
            levels = asks if side == 'BUY' else bids

            if not levels:
                return 0.05  # 5% penalty for no liquidity

            cumulative_size = Decimal('0')
            weighted_price = Decimal('0')

            for level in levels:
                level_size = level.size
                level_price = level.price

                if cumulative_size + level_size >= size:
                    # Partial fill of this level
                    remaining = size - cumulative_size
                    weighted_price += remaining * level_price
                    cumulative_size = size
                    break
                else:
                    # Full consumption of this level
                    weighted_price += level_size * level_price
                    cumulative_size += level_size

            if cumulative_size >= size and cumulative_size > 0:
                avg_price = weighted_price / size
                mid_price = (bids[0].price + asks[0].price) / Decimal('2')
                impact = abs(avg_price - mid_price) / mid_price
                return float(impact)
            else:
                # Not enough liquidity
                return 0.05  # 5% penalty

        return impact_function

    def _calculate_effective_spread(self) -> Optional[float]:
        """
        Calculate effective spread based on recent trade history

        Returns:
            Optional[float]: Effective spread in basis points
        """
        if len(self._trade_history) < 3:  # Reduced minimum requirement
            return None

        effective_spreads = []

        for trade in self._trade_history[-50:]:  # Last 50 trades
            try:
                mid_at_trade = trade['mid_price_at_execution']
                trade_price = trade['price']
                side_sign = 1 if trade.get('side') == 'BUY' else -1

                # Effective spread formula: 2 * |trade_price - mid_price| / mid_price
                eff_spread = 2.0 * abs(trade_price - mid_at_trade) / mid_at_trade
                effective_spreads.append(eff_spread)
            except (KeyError, ZeroDivisionError, TypeError):
                continue

        if effective_spreads:
            return float(np.mean(effective_spreads) * 10000)  # Convert to BPS
        else:
            return None

    def _analyze_book_shape(self, bids: List[OrderLevel],
                           asks: List[OrderLevel]) -> tuple:
        """
        Analyze order book shape and slope

        Args:
            bids: Bid side order levels
            asks: Ask side order levels

        Returns:
            tuple: (BookShape, bid_slope, ask_slope)
        """
        bid_slope = None
        ask_slope = None

        # Calculate bid slope (top 5 levels)
        if len(bids) >= 5:
            bid_prices = [float(b.price) for b in bids[:5]]
            bid_volumes = [float(b.size) for b in bids[:5]]
            if len(set(bid_prices)) > 1:  # Avoid perfect correlation
                bid_slope = float(np.polyfit(bid_prices, bid_volumes, 1)[0])

        # Calculate ask slope (top 5 levels)
        if len(asks) >= 5:
            ask_prices = [float(a.price) for a in asks[:5]]
            ask_volumes = [float(a.size) for a in asks[:5]]
            if len(set(ask_prices)) > 1:  # Avoid perfect correlation
                ask_slope = float(np.polyfit(ask_prices, ask_volumes, 1)[0])

        # Determine book shape
        book_shape = BookShape.FLAT
        if bid_slope is not None and ask_slope is not None:
            if abs(bid_slope) < 0.1 and abs(ask_slope) < 0.1:
                book_shape = BookShape.FLAT
            elif bid_slope > ask_slope:
                book_shape = BookShape.BID_HEAVY
            else:
                book_shape = BookShape.ASK_HEAVY

        return book_shape, bid_slope, ask_slope

    def _detect_large_orders(self, bids: List[OrderLevel],
                            asks: List[OrderLevel]) -> List[Dict]:
        """
        Detect unusually large orders in the book

        Args:
            bids: Bid side order levels
            asks: Ask side order levels

        Returns:
            List[Dict]: List of detected large orders
        """
        large_orders = []

        # Calculate average size from top 20 levels
        all_sizes = ([float(b.size) for b in bids[:20]] +
                     [float(a.size) for a in asks[:20]])

        if not all_sizes:
            return large_orders

        avg_size = np.mean(all_sizes)
        threshold = avg_size * 3  # 3x average size

        # Check bids for large orders
        for i, bid in enumerate(bids[:10]):
            if float(bid.size) > threshold:
                large_orders.append({
                    'side': 'BID',
                    'level': i,
                    'price': float(bid.price),
                    'size': float(bid.size),
                    'size_ratio': float(bid.size) / avg_size
                })

        # Check asks for large orders
        for i, ask in enumerate(asks[:10]):
            if float(ask.size) > threshold:
                large_orders.append({
                    'side': 'ASK',
                    'level': i,
                    'price': float(ask.price),
                    'size': float(ask.size),
                    'size_ratio': float(ask.size) / avg_size
                })

        return large_orders

    def update_trade_history(self, trade_data: Dict) -> None:
        """
        Update trade history for effective spread calculation

        Args:
            trade_data: Trade execution data
        """
        self._trade_history.append(trade_data)

        # Maintain history limit
        if len(self._trade_history) > self._trade_history_limit:
            self._trade_history = self._trade_history[-self._trade_history_limit:]

    def get_bid_ask_pressure(self, orderbook: OrderBookSnapshot,
                           depth_levels: int = 5) -> Dict[str, float]:
        """
        Calculate bid/ask pressure metrics

        Args:
            orderbook: Order book snapshot
            depth_levels: Number of levels to include in calculation

        Returns:
            Dict: Pressure metrics
        """
        bid_volume = sum(float(level.size) for level in orderbook.bids[:depth_levels])
        ask_volume = sum(float(level.size) for level in orderbook.asks[:depth_levels])

        total_volume = bid_volume + ask_volume

        if total_volume > 0:
            bid_pressure = bid_volume / total_volume
            ask_pressure = ask_volume / total_volume
            pressure_ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        else:
            bid_pressure = ask_pressure = 0.5
            pressure_ratio = 1.0

        return {
            'bid_pressure': bid_pressure,
            'ask_pressure': ask_pressure,
            'pressure_ratio': pressure_ratio,
            'total_volume': total_volume,
            'pressure_imbalance': bid_pressure - ask_pressure
        }

    def calculate_book_stability(self, orderbook_history: List[OrderBookSnapshot],
                               window_size: int = 10) -> float:
        """
        Calculate order book stability score

        Args:
            orderbook_history: Recent order book snapshots
            window_size: Number of snapshots to analyze

        Returns:
            float: Stability score (0-1, higher = more stable)
        """
        if len(orderbook_history) < 2:
            return 0.5

        recent_books = orderbook_history[-window_size:]
        spread_changes = []
        imbalance_changes = []

        for i in range(1, len(recent_books)):
            prev_book = recent_books[i - 1]
            curr_book = recent_books[i]

            # Spread stability
            if prev_book.bids and prev_book.asks and curr_book.bids and curr_book.asks:
                prev_spread = float(prev_book.asks[0].price - prev_book.bids[0].price)
                curr_spread = float(curr_book.asks[0].price - curr_book.bids[0].price)

                if prev_spread > 0:
                    spread_change = abs(curr_spread - prev_spread) / prev_spread
                    spread_changes.append(spread_change)

                # Imbalance stability
                prev_metrics = self.analyze_orderbook(prev_book)
                curr_metrics = self.analyze_orderbook(curr_book)
                imbalance_change = abs(curr_metrics.imbalance - prev_metrics.imbalance)
                imbalance_changes.append(imbalance_change)

        if spread_changes and imbalance_changes:
            spread_stability = 1.0 / (1.0 + np.mean(spread_changes))
            imbalance_stability = 1.0 / (1.0 + np.mean(imbalance_changes))
            return (spread_stability + imbalance_stability) / 2.0
        else:
            return 0.5