# src/market_data/enhanced_market_analyzer.py

import logging
from typing import Dict, Optional, List
from decimal import Decimal
from datetime import datetime

from .data_aggregator import DataAggregator
from .models import MarketMetrics, MicrostructurePatterns
from ..execution.market_analyzer import MarketConditionAnalyzer as BaseMarketAnalyzer


class EnhancedMarketConditionAnalyzer:
    """
    Enhanced market condition analyzer that integrates with the real-time market data pipeline
    """

    def __init__(self, data_aggregator: DataAggregator):
        self.logger = logging.getLogger(__name__)
        self.data_aggregator = data_aggregator
        self.base_analyzer = BaseMarketAnalyzer()

        # Cache for enhanced metrics
        self._cached_metrics: Dict[str, Dict] = {}
        self._cache_timeout = 5  # seconds

    async def analyze_market_conditions(self, symbol: str) -> Dict:
        """
        Analyze market conditions using enhanced real-time data

        Args:
            symbol: Trading symbol

        Returns:
            Dict: Comprehensive market analysis
        """
        try:
            # Get aggregated market data
            market_data = await self.data_aggregator.get_market_data(symbol)
            if not market_data or not market_data.current_metrics:
                return self._fallback_analysis(symbol)

            metrics = market_data.current_metrics
            patterns = market_data.patterns

            # Build comprehensive analysis
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'data_quality': self._assess_data_quality(market_data),

                # Basic market metrics
                'spread_bps': metrics.spread_bps,
                'liquidity_score': metrics.liquidity_score,
                'imbalance': metrics.imbalance,
                'book_shape': metrics.book_shape.value,

                # Enhanced analytics
                'microstructure_analysis': self._analyze_microstructure(patterns) if patterns else {},
                'liquidity_analysis': self._analyze_liquidity(metrics, market_data),
                'execution_analysis': await self._analyze_execution_conditions(symbol, market_data),
                'risk_assessment': self._assess_risk_factors(metrics, patterns),

                # Trading recommendations
                'execution_recommendation': self._generate_execution_recommendation(metrics, patterns),
                'optimal_windows': await self._get_execution_windows(symbol),

                # Performance metrics
                'processing_latency': self._calculate_processing_latency(market_data),
                'confidence_score': self._calculate_confidence_score(market_data)
            }

            # Cache the analysis
            self._cached_metrics[symbol] = {
                'analysis': analysis,
                'timestamp': datetime.utcnow()
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing market conditions for {symbol}: {e}")
            return self._fallback_analysis(symbol)

    def _assess_data_quality(self, market_data) -> Dict:
        """Assess the quality of market data"""
        return {
            'orderbook_age_ms': (
                datetime.utcnow() - market_data.current_metrics.timestamp
            ).total_seconds() * 1000,
            'tick_count': len(market_data.tick_history),
            'orderbook_count': len(market_data.orderbook_history),
            'cache_valid': market_data.is_cache_valid(),
            'data_completeness': min(1.0, len(market_data.tick_history) / 100),
            'quality_score': self._calculate_data_quality_score(market_data)
        }

    def _calculate_data_quality_score(self, market_data) -> float:
        """Calculate overall data quality score (0-1)"""
        age_penalty = min(1.0, (
            datetime.utcnow() - market_data.current_metrics.timestamp
        ).total_seconds() / 10)  # Penalty for data older than 10s

        completeness = min(1.0, len(market_data.tick_history) / 50)
        cache_bonus = 0.1 if market_data.is_cache_valid() else 0

        return max(0, 1.0 - age_penalty + completeness + cache_bonus)

    def _analyze_microstructure(self, patterns: MicrostructurePatterns) -> Dict:
        """Analyze microstructure patterns"""
        if not patterns:
            return {}

        return {
            'vpin_score': patterns.vpin_score,
            'vpin_interpretation': self._interpret_vpin(patterns.vpin_score),
            'pattern_flags': {
                'quote_stuffing': patterns.quote_stuffing,
                'layering': patterns.layering,
                'momentum_ignition': patterns.momentum_ignition,
                'ping_pong': patterns.ping_pong
            },
            'alert_level': patterns.alert_level,
            'pattern_confidence': patterns.pattern_confidence,
            'trade_flow_imbalance': patterns.trade_flow_imbalance,
            'market_manipulation_risk': self._assess_manipulation_risk(patterns),
            'information_flow': self._assess_information_flow(patterns)
        }

    def _interpret_vpin(self, vpin_score: float) -> str:
        """Interpret VPIN score"""
        if vpin_score > 0.8:
            return "HIGH_INFORMED_TRADING"
        elif vpin_score > 0.6:
            return "MODERATE_INFORMED_TRADING"
        elif vpin_score > 0.4:
            return "SLIGHT_INFORMED_TRADING"
        else:
            return "NORMAL_TRADING"

    def _assess_manipulation_risk(self, patterns: MicrostructurePatterns) -> str:
        """Assess market manipulation risk"""
        risk_factors = sum([
            patterns.quote_stuffing,
            patterns.layering,
            patterns.momentum_ignition
        ])

        if risk_factors >= 2:
            return "HIGH"
        elif risk_factors == 1:
            return "MEDIUM"
        else:
            return "LOW"

    def _assess_information_flow(self, patterns: MicrostructurePatterns) -> Dict:
        """Assess information flow in the market"""
        return {
            'informed_trading_probability': patterns.vpin_score,
            'information_intensity': abs(patterns.trade_flow_imbalance),
            'price_discovery_efficiency': 1.0 - patterns.vpin_score,  # Inverse relationship
            'market_stress_level': self._calculate_stress_level(patterns)
        }

    def _calculate_stress_level(self, patterns: MicrostructurePatterns) -> float:
        """Calculate market stress level (0-1)"""
        stress_factors = [
            patterns.vpin_score,
            abs(patterns.trade_flow_imbalance),
            patterns.pattern_confidence,
            patterns.quote_rate / 200  # Normalize quote rate
        ]

        return min(1.0, sum(stress_factors) / len(stress_factors))

    def _analyze_liquidity(self, metrics: MarketMetrics, market_data) -> Dict:
        """Analyze liquidity conditions"""
        return {
            'current_liquidity_score': metrics.liquidity_score,
            'liquidity_grade': self._grade_liquidity(metrics.liquidity_score),
            'depth_analysis': {
                'bid_volume_5': float(metrics.bid_volume_5),
                'ask_volume_5': float(metrics.ask_volume_5),
                'total_depth': float(metrics.top_5_liquidity),
                'imbalance': metrics.imbalance,
                'imbalance_severity': self._assess_imbalance_severity(metrics.imbalance)
            },
            'spread_analysis': {
                'current_spread_bps': metrics.spread_bps,
                'spread_grade': self._grade_spread(metrics.spread_bps),
                'effective_spread': metrics.effective_spread,
                'bid_ask_ratio': float(metrics.bid_volume_5 / metrics.ask_volume_5) if metrics.ask_volume_5 > 0 else float('inf')
            },
            'large_orders': {
                'count': len(metrics.large_orders),
                'total_size': sum(order['size'] for order in metrics.large_orders),
                'avg_size_ratio': sum(order['size_ratio'] for order in metrics.large_orders) / max(1, len(metrics.large_orders))
            },
            'historical_context': self._analyze_liquidity_trend(market_data)
        }

    def _grade_liquidity(self, score: float) -> str:
        """Grade liquidity score"""
        if score >= 0.8:
            return "EXCELLENT"
        elif score >= 0.6:
            return "GOOD"
        elif score >= 0.4:
            return "FAIR"
        elif score >= 0.2:
            return "POOR"
        else:
            return "VERY_POOR"

    def _grade_spread(self, spread_bps: float) -> str:
        """Grade spread tightness"""
        if spread_bps <= 2:
            return "VERY_TIGHT"
        elif spread_bps <= 5:
            return "TIGHT"
        elif spread_bps <= 10:
            return "MODERATE"
        elif spread_bps <= 20:
            return "WIDE"
        else:
            return "VERY_WIDE"

    def _assess_imbalance_severity(self, imbalance: float) -> str:
        """Assess order book imbalance severity"""
        abs_imbalance = abs(imbalance)
        if abs_imbalance >= 0.3:
            return "SEVERE"
        elif abs_imbalance >= 0.15:
            return "MODERATE"
        elif abs_imbalance >= 0.05:
            return "MILD"
        else:
            return "BALANCED"

    def _analyze_liquidity_trend(self, market_data) -> Dict:
        """Analyze liquidity trend from historical data"""
        if len(market_data.metrics_history) < 5:
            return {'trend': 'INSUFFICIENT_DATA'}

        recent_scores = [m.liquidity_score for m in market_data.metrics_history[-5:]]
        avg_score = sum(recent_scores) / len(recent_scores)

        if len(recent_scores) >= 2:
            trend_slope = (recent_scores[-1] - recent_scores[0]) / (len(recent_scores) - 1)
        else:
            trend_slope = 0

        return {
            'trend': 'IMPROVING' if trend_slope > 0.05 else 'DETERIORATING' if trend_slope < -0.05 else 'STABLE',
            'avg_recent_score': avg_score,
            'trend_slope': trend_slope,
            'volatility': max(recent_scores) - min(recent_scores)
        }

    async def _analyze_execution_conditions(self, symbol: str, market_data) -> Dict:
        """Analyze execution conditions"""
        try:
            # Get market impact estimate for a standard order size
            impact_estimate = await self.data_aggregator.estimate_market_impact(
                symbol, Decimal('1.0')
            )

            return {
                'execution_favorability': self._assess_execution_favorability(market_data.current_metrics),
                'recommended_order_size': self._recommend_order_size(market_data.current_metrics),
                'impact_analysis': {
                    'temporary_impact': impact_estimate.temporary_impact if impact_estimate else None,
                    'permanent_impact': impact_estimate.permanent_impact if impact_estimate else None,
                    'total_impact': impact_estimate.total_impact if impact_estimate else None
                } if impact_estimate else {},
                'execution_urgency': self._assess_execution_urgency(market_data),
                'slippage_expectation': self._estimate_slippage(market_data.current_metrics)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing execution conditions: {e}")
            return {'error': str(e)}

    def _assess_execution_favorability(self, metrics: MarketMetrics) -> str:
        """Assess how favorable current conditions are for execution"""
        score = 0

        # Liquidity factor
        if metrics.liquidity_score >= 0.7:
            score += 3
        elif metrics.liquidity_score >= 0.5:
            score += 2
        elif metrics.liquidity_score >= 0.3:
            score += 1

        # Spread factor
        if metrics.spread_bps <= 3:
            score += 3
        elif metrics.spread_bps <= 7:
            score += 2
        elif metrics.spread_bps <= 15:
            score += 1

        # Imbalance factor (lower imbalance is better)
        if abs(metrics.imbalance) <= 0.05:
            score += 2
        elif abs(metrics.imbalance) <= 0.15:
            score += 1

        # Convert score to rating
        if score >= 7:
            return "EXCELLENT"
        elif score >= 5:
            return "GOOD"
        elif score >= 3:
            return "FAIR"
        else:
            return "POOR"

    def _recommend_order_size(self, metrics: MarketMetrics) -> Dict:
        """Recommend optimal order size based on current conditions"""
        base_size = float(metrics.top_5_liquidity) * 0.1  # 10% of top 5 liquidity

        return {
            'conservative': base_size * 0.5,
            'normal': base_size,
            'aggressive': base_size * 2,
            'max_recommended': float(metrics.top_5_liquidity) * 0.3
        }

    def _assess_execution_urgency(self, market_data) -> str:
        """Assess execution urgency based on market dynamics"""
        if not market_data.patterns:
            return "LOW"

        patterns = market_data.patterns

        if patterns.momentum_ignition or patterns.alert_level == "HIGH":
            return "URGENT"
        elif patterns.quote_stuffing or patterns.alert_level == "MEDIUM":
            return "HIGH"
        elif patterns.vpin_score > 0.7:
            return "MEDIUM"
        else:
            return "LOW"

    def _estimate_slippage(self, metrics: MarketMetrics) -> Dict:
        """Estimate expected slippage"""
        base_slippage = metrics.spread_bps / 2  # Half spread as base

        return {
            'expected_slippage_bps': base_slippage,
            'slippage_range': {
                'best_case': base_slippage * 0.5,
                'expected': base_slippage,
                'worst_case': base_slippage * 2
            },
            'confidence': min(1.0, metrics.liquidity_score)
        }

    def _assess_risk_factors(self, metrics: MarketMetrics, patterns: Optional[MicrostructurePatterns]) -> Dict:
        """Assess various risk factors"""
        risks = {
            'liquidity_risk': 'HIGH' if metrics.liquidity_score < 0.3 else 'MEDIUM' if metrics.liquidity_score < 0.6 else 'LOW',
            'spread_risk': 'HIGH' if metrics.spread_bps > 20 else 'MEDIUM' if metrics.spread_bps > 10 else 'LOW',
            'imbalance_risk': self._assess_imbalance_severity(metrics.imbalance),
            'large_order_risk': 'HIGH' if len(metrics.large_orders) > 3 else 'MEDIUM' if len(metrics.large_orders) > 1 else 'LOW'
        }

        if patterns:
            risks.update({
                'manipulation_risk': self._assess_manipulation_risk(patterns),
                'information_risk': 'HIGH' if patterns.vpin_score > 0.8 else 'MEDIUM' if patterns.vpin_score > 0.6 else 'LOW'
            })

        # Overall risk assessment
        risk_count = sum(1 for risk in risks.values() if risk == 'HIGH')
        if risk_count >= 3:
            risks['overall_risk'] = 'HIGH'
        elif risk_count >= 1:
            risks['overall_risk'] = 'MEDIUM'
        else:
            risks['overall_risk'] = 'LOW'

        return risks

    def _generate_execution_recommendation(self, metrics: MarketMetrics,
                                         patterns: Optional[MicrostructurePatterns]) -> Dict:
        """Generate execution recommendation"""
        favorability = self._assess_execution_favorability(metrics)

        if favorability == "EXCELLENT":
            return {
                'action': 'EXECUTE_IMMEDIATELY',
                'strategy': 'AGGRESSIVE',
                'confidence': 0.9,
                'reasoning': 'Excellent liquidity and tight spreads'
            }
        elif favorability == "GOOD":
            return {
                'action': 'EXECUTE_SOON',
                'strategy': 'MODERATE',
                'confidence': 0.7,
                'reasoning': 'Good market conditions with acceptable costs'
            }
        elif favorability == "FAIR":
            return {
                'action': 'WAIT_FOR_BETTER_CONDITIONS',
                'strategy': 'PASSIVE',
                'confidence': 0.5,
                'reasoning': 'Suboptimal conditions, consider waiting'
            }
        else:
            return {
                'action': 'AVOID_EXECUTION',
                'strategy': 'WAIT',
                'confidence': 0.8,
                'reasoning': 'Poor market conditions, high execution costs expected'
            }

    async def _get_execution_windows(self, symbol: str) -> List[Dict]:
        """Get optimal execution windows"""
        try:
            windows = await self.data_aggregator.get_optimal_execution_windows(
                symbol, Decimal('1.0'), 6
            )
            return [
                {
                    'hour': w.hour,
                    'avg_spread': w.avg_spread,
                    'avg_depth': float(w.avg_depth),
                    'cost_score': w.cost_score(),
                    'confidence': w.confidence
                }
                for w in windows[:3]  # Top 3 windows
            ]
        except Exception:
            return []

    def _calculate_processing_latency(self, market_data) -> float:
        """Calculate processing latency in milliseconds"""
        if market_data.cache_timestamp:
            return (datetime.utcnow() - market_data.cache_timestamp).total_seconds() * 1000
        return 0.0

    def _calculate_confidence_score(self, market_data) -> float:
        """Calculate overall confidence score for the analysis"""
        data_quality = self._calculate_data_quality_score(market_data)
        data_freshness = max(0, 1 - self._calculate_processing_latency(market_data) / 10000)  # Penalty for old data
        data_completeness = min(1.0, len(market_data.tick_history) / 100)

        return (data_quality + data_freshness + data_completeness) / 3

    def _fallback_analysis(self, symbol: str) -> Dict:
        """Provide fallback analysis when enhanced data is unavailable"""
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'status': 'FALLBACK_MODE',
            'data_quality': {'quality_score': 0.1},
            'execution_recommendation': {
                'action': 'USE_CAUTION',
                'strategy': 'CONSERVATIVE',
                'confidence': 0.2,
                'reasoning': 'Insufficient market data for analysis'
            },
            'risk_assessment': {'overall_risk': 'HIGH'},
            'confidence_score': 0.1
        }