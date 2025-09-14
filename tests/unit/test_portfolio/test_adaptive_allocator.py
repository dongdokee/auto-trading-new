"""
Tests for Adaptive Allocator

Tests the performance-based dynamic strategy allocation and rebalancing system.
Follows TDD methodology with comprehensive test coverage.
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import asdict
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.portfolio.adaptive_allocator import (
    AdaptiveAllocator,
    AllocationUpdate,
    AdaptiveConfig,
    PerformanceWindow,
    RebalanceRecommendation
)


class TestAdaptiveAllocatorInitialization:
    """Test adaptive allocator initialization and configuration"""

    def test_should_create_allocator_with_default_config(self):
        """Test adaptive allocator creation with default parameters"""
        allocator = AdaptiveAllocator()

        assert allocator.performance_lookback == 126  # 6 months
        assert allocator.rebalance_threshold == 0.05  # 5%
        assert allocator.min_rebalance_interval == 21  # 21 days
        assert allocator.max_strategy_weight == 0.6
        assert allocator.min_strategy_weight == 0.05
        assert allocator.decay_factor == 0.94

    def test_should_create_allocator_with_custom_config(self):
        """Test adaptive allocator with custom configuration"""
        config = AdaptiveConfig(
            performance_lookback=252,  # 1 year
            rebalance_threshold=0.03,  # 3%
            min_rebalance_interval=14,  # 2 weeks
            max_strategy_weight=0.7,
            min_strategy_weight=0.03,
            decay_factor=0.9
        )

        allocator = AdaptiveAllocator(config)

        assert allocator.performance_lookback == 252
        assert allocator.rebalance_threshold == 0.03
        assert allocator.min_rebalance_interval == 14
        assert allocator.max_strategy_weight == 0.7
        assert allocator.min_strategy_weight == 0.03
        assert allocator.decay_factor == 0.9

    def test_should_validate_config_parameters(self):
        """Test configuration parameter validation"""
        with pytest.raises(ValueError, match="Performance lookback must be positive"):
            AdaptiveConfig(performance_lookback=0)

        with pytest.raises(ValueError, match="Rebalance threshold must be positive"):
            AdaptiveConfig(rebalance_threshold=0)

        with pytest.raises(ValueError, match="Min rebalance interval must be positive"):
            AdaptiveConfig(min_rebalance_interval=0)

        with pytest.raises(ValueError, match="Max strategy weight must be between 0 and 1"):
            AdaptiveConfig(max_strategy_weight=1.5)

        with pytest.raises(ValueError, match="Min strategy weight must be positive"):
            AdaptiveConfig(min_strategy_weight=0)

        with pytest.raises(ValueError, match="Decay factor must be between 0 and 1"):
            AdaptiveConfig(decay_factor=2.0)


class TestPerformanceTracking:
    """Test strategy performance tracking"""

    @pytest.fixture
    def allocator(self):
        """Create allocator instance for testing"""
        return AdaptiveAllocator()

    @pytest.fixture
    def sample_performance_data(self):
        """Create sample strategy performance data"""
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=170)
        dates = pd.date_range(start_date, periods=150, freq='D')
        np.random.seed(42)

        # Create strategies with different performance patterns
        performance_data = {
            'TrendFollowing': {
                'returns': pd.Series(np.random.normal(0.0008, 0.02, 150), index=dates),
                'sharpe_ratio': pd.Series(np.random.normal(1.2, 0.3, 150), index=dates),
                'max_drawdown': pd.Series(np.random.uniform(0.02, 0.08, 150), index=dates)
            },
            'MeanReversion': {
                'returns': pd.Series(np.random.normal(0.0006, 0.015, 150), index=dates),
                'sharpe_ratio': pd.Series(np.random.normal(1.0, 0.25, 150), index=dates),
                'max_drawdown': pd.Series(np.random.uniform(0.01, 0.06, 150), index=dates)
            },
            'RangeTrading': {
                'returns': pd.Series(np.random.normal(0.0004, 0.01, 150), index=dates),
                'sharpe_ratio': pd.Series(np.random.normal(0.8, 0.2, 150), index=dates),
                'max_drawdown': pd.Series(np.random.uniform(0.005, 0.03, 150), index=dates)
            }
        }

        return performance_data

    def test_should_add_strategy_performance(self, allocator, sample_performance_data):
        """Test adding strategy performance data"""
        for strategy_name, data in sample_performance_data.items():
            allocator.add_strategy_performance(strategy_name, data)

        assert len(allocator.strategy_performance) == 3
        assert 'TrendFollowing' in allocator.strategy_performance
        assert 'MeanReversion' in allocator.strategy_performance
        assert 'RangeTrading' in allocator.strategy_performance

        # Check data integrity
        trend_data = allocator.strategy_performance['TrendFollowing']
        assert 'returns' in trend_data
        assert 'sharpe_ratio' in trend_data
        assert 'max_drawdown' in trend_data
        assert len(trend_data['returns']) == 150

    def test_should_validate_performance_data_format(self, allocator):
        """Test performance data format validation"""
        # Missing required fields
        with pytest.raises(ValueError, match="Performance data must contain 'returns'"):
            allocator.add_strategy_performance('TestStrategy', {'sharpe_ratio': []})

        # Invalid returns type
        with pytest.raises(TypeError, match="Returns must be a pandas Series"):
            allocator.add_strategy_performance('TestStrategy', {'returns': [1, 2, 3]})

    def test_should_calculate_performance_scores(self, allocator, sample_performance_data):
        """Test calculation of composite performance scores"""
        for strategy_name, data in sample_performance_data.items():
            allocator.add_strategy_performance(strategy_name, data)

        from datetime import datetime
        performance_scores = allocator._calculate_performance_scores(datetime.now())

        assert len(performance_scores) == 3
        for strategy, score in performance_scores.items():
            assert isinstance(score, float)
            assert not np.isnan(score)

        # Scores should reflect relative performance differences
        # All scores should be positive and show differentiation
        all_scores = list(performance_scores.values())
        assert all(score > 0 for score in all_scores)

        # Scores should show some variance (not all equal)
        score_variance = np.var(all_scores)
        assert score_variance > 0

    def test_should_apply_exponential_decay_to_performance(self, allocator, sample_performance_data):
        """Test exponential decay weighting of performance metrics"""
        for strategy_name, data in sample_performance_data.items():
            allocator.add_strategy_performance(strategy_name, data)

        from datetime import datetime
        # Calculate scores with different decay factors
        high_decay_scores = allocator._calculate_performance_scores(datetime.now(), decay_factor=0.99)
        low_decay_scores = allocator._calculate_performance_scores(datetime.now(), decay_factor=0.8)

        # Scores should be different (recent performance weighted differently)
        for strategy in high_decay_scores:
            assert high_decay_scores[strategy] != low_decay_scores[strategy]


class TestAllocationCalculation:
    """Test dynamic allocation calculation"""

    @pytest.fixture
    def allocator_with_data(self):
        """Create allocator with performance data"""
        allocator = AdaptiveAllocator()

        # Add performance data with clear performance differences
        # Use recent dates so they don't get filtered out
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        dates = pd.date_range(start_date, periods=100, freq='D')
        np.random.seed(42)

        # High performing strategy
        allocator.add_strategy_performance('HighPerformer', {
            'returns': pd.Series(np.random.normal(0.002, 0.015, 100), index=dates),
            'sharpe_ratio': pd.Series(np.random.normal(1.8, 0.1, 100), index=dates),
            'max_drawdown': pd.Series(np.random.uniform(0.01, 0.03, 100), index=dates)
        })

        # Medium performing strategy
        allocator.add_strategy_performance('MediumPerformer', {
            'returns': pd.Series(np.random.normal(0.001, 0.02, 100), index=dates),
            'sharpe_ratio': pd.Series(np.random.normal(1.0, 0.2, 100), index=dates),
            'max_drawdown': pd.Series(np.random.uniform(0.02, 0.05, 100), index=dates)
        })

        # Low performing strategy
        allocator.add_strategy_performance('LowPerformer', {
            'returns': pd.Series(np.random.normal(0.0003, 0.025, 100), index=dates),
            'sharpe_ratio': pd.Series(np.random.normal(0.5, 0.3, 100), index=dates),
            'max_drawdown': pd.Series(np.random.uniform(0.04, 0.08, 100), index=dates)
        })

        return allocator

    def test_should_calculate_adaptive_allocation(self, allocator_with_data):
        """Test calculation of performance-based allocation"""
        # Current equal allocation
        current_allocation = {'HighPerformer': 1/3, 'MediumPerformer': 1/3, 'LowPerformer': 1/3}

        allocation_update = allocator_with_data.calculate_allocation_update(current_allocation)

        assert isinstance(allocation_update, AllocationUpdate)
        assert len(allocation_update.new_weights) == 3

        # Weights should sum to 1
        total_weight = sum(allocation_update.new_weights.values())
        assert abs(total_weight - 1.0) < 1e-6

        # High performer should get higher allocation
        assert allocation_update.new_weights['HighPerformer'] > allocation_update.new_weights['LowPerformer']

        # All weights should be within bounds (with small tolerance for floating-point precision)
        for weight in allocation_update.new_weights.values():
            assert allocator_with_data.min_strategy_weight - 1e-10 <= weight <= allocator_with_data.max_strategy_weight + 1e-10

    def test_should_respect_allocation_constraints(self, allocator_with_data):
        """Test that allocation respects min/max weight constraints"""
        current_allocation = {'HighPerformer': 0.5, 'MediumPerformer': 0.3, 'LowPerformer': 0.2}

        allocation_update = allocator_with_data.calculate_allocation_update(current_allocation)

        # Check constraints (with small tolerance for floating-point precision)
        for strategy, weight in allocation_update.new_weights.items():
            assert weight >= allocator_with_data.min_strategy_weight - 1e-10
            assert weight <= allocator_with_data.max_strategy_weight + 1e-10

        # Total should equal 1
        total_weight = sum(allocation_update.new_weights.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_should_calculate_allocation_changes(self, allocator_with_data):
        """Test calculation of allocation changes"""
        current_allocation = {'HighPerformer': 0.2, 'MediumPerformer': 0.4, 'LowPerformer': 0.4}

        allocation_update = allocator_with_data.calculate_allocation_update(current_allocation)

        # Should have weight changes
        assert len(allocation_update.weight_changes) == 3

        # Changes should match difference between new and current
        for strategy in current_allocation:
            expected_change = allocation_update.new_weights[strategy] - current_allocation[strategy]
            assert abs(allocation_update.weight_changes[strategy] - expected_change) < 1e-10

    def test_should_calculate_turnover(self, allocator_with_data):
        """Test turnover calculation"""
        current_allocation = {'HighPerformer': 0.3, 'MediumPerformer': 0.3, 'LowPerformer': 0.4}

        allocation_update = allocator_with_data.calculate_allocation_update(current_allocation)

        # Turnover should be positive
        assert allocation_update.turnover >= 0

        # Turnover should be sum of absolute changes
        expected_turnover = sum(abs(change) for change in allocation_update.weight_changes.values())
        assert abs(allocation_update.turnover - expected_turnover) < 1e-10


class TestRebalanceDecision:
    """Test rebalancing decision logic"""

    @pytest.fixture
    def allocator_with_history(self):
        """Create allocator with rebalancing history"""
        allocator = AdaptiveAllocator()

        # Add some rebalancing history
        allocator.last_rebalance_date = datetime.now() - timedelta(days=30)

        return allocator

    def test_should_decide_when_to_rebalance(self, allocator_with_history):
        """Test rebalancing decision based on threshold and time"""
        current_allocation = {'Strategy1': 0.4, 'Strategy2': 0.6}
        target_allocation = {'Strategy1': 0.5, 'Strategy2': 0.5}

        # Large deviation - should rebalance
        large_deviation_target = {'Strategy1': 0.3, 'Strategy2': 0.7}
        should_rebalance_large = allocator_with_history._should_rebalance(
            current_allocation, large_deviation_target
        )
        assert should_rebalance_large is True

        # Small deviation - should not rebalance
        small_deviation_target = {'Strategy1': 0.42, 'Strategy2': 0.58}
        should_rebalance_small = allocator_with_history._should_rebalance(
            current_allocation, small_deviation_target
        )
        assert should_rebalance_small is False

    def test_should_respect_minimum_rebalance_interval(self):
        """Test minimum time interval between rebalances"""
        allocator = AdaptiveAllocator()

        # Recent rebalance
        allocator.last_rebalance_date = datetime.now() - timedelta(days=10)

        current_allocation = {'Strategy1': 0.3, 'Strategy2': 0.7}
        target_allocation = {'Strategy1': 0.6, 'Strategy2': 0.4}  # Large deviation

        # Should not rebalance due to time constraint
        should_rebalance = allocator._should_rebalance(current_allocation, target_allocation)
        assert should_rebalance is False

    def test_should_create_rebalance_recommendation(self, allocator_with_history):
        """Test creation of rebalance recommendations"""
        current_allocation = {'Strategy1': 0.3, 'Strategy2': 0.4, 'Strategy3': 0.3}
        target_allocation = {'Strategy1': 0.4, 'Strategy2': 0.3, 'Strategy3': 0.3}

        recommendation = allocator_with_history._create_rebalance_recommendation(
            current_allocation, target_allocation, urgency='HIGH'
        )

        assert isinstance(recommendation, RebalanceRecommendation)
        assert recommendation.urgency == 'HIGH'
        assert len(recommendation.trades) > 0

        # Check trades make sense
        for trade in recommendation.trades:
            assert trade['strategy'] in current_allocation
            assert trade['action'] in ['BUY', 'SELL']
            assert trade['amount'] >= 0


class TestTransactionCostAwareness:
    """Test transaction cost-aware rebalancing"""

    @pytest.fixture
    def cost_aware_allocator(self):
        """Create allocator with transaction costs"""
        config = AdaptiveConfig(transaction_cost_rate=0.002)  # 0.2% transaction cost
        return AdaptiveAllocator(config)

    def test_should_account_for_transaction_costs_in_rebalancing(self, cost_aware_allocator):
        """Test that transaction costs are considered in rebalancing decisions"""
        current_allocation = {'Strategy1': 0.4, 'Strategy2': 0.6}
        target_allocation = {'Strategy1': 0.45, 'Strategy2': 0.55}  # Small change

        # Without transaction costs, this might trigger rebalancing
        # With costs, it should not
        recommendation = cost_aware_allocator.get_rebalance_recommendation(
            current_allocation, target_allocation
        )

        # Should consider transaction costs in decision
        if recommendation.should_rebalance:
            # If rebalancing is recommended, benefit should outweigh costs
            expected_benefit = recommendation.expected_performance_improvement
            expected_cost = recommendation.estimated_transaction_cost
            assert expected_benefit > expected_cost

    def test_should_calculate_transaction_costs(self, cost_aware_allocator):
        """Test transaction cost calculation"""
        current_allocation = {'Strategy1': 0.3, 'Strategy2': 0.7}
        target_allocation = {'Strategy1': 0.6, 'Strategy2': 0.4}

        turnover = sum(abs(target_allocation[s] - current_allocation[s]) for s in current_allocation)
        expected_cost = turnover * cost_aware_allocator.transaction_cost_rate

        calculated_cost = cost_aware_allocator._calculate_transaction_cost(
            current_allocation, target_allocation
        )

        assert abs(calculated_cost - expected_cost) < 1e-10

    def test_should_optimize_rebalancing_frequency(self, cost_aware_allocator):
        """Test optimization of rebalancing frequency based on costs"""
        # High transaction costs should reduce rebalancing frequency
        high_cost_config = AdaptiveConfig(transaction_cost_rate=0.01)  # 1%
        high_cost_allocator = AdaptiveAllocator(high_cost_config)

        # Low transaction costs should increase rebalancing frequency
        low_cost_config = AdaptiveConfig(transaction_cost_rate=0.0001)  # 0.01%
        low_cost_allocator = AdaptiveAllocator(low_cost_config)

        current_allocation = {'Strategy1': 0.45, 'Strategy2': 0.55}
        target_allocation = {'Strategy1': 0.52, 'Strategy2': 0.48}

        high_cost_rec = high_cost_allocator.get_rebalance_recommendation(
            current_allocation, target_allocation
        )

        low_cost_rec = low_cost_allocator.get_rebalance_recommendation(
            current_allocation, target_allocation
        )

        # Low cost allocator should be more likely to recommend rebalancing
        if high_cost_rec.should_rebalance != low_cost_rec.should_rebalance:
            assert low_cost_rec.should_rebalance is True


class TestRiskAdjustedAllocation:
    """Test risk-adjusted allocation strategies"""

    @pytest.fixture
    def risk_aware_allocator(self):
        """Create allocator with risk-adjusted performance metrics"""
        config = AdaptiveConfig(
            risk_adjustment_factor=0.5,  # 50% risk adjustment
            max_strategy_volatility=0.3   # 30% max volatility
        )
        return AdaptiveAllocator(config)

    def test_should_adjust_allocation_for_risk(self, risk_aware_allocator):
        """Test risk-adjusted allocation calculation"""
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        dates = pd.date_range(start_date, periods=100, freq='D')
        np.random.seed(42)

        # High return, high risk strategy
        risk_aware_allocator.add_strategy_performance('HighRiskStrategy', {
            'returns': pd.Series(np.random.normal(0.003, 0.04, 100), index=dates),  # High vol
            'sharpe_ratio': pd.Series(np.random.normal(0.8, 0.1, 100), index=dates),
            'max_drawdown': pd.Series(np.random.uniform(0.05, 0.15, 100), index=dates)
        })

        # Medium return, low risk strategy
        risk_aware_allocator.add_strategy_performance('LowRiskStrategy', {
            'returns': pd.Series(np.random.normal(0.001, 0.01, 100), index=dates),  # Low vol
            'sharpe_ratio': pd.Series(np.random.normal(1.2, 0.1, 100), index=dates),
            'max_drawdown': pd.Series(np.random.uniform(0.01, 0.03, 100), index=dates)
        })

        current_allocation = {'HighRiskStrategy': 0.5, 'LowRiskStrategy': 0.5}
        allocation_update = risk_aware_allocator.calculate_allocation_update(current_allocation)

        # Low risk strategy should get higher allocation due to risk adjustment
        assert allocation_update.new_weights['LowRiskStrategy'] > allocation_update.new_weights['HighRiskStrategy']

    def test_should_respect_volatility_constraints(self, risk_aware_allocator):
        """Test volatility constraint enforcement"""
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        dates = pd.date_range(start_date, periods=100, freq='D')
        np.random.seed(42)

        # Extremely high volatility strategy
        risk_aware_allocator.add_strategy_performance('ExtremeVolStrategy', {
            'returns': pd.Series(np.random.normal(0.002, 0.1, 100), index=dates),  # 10% daily vol
            'sharpe_ratio': pd.Series(np.random.normal(0.5, 0.1, 100), index=dates),
            'max_drawdown': pd.Series(np.random.uniform(0.2, 0.5, 100), index=dates)
        })

        # Normal volatility strategy
        risk_aware_allocator.add_strategy_performance('NormalVolStrategy', {
            'returns': pd.Series(np.random.normal(0.001, 0.02, 100), index=dates),
            'sharpe_ratio': pd.Series(np.random.normal(1.0, 0.1, 100), index=dates),
            'max_drawdown': pd.Series(np.random.uniform(0.02, 0.05, 100), index=dates)
        })

        current_allocation = {'ExtremeVolStrategy': 0.5, 'NormalVolStrategy': 0.5}
        allocation_update = risk_aware_allocator.calculate_allocation_update(current_allocation)

        # Extreme volatility strategy should get lower allocation due to risk adjustment
        # But still within constraints
        assert allocation_update.new_weights['ExtremeVolStrategy'] >= risk_aware_allocator.min_strategy_weight - 1e-10
        assert allocation_update.new_weights['NormalVolStrategy'] >= allocation_update.new_weights['ExtremeVolStrategy']


class TestAllocationUpdate:
    """Test AllocationUpdate data structure"""

    def test_should_create_allocation_update_with_required_fields(self):
        """Test AllocationUpdate creation"""
        new_weights = {'Strategy1': 0.4, 'Strategy2': 0.6}
        weight_changes = {'Strategy1': 0.1, 'Strategy2': -0.1}

        allocation_update = AllocationUpdate(
            new_weights=new_weights,
            weight_changes=weight_changes,
            turnover=0.2,
            confidence_score=0.8,
            expected_improvement=0.05,
            timestamp=datetime.now()
        )

        assert allocation_update.new_weights == new_weights
        assert allocation_update.weight_changes == weight_changes
        assert allocation_update.turnover == 0.2
        assert allocation_update.confidence_score == 0.8
        assert allocation_update.expected_improvement == 0.05

    def test_should_validate_allocation_update_consistency(self):
        """Test allocation update internal consistency"""
        new_weights = {'Strategy1': 0.3, 'Strategy2': 0.7}
        weight_changes = {'Strategy1': 0.1, 'Strategy2': -0.1}
        turnover = abs(weight_changes['Strategy1']) + abs(weight_changes['Strategy2'])

        allocation_update = AllocationUpdate(
            new_weights=new_weights,
            weight_changes=weight_changes,
            turnover=turnover,
            confidence_score=0.7,
            expected_improvement=0.02,
            timestamp=datetime.now()
        )

        # Weights should sum to 1
        assert abs(sum(allocation_update.new_weights.values()) - 1.0) < 1e-10

        # Turnover should match sum of absolute changes
        expected_turnover = sum(abs(change) for change in allocation_update.weight_changes.values())
        assert abs(allocation_update.turnover - expected_turnover) < 1e-10


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_should_handle_single_strategy_portfolio(self):
        """Test allocation with single strategy"""
        allocator = AdaptiveAllocator()

        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=70)
        dates = pd.date_range(start_date, periods=50, freq='D')
        allocator.add_strategy_performance('OnlyStrategy', {
            'returns': pd.Series(np.random.normal(0.001, 0.02, 50), index=dates),
            'sharpe_ratio': pd.Series(np.random.normal(1.0, 0.1, 50), index=dates),
            'max_drawdown': pd.Series(np.random.uniform(0.01, 0.05, 50), index=dates)
        })

        current_allocation = {'OnlyStrategy': 1.0}
        allocation_update = allocator.calculate_allocation_update(current_allocation)

        # Should maintain full allocation to single strategy
        assert allocation_update.new_weights['OnlyStrategy'] == 1.0
        assert allocation_update.turnover == 0.0

    def test_should_handle_poor_performance_across_all_strategies(self):
        """Test handling when all strategies perform poorly"""
        allocator = AdaptiveAllocator()

        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=70)
        dates = pd.date_range(start_date, periods=50, freq='D')

        # All strategies with poor performance
        for i, strategy_name in enumerate(['Strategy1', 'Strategy2', 'Strategy3']):
            allocator.add_strategy_performance(strategy_name, {
                'returns': pd.Series(np.random.normal(-0.001, 0.03, 50), index=dates),  # Negative returns
                'sharpe_ratio': pd.Series(np.random.normal(-0.5, 0.1, 50), index=dates),  # Negative Sharpe
                'max_drawdown': pd.Series(np.random.uniform(0.1, 0.2, 50), index=dates)   # High drawdown
            })

        current_allocation = {'Strategy1': 1/3, 'Strategy2': 1/3, 'Strategy3': 1/3}
        allocation_update = allocator.calculate_allocation_update(current_allocation)

        # Should still produce valid allocation (equal weights as fallback)
        assert len(allocation_update.new_weights) == 3
        assert abs(sum(allocation_update.new_weights.values()) - 1.0) < 1e-6

        # All weights should be within bounds (with small tolerance for floating-point precision)
        for weight in allocation_update.new_weights.values():
            assert allocator.min_strategy_weight - 1e-10 <= weight <= allocator.max_strategy_weight + 1e-10

    def test_should_handle_insufficient_performance_data(self):
        """Test handling of insufficient performance history"""
        allocator = AdaptiveAllocator()

        # Very short performance history
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        dates = pd.date_range(start_date, periods=5, freq='D')
        allocator.add_strategy_performance('Strategy1', {
            'returns': pd.Series(np.random.normal(0.001, 0.02, 5), index=dates),
            'sharpe_ratio': pd.Series(np.random.normal(1.0, 0.1, 5), index=dates),
            'max_drawdown': pd.Series(np.random.uniform(0.01, 0.03, 5), index=dates)
        })

        current_allocation = {'Strategy1': 1.0}

        # Should handle gracefully
        allocation_update = allocator.calculate_allocation_update(current_allocation)
        assert isinstance(allocation_update, AllocationUpdate)
        assert allocation_update.confidence_score <= 0.5  # Low confidence due to limited data

    def test_should_handle_extreme_weight_constraints(self):
        """Test handling of extreme weight constraints"""
        # Very restrictive constraints
        config = AdaptiveConfig(
            max_strategy_weight=0.35,  # No strategy can exceed 35%
            min_strategy_weight=0.15   # All strategies must have at least 15%
        )
        allocator = AdaptiveAllocator(config)

        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=70)
        dates = pd.date_range(start_date, periods=50, freq='D')

        # Add 4 strategies (should be challenging to satisfy constraints)
        for i, strategy_name in enumerate(['S1', 'S2', 'S3', 'S4']):
            allocator.add_strategy_performance(strategy_name, {
                'returns': pd.Series(np.random.normal(0.001 + i * 0.0005, 0.02, 50), index=dates),
                'sharpe_ratio': pd.Series(np.random.normal(1.0 + i * 0.2, 0.1, 50), index=dates),
                'max_drawdown': pd.Series(np.random.uniform(0.01, 0.05, 50), index=dates)
            })

        current_allocation = {'S1': 0.25, 'S2': 0.25, 'S3': 0.25, 'S4': 0.25}
        allocation_update = allocator.calculate_allocation_update(current_allocation)

        # Should respect constraints (with small tolerance for floating-point precision)
        for weight in allocation_update.new_weights.values():
            assert 0.15 - 1e-10 <= weight <= 0.35 + 1e-10

        # Should sum to 1
        assert abs(sum(allocation_update.new_weights.values()) - 1.0) < 1e-6