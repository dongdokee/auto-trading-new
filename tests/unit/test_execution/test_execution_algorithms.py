# tests/unit/test_execution/test_execution_algorithms.py
import pytest
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from src.execution.execution_algorithms import ExecutionAlgorithms
from src.execution.models import Order, OrderSide, OrderUrgency


class TestExecutionAlgorithms:
    """ExecutionAlgorithms 클래스에 대한 TDD 테스트"""

    @pytest.fixture
    def algorithms(self):
        """테스트용 ExecutionAlgorithms 인스턴스"""
        return ExecutionAlgorithms()

    @pytest.fixture
    def sample_order(self):
        """테스트용 기본 주문"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("10.0"),
            urgency=OrderUrgency.MEDIUM,
            price=Decimal("50000.0")
        )

    @pytest.fixture
    def market_analysis(self):
        """테스트용 시장 분석 데이터"""
        return {
            'spread_bps': 2.0,
            'liquidity_score': 0.8,
            'avg_volume_1min': 5000.0,
            'best_bid': Decimal('50000.0'),
            'best_ask': Decimal('50001.0'),
            'top_5_liquidity': 25000.0,
            'daily_volume': 10000000,
            'volatility': 0.02,
            'imbalance': 0.1,
            'vwap_bid': 49999.5,
            'vwap_ask': 50001.5
        }

    @pytest.fixture
    def volume_profile(self):
        """테스트용 볼륨 프로파일"""
        return {
            'hourly_volumes': [1000, 1200, 800, 1500, 2000, 1800],  # 6시간 프로파일
            'current_hour': 2,
            'total_remaining_volume': 8300,
            'peak_hours': [4, 5],  # 4시, 5시가 피크
            'off_peak_hours': [2, 3]  # 2시, 3시가 저조
        }

    # TWAP Algorithm Tests
    @pytest.mark.asyncio
    async def test_should_execute_twap_with_optimal_duration_calculation(self, algorithms, sample_order, market_analysis):
        """TWAP이 최적 지속시간을 계산하여 실행되어야 함"""
        # Given
        algorithms.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('2.0'),
            'avg_price': Decimal('50000.5'),
            'commission': Decimal('40.0'),
            'status': 'FILLED'
        })

        # When
        result = await algorithms.execute_twap(sample_order, market_analysis)

        # Then
        assert result['strategy'] == 'TWAP'
        assert 'duration' in result
        assert result['duration'] >= 30  # At least 30 seconds
        assert result['duration'] <= 1800  # At most 30 minutes
        assert len(result['slices']) >= 5  # At least 5 slices
        assert result['total_filled'] <= sample_order.size

    @pytest.mark.asyncio
    async def test_should_adjust_twap_slices_based_on_market_volatility(self, algorithms, sample_order, market_analysis):
        """TWAP이 시장 변동성에 따라 슬라이스를 조정해야 함"""
        # Given - High volatility market
        high_vol_analysis = market_analysis.copy()
        high_vol_analysis['volatility'] = 0.05  # High volatility

        algorithms.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('1.0'),
            'avg_price': Decimal('50000.0'),
            'commission': Decimal('20.0'),
            'status': 'FILLED'
        })

        # When
        high_vol_result = await algorithms.execute_twap(sample_order, high_vol_analysis)

        # Then
        # High volatility should result in more slices (smaller slices)
        assert len(high_vol_result['slices']) >= 8
        assert high_vol_result['strategy'] == 'TWAP'

    @pytest.mark.asyncio
    async def test_should_implement_dynamic_twap_with_market_feedback(self, algorithms, sample_order, market_analysis):
        """동적 TWAP이 시장 피드백을 반영해야 함"""
        # Given
        algorithms.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('1.5'),
            'avg_price': Decimal('50001.0'),
            'commission': Decimal('30.0'),
            'status': 'FILLED'
        })

        # Mock market condition updates
        algorithms.get_updated_market_conditions = AsyncMock(side_effect=[
            market_analysis,  # First call
            {**market_analysis, 'spread_bps': 5.0, 'volatility': 0.03},  # Worsening conditions
            {**market_analysis, 'spread_bps': 1.5, 'volatility': 0.015}  # Improving conditions
        ])

        # When
        result = await algorithms.execute_dynamic_twap(sample_order, market_analysis)

        # Then
        assert result['strategy'] == 'DYNAMIC_TWAP'
        assert 'adaptations' in result
        assert len(result['adaptations']) > 0  # Should have made adaptations
        assert result['total_filled'] <= sample_order.size

    # VWAP Algorithm Tests
    @pytest.mark.asyncio
    async def test_should_execute_vwap_following_volume_profile(self, algorithms, sample_order, market_analysis, volume_profile):
        """VWAP이 볼륨 프로파일을 따라 실행되어야 함"""
        # Given
        algorithms.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('1.8'),
            'avg_price': Decimal('50000.2'),
            'commission': Decimal('36.0'),
            'status': 'FILLED'
        })

        # When
        result = await algorithms.execute_vwap(sample_order, market_analysis, volume_profile)

        # Then
        assert result['strategy'] == 'VWAP'
        assert 'volume_schedule' in result
        assert len(result['slices']) > 0

        # Should allocate more volume during peak hours
        volume_schedule = result['volume_schedule']
        peak_allocation = sum(volume_schedule[hour] for hour in volume_profile['peak_hours'])
        off_peak_allocation = sum(volume_schedule[hour] for hour in volume_profile['off_peak_hours'])
        assert peak_allocation > off_peak_allocation

    @pytest.mark.asyncio
    async def test_should_calculate_vwap_benchmark_correctly(self, algorithms, sample_order, market_analysis, volume_profile):
        """VWAP 벤치마크를 올바르게 계산해야 함"""
        # Given
        algorithms.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('2.0'),
            'avg_price': Decimal('50000.8'),
            'commission': Decimal('40.0'),
            'status': 'FILLED'
        })

        # When
        result = await algorithms.execute_vwap(sample_order, market_analysis, volume_profile)

        # Then
        assert 'vwap_benchmark' in result
        assert 'vwap_slippage' in result
        assert isinstance(result['vwap_benchmark'], (float, Decimal))
        assert isinstance(result['vwap_slippage'], (float, Decimal))

        # VWAP slippage should be reasonable
        assert abs(result['vwap_slippage']) < 0.01  # Less than 1%

    @pytest.mark.asyncio
    async def test_should_adjust_vwap_pace_based_on_execution_progress(self, algorithms, sample_order, market_analysis, volume_profile):
        """VWAP이 실행 진행률에 따라 페이스를 조정해야 함"""
        # Given
        fill_sequence = [
            {'filled_qty': Decimal('0.5'), 'avg_price': Decimal('50000.0')},  # Slow start
            {'filled_qty': Decimal('1.5'), 'avg_price': Decimal('50000.5')},  # Catch up
            {'filled_qty': Decimal('2.0'), 'avg_price': Decimal('50001.0')},  # Normal pace
        ]

        algorithms.place_order = AsyncMock(side_effect=[
            {**fill, 'commission': Decimal('20.0'), 'status': 'FILLED'} for fill in fill_sequence
        ])

        # When
        result = await algorithms.execute_adaptive_vwap(sample_order, market_analysis, volume_profile)

        # Then
        assert result['strategy'] == 'ADAPTIVE_VWAP'
        assert 'pace_adjustments' in result
        assert len(result['pace_adjustments']) > 0

    # Adaptive Algorithm Tests
    @pytest.mark.asyncio
    async def test_should_implement_adaptive_execution_with_multiple_signals(self, algorithms, sample_order, market_analysis):
        """적응형 실행이 다중 신호를 활용해야 함"""
        # Given
        algorithms.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('1.0'),
            'avg_price': Decimal('50000.0'),
            'commission': Decimal('20.0'),
            'status': 'FILLED'
        })

        # Mock signal generators
        algorithms.calculate_momentum_signal = MagicMock(return_value=0.3)  # Positive momentum
        algorithms.calculate_liquidity_signal = MagicMock(return_value=0.7)  # Good liquidity
        algorithms.calculate_volatility_signal = MagicMock(return_value=-0.2)  # Rising volatility

        # When
        result = await algorithms.execute_adaptive(sample_order, market_analysis)

        # Then
        assert result['strategy'] == 'ADAPTIVE'
        assert 'signals' in result
        assert 'momentum' in result['signals']
        assert 'liquidity' in result['signals']
        assert 'volatility' in result['signals']

    @pytest.mark.asyncio
    async def test_should_adapt_slice_size_based_on_market_conditions(self, algorithms, sample_order, market_analysis):
        """적응형 실행이 시장 조건에 따라 슬라이스 크기를 조정해야 함"""
        # Given
        changing_conditions = [
            market_analysis,  # Initial conditions
            {**market_analysis, 'top_5_liquidity': 50000.0, 'spread_bps': 1.5},  # Better liquidity
            {**market_analysis, 'top_5_liquidity': 10000.0, 'spread_bps': 8.0},  # Worse liquidity
        ]

        algorithms.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('2.0'),
            'avg_price': Decimal('50000.0'),
            'commission': Decimal('40.0'),
            'status': 'FILLED'
        })

        algorithms.get_updated_market_conditions = AsyncMock(side_effect=changing_conditions)

        # When
        result = await algorithms.execute_adaptive(sample_order, market_analysis)

        # Then
        assert result['strategy'] == 'ADAPTIVE'
        assert 'slice_adjustments' in result
        assert len(result['slice_adjustments']) > 0

    @pytest.mark.asyncio
    async def test_should_implement_participation_rate_control(self, algorithms, sample_order, market_analysis):
        """참여율 제어를 구현해야 함"""
        # Given
        target_participation_rate = 0.2  # 20% of market volume

        algorithms.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('0.8'),
            'avg_price': Decimal('50000.0'),
            'commission': Decimal('16.0'),
            'status': 'FILLED'
        })

        # When
        result = await algorithms.execute_participation_rate(
            sample_order, market_analysis, target_participation_rate
        )

        # Then
        assert result['strategy'] == 'PARTICIPATION_RATE'
        assert 'target_rate' in result
        assert 'actual_rate' in result
        assert result['target_rate'] == target_participation_rate

        # Actual rate should be close to target
        assert abs(result['actual_rate'] - target_participation_rate) < 0.05

    # Algorithm Performance Tests
    @pytest.mark.asyncio
    async def test_should_calculate_implementation_shortfall(self, algorithms, sample_order, market_analysis):
        """구현 부족분(Implementation Shortfall)을 계산해야 함"""
        # Given
        decision_price = Decimal('50000.0')
        algorithms.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('10.0'),
            'avg_price': Decimal('50005.0'),
            'commission': Decimal('200.0'),
            'status': 'FILLED'
        })

        # When
        result = await algorithms.execute_twap(sample_order, market_analysis)
        shortfall = algorithms.calculate_implementation_shortfall(
            result, decision_price, sample_order
        )

        # Then
        assert 'market_impact' in shortfall
        assert 'timing_cost' in shortfall
        assert 'commission_cost' in shortfall
        assert 'total_shortfall' in shortfall
        assert isinstance(shortfall['total_shortfall'], (int, float, Decimal))

    def test_should_optimize_slice_timing_with_market_microstructure(self, algorithms, market_analysis):
        """시장 미세구조를 고려한 슬라이스 타이밍 최적화"""
        # Given
        microstructure_data = {
            'bid_ask_spread_pattern': [2.0, 1.8, 1.5, 2.2, 3.0],  # Spread over time
            'volume_pattern': [100, 150, 200, 120, 80],  # Volume over time
            'volatility_pattern': [0.02, 0.018, 0.015, 0.025, 0.03]  # Volatility over time
        }

        # When
        optimal_timing = algorithms.optimize_slice_timing(
            slice_count=5,
            total_duration=300,  # 5 minutes
            microstructure_data=microstructure_data
        )

        # Then
        assert len(optimal_timing) == 5
        assert all(isinstance(t, (int, float)) for t in optimal_timing)
        assert sum(optimal_timing) <= 300  # Total time within limit

        # Should concentrate execution during favorable conditions
        # (low spread, high volume, low volatility periods)
        peak_volume_index = microstructure_data['volume_pattern'].index(
            max(microstructure_data['volume_pattern'])
        )
        assert optimal_timing[peak_volume_index] > 0

    @pytest.mark.asyncio
    async def test_should_handle_execution_algorithm_failures_gracefully(self, algorithms, sample_order, market_analysis):
        """실행 알고리즘 실패를 우아하게 처리해야 함"""
        # Given
        algorithms.place_order = AsyncMock(side_effect=[
            Exception("Network error"),
            {'filled_qty': Decimal('2.0'), 'avg_price': Decimal('50001.0'), 'commission': Decimal('40.0'), 'status': 'FILLED'}
        ])

        # When
        result = await algorithms.execute_twap_with_fallback(sample_order, market_analysis)

        # Then
        assert result['strategy'] == 'TWAP_WITH_FALLBACK'
        assert 'errors' in result
        assert 'recovery_actions' in result
        assert len(result['errors']) > 0
        assert result['total_filled'] > 0  # Should have recovered and filled some

    def test_should_validate_algorithm_parameters(self, algorithms):
        """알고리즘 매개변수를 검증해야 함"""
        # Given
        invalid_params = {
            'participation_rate': 1.5,  # > 1.0 (invalid)
            'max_slice_size': -1.0,     # negative (invalid)
            'min_slice_interval': 0     # zero (invalid)
        }

        # When & Then
        with pytest.raises(ValueError, match="Participation rate must be between 0 and 1"):
            algorithms.validate_participation_rate_params(invalid_params)

        with pytest.raises(ValueError, match="Slice size must be positive"):
            algorithms.validate_slice_size_params(invalid_params)

        with pytest.raises(ValueError, match="Slice interval must be positive"):
            algorithms.validate_timing_params(invalid_params)

    @pytest.mark.asyncio
    async def test_should_benchmark_algorithm_performance(self, algorithms, sample_order, market_analysis):
        """알고리즘 성능을 벤치마크해야 함"""
        # Given
        algorithms.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('10.0'),
            'avg_price': Decimal('50002.0'),
            'commission': Decimal('200.0'),
            'status': 'FILLED'
        })

        benchmarks = ['TWAP', 'VWAP', 'ARRIVAL_PRICE']

        # When
        result = await algorithms.execute_twap(sample_order, market_analysis)
        performance = algorithms.calculate_performance_metrics(result, benchmarks)

        # Then
        assert 'vs_twap' in performance
        assert 'vs_vwap' in performance
        assert 'vs_arrival_price' in performance
        assert 'sharpe_ratio' in performance
        assert 'information_ratio' in performance

    def test_should_provide_execution_analytics(self, algorithms):
        """실행 분석을 제공해야 함"""
        # Given
        execution_history = [
            {
                'timestamp': '2023-01-01T10:00:00',
                'strategy': 'TWAP',
                'filled_qty': Decimal('5.0'),
                'avg_price': Decimal('50000.0'),
                'slippage': 0.002
            },
            {
                'timestamp': '2023-01-01T10:05:00',
                'strategy': 'VWAP',
                'filled_qty': Decimal('3.0'),
                'avg_price': Decimal('50001.0'),
                'slippage': 0.001
            }
        ]

        # When
        analytics = algorithms.generate_execution_analytics(execution_history)

        # Then
        assert 'average_slippage' in analytics
        assert 'strategy_performance' in analytics
        assert 'volume_statistics' in analytics
        assert 'execution_efficiency' in analytics
        assert analytics['average_slippage'] > 0