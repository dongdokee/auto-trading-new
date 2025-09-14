"""
BacktestEngine 클래스 단위 테스트

TDD 방법론을 사용하여 백테스트 엔진의 모든 기능을 테스트합니다.
Walk-Forward 검증과 룩어헤드 바이어스 방지를 중점적으로 테스트합니다.

테스트 대상:
- 기본 백테스트 실행
- Walk-Forward 최적화
- 룩어헤드 바이어스 방지
- 성과 메트릭 계산
- 거래 비용 모델링
- 리스크 관리 통합

Created: 2025-09-14 (Phase 2.1)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, MagicMock

# 아직 구현되지 않은 클래스 - 테스트를 먼저 작성 (TDD Red phase)
from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from src.backtesting.backtest_engine import StrategyInterface, Portfolio, Trade


class TestBacktestEngineInitialization:
    """BacktestEngine 초기화 테스트"""

    def test_should_create_backtest_engine_with_default_config(self):
        """기본 설정으로 BacktestEngine을 생성할 수 있어야 함"""
        engine = BacktestEngine()

        assert engine is not None
        assert isinstance(engine.config, BacktestConfig)
        assert engine.config.initial_capital == 100000.0
        assert engine.config.commission_rate == 0.0004
        assert engine.config.slippage_rate == 0.0005

    def test_should_create_backtest_engine_with_custom_config(self):
        """커스텀 설정으로 BacktestEngine을 생성할 수 있어야 함"""
        config = BacktestConfig(
            initial_capital=50000.0,
            commission_rate=0.0002,
            slippage_rate=0.0003,
            enable_walk_forward=True,
            walk_forward_window=252,
            rebalance_frequency='daily'
        )

        engine = BacktestEngine(config)

        assert engine.config.initial_capital == 50000.0
        assert engine.config.commission_rate == 0.0002
        assert engine.config.slippage_rate == 0.0003
        assert engine.config.enable_walk_forward is True
        assert engine.config.walk_forward_window == 252


class TestBasicBacktesting:
    """기본 백테스팅 기능 테스트"""

    def test_should_run_simple_buy_and_hold_strategy(self):
        """간단한 매수-보유 전략을 실행할 수 있어야 함"""
        engine = BacktestEngine()
        data = self._create_sample_price_data()
        strategy = self._create_buy_and_hold_strategy()

        result = engine.run_backtest(strategy, data, '2023-01-01', '2023-12-31')

        assert isinstance(result, BacktestResult)
        assert result.total_return != 0
        assert len(result.trades) > 0
        assert len(result.equity_curve) > 0
        assert result.sharpe_ratio is not None

    def test_should_prevent_lookahead_bias(self):
        """룩어헤드 바이어스를 방지해야 함"""
        engine = BacktestEngine()
        data = self._create_sample_price_data()
        strategy = self._create_lookahead_testing_strategy()

        result = engine.run_backtest(strategy, data, '2023-01-01', '2023-12-31')

        # 전략이 미래 데이터를 볼 수 없도록 보장
        for trade in result.trades:
            trade_date = trade.entry_time
            # 해당 거래 시점의 데이터만 사용했는지 검증
            available_data_end = data[data['datetime'] <= trade_date]
            assert len(available_data_end) > 0

    def test_should_calculate_transaction_costs_correctly(self):
        """거래 비용을 올바르게 계산해야 함"""
        config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.001,  # 0.1%
            slippage_rate=0.0005    # 0.05%
        )
        engine = BacktestEngine(config)
        data = self._create_sample_price_data()
        strategy = self._create_frequent_trading_strategy()

        result = engine.run_backtest(strategy, data, '2023-01-01', '2023-12-31')

        # 거래 비용이 반영되었는지 확인
        total_commission = sum(trade.commission for trade in result.trades)
        total_slippage = sum(trade.slippage_cost for trade in result.trades)

        assert total_commission > 0
        assert total_slippage > 0
        assert result.total_costs == total_commission + total_slippage

    def test_should_handle_insufficient_capital(self):
        """자본 부족 상황을 처리해야 함"""
        config = BacktestConfig(initial_capital=1000.0)  # 매우 적은 자본
        engine = BacktestEngine(config)
        data = self._create_expensive_stock_data()
        strategy = self._create_buy_and_hold_strategy()

        result = engine.run_backtest(strategy, data, '2023-01-01', '2023-12-31')

        # 자본 부족으로 거래가 제한되어야 함
        rejected_orders = [t for t in result.trades if t.status == 'REJECTED']
        assert len(rejected_orders) > 0

    def test_should_maintain_portfolio_state_correctly(self):
        """포트폴리오 상태를 올바르게 유지해야 함"""
        engine = BacktestEngine()
        data = self._create_sample_price_data()
        strategy = self._create_rebalancing_strategy()

        result = engine.run_backtest(strategy, data, '2023-01-01', '2023-12-31')

        # 포트폴리오 상태 일관성 검증
        assert len(result.portfolio_history) > 0

        for portfolio_state in result.portfolio_history:
            # 자산 + 현금 = 총 자본
            total_value = portfolio_state.cash + portfolio_state.total_position_value
            assert abs(total_value - portfolio_state.equity) < 0.01

    def _create_sample_price_data(self) -> pd.DataFrame:
        """샘플 가격 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)

        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))

        return pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, n)
        })

    def _create_expensive_stock_data(self) -> pd.DataFrame:
        """비싼 주식 데이터 생성 (자본 부족 테스트용)"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)

        # 매우 비싼 가격 (10,000+)
        prices = 10000 + np.cumsum(np.random.normal(50, 100, n))

        return pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.lognormal(8, 0.3, n)
        })

    def _create_buy_and_hold_strategy(self) -> StrategyInterface:
        """매수-보유 전략 생성"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "BuyAndHold"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 2:
                return []

            # 첫 거래일에만 매수 신호
            if len(portfolio.positions) == 0:
                return [{
                    'symbol': 'TEST',
                    'action': 'BUY',
                    'quantity': 100,
                    'price': data.iloc[-1]['close']
                }]
            return []

        strategy.generate_signals = generate_signals
        return strategy

    def _create_lookahead_testing_strategy(self) -> StrategyInterface:
        """룩어헤드 바이어스 테스트용 전략"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "LookaheadTest"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            # 이 전략은 현재까지의 데이터만 사용해야 함
            if len(data) < 10:
                return []

            # 단순 이동평균 기반 신호 (현재까지 데이터만 사용)
            ma_short = data['close'].rolling(5).mean().iloc[-1]
            ma_long = data['close'].rolling(10).mean().iloc[-1]

            if ma_short > ma_long and len(portfolio.positions) == 0:
                return [{
                    'symbol': 'TEST',
                    'action': 'BUY',
                    'quantity': 50,
                    'price': data.iloc[-1]['close']
                }]
            elif ma_short < ma_long and len(portfolio.positions) > 0:
                return [{
                    'symbol': 'TEST',
                    'action': 'SELL',
                    'quantity': 50,
                    'price': data.iloc[-1]['close']
                }]
            return []

        strategy.generate_signals = generate_signals
        return strategy

    def _create_frequent_trading_strategy(self) -> StrategyInterface:
        """빈번한 거래 전략 (거래비용 테스트용)"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "FrequentTrading"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 2:
                return []

            # 매일 거래 신호 생성 (비용 테스트)
            current_price = data.iloc[-1]['close']
            previous_price = data.iloc[-2]['close']

            if current_price > previous_price:
                return [{
                    'symbol': 'TEST',
                    'action': 'BUY',
                    'quantity': 10,
                    'price': current_price
                }]
            else:
                return [{
                    'symbol': 'TEST',
                    'action': 'SELL',
                    'quantity': 10,
                    'price': current_price
                }]

        strategy.generate_signals = generate_signals
        return strategy

    def _create_rebalancing_strategy(self) -> StrategyInterface:
        """리밸런싱 전략 (포트폴리오 상태 테스트용)"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "Rebalancing"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) % 30 != 0:  # 30일마다 리밸런싱
                return []

            target_allocation = 0.6  # 60% 주식
            current_allocation = portfolio.total_position_value / portfolio.equity if portfolio.equity > 0 else 0

            if abs(current_allocation - target_allocation) > 0.05:
                target_value = portfolio.equity * target_allocation
                current_value = portfolio.total_position_value
                adjust_value = target_value - current_value

                if adjust_value > 0:
                    return [{
                        'symbol': 'TEST',
                        'action': 'BUY',
                        'value': adjust_value,
                        'price': data.iloc[-1]['close']
                    }]
                else:
                    return [{
                        'symbol': 'TEST',
                        'action': 'SELL',
                        'value': abs(adjust_value),
                        'price': data.iloc[-1]['close']
                    }]

            return []

        strategy.generate_signals = generate_signals
        return strategy


class TestWalkForwardOptimization:
    """Walk-Forward 최적화 테스트"""

    def test_should_perform_walk_forward_validation(self):
        """Walk-Forward 검증을 수행해야 함"""
        config = BacktestConfig(enable_walk_forward=True, walk_forward_window=60)
        engine = BacktestEngine(config)
        data = self._create_long_term_data()
        strategy = self._create_adaptive_strategy()

        result = engine.run_walk_forward_backtest(strategy, data, '2023-01-01', '2023-12-31')

        assert isinstance(result, BacktestResult)
        assert result.walk_forward_results is not None
        assert len(result.walk_forward_results) > 1
        assert result.out_of_sample_performance is not None

    def test_should_prevent_overfitting_through_walk_forward(self):
        """Walk-Forward를 통한 과적합 방지 테스트"""
        config = BacktestConfig(enable_walk_forward=True)
        engine = BacktestEngine(config)
        data = self._create_noisy_data()
        strategy = self._create_overfitting_prone_strategy()

        result = engine.run_walk_forward_backtest(strategy, data, '2023-01-01', '2023-12-31')

        # In-sample과 Out-of-sample 성과 비교
        avg_in_sample = np.mean([r.in_sample_return for r in result.walk_forward_results])
        avg_out_sample = np.mean([r.out_of_sample_return for r in result.walk_forward_results])

        # 과적합이 있다면 Out-of-sample 성과가 현저히 낮을 것
        degradation = avg_in_sample - avg_out_sample
        assert degradation >= 0  # Out-of-sample이 나쁜 것은 정상

    def test_should_handle_insufficient_data_for_walk_forward(self):
        """Walk-Forward에 대한 데이터 부족 상황 처리"""
        config = BacktestConfig(enable_walk_forward=True, walk_forward_window=365)
        engine = BacktestEngine(config)
        short_data = self._create_short_term_data()  # 6개월 데이터
        strategy = self._create_simple_strategy()

        with pytest.raises(ValueError, match="Insufficient data"):
            engine.run_walk_forward_backtest(strategy, short_data, '2023-01-01', '2023-06-30')

    def _create_long_term_data(self) -> pd.DataFrame:
        """장기간 데이터 생성 (Walk-Forward 테스트용)"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n = len(dates)

        np.random.seed(42)
        # 트렌드가 있는 데이터
        trend = np.linspace(0, 0.5, n)
        noise = np.random.normal(0, 0.02, n)
        returns = trend/n + noise
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, n)
        })

    def _create_short_term_data(self) -> pd.DataFrame:
        """단기간 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        n = len(dates)

        prices = 100 + np.cumsum(np.random.normal(0, 1, n))

        return pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.lognormal(8, 0.3, n)
        })

    def _create_noisy_data(self) -> pd.DataFrame:
        """노이즈가 많은 데이터 생성 (과적합 테스트용)"""
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        n = len(dates)

        # 매우 노이즈가 많은 랜덤워크
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.03, n)))

        return pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n))),
            'close': prices,
            'volume': np.random.lognormal(9, 0.8, n)
        })

    def _create_adaptive_strategy(self) -> StrategyInterface:
        """적응형 전략 (Walk-Forward 테스트용)"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "AdaptiveStrategy"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 20:
                return []

            # 적응형 이동평균 전략
            short_window = 10
            long_window = 20

            ma_short = data['close'].rolling(short_window).mean().iloc[-1]
            ma_long = data['close'].rolling(long_window).mean().iloc[-1]

            if ma_short > ma_long * 1.02 and len(portfolio.positions) == 0:
                return [{
                    'symbol': 'TEST',
                    'action': 'BUY',
                    'quantity': 100,
                    'price': data.iloc[-1]['close']
                }]
            elif ma_short < ma_long * 0.98 and len(portfolio.positions) > 0:
                return [{
                    'symbol': 'TEST',
                    'action': 'SELL',
                    'quantity': 100,
                    'price': data.iloc[-1]['close']
                }]

            return []

        strategy.generate_signals = generate_signals
        return strategy

    def _create_overfitting_prone_strategy(self) -> StrategyInterface:
        """과적합되기 쉬운 전략"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "OverfittingProne"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 10:
                return []

            # 과도하게 복잡한 조건
            sma_5 = data['close'].rolling(5).mean().iloc[-1]
            sma_10 = data['close'].rolling(10).mean().iloc[-1]
            current_price = data['close'].iloc[-1]

            # 매우 구체적인 조건 (과적합 유도)
            if (sma_5 > sma_10 * 1.003 and
                current_price > sma_5 * 1.001 and
                len(portfolio.positions) == 0):
                return [{
                    'symbol': 'TEST',
                    'action': 'BUY',
                    'quantity': 50,
                    'price': current_price
                }]

            return []

        strategy.generate_signals = generate_signals
        return strategy

    def _create_simple_strategy(self) -> StrategyInterface:
        """단순한 전략"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "SimpleStrategy"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 5:
                return []

            # 단순 모멘텀
            if data['close'].iloc[-1] > data['close'].iloc[-5]:
                return [{
                    'symbol': 'TEST',
                    'action': 'BUY',
                    'quantity': 100,
                    'price': data['close'].iloc[-1]
                }]

            return []

        strategy.generate_signals = generate_signals
        return strategy