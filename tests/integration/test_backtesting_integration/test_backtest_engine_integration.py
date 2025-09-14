"""
BacktestEngine 통합 테스트

BacktestEngine의 핵심 기능을 종합적으로 테스트합니다.
실제 사용 시나리오를 기반으로 한 통합 테스트입니다.

Created: 2025-09-14 (Phase 2.1)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock

from src.backtesting.backtest_engine import (
    BacktestEngine, BacktestConfig, BacktestResult,
    StrategyInterface, Portfolio
)


class TestBacktestEngineIntegration:
    """BacktestEngine 통합 테스트"""

    def test_should_execute_complete_backtest_workflow(self):
        """완전한 백테스트 워크플로우 실행 테스트"""
        # 엔진 설정
        config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        engine = BacktestEngine(config)

        # 테스트 데이터와 전략
        data = self._create_realistic_market_data()
        strategy = self._create_momentum_strategy()

        # 백테스트 실행
        result = engine.run_backtest(strategy, data, '2023-01-01', '2023-12-31')

        # 기본 검증
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "Momentum"
        assert result.initial_capital == 100000.0
        assert result.total_trades > 0
        assert len(result.equity_curve) > 0
        assert len(result.trades) > 0

        # 성과 메트릭 검증
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert 0.0 <= result.win_rate <= 1.0

    def test_should_handle_transaction_costs_realistically(self):
        """현실적인 거래 비용 처리 테스트"""
        config = BacktestConfig(
            initial_capital=50000.0,
            commission_rate=0.002,  # 높은 수수료
            slippage_rate=0.001     # 높은 슬리피지
        )
        engine = BacktestEngine(config)

        data = self._create_realistic_market_data()
        strategy = self._create_high_frequency_strategy()

        result = engine.run_backtest(strategy, data, '2023-01-01', '2023-12-31')

        # 거래 비용이 적절히 반영되었는지 확인
        assert result.total_costs > 0
        total_calculated_costs = sum(
            trade.commission + trade.slippage_cost
            for trade in result.trades if trade.status == 'EXECUTED'
        )
        assert abs(result.total_costs - total_calculated_costs) < 0.01

        # 높은 거래 빈도로 인한 비용 부담 확인
        cost_ratio = result.total_costs / result.initial_capital
        assert cost_ratio > 0.01  # 1% 이상의 비용

    def test_should_prevent_lookahead_bias_strictly(self):
        """룩어헤드 바이어스 엄격 방지 테스트"""
        engine = BacktestEngine()
        data = self._create_trend_data_with_future_knowledge()
        strategy = self._create_perfect_foresight_strategy()

        result = engine.run_backtest(strategy, data, '2023-01-01', '2023-12-31')

        # 완벽한 예측 전략도 미래 정보 없이는 완벽할 수 없음
        assert result.total_return < 2.0  # 200% 미만 수익 (제한적)
        assert len([t for t in result.trades if t.status == 'REJECTED']) == 0

    def test_should_maintain_portfolio_consistency(self):
        """포트폴리오 일관성 유지 테스트"""
        engine = BacktestEngine()
        data = self._create_realistic_market_data()
        strategy = self._create_position_management_strategy()

        result = engine.run_backtest(strategy, data, '2023-01-01', '2023-12-31')

        # 각 시점의 포트폴리오 상태 일관성 확인
        for i, portfolio in enumerate(result.portfolio_history):
            if i == 0:
                continue

            # 자산 = 현금 + 포지션 가치
            expected_equity = portfolio.cash + portfolio.total_position_value
            assert abs(expected_equity - portfolio.equity) < 0.01

            # 포지션 가치가 음수가 아님
            assert portfolio.total_position_value >= 0
            assert portfolio.cash >= -0.01  # 약간의 허용 오차

    def test_should_handle_walk_forward_analysis(self):
        """Walk-Forward 분석 처리 테스트"""
        config = BacktestConfig(
            enable_walk_forward=True,
            walk_forward_window=120,  # 4개월
            out_of_sample_window=30   # 1개월
        )
        engine = BacktestEngine(config)

        # 충분한 기간의 데이터
        data = self._create_long_term_market_data()
        strategy = self._create_adaptive_strategy()

        result = engine.run_walk_forward_backtest(strategy, data, '2022-01-01', '2023-12-31')

        # Walk-Forward 결과 검증
        assert result.walk_forward_results is not None
        assert len(result.walk_forward_results) > 1
        assert result.out_of_sample_performance is not None

        # 각 Walk-Forward 기간 검증
        for wf_result in result.walk_forward_results:
            assert wf_result.in_sample_start < wf_result.in_sample_end
            assert wf_result.out_of_sample_start < wf_result.out_of_sample_end
            assert wf_result.in_sample_end <= wf_result.out_of_sample_start

        # Out-of-sample 메트릭 검증
        oos_metrics = result.out_of_sample_performance
        assert 'avg_out_of_sample_return' in oos_metrics
        assert 'performance_degradation' in oos_metrics
        assert oos_metrics['periods_count'] == len(result.walk_forward_results)

    def test_should_handle_edge_cases_gracefully(self):
        """경계 조건 우아한 처리 테스트"""
        engine = BacktestEngine()

        # 빈 데이터
        empty_data = pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        strategy = self._create_simple_strategy()

        with pytest.raises(ValueError, match="No data available"):
            engine.run_backtest(strategy, empty_data, '2023-01-01', '2023-12-31')

        # 단일 행 데이터
        single_row_data = self._create_realistic_market_data().iloc[:1]
        result = engine.run_backtest(strategy, single_row_data, '2023-01-01', '2023-01-01')
        assert isinstance(result, BacktestResult)

        # 자본 부족 시나리오
        poor_config = BacktestConfig(initial_capital=100.0)
        poor_engine = BacktestEngine(poor_config)
        expensive_data = self._create_expensive_asset_data()

        result = poor_engine.run_backtest(strategy, expensive_data, '2023-01-01', '2023-12-31')
        rejected_trades = [t for t in result.trades if t.status == 'REJECTED']
        assert len(rejected_trades) > 0

    def test_should_provide_comprehensive_trade_details(self):
        """포괄적인 거래 상세 정보 제공 테스트"""
        engine = BacktestEngine()
        data = self._create_realistic_market_data()
        strategy = self._create_detailed_tracking_strategy()

        result = engine.run_backtest(strategy, data, '2023-01-01', '2023-12-31')

        # 거래 기록 상세 검증
        for trade in result.trades:
            if trade.status == 'EXECUTED':
                assert trade.symbol is not None
                assert trade.action in ['BUY', 'SELL']
                assert trade.quantity > 0
                assert trade.price > 0
                assert trade.entry_time is not None
                assert trade.commission >= 0
                assert trade.slippage_cost >= 0

        # 거래 타이밍 일관성 검증
        trade_times = [t.entry_time for t in result.trades if t.status == 'EXECUTED']
        for i in range(1, len(trade_times)):
            assert trade_times[i] >= trade_times[i-1]  # 시간 순서

    def _create_realistic_market_data(self) -> pd.DataFrame:
        """현실적인 시장 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)

        # 현실적인 가격 변동 (GBM with drift)
        np.random.seed(42)
        mu = 0.0001  # 일간 드리프트
        sigma = 0.02  # 일간 변동성
        returns = np.random.normal(mu, sigma, n)

        # 가끔 큰 움직임 (fat tail)
        extreme_moves = np.random.binomial(1, 0.05, n)  # 5% 확률
        returns += extreme_moves * np.random.normal(0, 0.05, n)

        prices = 100 * np.exp(np.cumsum(returns))

        # 변동성 클러스터링 시뮬레이션
        volatility = np.ones(n) * sigma
        for i in range(1, n):
            volatility[i] = 0.05 + 0.9 * volatility[i-1] + 0.1 * abs(returns[i-1])

        return pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, volatility/4, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, volatility, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, volatility, n))),
            'close': prices,
            'volume': np.random.lognormal(12, 0.5, n)
        })

    def _create_long_term_market_data(self) -> pd.DataFrame:
        """장기간 시장 데이터 생성"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n = len(dates)

        # 장기 트렌드 + 사이클 + 노이즈
        t = np.arange(n)
        trend = 0.0001 * t
        cycle = 0.1 * np.sin(2 * np.pi * t / 252)  # 연간 사이클
        noise = np.random.normal(0, 0.02, n)

        returns = trend + cycle + noise
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.lognormal(10, 0.3, n)
        })

    def _create_trend_data_with_future_knowledge(self) -> pd.DataFrame:
        """미래 정보가 있는 트렌드 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)

        # 명확한 트렌드 (미래를 안다면 쉽게 예측 가능)
        trend_changes = [0, 100, 200, 300]  # 트렌드 변화 지점
        prices = []
        current_price = 100

        for i in range(n):
            if i in trend_changes:
                trend_direction = 1 if (trend_changes.index(i) % 2 == 0) else -1
            else:
                # 최근 트렌드 변화점 찾기
                recent_change = max([t for t in trend_changes if t <= i])
                trend_direction = 1 if (trend_changes.index(recent_change) % 2 == 0) else -1

            current_price *= (1 + trend_direction * 0.01 + np.random.normal(0, 0.01))
            prices.append(current_price)

        return pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': np.array(prices) * 1.02,
            'low': np.array(prices) * 0.98,
            'close': prices,
            'volume': np.random.lognormal(10, 0.3, n)
        })

    def _create_expensive_asset_data(self) -> pd.DataFrame:
        """비싼 자산 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)

        # 매우 비싼 가격
        prices = 50000 + np.cumsum(np.random.normal(100, 500, n))

        return pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.lognormal(8, 0.3, n)
        })

    def _create_momentum_strategy(self) -> StrategyInterface:
        """모멘텀 전략 생성"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "Momentum"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 20:
                return []

            # 20일 모멘텀
            current_price = data['close'].iloc[-1]
            price_20_days_ago = data['close'].iloc[-20]
            momentum = (current_price - price_20_days_ago) / price_20_days_ago

            signals = []
            if momentum > 0.05 and len(portfolio.positions) == 0:  # 5% 상승
                signals.append({
                    'symbol': 'TEST',
                    'action': 'BUY',
                    'quantity': int(portfolio.equity * 0.5 / current_price),
                    'price': current_price
                })
            elif momentum < -0.05 and len(portfolio.positions) > 0:  # 5% 하락
                for symbol, position in portfolio.positions.items():
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': position.quantity,
                        'price': current_price
                    })

            return signals

        strategy.generate_signals = generate_signals
        return strategy

    def _create_high_frequency_strategy(self) -> StrategyInterface:
        """고빈도 거래 전략"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "HighFrequency"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 2:
                return []

            # 매일 거래
            current_price = data['close'].iloc[-1]
            yesterday_price = data['close'].iloc[-2]
            change = (current_price - yesterday_price) / yesterday_price

            if abs(change) > 0.02:  # 2% 이상 변동시 거래
                action = 'BUY' if change > 0 else 'SELL'
                quantity = min(100, int(portfolio.cash / current_price / 10))

                if quantity > 0:
                    return [{
                        'symbol': 'TEST',
                        'action': action,
                        'quantity': quantity,
                        'price': current_price
                    }]

            return []

        strategy.generate_signals = generate_signals
        return strategy

    def _create_perfect_foresight_strategy(self) -> StrategyInterface:
        """완벽한 예측 전략 (룩어헤드 방지 테스트용)"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "PerfectForesight"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 5:
                return []

            # 최근 5일 트렌드 기반 (미래 정보 사용 불가)
            recent_prices = data['close'].iloc[-5:].values
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            if trend > 0.03 and len(portfolio.positions) == 0:
                return [{
                    'symbol': 'TEST',
                    'action': 'BUY',
                    'quantity': int(portfolio.equity * 0.8 / recent_prices[-1]),
                    'price': recent_prices[-1]
                }]

            return []

        strategy.generate_signals = generate_signals
        return strategy

    def _create_position_management_strategy(self) -> StrategyInterface:
        """포지션 관리 전략"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "PositionManagement"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 10:
                return []

            signals = []
            current_price = data['close'].iloc[-1]

            # 포지션이 없으면 매수
            if len(portfolio.positions) == 0:
                quantity = int(portfolio.cash * 0.3 / current_price)
                if quantity > 0:
                    signals.append({
                        'symbol': 'TEST',
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': current_price
                    })

            # 리밸런싱 (매 50일)
            elif len(data) % 50 == 0:
                target_value = portfolio.equity * 0.5
                current_value = portfolio.total_position_value

                if abs(target_value - current_value) > portfolio.equity * 0.1:
                    if target_value > current_value:
                        # 매수
                        buy_value = target_value - current_value
                        quantity = int(buy_value / current_price)
                        signals.append({
                            'symbol': 'TEST',
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': current_price
                        })

            return signals

        strategy.generate_signals = generate_signals
        return strategy

    def _create_adaptive_strategy(self) -> StrategyInterface:
        """적응형 전략"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "Adaptive"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 50:
                return []

            # 변동성 적응형
            recent_returns = data['close'].pct_change().dropna().iloc[-20:]
            volatility = recent_returns.std()

            # 변동성에 따른 포지션 사이징
            if volatility < 0.015:  # 낮은 변동성
                target_allocation = 0.8
            elif volatility > 0.03:  # 높은 변동성
                target_allocation = 0.3
            else:
                target_allocation = 0.5

            current_allocation = portfolio.total_position_value / portfolio.equity if portfolio.equity > 0 else 0
            current_price = data['close'].iloc[-1]

            if abs(current_allocation - target_allocation) > 0.1:
                if target_allocation > current_allocation:
                    # 매수
                    buy_value = (target_allocation - current_allocation) * portfolio.equity
                    quantity = int(buy_value / current_price)
                    if quantity > 0:
                        return [{
                            'symbol': 'TEST',
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': current_price
                        }]

            return []

        strategy.generate_signals = generate_signals
        return strategy

    def _create_simple_strategy(self) -> StrategyInterface:
        """단순한 전략"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "Simple"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            return []  # 거래하지 않음

        strategy.generate_signals = generate_signals
        return strategy

    def _create_detailed_tracking_strategy(self) -> StrategyInterface:
        """상세 추적 전략"""
        strategy = Mock(spec=StrategyInterface)
        strategy.name = "DetailedTracking"

        def generate_signals(data: pd.DataFrame, portfolio: Portfolio) -> List[Dict]:
            if len(data) < 10 or len(data) % 20 != 0:
                return []

            current_price = data['close'].iloc[-1]

            if len(portfolio.positions) == 0:
                return [{
                    'symbol': 'TEST',
                    'action': 'BUY',
                    'quantity': 100,
                    'price': current_price
                }]
            else:
                # 매도
                for symbol, position in portfolio.positions.items():
                    return [{
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': position.quantity,
                        'price': current_price
                    }]

            return []

        strategy.generate_signals = generate_signals
        return strategy