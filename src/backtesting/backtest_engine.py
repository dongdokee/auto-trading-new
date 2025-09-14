"""
백테스트 엔진

거래 전략의 백테스트를 실행하는 핵심 엔진입니다.
룩어헤드 바이어스 방지, Walk-Forward 검증, 현실적인 거래 비용 모델링을 지원합니다.

주요 기능:
- 룩어헤드 바이어스 방지 백테스트
- Walk-Forward 최적화 및 검증
- 현실적 거래 비용 계산 (수수료, 슬리피지)
- 포트폴리오 상태 추적
- 상세한 성과 메트릭 계산

TDD 방법론으로 구현 - Phase 2.1
Created: 2025-09-14
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Protocol, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time
from abc import ABC, abstractmethod


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.0004  # 0.04%
    slippage_rate: float = 0.0005    # 0.05%
    enable_walk_forward: bool = False
    walk_forward_window: int = 252   # 252 trading days
    out_of_sample_window: int = 63   # 63 trading days
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    min_trade_size: float = 100.0
    max_position_size: float = 0.2   # 20% of portfolio


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    def update_current_price(self, price: float):
        """현재 가격 업데이트"""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity


@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    action: str  # 'BUY', 'SELL'
    quantity: float
    price: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    commission: float = 0.0
    slippage_cost: float = 0.0
    realized_pnl: float = 0.0
    status: str = 'EXECUTED'  # 'EXECUTED', 'REJECTED'


@dataclass
class Portfolio:
    """포트폴리오 상태"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_position_value: float = 0.0
    equity: float = 0.0

    def update_equity(self):
        """총 자산 업데이트"""
        self.total_position_value = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        self.equity = self.cash + self.total_position_value


@dataclass
class WalkForwardResult:
    """Walk-Forward 결과"""
    period: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    in_sample_return: float
    out_of_sample_return: float
    in_sample_sharpe: float
    out_of_sample_sharpe: float


@dataclass
class BacktestResult:
    """백테스트 결과"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    total_costs: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    portfolio_history: List[Portfolio] = field(default_factory=list)
    walk_forward_results: Optional[List[WalkForwardResult]] = None
    out_of_sample_performance: Optional[Dict[str, float]] = None


class StrategyInterface(Protocol):
    """전략 인터페이스"""
    name: str

    def generate_signals(self, data: pd.DataFrame, portfolio: Portfolio) -> List[Dict[str, Any]]:
        """
        거래 신호 생성

        Args:
            data: 현재 시점까지의 시장 데이터
            portfolio: 현재 포트폴리오 상태

        Returns:
            List of signal dictionaries with keys: symbol, action, quantity, price
        """
        ...


class BacktestEngine:
    """백테스트 엔진"""

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        BacktestEngine 초기화

        Args:
            config: 백테스트 설정
        """
        self.config = config or BacktestConfig()
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, strategy: StrategyInterface, data: pd.DataFrame,
                    start_date: str, end_date: str) -> BacktestResult:
        """
        기본 백테스트 실행

        Args:
            strategy: 거래 전략
            data: 시장 데이터
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일

        Returns:
            BacktestResult: 백테스트 결과
        """
        # 데이터 필터링
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        test_data = data[(data['datetime'] >= start_dt) & (data['datetime'] <= end_dt)].copy()

        if len(test_data) == 0:
            raise ValueError("No data available for the specified date range")

        # 포트폴리오 초기화
        portfolio = Portfolio(cash=self.config.initial_capital)
        portfolio.update_equity()

        # 결과 추적
        trades = []
        equity_curve = [portfolio.equity]
        portfolio_history = []

        # 백테스트 실행 (룩어헤드 바이어스 방지)
        for i in range(len(test_data)):
            current_time = test_data.iloc[i]['datetime']
            current_data = test_data.iloc[:i+1]  # 현재 시점까지만

            # 포지션 현재가 업데이트
            current_price = test_data.iloc[i]['close']
            for position in portfolio.positions.values():
                position.update_current_price(current_price)

            # 포트폴리오 업데이트
            portfolio.update_equity()

            # 전략 신호 생성
            signals = strategy.generate_signals(current_data, portfolio)

            # 신호 실행
            for signal in signals:
                trade = self._execute_signal(signal, current_time, current_price, portfolio)
                if trade:
                    trades.append(trade)

            # 상태 기록
            equity_curve.append(portfolio.equity)
            portfolio_history.append(self._copy_portfolio(portfolio))

        # 성과 메트릭 계산
        result = self._calculate_performance_metrics(
            strategy.name, start_dt, end_dt, trades, equity_curve, portfolio_history
        )

        return result

    def run_walk_forward_backtest(self, strategy: StrategyInterface, data: pd.DataFrame,
                                 start_date: str, end_date: str) -> BacktestResult:
        """
        Walk-Forward 백테스트 실행

        Args:
            strategy: 거래 전략
            data: 시장 데이터
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일

        Returns:
            BacktestResult: Walk-Forward 백테스트 결과
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        test_data = data[(data['datetime'] >= start_dt) & (data['datetime'] <= end_dt)].copy()

        total_days = len(test_data)
        in_sample_window = self.config.walk_forward_window
        out_of_sample_window = self.config.out_of_sample_window

        if total_days < in_sample_window + out_of_sample_window:
            raise ValueError(f"Insufficient data for walk-forward analysis. "
                           f"Need at least {in_sample_window + out_of_sample_window} days, "
                           f"got {total_days}")

        walk_forward_results = []
        all_trades = []
        all_equity_curve = []
        all_portfolio_history = []

        # Walk-Forward 윈도우 반복
        start_idx = 0
        period = 0

        while start_idx + in_sample_window + out_of_sample_window <= total_days:
            period += 1

            # In-sample 기간
            in_sample_end_idx = start_idx + in_sample_window
            in_sample_data = test_data.iloc[start_idx:in_sample_end_idx]

            # Out-of-sample 기간
            out_sample_end_idx = min(in_sample_end_idx + out_of_sample_window, total_days)
            out_sample_data = test_data.iloc[in_sample_end_idx:out_sample_end_idx]

            # In-sample 백테스트 (전략 최적화 시뮬레이션)
            in_sample_result = self._run_period_backtest(strategy, in_sample_data)

            # Out-of-sample 백테스트 (실제 성과 검증)
            out_sample_result = self._run_period_backtest(strategy, out_sample_data)

            # Walk-Forward 결과 저장
            wf_result = WalkForwardResult(
                period=period,
                in_sample_start=in_sample_data.iloc[0]['datetime'],
                in_sample_end=in_sample_data.iloc[-1]['datetime'],
                out_of_sample_start=out_sample_data.iloc[0]['datetime'],
                out_of_sample_end=out_sample_data.iloc[-1]['datetime'],
                in_sample_return=in_sample_result.total_return,
                out_of_sample_return=out_sample_result.total_return,
                in_sample_sharpe=in_sample_result.sharpe_ratio,
                out_of_sample_sharpe=out_sample_result.sharpe_ratio
            )
            walk_forward_results.append(wf_result)

            # 거래 기록 누적
            all_trades.extend(out_sample_result.trades)
            all_equity_curve.extend(out_sample_result.equity_curve[1:])  # 중복 제거
            all_portfolio_history.extend(out_sample_result.portfolio_history)

            # 다음 윈도우로 이동
            start_idx += out_of_sample_window

        # 전체 결과 계산
        final_result = self._calculate_performance_metrics(
            strategy.name, start_dt, end_dt, all_trades, all_equity_curve, all_portfolio_history
        )

        # Walk-Forward 특화 메트릭 추가
        final_result.walk_forward_results = walk_forward_results
        final_result.out_of_sample_performance = self._calculate_out_of_sample_metrics(
            walk_forward_results
        )

        return final_result

    def _execute_signal(self, signal: Dict[str, Any], current_time: datetime,
                       current_price: float, portfolio: Portfolio) -> Optional[Trade]:
        """신호 실행"""
        symbol = signal.get('symbol', 'TEST')
        action = signal.get('action', 'BUY')
        quantity = signal.get('quantity', 0)
        signal_price = signal.get('price', current_price)

        if quantity <= 0:
            return None

        # 거래 비용 계산
        notional = quantity * signal_price
        commission = notional * self.config.commission_rate
        slippage = notional * self.config.slippage_rate

        # 자본 충분성 확인
        if action == 'BUY':
            total_cost = notional + commission + slippage
            if total_cost > portfolio.cash:
                return Trade(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=signal_price,
                    entry_time=current_time,
                    status='REJECTED'
                )

            # 매수 실행
            portfolio.cash -= total_cost
            if symbol in portfolio.positions:
                # 기존 포지션에 추가
                existing_pos = portfolio.positions[symbol]
                total_quantity = existing_pos.quantity + quantity
                avg_price = ((existing_pos.quantity * existing_pos.entry_price) +
                           (quantity * signal_price)) / total_quantity
                existing_pos.quantity = total_quantity
                existing_pos.entry_price = avg_price
            else:
                # 새 포지션 생성
                portfolio.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=signal_price,
                    entry_time=current_time,
                    current_price=signal_price
                )

        elif action == 'SELL':
            # 매도 가능 여부 확인
            if symbol not in portfolio.positions or portfolio.positions[symbol].quantity < quantity:
                return Trade(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=signal_price,
                    entry_time=current_time,
                    status='REJECTED'
                )

            # 매도 실행
            position = portfolio.positions[symbol]
            proceeds = notional - commission - slippage
            portfolio.cash += proceeds

            # PnL 계산
            realized_pnl = (signal_price - position.entry_price) * quantity - commission - slippage

            # 포지션 업데이트
            position.quantity -= quantity
            if position.quantity <= 0:
                del portfolio.positions[symbol]

            return Trade(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=signal_price,
                entry_time=current_time,
                commission=commission,
                slippage_cost=slippage,
                realized_pnl=realized_pnl,
                status='EXECUTED'
            )

        # 매수의 경우
        return Trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=signal_price,
            entry_time=current_time,
            commission=commission,
            slippage_cost=slippage,
            status='EXECUTED'
        )

    def _run_period_backtest(self, strategy: StrategyInterface,
                           data: pd.DataFrame) -> BacktestResult:
        """특정 기간 백테스트 실행"""
        if len(data) == 0:
            return self._create_empty_result(strategy.name)

        start_date = data.iloc[0]['datetime'].strftime('%Y-%m-%d')
        end_date = data.iloc[-1]['datetime'].strftime('%Y-%m-%d')

        return self.run_backtest(strategy, data, start_date, end_date)

    def _copy_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """포트폴리오 복사"""
        new_positions = {}
        for symbol, pos in portfolio.positions.items():
            new_positions[symbol] = Position(
                symbol=pos.symbol,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl
            )

        new_portfolio = Portfolio(
            cash=portfolio.cash,
            positions=new_positions,
            total_position_value=portfolio.total_position_value,
            equity=portfolio.equity
        )
        return new_portfolio

    def _calculate_performance_metrics(self, strategy_name: str, start_date: datetime,
                                     end_date: datetime, trades: List[Trade],
                                     equity_curve: List[float],
                                     portfolio_history: List[Portfolio]) -> BacktestResult:
        """성과 메트릭 계산"""
        if len(equity_curve) < 2:
            return self._create_empty_result(strategy_name)

        initial_capital = self.config.initial_capital
        final_capital = equity_curve[-1]
        total_return = (final_capital - initial_capital) / initial_capital

        # 일간 수익률 계산
        daily_returns = []
        for i in range(1, len(equity_curve)):
            daily_return = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            daily_returns.append(daily_return)

        daily_returns = np.array(daily_returns)

        # Sharpe Ratio 계산
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # 연간 수익률
        days = (end_date - start_date).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0

        # 최대 낙폭
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (np.array(equity_curve) - running_max) / running_max
        max_drawdown = drawdowns.min()

        # 승률
        executed_trades = [t for t in trades if t.status == 'EXECUTED' and t.realized_pnl != 0]
        if executed_trades:
            winning_trades = [t for t in executed_trades if t.realized_pnl > 0]
            win_rate = len(winning_trades) / len(executed_trades)
        else:
            win_rate = 0.0

        # 총 거래 비용
        total_costs = sum(t.commission + t.slippage_cost for t in trades)

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            total_costs=total_costs,
            trades=trades,
            equity_curve=equity_curve,
            portfolio_history=portfolio_history
        )

    def _calculate_out_of_sample_metrics(self,
                                       walk_forward_results: List[WalkForwardResult]) -> Dict[str, float]:
        """Out-of-sample 성과 메트릭 계산"""
        if not walk_forward_results:
            return {}

        out_returns = [r.out_of_sample_return for r in walk_forward_results]
        in_returns = [r.in_sample_return for r in walk_forward_results]

        return {
            'avg_out_of_sample_return': np.mean(out_returns),
            'avg_in_sample_return': np.mean(in_returns),
            'performance_degradation': np.mean(in_returns) - np.mean(out_returns),
            'out_of_sample_consistency': 1 - (np.std(out_returns) / (abs(np.mean(out_returns)) + 1e-6)),
            'periods_count': len(walk_forward_results)
        }

    def _create_empty_result(self, strategy_name: str) -> BacktestResult:
        """빈 결과 생성"""
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_capital=self.config.initial_capital,
            final_capital=self.config.initial_capital,
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            total_costs=0.0
        )