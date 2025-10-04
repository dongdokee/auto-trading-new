"""
Performance Analytics Module

Comprehensive performance analysis and reporting system for trading operations.
Supports both paper trading validation and live trading analytics.

Key Features:
- Real-time performance calculation
- Paper vs backtest comparison
- Risk-adjusted metrics
- Detailed attribution analysis
- Interactive reporting
- Benchmark comparison
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path

from .financial_math import (
    calculate_returns, calculate_sharpe_ratio, calculate_sortino_ratio,
    calculate_max_drawdown, calculate_volatility, calculate_calmar_ratio,
    calculate_information_ratio, calculate_beta
)
from .time_utils import get_utc_now
from .trade_journal import TradeJournal, JournalEntry


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Time period
    start_date: datetime
    end_date: datetime
    period_days: int

    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L metrics
    total_pnl: Decimal
    total_fees: Decimal
    net_pnl: Decimal
    gross_return: float
    net_return: float

    # Trade metrics
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    profit_factor: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    volatility: float

    # Additional metrics
    avg_trade_duration_hours: float
    daily_pnl_std: float
    consecutive_wins_max: int
    consecutive_losses_max: int

    # Mode-specific
    mode: str  # paper/live/demo
    benchmark_return: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary"""
        # Convert datetime strings
        for field in ['start_date', 'end_date']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])

        # Convert to Decimal
        for field in ['total_pnl', 'total_fees', 'net_pnl', 'avg_win', 'avg_loss',
                     'largest_win', 'largest_loss']:
            if field in data and data[field] is not None:
                data[field] = Decimal(str(data[field]))

        return cls(**data)


@dataclass
class ComparisonAnalysis:
    """Comparison analysis between paper and backtest/live"""
    paper_metrics: PerformanceMetrics
    reference_metrics: PerformanceMetrics

    # Difference metrics
    return_difference: float
    sharpe_difference: float
    drawdown_difference: float
    win_rate_difference: float

    # Correlation metrics
    pnl_correlation: float
    timing_correlation: float

    # Attribution analysis
    execution_impact: float  # Slippage + fees impact
    timing_impact: float     # Market timing differences
    size_impact: float       # Position sizing differences

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'paper_metrics': self.paper_metrics.to_dict(),
            'reference_metrics': self.reference_metrics.to_dict(),
            'return_difference': self.return_difference,
            'sharpe_difference': self.sharpe_difference,
            'drawdown_difference': self.drawdown_difference,
            'win_rate_difference': self.win_rate_difference,
            'pnl_correlation': self.pnl_correlation,
            'timing_correlation': self.timing_correlation,
            'execution_impact': self.execution_impact,
            'timing_impact': self.timing_impact,
            'size_impact': self.size_impact
        }


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis system

    Features:
    - Real-time performance calculation
    - Historical analysis
    - Paper vs backtest comparison
    - Risk attribution
    - Interactive reporting
    """

    def __init__(
        self,
        mode: str = "paper",
        benchmark_symbol: str = "BTCUSDT",
        enable_comparison: bool = True,
        cache_results: bool = True
    ):
        """
        Initialize performance analyzer

        Args:
            mode: Trading mode (paper/live/demo)
            benchmark_symbol: Benchmark symbol for comparison
            enable_comparison: Enable comparison analysis
            cache_results: Enable result caching
        """
        self.mode = mode
        self.benchmark_symbol = benchmark_symbol
        self.enable_comparison = enable_comparison
        self.cache_results = cache_results

        # Logger
        self.logger = logging.getLogger(f"performance_analyzer_{mode}")

        # Cache
        self.cached_metrics: Dict[str, PerformanceMetrics] = {}
        self.cached_comparisons: Dict[str, ComparisonAnalysis] = {}

        # Benchmark data
        self.benchmark_returns: Optional[pd.Series] = None

        self.logger.info(f"PerformanceAnalyzer initialized for {mode} mode")

    def calculate_metrics(
        self,
        journal_entries: List[JournalEntry],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics

        Args:
            journal_entries: List of trade journal entries
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            PerformanceMetrics object
        """
        try:
            # Filter entries by date if specified
            filtered_entries = self._filter_entries_by_date(
                journal_entries, start_date, end_date
            )

            if not filtered_entries:
                return self._create_empty_metrics(start_date, end_date)

            # Calculate basic metrics
            completed_trades = [e for e in filtered_entries if e.pnl is not None]

            if not completed_trades:
                return self._create_empty_metrics(start_date, end_date)

            # Time period
            actual_start = min(e.entry_time for e in completed_trades if e.entry_time)
            actual_end = max(e.exit_time for e in completed_trades if e.exit_time)
            period_days = (actual_end - actual_start).days if actual_end and actual_start else 0

            # Trade statistics
            total_trades = len(completed_trades)
            winning_trades = len([e for e in completed_trades if e.pnl and e.pnl > 0])
            losing_trades = len([e for e in completed_trades if e.pnl and e.pnl < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # P&L calculations
            total_pnl = sum(e.pnl for e in completed_trades if e.pnl)
            total_fees = sum(e.fees for e in completed_trades if e.fees)
            net_pnl = total_pnl - total_fees

            # Calculate returns (assuming starting capital)
            starting_capital = Decimal('100000')  # Default assumption
            gross_return = float(total_pnl / starting_capital) if starting_capital > 0 else 0
            net_return = float(net_pnl / starting_capital) if starting_capital > 0 else 0

            # Win/Loss analysis
            wins = [e.pnl for e in completed_trades if e.pnl and e.pnl > 0]
            losses = [abs(e.pnl) for e in completed_trades if e.pnl and e.pnl < 0]

            avg_win = sum(wins) / len(wins) if wins else Decimal('0')
            avg_loss = sum(losses) / len(losses) if losses else Decimal('0')
            largest_win = max(wins) if wins else Decimal('0')
            largest_loss = max(losses) if losses else Decimal('0')

            profit_factor = float(sum(wins) / sum(losses)) if losses and sum(losses) > 0 else float('inf') if wins else 0

            # Risk metrics
            pnl_series = pd.Series([float(e.pnl) for e in completed_trades if e.pnl])

            if len(pnl_series) > 1:
                returns_series = pnl_series / float(starting_capital)
                sharpe_ratio = calculate_sharpe_ratio(returns_series, risk_free_rate=0.02)
                sortino_ratio = calculate_sortino_ratio(returns_series, risk_free_rate=0.02)
                calmar_ratio = calculate_calmar_ratio(returns_series)
                volatility = calculate_volatility(returns_series)

                # Drawdown analysis
                cumulative_pnl = pnl_series.cumsum()
                max_dd, dd_series = calculate_max_drawdown(cumulative_pnl)
                max_dd_duration = self._calculate_max_drawdown_duration(dd_series)
            else:
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                calmar_ratio = 0.0
                volatility = 0.0
                max_dd = 0.0
                max_dd_duration = 0

            # Trade duration analysis
            durations = [
                (e.exit_time - e.entry_time).total_seconds() / 3600
                for e in completed_trades
                if e.entry_time and e.exit_time
            ]
            avg_duration = sum(durations) / len(durations) if durations else 0

            # Daily P&L analysis
            daily_pnl_std = float(pnl_series.std()) if len(pnl_series) > 1 else 0.0

            # Consecutive wins/losses
            consecutive_wins_max = self._calculate_max_consecutive(completed_trades, True)
            consecutive_losses_max = self._calculate_max_consecutive(completed_trades, False)

            # Benchmark comparison (if available)
            benchmark_return = None
            beta = None
            information_ratio = None

            if self.benchmark_returns is not None and len(pnl_series) > 1:
                try:
                    benchmark_return = float(self.benchmark_returns.mean())
                    beta = calculate_beta(pnl_series, self.benchmark_returns)
                    information_ratio = calculate_information_ratio(pnl_series, self.benchmark_returns)
                except Exception as e:
                    self.logger.warning(f"Benchmark calculation failed: {e}")

            return PerformanceMetrics(
                start_date=actual_start or (start_date or get_utc_now()),
                end_date=actual_end or (end_date or get_utc_now()),
                period_days=period_days,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_fees=total_fees,
                net_pnl=net_pnl,
                gross_return=gross_return,
                net_return=net_return,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_dd,
                max_drawdown_duration_days=max_dd_duration,
                volatility=volatility,
                avg_trade_duration_hours=avg_duration,
                daily_pnl_std=daily_pnl_std,
                consecutive_wins_max=consecutive_wins_max,
                consecutive_losses_max=consecutive_losses_max,
                mode=self.mode,
                benchmark_return=benchmark_return,
                beta=beta,
                information_ratio=information_ratio
            )

        except Exception as e:
            self.logger.error(f"Performance calculation failed: {e}")
            return self._create_empty_metrics(start_date, end_date)

    def compare_performance(
        self,
        paper_entries: List[JournalEntry],
        reference_entries: List[JournalEntry],
        reference_type: str = "backtest"
    ) -> ComparisonAnalysis:
        """
        Compare paper trading performance with reference (backtest/live)

        Args:
            paper_entries: Paper trading journal entries
            reference_entries: Reference journal entries
            reference_type: Type of reference (backtest/live)

        Returns:
            ComparisonAnalysis object
        """
        try:
            # Calculate metrics for both
            paper_metrics = self.calculate_metrics(paper_entries)
            reference_metrics = self.calculate_metrics(reference_entries)
            reference_metrics.mode = reference_type

            # Calculate differences
            return_diff = paper_metrics.net_return - reference_metrics.net_return
            sharpe_diff = paper_metrics.sharpe_ratio - reference_metrics.sharpe_ratio
            drawdown_diff = paper_metrics.max_drawdown - reference_metrics.max_drawdown
            win_rate_diff = paper_metrics.win_rate - reference_metrics.win_rate

            # Correlation analysis
            paper_pnl = [float(e.pnl) for e in paper_entries if e.pnl]
            reference_pnl = [float(e.pnl) for e in reference_entries if e.pnl]

            pnl_correlation = 0.0
            timing_correlation = 0.0

            if len(paper_pnl) > 1 and len(reference_pnl) > 1:
                min_length = min(len(paper_pnl), len(reference_pnl))
                paper_series = pd.Series(paper_pnl[:min_length])
                reference_series = pd.Series(reference_pnl[:min_length])

                pnl_correlation = paper_series.corr(reference_series)

                # Timing correlation (based on trade timing)
                timing_correlation = self._calculate_timing_correlation(
                    paper_entries, reference_entries
                )

            # Attribution analysis
            execution_impact = self._calculate_execution_impact(paper_entries, reference_entries)
            timing_impact = self._calculate_timing_impact(paper_entries, reference_entries)
            size_impact = self._calculate_size_impact(paper_entries, reference_entries)

            return ComparisonAnalysis(
                paper_metrics=paper_metrics,
                reference_metrics=reference_metrics,
                return_difference=return_diff,
                sharpe_difference=sharpe_diff,
                drawdown_difference=drawdown_diff,
                win_rate_difference=win_rate_diff,
                pnl_correlation=pnl_correlation,
                timing_correlation=timing_correlation,
                execution_impact=execution_impact,
                timing_impact=timing_impact,
                size_impact=size_impact
            )

        except Exception as e:
            self.logger.error(f"Performance comparison failed: {e}")
            raise

    def generate_report(
        self,
        metrics: PerformanceMetrics,
        output_path: str,
        format: str = "html",
        include_charts: bool = True
    ) -> bool:
        """
        Generate performance report

        Args:
            metrics: Performance metrics to report
            output_path: Output file path
            format: Report format (html/json/csv)
            include_charts: Include charts in HTML report

        Returns:
            True if successful
        """
        try:
            if format.lower() == "html":
                return self._generate_html_report(metrics, output_path, include_charts)
            elif format.lower() == "json":
                return self._generate_json_report(metrics, output_path)
            elif format.lower() == "csv":
                return self._generate_csv_report(metrics, output_path)
            else:
                self.logger.error(f"Unsupported format: {format}")
                return False

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return False

    def _filter_entries_by_date(
        self,
        entries: List[JournalEntry],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[JournalEntry]:
        """Filter entries by date range"""
        filtered = entries

        if start_date:
            filtered = [e for e in filtered if e.entry_time and e.entry_time >= start_date]

        if end_date:
            filtered = [e for e in filtered if e.entry_time and e.entry_time <= end_date]

        return filtered

    def _create_empty_metrics(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> PerformanceMetrics:
        """Create empty performance metrics"""
        now = get_utc_now()

        return PerformanceMetrics(
            start_date=start_date or now,
            end_date=end_date or now,
            period_days=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=Decimal('0'),
            total_fees=Decimal('0'),
            net_pnl=Decimal('0'),
            gross_return=0.0,
            net_return=0.0,
            avg_win=Decimal('0'),
            avg_loss=Decimal('0'),
            largest_win=Decimal('0'),
            largest_loss=Decimal('0'),
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration_days=0,
            volatility=0.0,
            avg_trade_duration_hours=0.0,
            daily_pnl_std=0.0,
            consecutive_wins_max=0,
            consecutive_losses_max=0,
            mode=self.mode
        )

    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        if len(drawdown_series) == 0:
            return 0

        in_drawdown = False
        current_duration = 0
        max_duration = 0

        for dd in drawdown_series:
            if dd < 0:  # In drawdown
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:  # Out of drawdown
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    in_drawdown = False
                    current_duration = 0

        # Check if still in drawdown at the end
        if in_drawdown:
            max_duration = max(max_duration, current_duration)

        return max_duration

    def _calculate_max_consecutive(self, entries: List[JournalEntry], wins: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        if not entries:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for entry in entries:
            if entry.pnl is None:
                continue

            is_win = entry.pnl > 0
            if (wins and is_win) or (not wins and not is_win):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_timing_correlation(
        self,
        paper_entries: List[JournalEntry],
        reference_entries: List[JournalEntry]
    ) -> float:
        """Calculate timing correlation between paper and reference trades"""
        try:
            # Create time-based correlation by matching trades
            paper_times = [(e.entry_time, float(e.pnl or 0)) for e in paper_entries if e.entry_time and e.pnl]
            reference_times = [(e.entry_time, float(e.pnl or 0)) for e in reference_entries if e.entry_time and e.pnl]

            if len(paper_times) < 2 or len(reference_times) < 2:
                return 0.0

            # Simple implementation: correlation of PnL ordered by time
            paper_pnl_ordered = [pnl for _, pnl in sorted(paper_times)]
            reference_pnl_ordered = [pnl for _, pnl in sorted(reference_times)]

            min_length = min(len(paper_pnl_ordered), len(reference_pnl_ordered))
            if min_length < 2:
                return 0.0

            paper_series = pd.Series(paper_pnl_ordered[:min_length])
            reference_series = pd.Series(reference_pnl_ordered[:min_length])

            return paper_series.corr(reference_series)

        except Exception:
            return 0.0

    def _calculate_execution_impact(
        self,
        paper_entries: List[JournalEntry],
        reference_entries: List[JournalEntry]
    ) -> float:
        """Calculate execution impact (slippage + fees)"""
        try:
            paper_fees = sum(float(e.fees or 0) for e in paper_entries)
            reference_fees = sum(float(e.fees or 0) for e in reference_entries)

            paper_slippage = sum(float(e.slippage_bps or 0) for e in paper_entries)
            reference_slippage = sum(float(e.slippage_bps or 0) for e in reference_entries)

            execution_cost_diff = (paper_fees + paper_slippage) - (reference_fees + reference_slippage)

            # Normalize by total traded volume
            paper_volume = sum(float((e.quantity or 0) * (e.entry_price or 0)) for e in paper_entries)
            if paper_volume > 0:
                return execution_cost_diff / paper_volume
            else:
                return 0.0

        except Exception:
            return 0.0

    def _calculate_timing_impact(
        self,
        paper_entries: List[JournalEntry],
        reference_entries: List[JournalEntry]
    ) -> float:
        """Calculate timing impact"""
        # Simplified implementation
        # In practice, this would compare entry/exit timing differences
        return 0.0

    def _calculate_size_impact(
        self,
        paper_entries: List[JournalEntry],
        reference_entries: List[JournalEntry]
    ) -> float:
        """Calculate position size impact"""
        try:
            paper_avg_size = np.mean([float(e.quantity or 0) for e in paper_entries])
            reference_avg_size = np.mean([float(e.quantity or 0) for e in reference_entries])

            if reference_avg_size > 0:
                return (paper_avg_size - reference_avg_size) / reference_avg_size
            else:
                return 0.0

        except Exception:
            return 0.0

    def _generate_html_report(
        self,
        metrics: PerformanceMetrics,
        output_path: str,
        include_charts: bool
    ) -> bool:
        """Generate HTML performance report"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Analytics Report - {metrics.mode.upper()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; margin-bottom: 30px; border-radius: 5px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background-color: #e8f4fd; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; margin-bottom: 5px; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .section {{ margin-bottom: 40px; }}
        .section-title {{ font-size: 24px; color: #2c3e50; margin-bottom: 20px; border-bottom: 2px solid #3498db; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .neutral {{ color: #34495e; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Analytics Report</h1>
        <p><strong>Mode:</strong> {metrics.mode.upper()}</p>
        <p><strong>Period:</strong> {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')} ({metrics.period_days} days)</p>
        <p><strong>Generated:</strong> {get_utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>

    <div class="section">
        <h2 class="section-title">Key Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.net_return >= 0 else 'negative'}">{metrics.net_return:.2%}</div>
                <div class="metric-label">Net Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{metrics.max_drawdown:.1%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2 class="section-title">Detailed Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total P&L</td><td class="{'positive' if metrics.total_pnl >= 0 else 'negative'}">${metrics.total_pnl:.2f}</td></tr>
            <tr><td>Total Fees</td><td class="negative">${metrics.total_fees:.2f}</td></tr>
            <tr><td>Net P&L</td><td class="{'positive' if metrics.net_pnl >= 0 else 'negative'}">${metrics.net_pnl:.2f}</td></tr>
            <tr><td>Winning Trades</td><td class="positive">{metrics.winning_trades}</td></tr>
            <tr><td>Losing Trades</td><td class="negative">{metrics.losing_trades}</td></tr>
            <tr><td>Average Win</td><td class="positive">${metrics.avg_win:.2f}</td></tr>
            <tr><td>Average Loss</td><td class="negative">${metrics.avg_loss:.2f}</td></tr>
            <tr><td>Largest Win</td><td class="positive">${metrics.largest_win:.2f}</td></tr>
            <tr><td>Largest Loss</td><td class="negative">${metrics.largest_loss:.2f}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2 class="section-title">Risk Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Volatility</td><td>{metrics.volatility:.1%}</td></tr>
            <tr><td>Sortino Ratio</td><td>{metrics.sortino_ratio:.2f}</td></tr>
            <tr><td>Calmar Ratio</td><td>{metrics.calmar_ratio:.2f}</td></tr>
            <tr><td>Max Drawdown Duration</td><td>{metrics.max_drawdown_duration_days} days</td></tr>
            <tr><td>Daily P&L Std Dev</td><td>${metrics.daily_pnl_std:.2f}</td></tr>
            <tr><td>Max Consecutive Wins</td><td>{metrics.consecutive_wins_max}</td></tr>
            <tr><td>Max Consecutive Losses</td><td>{metrics.consecutive_losses_max}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2 class="section-title">Trading Behavior</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Average Trade Duration</td><td>{metrics.avg_trade_duration_hours:.1f} hours</td></tr>
            <tr><td>Trades per Day</td><td>{metrics.total_trades / max(metrics.period_days, 1):.1f}</td></tr>
"""

            if metrics.benchmark_return is not None:
                html_content += f"""
            <tr><td>Benchmark Return</td><td>{metrics.benchmark_return:.2%}</td></tr>
            <tr><td>Beta</td><td>{metrics.beta:.2f}</td></tr>
            <tr><td>Information Ratio</td><td>{metrics.information_ratio:.2f}</td></tr>
"""

            html_content += """
        </table>
    </div>

</body>
</html>
"""

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"HTML report generated: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"HTML report generation failed: {e}")
            return False

    def _generate_json_report(self, metrics: PerformanceMetrics, output_path: str) -> bool:
        """Generate JSON performance report"""
        try:
            report_data = {
                'generated_at': get_utc_now().isoformat(),
                'mode': self.mode,
                'analyzer_config': {
                    'benchmark_symbol': self.benchmark_symbol,
                    'enable_comparison': self.enable_comparison
                },
                'performance_metrics': metrics.to_dict()
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"JSON report generated: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"JSON report generation failed: {e}")
            return False

    def _generate_csv_report(self, metrics: PerformanceMetrics, output_path: str) -> bool:
        """Generate CSV performance report"""
        try:
            # Flatten metrics to single row
            metrics_dict = metrics.to_dict()

            df = pd.DataFrame([metrics_dict])
            df.to_csv(output_path, index=False)

            self.logger.info(f"CSV report generated: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"CSV report generation failed: {e}")
            return False


# Convenience functions

def analyze_journal_performance(
    journal_path: str,
    mode: str = "paper",
    session_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> PerformanceMetrics:
    """
    Analyze performance from trade journal

    Args:
        journal_path: Path to trade journal database
        mode: Trading mode
        session_id: Specific session to analyze
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        PerformanceMetrics object
    """
    try:
        with TradeJournal(journal_path, mode=mode) as journal:
            entries = journal.get_entries(
                session_id=session_id,
                start_date=start_date,
                end_date=end_date
            )

            analyzer = PerformanceAnalyzer(mode=mode)
            return analyzer.calculate_metrics(entries, start_date, end_date)

    except Exception as e:
        logging.error(f"Journal performance analysis failed: {e}")
        raise


def compare_paper_vs_backtest(
    paper_journal_path: str,
    backtest_journal_path: str
) -> ComparisonAnalysis:
    """
    Compare paper trading vs backtest results

    Args:
        paper_journal_path: Path to paper trading journal
        backtest_journal_path: Path to backtest journal

    Returns:
        ComparisonAnalysis object
    """
    try:
        with TradeJournal(paper_journal_path, mode="paper") as paper_journal:
            paper_entries = paper_journal.get_entries()

        with TradeJournal(backtest_journal_path, mode="backtest") as backtest_journal:
            backtest_entries = backtest_journal.get_entries()

        analyzer = PerformanceAnalyzer(mode="paper", enable_comparison=True)
        return analyzer.compare_performance(paper_entries, backtest_entries, "backtest")

    except Exception as e:
        logging.error(f"Paper vs backtest comparison failed: {e}")
        raise


def generate_performance_report(
    metrics: PerformanceMetrics,
    output_dir: str = "reports",
    formats: List[str] = ["html", "json"]
) -> Dict[str, bool]:
    """
    Generate performance reports in multiple formats

    Args:
        metrics: Performance metrics to report
        output_dir: Output directory
        formats: List of formats to generate

    Returns:
        Dictionary with format -> success status
    """
    os.makedirs(output_dir, exist_ok=True)
    analyzer = PerformanceAnalyzer(mode=metrics.mode)

    results = {}
    timestamp = get_utc_now().strftime("%Y%m%d_%H%M%S")

    for format_type in formats:
        try:
            filename = f"performance_report_{metrics.mode}_{timestamp}.{format_type}"
            output_path = os.path.join(output_dir, filename)

            success = analyzer.generate_report(metrics, output_path, format_type)
            results[format_type] = success

        except Exception as e:
            logging.error(f"Report generation failed for {format_type}: {e}")
            results[format_type] = False

    return results