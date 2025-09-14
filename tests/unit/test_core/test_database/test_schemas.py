"""
Tests for database schema definitions and models.
Following TDD methodology: Red -> Green -> Refactor
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import Optional

# These imports will fail initially (Red phase)
# We'll implement them to make tests pass (Green phase)
try:
    from src.core.database.schemas.trading_schemas import (
        Position, Trade, Order, MarketData, Portfolio,
        RiskMetrics, StrategyPerformance
    )
    from src.core.database.schemas.base import BaseModel, TimestampMixin
except ImportError:
    pytest.skip("Database schemas not yet implemented", allow_module_level=True)


class TestBaseSchemaModels:
    """Test base schema functionality"""

    def test_should_create_base_model_with_id(self):
        """Base models should have auto-generated ID field"""
        # This test will fail initially - we need to implement BaseModel
        model = BaseModel()
        assert hasattr(model, 'id')
        assert model.id is not None

    def test_should_create_timestamp_mixin_with_created_updated(self):
        """TimestampMixin should provide created_at and updated_at fields"""
        # This test will fail initially - we need to implement TimestampMixin
        mixin = TimestampMixin()
        assert hasattr(mixin, 'created_at')
        assert hasattr(mixin, 'updated_at')
        assert isinstance(mixin.created_at, datetime)
        assert isinstance(mixin.updated_at, datetime)


class TestTradingSchemas:
    """Test trading-related database schemas"""

    def test_should_create_position_model_with_required_fields(self):
        """Position model should have all required trading fields"""
        position = Position(
            symbol='BTCUSDT',
            side='LONG',
            size=Decimal('1.5'),
            entry_price=Decimal('50000.00'),
            leverage=Decimal('3.0'),
            margin_usdt=Decimal('25000.00'),
            liquidation_price=Decimal('33333.33'),
            unrealized_pnl=Decimal('0.00'),
            status='OPEN'
        )

        assert position.symbol == 'BTCUSDT'
        assert position.side == 'LONG'
        assert position.size == Decimal('1.5')
        assert position.entry_price == Decimal('50000.00')
        assert position.leverage == Decimal('3.0')
        assert position.status == 'OPEN'

    def test_should_create_trade_model_with_execution_details(self):
        """Trade model should track individual trade executions"""
        trade = Trade(
            position_id=1,
            symbol='BTCUSDT',
            side='LONG',
            quantity=Decimal('1.0'),
            price=Decimal('50000.00'),
            fee=Decimal('15.00'),
            fee_currency='USDT',
            execution_time=datetime.utcnow(),
            trade_type='MARKET'
        )

        assert trade.position_id == 1
        assert trade.symbol == 'BTCUSDT'
        assert trade.quantity == Decimal('1.0')
        assert trade.price == Decimal('50000.00')
        assert trade.fee == Decimal('15.00')

    def test_should_create_order_model_with_order_management_fields(self):
        """Order model should track order lifecycle"""
        order = Order(
            symbol='BTCUSDT',
            side='BUY',
            order_type='LIMIT',
            quantity=Decimal('1.0'),
            price=Decimal('49000.00'),
            status='PENDING',
            time_in_force='GTC'
        )

        assert order.symbol == 'BTCUSDT'
        assert order.side == 'BUY'
        assert order.order_type == 'LIMIT'
        assert order.quantity == Decimal('1.0')
        assert order.price == Decimal('49000.00')
        assert order.status == 'PENDING'

    def test_should_create_market_data_model_with_ohlcv_fields(self):
        """MarketData model should store OHLCV and market microstructure data"""
        market_data = MarketData(
            symbol='BTCUSDT',
            timestamp=datetime.utcnow(),
            open_price=Decimal('49500.00'),
            high_price=Decimal('50500.00'),
            low_price=Decimal('49000.00'),
            close_price=Decimal('50000.00'),
            volume=Decimal('1250.5'),
            quote_volume=Decimal('62525000.00'),
            trades_count=5500,
            timeframe='1m'
        )

        assert market_data.symbol == 'BTCUSDT'
        assert market_data.open_price == Decimal('49500.00')
        assert market_data.volume == Decimal('1250.5')
        assert market_data.timeframe == '1m'


class TestPortfolioSchemas:
    """Test portfolio and risk management schemas"""

    def test_should_create_portfolio_model_with_equity_tracking(self):
        """Portfolio model should track equity and performance metrics"""
        portfolio = Portfolio(
            total_equity_usdt=Decimal('100000.00'),
            available_balance_usdt=Decimal('75000.00'),
            used_margin_usdt=Decimal('25000.00'),
            unrealized_pnl_usdt=Decimal('2500.00'),
            realized_pnl_usdt=Decimal('1500.00'),
            total_positions=3,
            leverage_ratio=Decimal('2.5')
        )

        assert portfolio.total_equity_usdt == Decimal('100000.00')
        assert portfolio.available_balance_usdt == Decimal('75000.00')
        assert portfolio.unrealized_pnl_usdt == Decimal('2500.00')
        assert portfolio.total_positions == 3

    def test_should_create_risk_metrics_model_with_risk_calculations(self):
        """RiskMetrics model should store risk calculations and monitoring data"""
        risk_metrics = RiskMetrics(
            portfolio_id=1,
            var_daily_usdt=Decimal('2000.00'),
            var_utilization_pct=Decimal('0.15'),
            current_drawdown_pct=Decimal('0.05'),
            max_drawdown_pct=Decimal('0.08'),
            high_water_mark=Decimal('105000.00'),
            consecutive_loss_days=2,
            sharpe_ratio=Decimal('1.85'),
            kelly_fraction=Decimal('0.25')
        )

        assert risk_metrics.var_daily_usdt == Decimal('2000.00')
        assert risk_metrics.current_drawdown_pct == Decimal('0.05')
        assert risk_metrics.sharpe_ratio == Decimal('1.85')
        assert risk_metrics.kelly_fraction == Decimal('0.25')

    def test_should_create_strategy_performance_model_with_metrics(self):
        """StrategyPerformance model should track individual strategy metrics"""
        strategy_perf = StrategyPerformance(
            strategy_name='TrendFollowing',
            total_trades=150,
            winning_trades=95,
            losing_trades=55,
            win_rate_pct=Decimal('63.33'),
            avg_win_usdt=Decimal('250.00'),
            avg_loss_usdt=Decimal('150.00'),
            profit_factor=Decimal('1.92'),
            max_consecutive_wins=8,
            max_consecutive_losses=4
        )

        assert strategy_perf.strategy_name == 'TrendFollowing'
        assert strategy_perf.total_trades == 150
        assert strategy_perf.win_rate_pct == Decimal('63.33')
        assert strategy_perf.profit_factor == Decimal('1.92')


class TestSchemaValidation:
    """Test schema validation and constraints"""

    def test_should_validate_position_side_values(self):
        """Position side should only allow LONG or SHORT"""
        with pytest.raises(ValueError):
            Position(
                symbol='BTCUSDT',
                side='INVALID',  # Should raise ValueError
                size=Decimal('1.0'),
                entry_price=Decimal('50000.00'),
                status='OPEN'
            )

    def test_should_validate_positive_decimal_values(self):
        """Financial values should be positive where required"""
        with pytest.raises(ValueError):
            Position(
                symbol='BTCUSDT',
                side='LONG',
                size=Decimal('-1.0'),  # Negative size should raise ValueError
                entry_price=Decimal('50000.00'),
                status='OPEN'
            )

    def test_should_validate_order_status_transitions(self):
        """Order status should follow valid state transitions"""
        order = Order(
            symbol='BTCUSDT',
            side='BUY',
            order_type='LIMIT',
            quantity=Decimal('1.0'),
            price=Decimal('49000.00'),
            status='PENDING'
        )

        # Valid transitions using update_status method
        order.update_status('PARTIALLY_FILLED')
        assert order.status == 'PARTIALLY_FILLED'

        order.update_status('FILLED')
        assert order.status == 'FILLED'

        # Invalid transition should raise error
        with pytest.raises(ValueError):
            order.update_status('PENDING')  # Can't go back to PENDING from FILLED


class TestSchemaRelationships:
    """Test relationships between schema models"""

    def test_should_establish_position_trade_relationship(self):
        """Trades should be linked to positions via foreign key"""
        position = Position(
            symbol='BTCUSDT',
            side='LONG',
            size=Decimal('1.0'),
            entry_price=Decimal('50000.00'),
            status='OPEN'
        )

        trade = Trade(
            position_id=position.id,
            symbol='BTCUSDT',
            side='LONG',
            quantity=Decimal('0.5'),
            price=Decimal('50000.00'),
            fee=Decimal('7.50'),
            execution_time=datetime.utcnow()
        )

        assert trade.position_id == position.id

    def test_should_establish_portfolio_risk_metrics_relationship(self):
        """RiskMetrics should be linked to Portfolio"""
        portfolio = Portfolio(
            total_equity_usdt=Decimal('100000.00'),
            available_balance_usdt=Decimal('75000.00')
        )

        risk_metrics = RiskMetrics(
            portfolio_id=portfolio.id,
            var_daily_usdt=Decimal('2000.00'),
            current_drawdown_pct=Decimal('0.05')
        )

        assert risk_metrics.portfolio_id == portfolio.id