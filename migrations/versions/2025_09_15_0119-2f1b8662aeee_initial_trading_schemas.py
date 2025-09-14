"""Initial trading schemas

Revision ID: 2f1b8662aeee
Revises:
Create Date: 2025-09-15 01:19:56.165662

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.types import DECIMAL


# revision identifiers, used by Alembic.
revision: str = '2f1b8662aeee'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Create all trading tables."""

    # Create enum types
    position_status_enum = sa.Enum('OPEN', 'CLOSED', 'LIQUIDATED', name='positionstatus')
    position_side_enum = sa.Enum('LONG', 'SHORT', name='positionside')
    order_status_enum = sa.Enum('NEW', 'PENDING', 'PARTIALLY_FILLED', 'FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED', name='orderstatus')
    order_type_enum = sa.Enum('MARKET', 'LIMIT', 'STOP_MARKET', 'STOP_LIMIT', name='ordertype')
    order_side_enum = sa.Enum('BUY', 'SELL', name='orderside')
    trade_side_enum = sa.Enum('BUY', 'SELL', name='tradeside')

    position_status_enum.create(op.get_bind())
    position_side_enum.create(op.get_bind())
    order_status_enum.create(op.get_bind())
    order_type_enum.create(op.get_bind())
    order_side_enum.create(op.get_bind())
    trade_side_enum.create(op.get_bind())

    # Create portfolios table
    op.create_table(
        'portfolios',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('uuid', sa.String(36), unique=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('initial_capital', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('current_capital', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('available_balance', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('used_margin', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('unrealized_pnl', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('realized_pnl', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('high_water_mark', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('max_drawdown', DECIMAL(precision=5, scale=4), server_default='0', nullable=False),
        sa.Column('is_paper_trading', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('current_var', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('max_var_limit', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('max_leverage', DECIMAL(precision=10, scale=2), server_default='1', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('created_by', sa.String(100), nullable=True),
        sa.Column('updated_by', sa.String(100), nullable=True),
        sa.Column('version', sa.Integer(), server_default='1', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
    )

    # Create positions table
    op.create_table(
        'positions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('uuid', sa.String(36), unique=True, nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', position_side_enum, nullable=False),
        sa.Column('size', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('entry_price', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('exit_price', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('leverage', DECIMAL(precision=10, scale=2), nullable=False),
        sa.Column('status', position_status_enum, nullable=False),
        sa.Column('unrealized_pnl', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('realized_pnl', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('margin_used', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('liquidation_price', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('created_by', sa.String(100), nullable=True),
        sa.Column('updated_by', sa.String(100), nullable=True),
        sa.Column('version', sa.Integer(), server_default='1', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
    )

    # Create orders table
    op.create_table(
        'orders',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('uuid', sa.String(36), unique=True, nullable=False),
        sa.Column('position_id', sa.Integer(), sa.ForeignKey('positions.id'), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', order_side_enum, nullable=False),
        sa.Column('type', order_type_enum, nullable=False),
        sa.Column('status', order_status_enum, nullable=False),
        sa.Column('quantity', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('filled_quantity', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('price', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('stop_price', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('time_in_force', sa.String(10), server_default='GTC', nullable=False),
        sa.Column('exchange_order_id', sa.String(100), nullable=True),
        sa.Column('client_order_id', sa.String(100), nullable=True),
        sa.Column('submitted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('filled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancelled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('created_by', sa.String(100), nullable=True),
        sa.Column('updated_by', sa.String(100), nullable=True),
        sa.Column('version', sa.Integer(), server_default='1', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
    )

    # Create trades table
    op.create_table(
        'trades',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('uuid', sa.String(36), unique=True, nullable=False),
        sa.Column('position_id', sa.Integer(), sa.ForeignKey('positions.id'), nullable=True),
        sa.Column('order_id', sa.Integer(), sa.ForeignKey('orders.id'), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', trade_side_enum, nullable=False),
        sa.Column('quantity', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('price', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('fee', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('fee_currency', sa.String(10), nullable=True),
        sa.Column('realized_pnl', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('execution_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('exchange_trade_id', sa.String(100), nullable=True),
        sa.Column('exchange_order_id', sa.String(100), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('created_by', sa.String(100), nullable=True),
        sa.Column('updated_by', sa.String(100), nullable=True),
        sa.Column('version', sa.Integer(), server_default='1', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
    )

    # Create market_data table
    op.create_table(
        'market_data',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('open_price', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('high_price', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('low_price', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('close_price', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('volume', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('quote_volume', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('number_of_trades', sa.Integer(), nullable=True),
        sa.Column('taker_buy_base_volume', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('taker_buy_quote_volume', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('sma_20', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('sma_50', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('ema_12', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('ema_26', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('rsi_14', DECIMAL(precision=5, scale=2), nullable=True),
        sa.Column('atr_14', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Create risk_metrics table
    op.create_table(
        'risk_metrics',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('portfolio_id', sa.Integer(), sa.ForeignKey('portfolios.id'), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('portfolio_value', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('var_1d', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('var_1w', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('expected_shortfall', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('max_drawdown', DECIMAL(precision=5, scale=4), nullable=True),
        sa.Column('current_drawdown', DECIMAL(precision=5, scale=4), server_default='0', nullable=False),
        sa.Column('volatility', DECIMAL(precision=5, scale=4), nullable=True),
        sa.Column('sharpe_ratio', DECIMAL(precision=8, scale=4), nullable=True),
        sa.Column('sortino_ratio', DECIMAL(precision=8, scale=4), nullable=True),
        sa.Column('largest_position_pct', DECIMAL(precision=5, scale=4), nullable=True),
        sa.Column('total_leverage', DECIMAL(precision=10, scale=2), nullable=True),
        sa.Column('correlation_risk', DECIMAL(precision=5, scale=4), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Create strategy_performances table
    op.create_table(
        'strategy_performances',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('portfolio_id', sa.Integer(), sa.ForeignKey('portfolios.id'), nullable=False),
        sa.Column('strategy_name', sa.String(100), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('allocated_capital', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('current_value', DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('unrealized_pnl', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('realized_pnl', DECIMAL(precision=20, scale=8), server_default='0', nullable=False),
        sa.Column('total_return', DECIMAL(precision=8, scale=6), nullable=True),
        sa.Column('sharpe_ratio', DECIMAL(precision=8, scale=4), nullable=True),
        sa.Column('max_drawdown', DECIMAL(precision=5, scale=4), nullable=True),
        sa.Column('win_rate', DECIMAL(precision=5, scale=4), nullable=True),
        sa.Column('profit_factor', DECIMAL(precision=8, scale=4), nullable=True),
        sa.Column('total_trades', sa.Integer(), server_default='0', nullable=False),
        sa.Column('winning_trades', sa.Integer(), server_default='0', nullable=False),
        sa.Column('losing_trades', sa.Integer(), server_default='0', nullable=False),
        sa.Column('largest_win', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('largest_loss', DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Create indexes
    op.create_index('idx_positions_symbol_status', 'positions', ['symbol', 'status'])
    op.create_index('idx_positions_status_created', 'positions', ['status', 'created_at'])
    op.create_index('idx_positions_uuid', 'positions', ['uuid'])

    op.create_index('idx_trades_symbol_execution_time', 'trades', ['symbol', 'execution_time'])
    op.create_index('idx_trades_position_id', 'trades', ['position_id'])
    op.create_index('idx_trades_exchange_trade_id', 'trades', ['exchange_trade_id'])

    op.create_index('idx_orders_status_symbol', 'orders', ['status', 'symbol'])
    op.create_index('idx_orders_exchange_order_id', 'orders', ['exchange_order_id'])
    op.create_index('idx_orders_client_order_id', 'orders', ['client_order_id'])

    op.create_index('idx_market_data_symbol_timeframe_timestamp', 'market_data', ['symbol', 'timeframe', 'timestamp'])
    op.create_index('idx_market_data_timestamp', 'market_data', ['timestamp'])

    op.create_index('idx_risk_metrics_portfolio_timestamp', 'risk_metrics', ['portfolio_id', 'timestamp'])

    op.create_index('idx_strategy_performance_portfolio_strategy', 'strategy_performances', ['portfolio_id', 'strategy_name'])
    op.create_index('idx_strategy_performance_timestamp', 'strategy_performances', ['timestamp'])

    # Create unique constraints
    op.create_unique_constraint('uq_trades_exchange_trade_id', 'trades', ['exchange_trade_id'])
    op.create_unique_constraint('uq_orders_exchange_order_id', 'orders', ['exchange_order_id'])
    op.create_unique_constraint('uq_market_data_symbol_timeframe_timestamp', 'market_data', ['symbol', 'timeframe', 'timestamp'])


def downgrade() -> None:
    """Downgrade schema - Drop all tables."""

    # Drop indexes first
    op.drop_index('idx_strategy_performance_timestamp', 'strategy_performances')
    op.drop_index('idx_strategy_performance_portfolio_strategy', 'strategy_performances')
    op.drop_index('idx_risk_metrics_portfolio_timestamp', 'risk_metrics')
    op.drop_index('idx_market_data_timestamp', 'market_data')
    op.drop_index('idx_market_data_symbol_timeframe_timestamp', 'market_data')
    op.drop_index('idx_orders_client_order_id', 'orders')
    op.drop_index('idx_orders_exchange_order_id', 'orders')
    op.drop_index('idx_orders_status_symbol', 'orders')
    op.drop_index('idx_trades_exchange_trade_id', 'trades')
    op.drop_index('idx_trades_position_id', 'trades')
    op.drop_index('idx_trades_symbol_execution_time', 'trades')
    op.drop_index('idx_positions_uuid', 'positions')
    op.drop_index('idx_positions_status_created', 'positions')
    op.drop_index('idx_positions_symbol_status', 'positions')

    # Drop tables
    op.drop_table('strategy_performances')
    op.drop_table('risk_metrics')
    op.drop_table('market_data')
    op.drop_table('trades')
    op.drop_table('orders')
    op.drop_table('positions')
    op.drop_table('portfolios')

    # Drop enum types
    sa.Enum(name='tradeside').drop(op.get_bind())
    sa.Enum(name='orderside').drop(op.get_bind())
    sa.Enum(name='ordertype').drop(op.get_bind())
    sa.Enum(name='orderstatus').drop(op.get_bind())
    sa.Enum(name='positionside').drop(op.get_bind())
    sa.Enum(name='positionstatus').drop(op.get_bind())
