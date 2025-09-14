"""
Tests for repository pattern implementation.
Following TDD methodology: Red -> Green -> Refactor
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional
import asyncio

# These imports will fail initially (Red phase)
# We'll implement them to make tests pass (Green phase)
try:
    from src.core.database.repository.base import BaseRepository, RepositoryError
    from src.core.database.repository.trading_repository import (
        PositionRepository, TradeRepository, OrderRepository, MarketDataRepository
    )
    from src.core.database.schemas.trading_schemas import Position, Trade, Order, MarketData
except ImportError:
    pytest.skip("Repository pattern not yet implemented", allow_module_level=True)


class MockDatabaseSession:
    """Mock database session for testing"""

    def __init__(self):
        self.data = {}
        self.auto_increment = 1
        self.committed = False
        self.rolled_back = False

    async def add(self, item):
        """Add item to mock session"""
        if not hasattr(item, 'id') or item.id is None:
            item.id = self.auto_increment
            self.auto_increment += 1
        self.data[item.id] = item

    async def commit(self):
        """Commit mock transaction"""
        self.committed = True

    async def rollback(self):
        """Rollback mock transaction"""
        self.rolled_back = True

    async def query(self, model_class):
        """Return mock query object"""
        return MockQuery(model_class, self.data)

    async def get(self, model_class, item_id):
        """Get item by ID"""
        return self.data.get(item_id)


class MockQuery:
    """Mock query object"""

    def __init__(self, model_class, data):
        self.model_class = model_class
        self.data = data
        self.filters = []
        self.order_by_field = None
        self.limit_count = None

    async def filter(self, **kwargs):
        """Add filter to query"""
        self.filters.append(kwargs)
        return self

    async def order_by(self, field):
        """Add ordering to query"""
        self.order_by_field = field
        return self

    async def limit(self, count):
        """Add limit to query"""
        self.limit_count = count
        return self

    async def all(self):
        """Return all matching items"""
        items = list(self.data.values())

        # Apply filters
        for filter_dict in self.filters:
            for key, value in filter_dict.items():
                items = [item for item in items if getattr(item, key, None) == value]

        # Apply limit
        if self.limit_count:
            items = items[:self.limit_count]

        return items

    async def first(self):
        """Return first matching item"""
        items = await self.all()
        return items[0] if items else None


class TestBaseRepository:
    """Test base repository functionality"""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session"""
        return MockDatabaseSession()

    @pytest.fixture
    def base_repository(self, mock_session):
        """Create base repository with mock session"""
        return BaseRepository(Position, mock_session)

    def test_should_create_repository_with_model_and_session(self, mock_session):
        """Should create repository with model class and session"""
        repository = BaseRepository(Position, mock_session)

        assert repository.model == Position
        assert repository.session == mock_session

    @pytest.mark.asyncio
    async def test_should_create_new_entity(self, base_repository):
        """Should create new entity in repository"""
        position_data = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'size': Decimal('1.5'),
            'entry_price': Decimal('50000.00'),
            'leverage': Decimal('3.0'),
            'status': 'OPEN'
        }

        position = await base_repository.create(position_data)

        assert position.symbol == 'BTCUSDT'
        assert position.side == 'LONG'
        assert position.size == Decimal('1.5')
        assert position.id is not None

    @pytest.mark.asyncio
    async def test_should_get_entity_by_id(self, base_repository):
        """Should retrieve entity by ID"""
        # First create a position
        position_data = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'size': Decimal('1.0'),
            'entry_price': Decimal('50000.00'),
            'leverage': Decimal('2.0'),
            'status': 'OPEN'
        }

        created_position = await base_repository.create(position_data)
        position_id = created_position.id

        # Then retrieve it
        retrieved_position = await base_repository.get_by_id(position_id)

        assert retrieved_position is not None
        assert retrieved_position.id == position_id
        assert retrieved_position.symbol == 'BTCUSDT'

    @pytest.mark.asyncio
    async def test_should_update_existing_entity(self, base_repository):
        """Should update existing entity"""
        # Create position
        position_data = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'size': Decimal('1.0'),
            'entry_price': Decimal('50000.00'),
            'leverage': Decimal('2.0'),
            'status': 'OPEN'
        }

        position = await base_repository.create(position_data)

        # Update position
        updates = {'status': 'CLOSED', 'size': Decimal('0.0')}
        updated_position = await base_repository.update(position.id, updates)

        assert updated_position.status == 'CLOSED'
        assert updated_position.size == Decimal('0.0')
        assert updated_position.symbol == 'BTCUSDT'  # Unchanged

    @pytest.mark.asyncio
    async def test_should_delete_entity(self, base_repository):
        """Should delete entity from repository"""
        # Create position
        position_data = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'size': Decimal('1.0'),
            'entry_price': Decimal('50000.00'),
            'leverage': Decimal('2.0'),
            'status': 'OPEN'
        }

        position = await base_repository.create(position_data)
        position_id = position.id

        # Delete position
        deleted = await base_repository.delete(position_id)

        assert deleted is True

        # Verify it's gone
        retrieved = await base_repository.get_by_id(position_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_should_list_all_entities(self, base_repository):
        """Should list all entities in repository"""
        # Create multiple positions
        for i in range(3):
            position_data = {
                'symbol': f'BTC{i}USDT',
                'side': 'LONG',
                'size': Decimal('1.0'),
                'entry_price': Decimal('50000.00'),
                'leverage': Decimal('2.0'),
                'status': 'OPEN'
            }
            await base_repository.create(position_data)

        # List all
        positions = await base_repository.list_all()

        assert len(positions) == 3
        symbols = [p.symbol for p in positions]
        assert 'BTC0USDT' in symbols
        assert 'BTC1USDT' in symbols
        assert 'BTC2USDT' in symbols

    @pytest.mark.asyncio
    async def test_should_find_entities_by_criteria(self, base_repository):
        """Should find entities matching criteria"""
        # Create positions with different statuses
        open_position_data = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'size': Decimal('1.0'),
            'entry_price': Decimal('50000.00'),
            'leverage': Decimal('2.0'),
            'status': 'OPEN'
        }

        closed_position_data = {
            'symbol': 'ETHUSDT',
            'side': 'SHORT',
            'size': Decimal('2.0'),
            'entry_price': Decimal('3000.00'),
            'leverage': Decimal('3.0'),
            'status': 'CLOSED'
        }

        await base_repository.create(open_position_data)
        await base_repository.create(closed_position_data)

        # Find open positions
        open_positions = await base_repository.find_by(status='OPEN')

        assert len(open_positions) == 1
        assert open_positions[0].symbol == 'BTCUSDT'
        assert open_positions[0].status == 'OPEN'


class TestPositionRepository:
    """Test position-specific repository functionality"""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session"""
        return MockDatabaseSession()

    @pytest.fixture
    def position_repository(self, mock_session):
        """Create position repository with mock session"""
        return PositionRepository(mock_session)

    @pytest.mark.asyncio
    async def test_should_find_open_positions_by_symbol(self, position_repository):
        """Should find open positions for specific symbol"""
        # Create mixed positions
        btc_open_data = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'size': Decimal('1.0'),
            'entry_price': Decimal('50000.00'),
            'leverage': Decimal('2.0'),
            'status': 'OPEN'
        }

        btc_closed_data = {
            'symbol': 'BTCUSDT',
            'side': 'SHORT',
            'size': Decimal('0.5'),
            'entry_price': Decimal('51000.00'),
            'leverage': Decimal('2.0'),
            'status': 'CLOSED'
        }

        eth_open_data = {
            'symbol': 'ETHUSDT',
            'side': 'LONG',
            'size': Decimal('2.0'),
            'entry_price': Decimal('3000.00'),
            'leverage': Decimal('3.0'),
            'status': 'OPEN'
        }

        await position_repository.create(btc_open_data)
        await position_repository.create(btc_closed_data)
        await position_repository.create(eth_open_data)

        # Find BTC open positions
        btc_open_positions = await position_repository.get_open_positions_by_symbol('BTCUSDT')

        assert len(btc_open_positions) == 1
        assert btc_open_positions[0].symbol == 'BTCUSDT'
        assert btc_open_positions[0].status == 'OPEN'
        assert btc_open_positions[0].side == 'LONG'

    @pytest.mark.asyncio
    async def test_should_calculate_total_exposure_by_symbol(self, position_repository):
        """Should calculate total exposure for symbol"""
        # Create multiple positions for same symbol
        position1_data = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'size': Decimal('1.5'),
            'entry_price': Decimal('50000.00'),
            'leverage': Decimal('2.0'),
            'status': 'OPEN'
        }

        position2_data = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'size': Decimal('0.8'),
            'entry_price': Decimal('51000.00'),
            'leverage': Decimal('3.0'),
            'status': 'OPEN'
        }

        await position_repository.create(position1_data)
        await position_repository.create(position2_data)

        # Calculate total exposure
        total_exposure = await position_repository.get_total_exposure_by_symbol('BTCUSDT')

        # Should be sum of sizes for open positions
        expected_exposure = Decimal('1.5') + Decimal('0.8')
        assert total_exposure == expected_exposure

    @pytest.mark.asyncio
    async def test_should_get_positions_by_status(self, position_repository):
        """Should get positions filtered by status"""
        # Create positions with different statuses (using valid PositionStatus values)
        for status in ['OPEN', 'CLOSED', 'LIQUIDATED']:
            position_data = {
                'symbol': 'BTCUSDT',
                'side': 'LONG',
                'size': Decimal('1.0'),
                'entry_price': Decimal('50000.00'),
                'leverage': Decimal('2.0'),
                'status': status
            }
            await position_repository.create(position_data)

        # Get only open positions
        open_positions = await position_repository.get_positions_by_status('OPEN')

        assert len(open_positions) == 1
        assert open_positions[0].status == 'OPEN'


class TestTradeRepository:
    """Test trade-specific repository functionality"""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session"""
        return MockDatabaseSession()

    @pytest.fixture
    def trade_repository(self, mock_session):
        """Create trade repository with mock session"""
        return TradeRepository(mock_session)

    @pytest.mark.asyncio
    async def test_should_get_trades_by_date_range(self, trade_repository):
        """Should get trades within date range"""
        # Create trades with different timestamps
        base_time = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        for i in range(3):
            trade_data = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': Decimal('1.0'),
                'price': Decimal('50000.00'),
                'fee': Decimal('25.0'),
                'fee_currency': 'USDT',
                'exchange_trade_id': f'trade_{i}',
                'execution_time': base_time.replace(hour=12 + i)
            }
            await trade_repository.create(trade_data)

        # Get trades in specific range (inclusive end)
        start_time = base_time.replace(hour=12)
        end_time = base_time.replace(hour=14)

        trades = await trade_repository.get_trades_by_date_range(start_time, end_time)

        assert len(trades) == 3  # Should include trades at 12:00, 13:00, and 14:00 (inclusive range)
        for trade in trades:
            assert start_time <= trade.execution_time <= end_time

    @pytest.mark.asyncio
    async def test_should_get_total_pnl_by_symbol(self, trade_repository):
        """Should calculate total PnL for symbol"""
        # Create buy and sell trades
        buy_trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': Decimal('1.0'),
            'price': Decimal('50000.00'),
            'fee': Decimal('25.0'),
            'fee_currency': 'USDT',
            'realized_pnl': Decimal('0.0'),
            'exchange_trade_id': 'buy_trade_1'
        }

        sell_trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'SELL',
            'quantity': Decimal('1.0'),
            'price': Decimal('52000.00'),
            'fee': Decimal('26.0'),
            'fee_currency': 'USDT',
            'realized_pnl': Decimal('1949.0'),  # 52000 - 50000 - 25 - 26 = 1949
            'exchange_trade_id': 'sell_trade_1'
        }

        await trade_repository.create(buy_trade_data)
        await trade_repository.create(sell_trade_data)

        # Calculate total PnL
        total_pnl = await trade_repository.get_total_pnl_by_symbol('BTCUSDT')

        assert total_pnl == Decimal('1949.0')


class TestRepositoryErrorHandling:
    """Test repository error handling"""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session"""
        return MockDatabaseSession()

    @pytest.mark.asyncio
    async def test_should_handle_entity_not_found(self, mock_session):
        """Should handle entity not found gracefully"""
        repository = BaseRepository(Position, mock_session)

        # Try to get non-existent entity
        result = await repository.get_by_id(999)

        assert result is None

    @pytest.mark.asyncio
    async def test_should_handle_invalid_update_id(self, mock_session):
        """Should handle update with invalid ID"""
        repository = BaseRepository(Position, mock_session)

        # Try to update non-existent entity
        with pytest.raises(RepositoryError, match="Entity with id 999 not found"):
            await repository.update(999, {'status': 'CLOSED'})

    @pytest.mark.asyncio
    async def test_should_handle_invalid_delete_id(self, mock_session):
        """Should handle delete with invalid ID"""
        repository = BaseRepository(Position, mock_session)

        # Try to delete non-existent entity
        result = await repository.delete(999)

        assert result is False