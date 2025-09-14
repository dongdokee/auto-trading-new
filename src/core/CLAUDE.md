# Core Infrastructure Module - CLAUDE.md

This file provides specific guidance for Claude Code when working on the core infrastructure module.

## Module Overview

**Location**: `src/core/`
**Purpose**: Core infrastructure components including database, configuration, and system foundations
**Status**: ✅ **PHASE 2.2 COMPLETED: Complete Database Migration System** 🚀
**Last Updated**: 2025-09-15 (Database schemas, configuration management, repository pattern, and Alembic migration system completed)

## ⭐ CRITICAL IMPLEMENTATION CONTEXT ⭐

### 🚀 Successfully Completed: Complete Core Infrastructure System

#### **1. Database System** ✅ **FOUNDATION COMPONENT**
**Location**: `src/core/database/`
**Tests**: `tests/unit/test_core/test_database/` (48 test cases passing)
**Implementation Date**: 2025-09-15 (Complete database foundation with schemas, repository pattern, and Alembic migration system)

#### **2. Configuration Management** ✅ **SYSTEM COMPONENT**
**Location**: `src/core/config/`
**Tests**: `tests/unit/test_core/test_config/` (15 test cases passing)
**Implementation Date**: 2025-09-15 (Pydantic v2 configuration system with validation)

#### **3. Database Migration System** ✅ **PRODUCTION COMPONENT**
**Location**: `migrations/`, `src/core/database/models.py`
**Tests**: `tests/unit/test_core/test_database/test_migrations.py` (19 test cases passing)
**Implementation Date**: 2025-09-15 (Complete Alembic migration system with PostgreSQL support)

## 🔧 **Implementation Details**

### **Database Schema System** ✅ **COMPLETED**

#### **Core Schema Models** - `src/core/database/schemas/`
```python
# Base mixins for all models
from src.core.database.schemas.base import BaseModel, TimestampMixin, AuditMixin

# Trading-specific schemas
from src.core.database.schemas.trading_schemas import (
    Position, Trade, Order, MarketData, Portfolio, RiskMetrics, StrategyPerformance
)

# Portfolio schemas
from src.core.database.schemas.portfolio_schemas import (
    PortfolioState, AllocationHistory, PerformanceMetrics
)
```

#### **Key Architecture Decisions:**

1. **Composition Over Inheritance Pattern** - Clean mixin architecture:
```python
@dataclass
class Position:
    symbol: str
    side: str  # LONG/SHORT
    size: Decimal
    entry_price: Decimal
    leverage: Decimal
    status: str  # OPEN/CLOSED/LIQUIDATED

    def __post_init__(self):
        # Initialize mixins manually to avoid inheritance issues
        BaseModel.__init__(self)
        TimestampMixin.__init__(self)
        AuditMixin.__init__(self)
        self._validate_position()
```

2. **Type-Safe Enums** - Comprehensive validation:
```python
class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
```

3. **Business Logic Validation** - Domain-specific constraints:
```python
def _validate_position(self):
    """Validate position business rules"""
    if self.size <= 0:
        raise ValueError(f"Position size must be positive: {self.size}")
    if self.entry_price <= 0:
        raise ValueError(f"Entry price must be positive: {self.entry_price}")
    if self.status not in [status.value for status in PositionStatus]:
        raise ValueError(f"Invalid position status: {self.status}")
```

### **Repository Pattern System** ✅ **COMPLETED**

#### **Base Repository** - `src/core/database/repository/base.py`
```python
from src.core.database.repository import BaseRepository, RepositoryError

# Generic repository with full CRUD operations
class BaseRepository(Generic[T]):
    async def create(self, data: Dict[str, Any]) -> T
    async def get_by_id(self, entity_id: int) -> Optional[T]
    async def update(self, entity_id: int, updates: Dict[str, Any]) -> T
    async def delete(self, entity_id: int) -> bool
    async def list_all(self, limit: Optional[int] = None) -> List[T]
    async def find_by(self, **kwargs) -> List[T]
```

#### **Specialized Trading Repositories** - `src/core/database/repository/trading_repository.py`
```python
from src.core.database.repository import (
    PositionRepository, TradeRepository, OrderRepository, MarketDataRepository
)

# Position management
position_repo = PositionRepository(session)
open_positions = await position_repo.get_open_positions_by_symbol('BTCUSDT')
total_exposure = await position_repo.get_total_exposure_by_symbol('BTCUSDT')

# Trade analysis
trade_repo = TradeRepository(session)
trades = await trade_repo.get_trades_by_date_range(start_date, end_date)
total_pnl = await trade_repo.get_total_pnl_by_symbol('BTCUSDT')
```

### **Configuration Management System** ✅ **COMPLETED**

#### **Pydantic v2 Configuration Models** - `src/core/config/models.py`
```python
from src.core.config import AppConfig, DatabaseConfig, ExchangeConfig, RiskConfig

# Complete application configuration
class AppConfig(BaseModel):
    database: DatabaseConfig
    exchanges: Dict[str, ExchangeConfig]
    risk: RiskConfig
    trading: TradingConfig
    monitoring: MonitoringConfig
    system: SystemConfig

    @model_validator(mode='after')
    def validate_cross_config_relationships(self):
        # Validate relationships between config sections
        return self
```

#### **Configuration Loading and Validation** - `src/core/config/loader.py`
```python
from src.core.config import ConfigLoader, ConfigValidator

# Load configuration with precedence: CLI -> ENV -> YAML -> defaults
loader = ConfigLoader()
config = loader.load_config(
    yaml_path="config/trading.yaml",
    env_prefix="TRADING_"
)

# Validate for production requirements
validator = ConfigValidator()
validator.validate_production_requirements(config)
validator.validate_security_requirements(config)
```

#### **Key Configuration Features:**

1. **Environment Variable Support**:
```python
# Automatic environment variable mapping
TRADING_DATABASE__HOST=localhost
TRADING_DATABASE__PORT=5432
TRADING_EXCHANGES__BINANCE__API_KEY=your_api_key
TRADING_RISK__MAX_PORTFOLIO_RISK_PCT=0.02
```

2. **YAML File Loading**:
```yaml
# config/trading.yaml
database:
  host: localhost
  port: 5432
  name: trading_db

exchanges:
  binance:
    api_key: ${BINANCE_API_KEY}
    testnet: true

risk:
  max_portfolio_risk_pct: 0.02
  max_position_risk_pct: 0.01
```

3. **Production Security Validation**:
```python
def validate_security_requirements(self, config: AppConfig):
    """Ensure production security standards"""
    for exchange_name, exchange_config in config.exchanges.items():
        if not exchange_config.api_key or len(exchange_config.api_key) < 10:
            raise ValueError(f"Invalid API key for {exchange_name}")
```

### **Database Migration System** ✅ **COMPLETED**

#### **Alembic Environment Setup** - `alembic.ini` & `migrations/env.py`
```python
# Dynamic database URL resolution with fallback hierarchy
def get_url():
    """Get database URL from configuration or environment."""
    # Priority: ENV -> Config System -> alembic.ini
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    try:
        from src.core.config import ConfigLoader
        loader = ConfigLoader()
        app_config = loader.load_config()
        db_config = app_config.database
        return f"postgresql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.name}"
    except Exception:
        return config.get_main_option("sqlalchemy.url")
```

#### **SQLAlchemy Models for Migrations** - `src/core/database/models.py`
```python
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.types import DECIMAL

Base = declarative_base()

class Position(Base, TimestampMixin, AuditMixin):
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    side = Column(SQLEnum(PositionSide), nullable=False)
    size = Column(DECIMAL(precision=20, scale=8), nullable=False)
    entry_price = Column(DECIMAL(precision=20, scale=8), nullable=False)
    leverage = Column(DECIMAL(precision=10, scale=2), nullable=False)
    status = Column(SQLEnum(PositionStatus), nullable=False)
```

#### **Initial Migration Script** - `migrations/versions/2025_09_15_*_initial_trading_schemas.py`
```python
def upgrade() -> None:
    """Upgrade schema - Create all trading tables."""
    # Create PostgreSQL enums
    position_status_enum = sa.Enum('OPEN', 'CLOSED', 'LIQUIDATED', name='positionstatus')
    position_status_enum.create(op.get_bind())

    # Create tables with proper indexes and constraints
    op.create_table('positions', ...)
    op.create_index('idx_positions_symbol_status', 'positions', ['symbol', 'status'])
    op.create_unique_constraint('uq_positions_uuid', 'positions', ['uuid'])

def downgrade() -> None:
    """Downgrade schema - Drop all tables."""
    # Complete rollback capability
    op.drop_table('positions')
    sa.Enum(name='positionstatus').drop(op.get_bind())
```

#### **Key Migration Features:**

1. **Production-Ready Migration System**:
- Timestamped migration files with proper versioning
- Complete upgrade() and downgrade() operations for rollback
- PostgreSQL-specific enums for type safety
- Performance-optimized indexes for trading queries

2. **Financial Data Precision**:
- DECIMAL(20,8) for all monetary amounts (prices, quantities, PnL)
- DECIMAL(10,2) for leverage ratios
- DECIMAL(5,4) for percentages and performance ratios
- Timezone-aware DATETIME for all timestamps

3. **Database Schema Coverage**:
- 7 core tables: positions, trades, orders, market_data, portfolios, risk_metrics, strategy_performances
- 6 PostgreSQL enums for data integrity
- 15 performance indexes for high-frequency trading
- 3 unique constraints for exchange integration
- Complete audit trail with created_by/updated_by tracking

## 📊 **Testing Coverage**

### **Database Tests** ✅ **48 TESTS PASSING**
- **Schema Validation Tests**: 14 tests covering all models and validation rules
- **Repository Pattern Tests**: 15 tests covering CRUD operations and domain queries
- **Migration System Tests**: 19 tests covering Alembic environment, scripts, operations, and integration

### **Configuration Tests** ✅ **15 TESTS PASSING**
- **Model Validation Tests**: 7 tests covering all config models and validators
- **Loading System Tests**: 4 tests covering environment and YAML loading
- **Security Validation Tests**: 4 tests covering production requirements

### **Migration Tests** ✅ **19 TESTS PASSING**
- **Migration Environment Tests**: 4 tests for Alembic setup validation
- **Migration Script Tests**: 3 tests for script syntax and metadata validation
- **Migration Operations Tests**: 3 tests for table creation and schema validation
- **Migration Rollback Tests**: 1 test for downgrade operations
- **Migration Configuration Tests**: 3 tests for URL and configuration handling
- **Migration Integration Tests**: 3 tests for system integration
- **Migration Performance Tests**: 2 tests for indexes and data types

### **Test Examples**
```python
def test_should_create_position_with_validation():
    """Position should validate business rules on creation"""
    position = Position(
        symbol='BTCUSDT',
        side='LONG',
        size=Decimal('1.5'),
        entry_price=Decimal('50000.00'),
        leverage=Decimal('3.0'),
        status='OPEN'
    )
    assert position.symbol == 'BTCUSDT'
    assert position.status == 'OPEN'

@pytest.mark.asyncio
async def test_should_calculate_total_exposure():
    """Repository should calculate total position exposure"""
    position_repo = PositionRepository(mock_session)

    # Create multiple positions
    await position_repo.create({...})
    await position_repo.create({...})

    total_exposure = await position_repo.get_total_exposure_by_symbol('BTCUSDT')
    assert total_exposure == expected_total
```

## 🚀 **Usage Patterns & Best Practices**

### **1. Database Schema Usage**
```python
# Always use proper validation
from src.core.database.schemas.trading_schemas import Position, PositionStatus

position = Position(
    symbol='BTCUSDT',
    side='LONG',
    size=Decimal('1.5'),
    entry_price=Decimal('50000.00'),
    leverage=Decimal('3.0'),
    status=PositionStatus.OPEN.value  # Use enum values
)
```

### **2. Repository Pattern Usage**
```python
# Use dependency injection for repositories
class TradingService:
    def __init__(self, position_repo: PositionRepository, trade_repo: TradeRepository):
        self.position_repo = position_repo
        self.trade_repo = trade_repo

    async def close_position(self, position_id: int, close_price: Decimal):
        # Update position
        position = await self.position_repo.update(position_id, {
            'status': PositionStatus.CLOSED.value,
            'exit_price': close_price
        })

        # Create closing trade
        trade_data = {
            'symbol': position.symbol,
            'side': 'SELL' if position.side == 'LONG' else 'BUY',
            'quantity': position.size,
            'price': close_price,
            'position_id': position_id
        }
        await self.trade_repo.create(trade_data)
```

### **3. Configuration Management Usage**
```python
# Load and validate configuration
from src.core.config import ConfigLoader, ConfigValidator

async def initialize_trading_system():
    # Load configuration
    loader = ConfigLoader()
    config = loader.load_config(
        yaml_path="config/production.yaml",
        env_prefix="TRADING_"
    )

    # Validate for production
    validator = ConfigValidator()
    validator.validate_production_requirements(config)
    validator.validate_security_requirements(config)

    # Initialize components with config
    database_session = create_database_session(config.database)
    exchange_clients = create_exchange_clients(config.exchanges)
    risk_controller = RiskController(config.risk)

    return TradingSystem(database_session, exchange_clients, risk_controller)
```

### **4. Database Migration Usage**
```python
# Running migrations in different environments
from alembic import command
from alembic.config import Config

# Development environment
# Set DATABASE_URL environment variable
os.environ["DATABASE_URL"] = "postgresql://dev_user:dev_pass@localhost/trading_dev"

# Run migration to latest version
alembic_cfg = Config("alembic.ini")
command.upgrade(alembic_cfg, "head")

# Production environment with environment variables
os.environ["DATABASE_URL"] = "postgresql://prod_user:prod_pass@prod_host/trading_prod"

# Create new migration (when schema changes)
command.revision(alembic_cfg, autogenerate=True, message="Add new trading feature")

# Rollback to previous version (if needed)
command.downgrade(alembic_cfg, "-1")
```

## 🔗 **Integration Points**

### **With Risk Management** ✅ **READY**
- Database schemas support all risk management data models
- Repository pattern provides risk metrics tracking
- Configuration system includes comprehensive risk parameters

### **With Strategy Engine** 🔄 **READY**
- Strategy performance tracking schemas implemented
- Portfolio state management ready for strategy allocation
- Configuration framework supports strategy parameters

### **With Order Execution** 🔄 **READY**
- Order management schemas with full lifecycle tracking
- Trade execution recording with detailed metadata
- Market data schemas for execution context

### **With Backtesting** 🔄 **READY**
- Historical data schemas support backtesting requirements
- Performance tracking ready for backtest result analysis
- Configuration system supports backtesting parameters

## 📋 **Configuration Guide**

### **Development Configuration**
```yaml
# config/development.yaml
database:
  host: localhost
  port: 5432
  name: trading_dev
  echo_queries: true

exchanges:
  binance:
    testnet: true
    api_key: ${DEV_BINANCE_API_KEY}

risk:
  max_portfolio_risk_pct: 0.05  # Higher for development
  enable_paper_trading: true

logging:
  level: DEBUG
  log_to_file: false
```

### **Production Configuration**
```yaml
# config/production.yaml
database:
  host: ${DB_HOST}
  port: ${DB_PORT}
  name: trading_prod
  pool_size: 20
  ssl_required: true

exchanges:
  binance:
    testnet: false
    api_key: ${PROD_BINANCE_API_KEY}
    api_secret: ${PROD_BINANCE_API_SECRET}

risk:
  max_portfolio_risk_pct: 0.02
  enable_paper_trading: false
  require_manual_approval: true

monitoring:
  enable_alerts: true
  alert_webhook: ${SLACK_WEBHOOK_URL}
```

## ⚠️ **Important Notes**

### **Database Considerations**
- ✅ All schemas use composition pattern to avoid multiple inheritance issues
- ✅ Decimal types used for all financial calculations
- ✅ Comprehensive validation prevents invalid data
- ✅ Audit trails and soft delete capabilities built-in

### **Repository Pattern Guidelines**
- ✅ Always use async methods for database operations
- ✅ Specialized repositories provide domain-specific query methods
- ✅ Error handling includes comprehensive logging
- ✅ Transaction management handled at repository level

### **Configuration Security**
- ✅ API keys and secrets never stored in plaintext
- ✅ Environment variable interpolation supported
- ✅ Production validation ensures security requirements
- ✅ Configuration encryption ready for implementation

### **Migration System Guidelines**
- ✅ Always test migrations on development data before production
- ✅ Use environment variables for database URLs in production
- ✅ Keep rollback capability for all schema changes
- ✅ Monitor migration performance on large datasets
- ✅ Validate data integrity after migration completion

## 🎯 **Next Development Priorities**

### **Immediate (Phase 3.1 - Business Logic Implementation)**
1. **Risk Management Integration**: Connect risk system with database models
2. **Strategy Engine Integration**: Implement strategy performance tracking
3. **Order Execution Integration**: Connect order system with database models

### **Medium-term (Phase 3.2 - Advanced Business Logic)**
1. **Database Connection Pooling**: Production-grade connection management
2. **Query Optimization**: Advanced query patterns for high-performance
3. **Caching Layer**: Redis integration for frequently accessed data

### **Long-term (Phase 3+ - Production)**
1. **Database Sharding**: Horizontal scaling for high-volume trading
2. **Event Sourcing**: Event-driven architecture for audit and replay
3. **Multi-Tenant Support**: Support for multiple trading accounts/strategies

---

## 📚 **Reference Documentation**

### **📋 Main Project References**
- **🎯 Development Guide**: `@CLAUDE.md` - Core development guidance and navigation
- **📊 Progress Status**: `@IMPLEMENTATION_PROGRESS.md` - Overall project status
- **🗺️ Implementation Plan**: `@docs/AGREED_IMPLEMENTATION_PLAN.md` - Complete roadmap

### **📂 Related Module Documentation**
- **⚠️ Risk Management**: `@src/risk_management/CLAUDE.md` - Risk system details
- **📈 Strategy Engine**: `@src/strategy_engine/CLAUDE.md` - Strategy implementation
- **🔧 Utility Functions**: `@src/utils/CLAUDE.md` - Enhanced utility functions

### **📖 Technical Documentation**
- **🧪 TDD Methodology**: `@docs/augmented-coding.md` - Development discipline used
- **🏛️ System Architecture**: `@docs/project-system-architecture.md` - Overall architecture
- **🔧 Engineering Guide**: `@docs/software-engineering-guide.md` - Best practices followed

---

**Status**: ✅ **Phase 2.2 Complete - Complete Database Migration System**
**Current Achievement**: Complete database schema system, configuration management, repository pattern, and Alembic migration system
**Next Phase**: 3.1 - Business Logic Implementation (Risk Management, Strategy Engine, Order Execution)
**Integration Ready**: All infrastructure components ready for business logic implementation with production-grade database migration system