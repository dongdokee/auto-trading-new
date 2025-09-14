# AutoTrading System - Project Structure Documentation
# 자동매매 시스템 프로젝트 구조 문서

**Generated**: 2025-09-14 (Updated: 2025-09-14)
**Phase**: 1.2 - Risk Management Module Completed ✅
**Status**: Position Sizing Engine Complete, Ready for Strategy Engine

## 📁 Complete Project Structure

```
AutoTradingNew/
├── 📋 Project Documentation
│   ├── README.md                     # Project overview and setup guide
│   ├── CLAUDE.md                     # Development guidance for Claude Code
│   ├── IMPLEMENTATION_PROGRESS.md    # Detailed progress tracking
│   └── PROJECT_STRUCTURE.md          # This file
│
├── 📚 Documentation (docs/)
│   ├── AGREED_IMPLEMENTATION_PLAN.md
│   ├── augmented-coding.md
│   ├── software-engineering-guide.md
│   ├── project-system-architecture.md
│   └── project-system-design/       # Detailed technical specifications
│       ├── 1-core-system.md
│       ├── 2-financial-engineering.md
│       ├── 3-strategy-engine.md
│       ├── 4-risk-management.md
│       ├── 5-portfolio-optimization.md
│       ├── 6-execution-engine.md
│       ├── 7-market-microstructure.md
│       ├── 8-backtesting.md
│       ├── 9-monitoring.md
│       ├── 10-infrastructure.md
│       ├── 11-data-quality.md
│       ├── 12-main-system.md
│       ├── 13-validation-checklist.md
│       └── 14-implementation-guide.md
│
├── 🐍 Source Code (src/)
│   ├── 🔧 core/                      # Core system components
│   │   └── __init__.py               # Configuration, logging, exceptions
│   ├── 🤖 trading_engine/            # Main trading coordination
│   │   └── __init__.py               # Coordinator, state manager
│   ├── ⚠️  risk_management/           # Risk control and Kelly optimization ✅ COMPLETED
│   │   ├── __init__.py               # Module initialization
│   │   ├── risk_management.py        # ✅ RiskController: Kelly, VaR, leverage, drawdown
│   │   ├── position_sizing.py        # ✅ PositionSizer: Multi-constraint position sizing
│   │   ├── position_management.py    # ✅ PositionManager: Position lifecycle management
│   │   └── CLAUDE.md                # ✅ Module-specific implementation context
│   ├── 📈 strategy_engine/           # Trading strategies and regime detection
│   │   ├── strategies/               # Individual strategy implementations
│   │   │   └── __init__.py
│   │   └── __init__.py               # Strategy manager, regime detector
│   ├── 💼 portfolio/                 # Portfolio management and optimization
│   │   └── __init__.py               # Optimizer, allocator
│   ├── ⚡ execution/                  # Order execution and routing
│   │   └── __init__.py               # Order executor, smart routing
│   ├── 📊 data/                      # Market data processing
│   │   └── __init__.py               # Market data, validation, feed handler
│   ├── 🏦 exchanges/                 # Exchange connectivity
│   │   └── __init__.py               # Base interface, Binance, Bybit connectors
│   ├── 📡 monitoring/                # System monitoring and metrics
│   │   └── __init__.py               # Metrics collector, alerter
│   └── 🛠️ utils/                      # Utility functions
│       └── __init__.py               # Math utils, time utils
│
├── 🧪 Testing Framework (tests/)
│   ├── unit/                         # Unit tests (TDD approach)
│   │   ├── test_risk_management/     # ✅ Risk management unit tests (51 tests)
│   │   │   ├── __init__.py
│   │   │   ├── test_risk_controller.py      # ✅ 22 tests - Kelly, VaR, leverage, drawdown
│   │   │   ├── test_position_sizing.py      # ✅ 15 tests - Multi-constraint sizing
│   │   │   └── test_position_management.py  # ✅ 14 tests - Position lifecycle
│   │   ├── test_strategy_engine/     # Strategy engine tests (planned)
│   │   │   └── __init__.py
│   │   ├── test_portfolio/           # Portfolio management tests (planned)
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── integration/                  # Integration tests
│   │   ├── test_risk_management_integration.py # ✅ 6 integration tests
│   │   ├── test_exchange_integration/ # Exchange connectivity tests (planned)
│   │   │   └── __init__.py
│   │   ├── test_data_pipeline/       # Data pipeline tests (planned)
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── fixtures/                     # Test data and fixtures
│   │   └── __init__.py
│   └── __init__.py
│
├── ⚙️ Configuration (config/)
│   ├── config.yaml                   # Main system configuration
│   ├── strategies.yaml               # Trading strategy parameters
│   └── risk_limits.yaml              # Risk management limits
│
├── 🚀 Scripts (scripts/)
│   ├── run_trading.py               # Main trading system launcher
│   ├── backtest.py                  # Backtesting system launcher
│   └── paper_trading.py             # Paper trading launcher
│
└── 📦 Project Configuration
    ├── requirements.txt              # Production dependencies
    ├── requirements-dev.txt          # Development dependencies
    ├── setup.py                     # Package configuration
    ├── pytest.ini                  # Testing configuration
    ├── .env.example                 # Environment variables template
    └── .gitignore                   # Git ignore patterns
```

## 🏗️ Architecture Principles

### 1. **Hexagonal Architecture** (Clean Architecture)
- **Core Business Logic**: Independent of external dependencies
- **Infrastructure Layer**: Database, API, external services
- **Application Layer**: Use cases and orchestration
- **Domain Layer**: Business rules and entities

### 2. **Test-Driven Development (TDD)**
- **Red-Green-Refactor**: Write failing test → Implement → Refactor
- **Test Categories**: Unit, Integration, End-to-end
- **Coverage Target**: 80%+ code coverage with meaningful tests

### 3. **Microservices Ready**
- **Loose Coupling**: Each component can be deployed independently
- **Event-Driven**: Asynchronous communication between services
- **Scalability**: Horizontal scaling capabilities built-in

### 4. **Domain-Driven Design**
- **Financial Concepts**: Code structure reflects trading domain
- **Bounded Contexts**: Clear separation of concerns
- **Ubiquitous Language**: Consistent terminology across codebase

## 🔧 Technology Stack

### **Core Dependencies** (requirements.txt)
```
📊 Data & Computation:     numpy, pandas, scipy, numba
💰 Quantitative Finance:   QuantLib, arch (GARCH), hmmlearn (HMM), cvxpy
🤖 Machine Learning:       scikit-learn, lightgbm
🗄️ Databases:             asyncpg (PostgreSQL), redis, sqlalchemy
⚡ Exchange Connectivity:  ccxt, websockets
🔄 Communication:          grpcio, aiohttp, httpx
📝 Configuration:          pydantic, python-dotenv, pyyaml
📡 Monitoring:             structlog, prometheus-client
```

### **Development Dependencies** (requirements-dev.txt)
```
🧪 Testing:               pytest, pytest-asyncio, pytest-cov, hypothesis
🎨 Code Quality:          black, isort, flake8, mypy, pylint
📚 Documentation:         sphinx, sphinx-rtd-theme
🔍 Profiling:             py-spy, memory-profiler, line-profiler
🏭 Test Data:             factory-boy, freezegun
```

## 📋 Implementation Priorities

### **✅ Phase 1.2: Risk Management Module (COMPLETED)**
1. ✅ **RiskController Implementation**: Kelly Criterion, VaR monitoring, leverage limits, drawdown tracking
2. ✅ **Position Sizing Engine**: Multi-constraint optimization (Kelly/ATR/VaR/liquidation safety)
3. ✅ **Position Management**: Complete position lifecycle with PnL tracking and stop management
4. ✅ **Comprehensive Testing**: 57 tests (51 unit + 6 integration) with full TDD methodology

### **🚀 Phase 2.1: Strategy Engine (NEXT PRIORITY)**
1. **Base Strategy Interface**: Abstract strategy pattern implementation
2. **Regime Detection System**: HMM/GARCH implementation for market state identification
3. **Signal Generation Pipeline**: Strategy signal processing and validation
4. **Strategy Integration**: Connect with Position Sizing Engine

### **Phase 2.2: Backtesting Framework**
1. **Historical Data Pipeline**: Data validation and preprocessing
2. **Walk-Forward Testing**: Time-series cross-validation framework
3. **Performance Analytics**: Risk-adjusted return metrics and reporting
4. **Strategy Validation**: Out-of-sample testing and optimization

### **Phase 3.1: Market Data Pipeline**
1. **Real-time Data Feeds**: WebSocket connections to exchanges
2. **Data Quality Framework**: Validation and cleaning pipelines
3. **Storage Integration**: TimescaleDB for time-series data
4. **Multi-exchange Aggregation**: Unified market data interface

## 🔒 Security & Risk Controls

### **API Security**
- Environment variable management (`.env.example` template provided)
- No hardcoded credentials or API keys
- Secure credential storage requirements

### **Risk Management**
- **Circuit Breakers**: Automatic trading halt on excessive losses
- **Position Limits**: Maximum exposure per asset/strategy
- **VaR Monitoring**: Real-time Value-at-Risk calculation
- **Kill Switches**: Emergency stop mechanisms

### **Data Security**
- Encrypted communication channels
- Audit logging for all trading activities
- Backup and disaster recovery procedures

## 🚀 Quick Start Commands

### **Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate    # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your actual values
```

### **Development Workflow**
```bash
# Run tests (TDD workflow)
pytest -v

# Run specific test category
pytest tests/unit/ -v              # Unit tests only
pytest tests/integration/ -v       # Integration tests only

# Code quality checks
black src/ tests/                  # Code formatting
isort src/ tests/                  # Import sorting
flake8 src/ tests/                # Linting
mypy src/                         # Type checking

# Run with coverage
pytest --cov=src --cov-report=html
```

### **System Execution**
```bash
# Main trading system
python scripts/run_trading.py

# Backtesting
python scripts/backtest.py

# Paper trading
python scripts/paper_trading.py
```

## 📊 Next Development Steps

### **✅ Completed Foundations**
1. ✅ **Environment Setup**: Anaconda environment `autotrading` configured with Python 3.10.18
2. ✅ **TDD Implementation**: Complete risk management module with 57 passing tests
3. ✅ **Risk Foundation**: RiskController, PositionSizer, and PositionManager fully implemented
4. ✅ **Integration Verified**: All components work together seamlessly

### **🚀 Next Immediate Actions (Phase 2.1)**
1. **Strategy Engine Foundation**: Create base strategy interface and abstract classes
2. **Market Regime Detection**: Implement HMM/GARCH models for market state identification
3. **Signal Processing Pipeline**: Build signal generation and validation framework
4. **Strategy-Risk Integration**: Connect strategy signals with position sizing engine

### **Development Methodology**
- **Follow TDD**: Red → Green → Refactor cycle
- **Commit Discipline**: Separate structural vs. behavioral changes
- **Documentation**: Update this file as structure evolves
- **Risk First**: Implement risk controls before trading logic

## 🔗 Key References

- **Development Guide**: `@docs/augmented-coding.md` (TDD methodology)
- **System Architecture**: `@docs/project-system-architecture.md`
- **Implementation Plan**: `@docs/AGREED_IMPLEMENTATION_PLAN.md`
- **Progress Tracking**: `@IMPLEMENTATION_PROGRESS.md`

---

**Status**: ✅ **Phase 1.2 Complete - Risk Management Module Fully Implemented**
**Current Achievement**: Complete position sizing engine with 57 passing tests
**Next Phase**: 2.1 - Strategy Engine Development
**Ready for**: Strategy development with established risk management foundation