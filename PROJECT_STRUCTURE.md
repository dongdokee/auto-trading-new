# AutoTrading System - Project Structure Documentation
# 자동매매 시스템 프로젝트 구조 문서

**Generated**: 2025-09-14
**Phase**: 1.1 - Project Structure Setup Completed ✅
**Status**: Ready for Implementation

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
│   ├── ⚠️  risk_management/           # Risk control and Kelly optimization
│   │   └── __init__.py               # Risk controller, VaR, position sizing
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
│   │   ├── test_risk_management/     # Risk management tests
│   │   │   └── __init__.py
│   │   ├── test_strategy_engine/     # Strategy engine tests
│   │   │   └── __init__.py
│   │   ├── test_portfolio/           # Portfolio management tests
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── integration/                  # Integration tests
│   │   ├── test_exchange_integration/ # Exchange connectivity tests
│   │   │   └── __init__.py
│   │   ├── test_data_pipeline/       # Data pipeline tests
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

### **Phase 1.2: TDD Foundation (Next)**
1. **First Failing Test**: Write `test_risk_controller.py`
2. **Core Configuration**: Implement config loading system
3. **Logging System**: Structured logging setup
4. **Basic Risk Controller**: Minimum viable implementation

### **Phase 1.3: Market Data Pipeline**
1. **Data Validation Framework**: Ensure data quality
2. **WebSocket Connections**: Real-time market data
3. **Data Storage**: TimescaleDB integration
4. **Feed Handler**: Multi-exchange data aggregation

### **Phase 1.4: Strategy Engine**
1. **Base Strategy Interface**: Abstract strategy pattern
2. **Regime Detection**: HMM/GARCH implementation
3. **Signal Generation**: Strategy signal pipeline
4. **Backtesting Framework**: Historical strategy validation

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

### **Immediate Actions Required**
1. **Environment Setup**: Copy `.env.example` to `.env` and configure
2. **First Test**: Write failing test for `RiskController`
3. **TDD Cycle**: Implement minimal passing code
4. **Configuration**: Load and validate system configuration

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

**Status**: ✅ **Project Structure Setup Complete**
**Next Phase**: 1.2 - TDD Foundation and Risk Controller Implementation
**Ready for**: Core component development with test-first approach