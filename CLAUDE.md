# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Korean cryptocurrency futures automated trading system (코인 선물 자동매매 시스템) currently transitioning from design phase to implementation. The system implements advanced quantitative trading strategies with sophisticated risk management and portfolio optimization.

## Current Development Status

**Project Phase**: Phase 1.2 - Core Risk Management Module Implemented
**Implementation Progress**: 25% (Design Complete, Core Risk Management Complete)
**Last Updated**: 2025-09-14
**Status**: 🟢 RiskController class fully implemented with TDD, ready for next modules

📋 **Progress Tracking**: `@IMPLEMENTATION_PROGRESS.md` - Detailed progress and next steps
📋 **Implementation Plan**: `@docs/AGREED_IMPLEMENTATION_PLAN.md` - Complete 5-phase roadmap

## 📁 **Modular Context Management Strategy** ⭐

**IMPORTANT**: This project uses **modular CLAUDE.md files** for better context management:

### 📂 Module-Specific Context Files:
- **`@src/risk_management/CLAUDE.md`** ✅ - Risk management module implementation details
- **`@src/strategy_engine/CLAUDE.md`** (planned) - Strategy implementation context
- **`@src/execution/CLAUDE.md`** (planned) - Order execution context
- **`@src/portfolio/CLAUDE.md`** (planned) - Portfolio management context
- **`@src/data/CLAUDE.md`** (planned) - Data pipeline context

### 🎯 **Context Storage Pattern**:
- **Main CLAUDE.md**: Overall project guidance, architecture, environment setup
- **Module CLAUDE.md**: Specific implementation details, completed work, API interfaces
- **When working on a module**: Always check both main + module CLAUDE.md files

### ✅ **Already Implemented**:
- **Risk Management Module**: Complete with TDD implementation, tests, and context documentation

## 📁 **Project Structure Adherence** ⭐

**CRITICAL**: Always follow the structure defined in `@PROJECT_STRUCTURE.md`:

### 🏗️ **Source Code Structure**:
```
src/
├── core/                 # Core system components (config, logging, exceptions)
├── risk_management/      # Risk control and Kelly optimization ⚠️
├── strategy_engine/      # Trading strategies and regime detection
├── portfolio/            # Portfolio management and optimization
├── execution/            # Order execution and routing
├── data/                # Market data processing
├── exchanges/           # Exchange connectivity
├── monitoring/          # System monitoring and metrics
└── utils/               # Utility functions
```

### 🧪 **Test Structure**:
```
tests/
├── unit/                # Unit tests (TDD approach)
│   ├── test_risk_management/     # Risk management tests ⚠️
│   ├── test_strategy_engine/     # Strategy engine tests
│   ├── test_portfolio/           # Portfolio management tests
│   └── ...
├── integration/         # Integration tests
└── fixtures/           # Test data and fixtures
```

### ⚠️ **IMPORTANT RULES**:
1. **NO files directly in `src/core/`** - Only configuration, logging, exceptions
2. **Each module gets its own directory** under `src/`
3. **Test structure mirrors source structure** under `tests/unit/`
4. **Always check `@PROJECT_STRUCTURE.md` before creating new files**

## Architecture Overview

The system follows a microservices architecture with these main components:

- **Trading Engine**: Main coordination hub using Python/asyncio
- **Risk Manager**: Kelly optimization and VaR-based risk control
- **Strategy Engine**: Multi-strategy execution with regime detection
- **Order Executor**: Smart order routing and execution
- **Data Feed Service**: Real-time market data collection
- **Infrastructure**: PostgreSQL, TimescaleDB, Redis, RabbitMQ

Key architectural patterns:
- Event-driven architecture for loose coupling
- CQRS (Command Query Responsibility Segregation) for data operations
- Hexagonal architecture for clean separation of concerns

📋 **Architecture Reference**: `@docs/project-system-architecture.md` - Complete C4 model documentation

## Technology Stack

- **Backend**: Python 3.10+, asyncio for concurrency
- **Databases**: PostgreSQL (transactional data), TimescaleDB (time series), Redis (caching/state)
- **Communication**: gRPC for inter-service, WebSocket for market data
- **Monitoring**: Prometheus + Grafana, AlertManager
- **Infrastructure**: Docker, potentially Kubernetes for orchestration

## Component Documentation Map

When working on specific components, refer to these documentation files:

### Core System Components
- **Main Trading System**: `@docs/project-system-design/1-core-system.md` & `@docs/project-system-design/12-main-system.md`
- **Financial Engineering Models**: `@docs/project-system-design/2-financial-engineering.md`
- **Strategy Engine**: `@docs/project-system-design/3-strategy-engine.md`
- **Risk Management**: `@docs/project-system-design/4-risk-management.md`
- **Portfolio Optimization**: `@docs/project-system-design/5-portfolio-optimization.md`
- **Order Execution**: `@docs/project-system-design/6-execution-engine.md`

### Infrastructure & Operations
- **Market Data & Microstructure**: `@docs/project-system-design/7-market-microstructure.md`
- **Backtesting System**: `@docs/project-system-design/8-backtesting.md`
- **Monitoring & Alerting**: `@docs/project-system-design/9-monitoring.md`
- **Infrastructure & Deployment**: `@docs/project-system-design/10-infrastructure.md`
- **Data Quality & Validation**: `@docs/project-system-design/11-data-quality.md`

### Implementation & Testing
- **Validation Checklist**: `@docs/project-system-design/13-validation-checklist.md`
- **Implementation Guide**: `@docs/project-system-design/14-implementation-guide.md`

## Development Methodology & Coding Standards

### TDD and Clean Code Principles
**MUST FOLLOW**: `@docs/augmented-coding.md` - Kent Beck's TDD and Tidy First principles

**Core Development Cycle**:
1. **Red**: Write a failing test first
2. **Green**: Write minimal code to make test pass
3. **Refactor**: Improve code structure (only after tests pass)

**Critical Rules**:
- Never mix structural changes with behavioral changes in same commit
- All tests must pass before any commit
- Separate commits for structural vs. behavioral changes
- Use meaningful test names that describe behavior
- One refactoring change at a time

### Engineering Best Practices
**Reference**: `@docs/software-engineering-guide.md` - Comprehensive engineering guidelines

**Key Principles**:
- **Separation of Concerns**: Each component has single responsibility
- **KISS & YAGNI**: Simple solutions, implement only when needed
- **Dependency Inversion**: High-level modules depend on abstractions
- **Clean Code**: Meaningful names, small functions, clear control flow

## Financial Engineering Components

The system implements sophisticated quantitative finance concepts:

- **Kelly Optimization**: Fractional Kelly criterion for position sizing
  - 📋 **Reference**: `@docs/project-system-design/2-financial-engineering.md` (섹션 2.1-2.2)

- **Risk Management**: VaR models, drawdown monitoring, liquidation probability
  - 📋 **Reference**: `@docs/project-system-design/4-risk-management.md`

- **Regime Detection**: HMM and GARCH models for market state identification
  - 📋 **Reference**: `@docs/project-system-design/3-strategy-engine.md` (섹션 4)

- **Strategy Matrix**: Multi-strategy approach with trend following, mean reversion, funding arbitrage
  - 📋 **Reference**: `@docs/project-system-design/3-strategy-engine.md`

- **Portfolio Optimization**: Dynamic allocation with risk constraints
  - 📋 **Reference**: `@docs/project-system-design/5-portfolio-optimization.md`

## Development Guidelines

### Code Organization
- Follow domain-driven design principles
- Implement clean architecture with clear boundaries
- Use dependency injection for testability
- Separate pure business logic from infrastructure concerns

📋 **Reference**: `@docs/software-engineering-guide.md` (섹션 1-4)

### TDD Workflow for Financial Components
Given the critical nature of financial calculations:

1. **Write comprehensive tests** with known expected outputs for mathematical functions
2. **Test edge cases**: Zero positions, extreme market conditions, boundary values
3. **Validate against benchmarks**: Compare risk metrics to established financial models
4. **Continuous refactoring**: Keep financial logic clean and maintainable

📋 **Reference**: `@docs/augmented-coding.md` for strict TDD discipline

### Performance Requirements
- Sub-100ms latency for order execution critical path
- Real-time processing of market data feeds
- Efficient handling of large time series datasets
- Scalable architecture for multiple trading pairs

📋 **Reference**: `@docs/project-system-design/7-market-microstructure.md` for latency requirements

### Security Considerations
- API keys stored in environment variables or secure key management
- No credentials in code or version control
- Encrypted communication for sensitive data
- Comprehensive audit logging for all trading activities

📋 **Reference**: `@docs/project-system-architecture.md` (섹션 8) for security architecture

## Testing Strategy

Given the financial nature of the system:

**Testing Pyramid**:
- **Unit Tests**: All mathematical/financial functions with known expected outputs
  - **Reference**: `@docs/augmented-coding.md` for TDD methodology
- **Integration Tests**: Exchange connectivity and data pipelines
- **Backtesting**: Strategy validation with historical data
  - **Reference**: `@docs/project-system-design/8-backtesting.md`
- **Paper Trading**: Live system validation without real money
- **Chaos Engineering**: Resilience testing

📋 **References**:
- `@docs/project-system-design/13-validation-checklist.md` for testing requirements
- `@docs/software-engineering-guide.md` (섹션 2.3) for testing best practices

## Next Immediate Tasks

When starting the next development session, begin with these tasks in order:

### 🚀 Phase 1.1: Project Structure Setup
1. **Create directory structure** - Standard Python project layout
2. **Write requirements.txt** - Install confirmed technology stack
3. **Setup basic configuration files** - `.env.example`, `config.yaml`
4. **Initialize TDD workflow** - First failing test for `RiskController`

### 📋 Implementation Checklist
- [ ] `src/`, `tests/`, `config/` directories created
- [ ] `requirements.txt` with confirmed packages
- [ ] First failing test written (`test_risk_controller.py`)
- [ ] Basic logging system setup
- [ ] Environment variable management

**Reference for next steps**: `@IMPLEMENTATION_PROGRESS.md` section "다음 즉시 작업"

## Development Commands

### Environment Setup ✅ **COMPLETED (2025-09-14)**

**🎯 CRITICAL: Use Anaconda Environment**
- **Environment Name**: `autotrading`
- **Python Version**: 3.10.18
- **Location**: `C:\Users\dongd\anaconda3\envs\autotrading`

**⚠️ CRITICAL: Environment Issues Discovered**

**Environment Activation Issues:**
```bash
# ❌ FAILED: conda commands not recognized in this environment
conda activate autotrading                           # Command not found
C:\Users\dongd\anaconda3\Scripts\conda.exe activate  # CondaError: Run 'conda init'
```

**⚠️ MANDATORY: Direct Path Execution Required**
```bash
# ❌ WRONG: Uses system Python 3.13 (causes compatibility issues)
python script.py
pip install package

# ✅ REQUIRED: Must use direct paths to autotrading environment
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" script.py
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/pip.exe" install package

# ✅ CONFIRMED WORKING: All commands tested and verified
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/ -v
```

### Package Versions (Installed 2025-09-14)
**Core packages successfully installed:**
- numpy: 2.2.5 (newer than requirements.txt 1.26.4)
- pandas: 2.3.2 (newer than requirements.txt 2.2.2)
- scipy: 1.15.3 (newer than requirements.txt 1.13.1)
- ccxt: 4.4.82 (newer than requirements.txt 4.3.26)
- scikit-learn: 1.7.1
- pydantic: 2.8.2
- aiohttp: 3.10.1
- websockets: 12.0
- All async libraries (asyncio-mqtt, aioredis, aiodns) ✅

### TDD Workflow Commands
```bash
# ⚠️ CRITICAL: ALWAYS use direct paths (conda activation fails)

# Run all tests
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/ -v

# Run specific test module
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_risk_management/ -v

# Run specific test file
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_risk_management/test_risk_controller.py -v

# Run tests with coverage (when setup)
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest --cov=src tests/

# Run linting (when setup)
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/flake8.exe" src/ tests/
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/mypy.exe" src/
```

### Package Installation
```bash
# ⚠️ CRITICAL: conda install may fail due to activation issues
# Use direct pip path for all installations

# Install packages directly with pip (VERIFIED WORKING)
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/pip.exe" install package_name

# For scientific packages (if conda fails, use pip)
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/pip.exe" install numpy pandas scipy

# Check installed packages
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pip list
```

## Key Implementation Areas by Priority

When working on specific areas, consult these documentation combinations:

### 1. Market Data Pipeline
- **TDD Approach**: Start with data validation tests
- **Primary**: `@docs/project-system-design/7-market-microstructure.md`
- **Supporting**: `@docs/project-system-design/11-data-quality.md`
- **Methodology**: `@docs/augmented-coding.md` for test-first development

### 2. Risk Management Framework
- **TDD Approach**: Test mathematical models with known outputs first
- **Primary**: `@docs/project-system-design/4-risk-management.md`
- **Supporting**: `@docs/project-system-design/2-financial-engineering.md`
- **Best Practices**: `@docs/software-engineering-guide.md` (섹션 7.2) for reliability

### 3. Strategy Implementation
- **TDD Approach**: Test strategy signals against historical data
- **Primary**: `@docs/project-system-design/3-strategy-engine.md`
- **Supporting**: `@docs/project-system-design/8-backtesting.md`
- **Process**: `@docs/augmented-coding.md` for clean refactoring

### 4. Order Execution System
- **TDD Approach**: Mock exchange responses for testing
- **Primary**: `@docs/project-system-design/6-execution-engine.md`
- **Supporting**: `@docs/project-system-design/7-market-microstructure.md`
- **Architecture**: `@docs/software-engineering-guide.md` (섹션 7.3) for performance

### 5. Portfolio Management
- **TDD Approach**: Test optimization algorithms with synthetic portfolios
- **Primary**: `@docs/project-system-design/5-portfolio-optimization.md`
- **Supporting**: `@docs/project-system-design/4-risk-management.md`

### 6. System Monitoring
- **TDD Approach**: Test alerting thresholds and metrics calculations
- **Primary**: `@docs/project-system-design/9-monitoring.md`
- **Supporting**: `@docs/project-system-design/10-infrastructure.md`

## Quality Assurance Workflow

### Commit Standards
**STRICTLY FOLLOW**: `@docs/augmented-coding.md` commit discipline

**Before Every Commit**:
1. ✅ All tests passing
2. ✅ No compiler/linter warnings
3. ✅ Single logical unit of work
4. ✅ Clear commit message indicating structural vs. behavioral change

### Code Review Process
**Reference**: `@docs/software-engineering-guide.md` (섹션 3.3 & 5.1)

**Financial Code Reviews Must Include**:
- Mathematical correctness verification
- Edge case handling validation
- Performance impact assessment
- Risk management compliance check

## Monitoring and Observability

The system requires comprehensive monitoring:
- Real-time performance metrics (Sharpe ratio, drawdown, PnL)
- System health metrics (latency, error rates, throughput)
- Risk metrics (VaR breaches, position sizes, leverage)
- Alert thresholds for critical system and financial events

📋 **Reference**: `@docs/project-system-design/9-monitoring.md` for complete monitoring specifications

## Deployment Considerations

- Proximity to exchange servers for low latency
- High availability with automated failover
- Comprehensive backup and disaster recovery
- Blue-green deployment for minimal downtime
- Feature flags for gradual rollout of new strategies

📋 **Reference**: `@docs/project-system-design/10-infrastructure.md` for infrastructure specifications

## Problem-Solving Approach

When debugging complex issues:

**Systematic Debugging Process** (`@docs/software-engineering-guide.md` 섹션 2):
1. **Reproduce the Issue**: Create reliable test case
2. **Gather Information**: Logs, traces, system state
3. **Form & Test Hypotheses**: One change at a time
4. **Document Findings**: Build institutional knowledge

**Persistence Guidelines** (`@docs/software-engineering-guide.md` 섹션 6):
- Break down complex problems into manageable pieces
- Use time-boxing to avoid infinite debugging loops
- Ask for help when stuck (especially for financial calculations)
- Maintain growth mindset and learn from failures

## Risk and Compliance

- Implement circuit breakers for extreme market conditions
- Position size limits and maximum drawdown thresholds
- Regulatory compliance for automated trading systems
- Comprehensive audit trail for all trading decisions
- Kill switches for emergency shutdown

📋 **References**:
- `@docs/project-system-design/4-risk-management.md` for risk controls
- `@docs/project-system-architecture.md` (섹션 13) for compliance procedures