# Quick Start Guide - AutoTrading System

**Purpose**: Essential commands and workflows for immediate productivity

**Last Updated**: 2025-10-06 (Updated: Environment separation strategy)

## Environment Setup

### Two Anaconda Environments

This project uses two separate conda environments for different purposes:

#### 🔬 **Development Environment**: `autotrading-dev`
- **Purpose**: Development, testing, code quality checks
- **Packages**: Production packages + development tools (pytest, black, mypy, etc.)
- **Setup**: `pip install -r requirements-dev.txt`

#### 🚀 **Production Environment**: `autotrading`
- **Purpose**: Running live/paper trading systems
- **Packages**: Production packages only (minimal dependencies)
- **Setup**: `pip install -r requirements.txt`

**For complete environment details**: `@PROJECT_STRUCTURE.md`

## Essential Commands

### ✅ Testing Commands (Development Environment)

```bash
# Activate development environment
conda activate autotrading-dev

# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/unit/test_risk_management/ -v
python -m pytest tests/unit/test_strategy_engine/ -v
python -m pytest tests/unit/test_portfolio/ -v
python -m pytest tests/integration/ -v

# Run with coverage
python -m pytest --cov=src tests/
```

### 📦 Package Management

```bash
# For development environment
conda activate autotrading-dev
pip install -r requirements-dev.txt

# For production environment
conda activate autotrading
pip install -r requirements.txt

# Check installed packages
pip list

# Verify environment
python --version
```

### 🎨 Code Quality (Development Environment)

```bash
# Activate development environment
conda activate autotrading-dev

# Code formatting
black src/ tests/
isort src/ tests/

# Linting
mypy src/
```

### 🚀 System Execution (Production Environment)

```bash
# Activate production environment
conda activate autotrading

# Paper trading (recommended for testing)
python scripts/paper_trading.py

# Backtesting
python scripts/backtest.py

# Live trading (use with caution!)
python scripts/run_trading.py
```

## Development Workflow

### TDD Workflow (Red → Green → Refactor)
1. **Activate dev environment**: `conda activate autotrading-dev`
2. **Write failing test** for new functionality
3. **Run test** to confirm it fails
4. **Write minimal code** to make test pass
5. **Run all tests** to ensure no regressions
6. **Refactor** if needed (keep tests passing)
7. **Commit** with clear message

**Complete TDD methodology**: `@docs/augmented-coding.md`

### Typical Development Session

```bash
# 1. Activate development environment
conda activate autotrading-dev

# 2. Check current status
python -m pytest tests/ -v

# 3. Work on specific module (example: risk management)
python -m pytest tests/unit/test_risk_management/ -v

# 4. Run integration tests
python -m pytest tests/integration/ -v

# 5. Format code before commit
black src/ tests/
isort src/ tests/
```

## Troubleshooting Common Issues

### Issue 1: "python not recognized"
**Solution**: Activate the appropriate conda environment first
```bash
# For development
conda activate autotrading-dev
python --version

# For production
conda activate autotrading
python --version
```

### Issue 2: Import errors
**Solution**: Ensure you have activated the correct environment and installed dependencies
```bash
# For development
conda activate autotrading-dev
pip install -r requirements-dev.txt

# For production
conda activate autotrading
pip install -r requirements.txt
```

### Issue 3: Tests not discovering
**Solution**: Activate dev environment and run from project root
```bash
conda activate autotrading-dev
python -m pytest --collect-only tests/
```

### Issue 4: Environment verification
**Quick check script**:
```bash
# Activate environment first
conda activate autotrading-dev

# Run verification
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import numpy, pandas, scipy, ccxt, pydantic
    print('✅ Core packages: OK')
    import pytest, black, mypy
    print('✅ Dev packages: OK')
except ImportError as e:
    print(f'❌ Missing: {e}')
"
```

## Project Structure Quick Reference

### Key Directories
```
src/
├── risk_management/     # ✅ COMPLETED (Phase 1)
├── strategy_engine/     # ✅ COMPLETED (Phase 3.1-3.2)
├── portfolio/           # ✅ COMPLETED (Phase 3.3)
├── core/                # ✅ COMPLETED (Phase 2.1-2.2)
├── backtesting/         # ✅ COMPLETED
├── utils/               # ✅ COMPLETED
└── execution/           # ✅ COMPLETED (Phase 4.1)

tests/
├── unit/                # Module-specific tests
├── integration/         # Cross-module tests
└── fixtures/            # Test data
```

### Documentation Quick Access
- **📊 Project Status**: `@PROJECT_STATUS.md` - Progress and roadmap
- **🏗️ Technical Details**: `@PROJECT_STRUCTURE.md` - Complete tech stack and environment
- **🎯 Development Guide**: `@CLAUDE.md` - Principles and navigation
- **📋 Doc Management**: `@DOCUMENT_MANAGEMENT_GUIDE.md` - Documentation rules

## Current Development Status

**Phase Completed**: 6.1 (100% overall progress)
**Tests Passing**: 924+ tests (100%)
**Status**: Production Ready with Paper Trading Validation

**Key Completed Modules**:
- ✅ Risk Management (RiskController, PositionSizer, PositionManager)
- ✅ Strategy Engine (4 strategies + regime detection)
- ✅ Portfolio Optimization (Markowitz + attribution)
- ✅ Core Infrastructure (Database + configuration)
- ✅ Backtesting Framework (Walk-forward validation)
- ✅ Order Execution (Smart routing + execution algorithms)
- ✅ API Integration (Binance + Paper trading)
- ✅ System Integration (Event-driven orchestration)
- ✅ Market Data Pipeline (Real-time analytics)
- ✅ Production Optimization (8-component suite)
- ✅ Enhanced Utilities (Logging + financial math)

## Getting Help

- **Environment Issues**: Check `@PROJECT_STRUCTURE.md` troubleshooting section
- **TDD Questions**: Review `@docs/augmented-coding.md`
- **Module Details**: Check specific `@src/[module]/CLAUDE.md` files
- **Project Status**: Check `@PROJECT_STATUS.md`

---

**Quick Start Priority**: Activate environment → Get tests running → Understand current module → Follow TDD workflow → Reference documentation as needed

**For detailed information, always reference the complete documentation rather than working from this quick reference.**
