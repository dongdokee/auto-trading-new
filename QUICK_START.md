# Quick Start Guide - AutoTrading System

**Purpose**: Essential commands and workflows for immediate productivity

**Last Updated**: 2025-09-19 (Created during documentation refactoring)

## Environment Setup

### Critical Environment Note ⚠️
**Standard conda activation fails** - must use direct paths to Python executable

**Environment Details**:
- **Name**: `autotrading`
- **Python**: 3.10.18
- **Status**: ✅ Fully configured (222 tests passing)

**For complete environment details**: `@PROJECT_STRUCTURE.md`

## Essential Commands

### ✅ Testing Commands (Most Used)

```bash
# Run all tests
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/ -v

# Run specific module tests
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_risk_management/ -v
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_strategy_engine/ -v
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_portfolio/ -v
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/integration/ -v

# Run with coverage
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest --cov=src tests/
```

### 📦 Package Management

```bash
# Install packages (ALWAYS use direct pip path)
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/pip.exe" install package_name

# Check installed packages
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pip list

# Verify environment
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" --version
```

### 🎨 Code Quality

```bash
# Code formatting (when installed)
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/black.exe" src/ tests/
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/isort.exe" src/ tests/

# Linting (when installed)
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/flake8.exe" src/ tests/
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/mypy.exe" src/
```

### 🚀 System Execution (Future)

```bash
# Main trading system (Phase 4+)
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" scripts/run_trading.py

# Backtesting
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" scripts/backtest.py

# Paper trading
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" scripts/paper_trading.py
```

## Development Workflow

### TDD Workflow (Red → Green → Refactor)
1. **Write failing test** for new functionality
2. **Run test** to confirm it fails
3. **Write minimal code** to make test pass
4. **Run all tests** to ensure no regressions
5. **Refactor** if needed (keep tests passing)
6. **Commit** with clear message

**Complete TDD methodology**: `@docs/augmented-coding.md`

### Typical Development Session

```bash
# 1. Check current status
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/ -v

# 2. Work on specific module (example: risk management)
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_risk_management/ -v

# 3. Run integration tests
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/integration/ -v

# 4. Format code before commit
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/black.exe" src/ tests/
```

## Troubleshooting Common Issues

### Issue 1: "python not recognized"
**Solution**: Always use full path to autotrading environment
```bash
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" --version
# Should output: Python 3.10.18
```

### Issue 2: Import errors
**Solution**: Install in correct environment
```bash
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/pip.exe" install package_name
```

### Issue 3: Tests not discovering
**Solution**: Run from project root with explicit test discovery
```bash
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest --collect-only tests/
```

### Issue 4: Environment verification
**Quick check script**:
```bash
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -c "
import sys
print(f'Python: {sys.version}')
try:
    import numpy, pandas, scipy, ccxt, pydantic
    print('✅ Core packages: OK')
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
└── execution/           # 🚀 NEXT (Phase 4.1)

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

**Phase Completed**: 3.3 (70% overall progress)
**Tests Passing**: 222 tests (100%)
**Next Priority**: Phase 4.1 - Order Execution Engine

**Key Completed Modules**:
- ✅ Risk Management (RiskController, PositionSizer, PositionManager)
- ✅ Strategy Engine (4 strategies + regime detection)
- ✅ Portfolio Optimization (Markowitz + attribution)
- ✅ Core Infrastructure (Database + configuration)
- ✅ Backtesting Framework (Walk-forward validation)

## Getting Help

- **Environment Issues**: Check `@PROJECT_STRUCTURE.md` troubleshooting section
- **TDD Questions**: Review `@docs/augmented-coding.md`
- **Module Details**: Check specific `@src/[module]/CLAUDE.md` files
- **Project Status**: Check `@PROJECT_STATUS.md`

---

**Quick Start Priority**: Get tests running → Understand current module → Follow TDD workflow → Reference documentation as needed

**For detailed information, always reference the complete documentation rather than working from this quick reference.**