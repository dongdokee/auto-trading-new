# Development Environment - AutoTrading System
# 개발 환경 설정 및 명령어 가이드

**Single Source of Truth for Environment Information** ⭐

**Last Updated**: 2025-09-15
**Status**: ✅ Environment Configured & Fully Tested
**Python Version**: 3.10.18
**Environment**: Anaconda `autotrading`

---

## 🐍 **Python Environment Setup**

### **Anaconda Environment Details** ✅ **COMPLETED**
- **Environment Name**: `autotrading`
- **Python Version**: 3.10.18
- **Environment Path**: `C:\Users\dongd\anaconda3\envs\autotrading`
- **Setup Date**: 2025-09-14
- **Verification Status**: ✅ All core packages installed and tested

### **⚠️ CRITICAL: Environment Activation Issues Discovered**

**Problem Identified**: Standard conda activation commands fail in this system environment
```bash
# ❌ FAILED: These commands don't work
conda activate autotrading                           # Command not found
C:\Users\dongd\anaconda3\Scripts\conda.exe activate  # CondaError: Run 'conda init'
```

**⚠️ MANDATORY SOLUTION: Direct Path Execution Required**
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

---

## 🛠️ **Development Commands**

### **Package Installation Commands**
```bash
# ⚠️ CRITICAL: conda install may fail due to activation issues
# Use direct pip path for all installations

# Install packages directly with pip (VERIFIED WORKING)
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/pip.exe" install package_name

# For scientific packages (if conda fails, use pip)
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/pip.exe" install numpy pandas scipy

# Check installed packages
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pip list

# Setup environment variables
cp .env.example .env
# Edit .env with your actual values
```

### **TDD Workflow Commands** ⚠️ **CRITICAL: ALWAYS use direct paths**
```bash
# Run all tests
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/ -v

# Run specific test modules
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_risk_management/ -v
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/integration/ -v

# Run specific test file
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_risk_management/test_risk_controller.py -v

# Run with coverage (when setup)
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest --cov=src tests/

# Run specific test cases
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_strategy_engine/ -v
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_portfolio/ -v
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_backtesting/ -v
```

### **Code Quality Tools Commands**
```bash
# Code quality checks (when setup)
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/flake8.exe" src/ tests/
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/mypy.exe" src/
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/black.exe" src/ tests/
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/isort.exe" src/ tests/
```

### **System Execution Commands** (Future Phases)
```bash
# Main trading system
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" scripts/run_trading.py

# Backtesting
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" scripts/backtest.py

# Paper trading
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" scripts/paper_trading.py
```

---

## 📦 **Installed Packages Status**

### **Core Dependencies** ✅ **VERIFIED INSTALLED**
- **numpy**: 2.2.5 ✅
- **pandas**: 2.3.2 ✅
- **scipy**: 1.15.3 ✅
- **scikit-learn**: 1.7.1 ✅

### **Financial & Async Libraries** ✅ **VERIFIED INSTALLED**
- **ccxt**: 4.4.82 ✅ (Cryptocurrency exchange library)
- **aiohttp**: ✅ (Async HTTP client)
- **aioredis**: ✅ (Async Redis client)
- **websockets**: 12.0 ✅ (WebSocket support)
- **httpx**: ✅ (Modern HTTP client)

### **Configuration & Validation** ✅ **VERIFIED INSTALLED**
- **pydantic**: 2.8.2 ✅ (Data validation)
- **python-dotenv**: ✅ (Environment variable management)
- **cryptography**: ✅ (Encryption support)

### **Logging & Structure** ✅ **VERIFIED INSTALLED**
- **structlog**: 24.2.0 ✅ (Structured logging)

### **Testing Framework** ✅ **READY**
- **pytest**: ✅ (Testing framework - ready for use)

### **Future Installation** (Install when needed)
- **arch**: GARCH models for volatility forecasting
- **hmmlearn**: Hidden Markov Models for regime detection
- **statsmodels**: Statistical analysis
- **PostgreSQL clients**: Database connectivity
- **TimescaleDB clients**: Time-series database
- **Redis clients**: Caching
- **prometheus-client**: Monitoring metrics

---

## 🔧 **IDE Configuration**

### **VS Code Settings** (Recommended)
```json
{
    "python.interpreterPath": "C:\\Users\\dongd\\anaconda3\\envs\\autotrading\\python.exe",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.formatting.provider": "black"
}
```

### **PyCharm Settings** (Alternative)
- **Interpreter**: `C:\Users\dongd\anaconda3\envs\autotrading\python.exe`
- **Test Runner**: pytest
- **Source Root**: `src/`

---

## 🚨 **Troubleshooting Guide**

### **Common Issues & Solutions**

#### **Issue 1: "python not recognized" or wrong Python version**
```bash
# Problem: System uses Python 3.13 instead of 3.10.18
# Solution: Always use full path
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" --version
# Should output: Python 3.10.18
```

#### **Issue 2: Import errors or package not found**
```bash
# Problem: Package installed in wrong environment
# Solution: Install with direct pip path
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/pip.exe" install package_name

# Verify installation
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -c "import package_name; print('OK')"
```

#### **Issue 3: Tests not running or failing unexpectedly**
```bash
# Check if using correct Python
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest --version

# Verify test discovery
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest --collect-only tests/

# Run with verbose output
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/ -v --tb=short
```

#### **Issue 4: Module import paths not working**
```bash
# Ensure PYTHONPATH includes src directory
# Add to your IDE or set environment variable:
export PYTHONPATH="${PYTHONPATH}:./src"

# Or run with module path
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/ -v
```

### **Environment Verification Script**
```bash
# Quick environment check
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python path: {sys.executable}')

try:
    import numpy, pandas, scipy
    print('✅ Core scientific packages: OK')
except ImportError as e:
    print(f'❌ Import error: {e}')

try:
    import ccxt, aiohttp, pydantic
    print('✅ Trading packages: OK')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

---

## 📊 **Environment Status Summary**

### **✅ WORKING & VERIFIED**
- Python 3.10.18 environment fully configured
- All core dependencies installed and tested
- Direct path execution method established and verified
- Testing framework operational (222 tests passing)
- Core trading system modules importable
- Database migration system ready

### **⚠️ KNOWN LIMITATIONS**
- conda activate commands don't work (use direct paths)
- Some advanced packages (arch, hmmlearn) not yet installed
- IDE integration requires manual interpreter configuration

### **🚀 READY FOR**
- Phase 4.1: Order Execution Engine development
- All TDD workflows and testing
- Production deployment preparation
- Additional package installations as needed

---

## 📚 **Related Documentation**

- **📋 Main Development Guide**: `@CLAUDE.md`
- **🏗️ Project Structure**: `@PROJECT_STRUCTURE.md`
- **📊 Implementation Progress**: `@IMPLEMENTATION_PROGRESS.md`
- **🧪 TDD Methodology**: `@docs/augmented-coding.md`

---

**Environment Maintainer**: AutoTrading Development Team
**Last Verification**: 2025-09-15 (All 222 tests passing)
**Next Review**: When adding new dependencies or tools