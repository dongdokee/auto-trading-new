# Risk Management Module - CLAUDE.md

This file provides specific guidance for Claude Code when working on the risk management module.

## Module Overview

**Location**: `src/risk_management/`
**Purpose**: Complete risk management framework including Kelly Criterion optimization, VaR monitoring, position sizing, and position lifecycle management
**Status**: ✅ **PHASE 1.2 COMPLETED: Full Risk Management Module with Position Sizing Engine** 🚀
**Last Updated**: 2025-09-14 (Position Sizing Engine and Position Management completed)

## ⭐ CRITICAL IMPLEMENTATION CONTEXT ⭐

### 🚀 Successfully Completed: Complete Risk Management Framework

#### **1. RiskController Class** ✅
**File**: `src/risk_management/risk_management.py`
**Tests**: `tests/unit/test_risk_management/test_risk_controller.py` (22 test cases, all passing)
**Implementation Date**: 2025-09-14 (Drawdown monitoring system completed)

#### **2. PositionSizer Class** ✅ **NEW COMPLETED**
**File**: `src/risk_management/position_sizing.py`
**Tests**: `tests/unit/test_risk_management/test_position_sizing.py` (15 test cases, all passing)
**Implementation Date**: 2025-09-14 (Multi-constraint position sizing engine)

#### **3. PositionManager Class** ✅ **NEW COMPLETED**
**File**: `src/risk_management/position_management.py`
**Tests**: `tests/unit/test_risk_management/test_position_management.py` (14 test cases, all passing)
**Implementation Date**: 2025-09-14 (Complete position lifecycle management)

#### **4. Integration Testing** ✅ **NEW COMPLETED**
**File**: `tests/integration/test_risk_management_integration.py`
**Tests**: 6 comprehensive integration tests covering complete workflows
**Implementation Date**: 2025-09-14 (Multi-component integration validation)

#### **Key Architecture Decisions:**

1. **Configurable Risk Parameters** - ESSENTIAL for production flexibility:
```python
RiskController(
    initial_capital_usdt=10000.0,
    var_daily_pct=0.02,           # Configurable VaR limit (default: 2%)
    cvar_daily_pct=0.03,          # Configurable CVaR limit (default: 3%)
    max_drawdown_pct=0.12,        # Configurable drawdown limit (default: 12%)
    correlation_threshold=0.7,     # Correlation threshold
    concentration_limit=0.2,       # Single asset concentration (20%)
    max_leverage=10.0,            # Maximum leverage (default: 10x)
    liquidation_prob_24h=0.005,   # 24h liquidation probability (0.5%)
    max_consecutive_loss_days=7,  # 🌟 NEW: Max consecutive loss days (default: 7)
    allow_short=False             # Long-only by default, short optional
)
```

2. **Kelly Criterion Implementation** - Advanced financial engineering:
   - **EMA weighting** (alpha=0.2) for recent data emphasis
   - **Shrinkage correction** for parameter estimation uncertainty
   - **Fractional Kelly** (default=0.25) to prevent over-betting
   - **Regime-based caps**:
     - BULL: 15% long, -5% short
     - BEAR: 5% long, -15% short
     - SIDEWAYS: 10% long, -10% short
     - NEUTRAL: 8% long, -8% short
   - **Minimum sample requirement**: 30 data points

3. **VaR Limit Checking** - Real-time risk monitoring:
   - Returns violations as `List[Tuple[str, float]]` format
   - Supports both return-based and USDT-based limits
   - Integrated with portfolio state dictionary structure

4. **🚀 COMPLETED: Leverage Limit System** - Advanced leverage management:
   - **Basic Leverage Checking**: `check_leverage_limit()` - detects portfolio leverage violations
   - **Total Leverage Calculation**: `_calculate_total_leverage()` - computes portfolio-wide leverage
   - **Safe Leverage Calculation**: `calculate_safe_leverage_limit()` - liquidation distance-based limits
   - **Volatility Adjustment**: `calculate_volatility_adjusted_leverage()` - market regime-aware scaling
   - **Multi-layered Safety**: 3-sigma safety margins, regime-based adjustments, minimum 1x leverage

5. **🌟 NEW: Drawdown Monitoring System** - Comprehensive drawdown tracking:
   - **Real-time Drawdown Calculation**: `update_drawdown()` - tracks current drawdown vs high water mark
   - **Drawdown Limit Checking**: `check_drawdown_limit()` - detects max drawdown violations
   - **Severity Classification**: `get_drawdown_severity_level()` - MILD/MODERATE/SEVERE categories
   - **Consecutive Loss Tracking**: `update_consecutive_loss_days()` - tracks loss streaks
   - **Loss Limit Checking**: `check_consecutive_loss_limit()` - detects excessive loss streaks
   - **Recovery Tracking**: `track_drawdown_recovery()` - measures recovery periods
   - **Recovery Statistics**: `get_recovery_statistics()` - aggregates recovery data

#### **Critical Technical Patterns:**

- **TDD Methodology**: Red → Green → Refactor cycles successfully applied
- **Test Naming**: `test_should_[behavior]_when_[condition]`
- **Edge Case Handling**: Insufficient data, zero variance, negative returns
- **Type Safety**: Full type annotations throughout
- **Documentation**: Comprehensive docstrings with Args/Returns

#### **API Interface:**

```python
# 1. VaR Limit Checking
violations = risk_controller.check_var_limit(portfolio_state)
# Returns: List[Tuple[str, float]] - violation type and value

# 2. Kelly Criterion Calculation
kelly_fraction = risk_controller.calculate_optimal_position_fraction(
    returns=np.array([...]),      # Required: return series
    regime='NEUTRAL',             # Optional: market regime
    fractional=0.25               # Optional: fractional Kelly multiplier
)
# Returns: float - optimal position fraction (+ for long, - for short, 0 for no position)

# 3. 🚀 NEW: Leverage Limit Checking
leverage_violations = risk_controller.check_leverage_limit(portfolio_state)
# Returns: List[Tuple[str, float]] - leverage violations

# 4. 🚀 NEW: Safe Leverage Calculation
safe_leverage = risk_controller.calculate_safe_leverage_limit(portfolio_state)
# Returns: float - max safe leverage based on liquidation distances

# 5. 🚀 COMPLETED: Volatility-Adjusted Leverage
adjusted_leverage = risk_controller.calculate_volatility_adjusted_leverage(
    base_leverage=5.0,
    market_state={'daily_volatility': 0.08, 'regime': 'VOLATILE'}
)
# Returns: float - leverage adjusted for market conditions

# 6. 🌟 NEW: Drawdown Monitoring
current_drawdown = risk_controller.update_drawdown(current_equity)
# Returns: float - current drawdown percentage (0.1 = 10% drawdown)

drawdown_violations = risk_controller.check_drawdown_limit(current_equity)
# Returns: List[Tuple[str, float]] - drawdown violations

severity = risk_controller.get_drawdown_severity_level()
# Returns: str - 'MILD' (0-5%), 'MODERATE' (5-10%), 'SEVERE' (10%+)

# 7. 🌟 NEW: Consecutive Loss Tracking
consecutive_days = risk_controller.update_consecutive_loss_days(daily_pnl)
# Returns: int - current consecutive loss days

loss_violations = risk_controller.check_consecutive_loss_limit()
# Returns: List[Tuple[str, int]] - consecutive loss violations

# 8. 🌟 NEW: Recovery Tracking
recovery_days = risk_controller.track_drawdown_recovery(current_equity, current_time)
# Returns: Optional[int] - recovery period in days, None if still in drawdown

recovery_stats = risk_controller.get_recovery_statistics()
# Returns: Dict - comprehensive recovery statistics
```

#### **Integration Points:**

- **Portfolio State Dictionary**: Established structure for `check_var_limit()`
- **Return Format**: Violations returned as tuples for consistent error handling
- **Regime Awareness**: Market regime parameter for context-aware risk management
- **USDT Calculations**: Direct integration with position sizing systems

### 🎉 **NEW: Position Sizing Engine Implementation**

#### **PositionSizer Class Architecture** ✅

**Core Concept**: Multi-constraint optimization combining all major risk factors

```python
# Complete position sizing workflow
position_sizer = PositionSizer(risk_controller)

position_size = position_sizer.calculate_position_size(
    signal={
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'strength': 0.8,        # Signal strength (0-1)
        'confidence': 0.7       # Signal confidence (0-1)
    },
    market_state={
        'price': 50000.0,       # Current price (USDT)
        'atr': 2000.0,          # Average True Range (USDT)
        'daily_volatility': 0.05, # Daily volatility (5%)
        'regime': 'NEUTRAL',    # Market regime
        'min_notional': 10.0,   # Minimum trade size
        'lot_size': 0.001,      # Exchange lot size
        'symbol_leverage': 10   # Max leverage for symbol
    },
    portfolio_state={
        'equity': 10000.0,      # Current equity (USDT)
        'recent_returns': returns_array,  # Recent return history
        'positions': [],        # Current positions
        'current_var_usdt': 0.0,  # Current VaR usage
        'symbol_volatilities': {'BTCUSDT': 0.05},
        'correlation_matrix': correlation_data
    }
)
# Returns: float - optimal position size in coin units
```

#### **Multi-Constraint Algorithm** 🧮

The position sizing engine implements sophisticated constraint optimization:

```python
# 1. Kelly-based sizing (from RiskController)
kelly_size = kelly_fraction * equity / price

# 2. ATR-based risk sizing (1% equity risk per trade)
atr_size = (equity * 0.01) / (2.0 * atr)

# 3. Liquidation safety sizing (3-ATR safety margin)
liquidation_safe_size = safe_leverage * equity / price

# 4. VaR-constrained sizing
var_size = available_var / (1.65 * volatility * price)

# 5. Final size = minimum of all constraints
base_size = min(kelly_size, atr_size, liquidation_safe_size, var_size)

# 6. Apply adjustments
final_size = base_size * correlation_factor * signal_strength
```

#### **Key Features Implemented**:

1. **Multi-Constraint Optimization**: Combines Kelly, ATR, liquidation safety, and VaR
2. **Correlation Adjustment**: Reduces size for correlated positions (0.3x-1.0x multiplier)
3. **Exchange Compliance**: Lot size rounding, minimum notional enforcement
4. **Maintenance Margin Tiers**: Binance-style tiered margin requirements
5. **Signal Strength Integration**: Scales position based on signal confidence
6. **Edge Case Handling**: Zero volatility, insufficient data, extreme scenarios

#### **API Interface**:

```python
# Main position sizing
size = position_sizer.calculate_position_size(signal, market_state, portfolio_state)

# Individual sizing methods (for debugging/analysis)
kelly_size = position_sizer._calculate_kelly_based_size(signal, market_state, portfolio_state)
atr_size = position_sizer._calculate_atr_based_size(signal, market_state, portfolio_state)
liquidation_size = position_sizer._calculate_liquidation_safe_size(symbol, side, market_state, portfolio_state)
var_size = position_sizer._calculate_var_constrained_size(symbol, market_state, portfolio_state)
correlation_factor = position_sizer._calculate_correlation_adjustment(symbol, portfolio_state)
```

### 🎉 **NEW: Position Management System**

#### **PositionManager Class Architecture** ✅

**Core Concept**: Complete position lifecycle management with real-time tracking

```python
# Complete position lifecycle
position_manager = PositionManager(risk_controller)

# 1. Open position
position = position_manager.open_position(
    symbol='BTCUSDT',
    side='LONG',
    size=0.1,           # Position size in coin units
    price=50000.0,      # Entry price
    leverage=5.0        # Leverage used
)

# 2. Update position with market prices
updated_position = position_manager.update_position('BTCUSDT', new_price=52000.0)

# 3. Check stop conditions
stop_reason = position_manager.check_stop_conditions('BTCUSDT', current_price=48000.0)
# Returns: 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', or None

# 4. Close position
closed_position = position_manager.close_position('BTCUSDT', close_price=53000.0, reason='MANUAL')
```

#### **Position Data Structure**:

```python
position = {
    'symbol': 'BTCUSDT',
    'side': 'LONG',
    'size': 0.1,
    'entry_price': 50000.0,
    'current_price': 52000.0,
    'leverage': 5.0,
    'margin': 1000.0,           # Required margin (USDT)
    'liquidation_price': 45250.0, # Calculated liquidation price
    'unrealized_pnl': 200.0,    # Current unrealized P&L (USDT)
    'realized_pnl': 0.0,        # Realized P&L on close
    'open_time': datetime.now(),
    'stop_loss': 48000.0,       # Optional stop loss
    'take_profit': 55000.0,     # Optional take profit
    'trailing_stop': 49000.0,   # Dynamic trailing stop
    'trailing_distance': 0.02   # 2% trailing distance
}
```

#### **Key Features Implemented**:

1. **Automatic Liquidation Calculation**: For both LONG and SHORT positions
2. **Real-time P&L Tracking**: Unrealized and realized P&L calculation
3. **Stop Management**: Stop-loss, take-profit, trailing stop support
4. **Risk Integration**: Uses RiskController for margin requirements
5. **Position Lifecycle**: Complete open → update → close workflow

#### **Stop Management System**:

```python
# Set stops after opening position
position['stop_loss'] = entry_price * 0.96      # 4% stop loss
position['take_profit'] = entry_price * 1.10    # 10% take profit
position['trailing_distance'] = 0.02            # 2% trailing stop

# Automatic stop checking
stop_reason = position_manager.check_stop_conditions(symbol, current_price)
if stop_reason:
    closed_position = position_manager.close_position(symbol, current_price, stop_reason)
```

## 🧪 Comprehensive Test Suite

**Total Tests**: ✅ 57 tests passing (51 unit + 6 integration) 🎉 **FULLY COMPLETED**

### **Unit Tests** (51 tests)
**Location**: `tests/unit/test_risk_management/`

#### **1. RiskController Tests** (22 tests) - `test_risk_controller.py`
- **Initialization Tests** (3 tests): Capital setup, default limits, custom parameters
- **VaR Limit Tests** (2 tests): Violation detection, within-limit validation
- **Kelly Criterion Tests** (4 tests): Long-only, short positions, insufficient data
- **Leverage Management Tests** (6 tests): Violation detection, total leverage, safe limits, volatility adjustment
- **Drawdown Monitoring Tests** (7 tests): Real-time tracking, severity classification, consecutive loss tracking

#### **2. PositionSizer Tests** (15 tests) - `test_position_sizing.py` ✅ **NEW**
- **Basic Functionality Tests** (6 tests): Initialization, basic sizing, signal strength, position limits, lot size rounding, insufficient data handling
- **Constraint-Specific Tests** (4 tests): ATR-based sizing accuracy, VaR-constrained sizing, correlation adjustment, liquidation safety
- **Edge Case Tests** (3 tests): Zero volatility, zero ATR, minimum constraint selection
- **Signal Integration Tests** (2 tests): Signal strength scaling, maintenance margin tiers

#### **3. PositionManager Tests** (14 tests) - `test_position_management.py` ✅ **NEW**
- **Lifecycle Management Tests** (5 tests): Initialization, position opening, closing, nonexistent position handling
- **P&L Calculation Tests** (2 tests): Long/short position P&L accuracy
- **Liquidation Price Tests** (2 tests): Long/short liquidation price calculation
- **Stop Management Tests** (3 tests): Stop-loss detection, take-profit detection, no-stop conditions
- **Trailing Stop Tests** (2 tests): Long position trailing stop updates, trailing stop persistence

### **Integration Tests** (6 tests) ✅ **NEW**
**Location**: `tests/integration/test_risk_management_integration.py`

- **Complete Workflow Test** (1 test): Full position lifecycle (sizing → open → update → close)
- **Multi-Asset Portfolio Test** (1 test): Correlation-adjusted sizing for multiple assets
- **Risk Limit Enforcement Test** (1 test): VaR budget exhaustion and constraint binding
- **Leverage Interaction Test** (1 test): Portfolio leverage limits with position sizing
- **Drawdown Integration Test** (1 test): Real-time drawdown tracking with position P&L
- **Constraint Validation Test** (1 test): All constraints (Kelly, ATR, VaR, correlation) working together

### Test Execution Commands:
**For complete environment commands**: 📋 `@PROJECT_STRUCTURE.md`

```bash
# Risk Management specific tests
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_risk_management/ -v
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/integration/test_risk_management_integration.py -v
```

## 🎉 **PHASE 1.2 COMPLETED** - Full Risk Management Module 🚀

### ✅ **ALL IMPLEMENTATIONS COMPLETED (2025-09-14)**

#### 1. ~~**RiskController Class**~~ ✅ **FULLY COMPLETED**:
```python
# Complete risk monitoring and Kelly optimization
risk_controller = RiskController(initial_capital_usdt=10000.0, ...)
violations = risk_controller.check_var_limit(portfolio_state)
kelly_fraction = risk_controller.calculate_optimal_position_fraction(returns)
leverage_violations = risk_controller.check_leverage_limit(portfolio_state)
drawdown = risk_controller.update_drawdown(current_equity)
```

#### 2. ~~**Position Sizing Engine**~~ ✅ **NEWLY COMPLETED** 🌟:
**File**: `src/risk_management/position_sizing.py`
**Tests**: `tests/unit/test_risk_management/test_position_sizing.py` (15 tests, all passing)

```python
# Multi-constraint position sizing combining all risk factors
position_sizer = PositionSizer(risk_controller)
position_size = position_sizer.calculate_position_size(
    signal=signal,           # Trading signal with strength
    market_state=market_state,   # Price, ATR, volatility
    portfolio_state=portfolio_state  # Current positions, equity
)
# Returns: float - optimal position size respecting ALL constraints
```

**Key Features**:
- **Kelly-based sizing**: Integrates RiskController Kelly Criterion
- **ATR-based risk sizing**: 1% equity risk per trade with ATR stop distance
- **Liquidation safety**: Maximum size to avoid liquidation (3-ATR safety margin)
- **VaR constraint**: Position size within available VaR budget
- **Correlation adjustment**: Reduces size for correlated positions
- **Maintenance margin tiers**: Binance-style tiered margin requirements
- **Exchange compliance**: Lot size rounding, minimum notional values

#### 3. ~~**Position Management System**~~ ✅ **NEWLY COMPLETED** 🌟:
**File**: `src/risk_management/position_management.py`
**Tests**: `tests/unit/test_risk_management/test_position_management.py` (14 tests, all passing)

```python
# Complete position lifecycle management
position_manager = PositionManager(risk_controller)

# Open position
position = position_manager.open_position('BTCUSDT', 'LONG', size, price, leverage)

# Update with market prices
updated_position = position_manager.update_position('BTCUSDT', new_price)

# Check stop conditions
stop_reason = position_manager.check_stop_conditions('BTCUSDT', current_price)
# Returns: 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', or None

# Close position
closed_position = position_manager.close_position('BTCUSDT', close_price, 'MANUAL')
```

**Key Features**:
- **Position lifecycle**: Open → Update → Close workflow
- **PnL tracking**: Real-time unrealized/realized PnL calculation
- **Liquidation price**: Automatic calculation for long/short positions
- **Stop management**: Stop-loss, take-profit, trailing stop support
- **Risk integration**: Uses RiskController for margin requirements

### 📊 **Complete Test Suite (57 tests, all passing)**:
- **RiskController**: 22 tests (Kelly, VaR, leverage, drawdown)
- **PositionSizer**: 15 tests (sizing methods, constraints, edge cases)
- **PositionManager**: 14 tests (lifecycle, PnL, stops, liquidation)
- **Integration**: 6 tests (complete workflows, multi-asset, constraints)

### 🎯 **PHASE 1.2 SUCCESS CRITERIA - ALL MET** ✅
- ✅ Position Sizing Engine fully implemented
- ✅ All risk constraints integrated (Kelly, ATR, VaR, leverage, correlation)
- ✅ Position lifecycle management complete
- ✅ Comprehensive test coverage (57 tests)
- ✅ Mathematical accuracy validated
- ✅ Exchange compliance implemented
- ✅ Ready for Strategy Engine integration

## 🚀 **READY FOR NEXT PHASE: Strategy Engine Integration**

The risk management module is **PRODUCTION-READY** and provides a complete foundation for:

### **Phase 2.1: Strategy Engine Integration** 🎯 **NEXT PRIORITY**

**Integration Points**:
```python
# Strategy Engine will call Position Sizing Engine
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.position_management import PositionManager

# Strategy generates signals
strategy_signal = {
    'symbol': 'BTCUSDT',
    'side': 'LONG',
    'strength': 0.8,
    'confidence': 0.7,
    'strategy_id': 'momentum_breakout'
}

# Position Sizer calculates optimal size
position_size = position_sizer.calculate_position_size(
    signal=strategy_signal,
    market_state=current_market_data,
    portfolio_state=current_portfolio
)

# Position Manager handles execution
position = position_manager.open_position(
    symbol=strategy_signal['symbol'],
    side=strategy_signal['side'],
    size=position_size,
    price=current_price,
    leverage=calculated_leverage
)
```

### **Future Integration Phases**:
- **Order Execution** (Phase 4.1): Will use `PositionManager` for position lifecycle
- **Portfolio Optimizer** (Phase 3.2): Will use risk limits and correlation matrices
- **Backtesting Framework** (Phase 2.2): Will use complete risk management for historical testing
- **Monitoring System** (Phase 5.1): Will use risk metrics and violation alerts

### **Risk Management API Ready For**:
- ✅ **Strategy Signal Processing**: Position sizing based on signal strength and confidence
- ✅ **Multi-Asset Portfolios**: Correlation-aware position sizing and risk limits
- ✅ **Real-time Risk Monitoring**: VaR, leverage, and drawdown tracking
- ✅ **Position Lifecycle**: Complete open-to-close position management
- ✅ **Exchange Integration**: Lot sizes, margin requirements, liquidation calculations

## 📚 **Related Documentation**

### **📋 Main Claude Code References**
- **🎯 Development Guide**: `@CLAUDE.md` - Core development guidance and document navigation
- **📊 Progress Status**: `@IMPLEMENTATION_PROGRESS.md` - Overall project progress and next steps
- **🏗️ Project Structure**: `@PROJECT_STRUCTURE.md` - Complete environment setup and commands

### **📖 Technical Specifications**
- **💰 Risk Management Design**: `@docs/project-system-design/4-risk-management.md` - Detailed design specifications
- **🔢 Financial Models**: `@docs/project-system-design/2-financial-engineering.md` - Kelly Criterion, VaR theory
- **🧪 TDD Methodology**: `@docs/augmented-coding.md` - Development discipline and practices

## ⚠️ Critical Dependencies

**For complete dependency information**: 📋 `@PROJECT_STRUCTURE.md`
**Key Requirements**: numpy, pandas, scipy, pytest

## 🔧 Development Patterns for This Module

When extending this module:

1. **Always TDD**: Write failing test first
2. **Type Everything**: Full type annotations required
3. **Document Edge Cases**: Handle insufficient data, zero variance, etc.
4. **Financial Accuracy**: Validate against known benchmarks
5. **Configuration First**: Make parameters configurable
6. **Test Both Directions**: Long-only and short scenarios

## 🎯 Performance Considerations

- **Real-time Requirements**: Risk calculations must complete in <10ms
- **Memory Efficiency**: Use numpy arrays for large return series
- **CPU Optimization**: EMA calculations are vectorized
- **Caching Opportunity**: Consider caching regime caps and weights

---
**Module Maintainer**: Risk Management Team
**Last Implementation**: Drawdown Monitoring System (2025-09-14)
**Next Priority**: Position Sizing Engine