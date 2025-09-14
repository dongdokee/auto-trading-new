# Risk Management Module - CLAUDE.md

This file provides specific guidance for Claude Code when working on the risk management module.

## Module Overview

**Location**: `src/risk_management/` and `src/core/risk_management.py`
**Purpose**: Core risk management functionality including Kelly Criterion optimization, VaR monitoring, and position sizing
**Status**: ✅ **RiskController class fully implemented with TDD + Leverage Limit Checking**
**Last Updated**: 2025-09-14 (Leverage system completed)

## ⭐ CRITICAL IMPLEMENTATION CONTEXT ⭐

### 🚀 Successfully Completed: RiskController Class

**File**: `src/risk_management/risk_management.py`
**Tests**: `tests/unit/test_risk_management/test_risk_controller.py` (15 test cases, all passing)
**Implementation Date**: 2025-09-14 (Updated: Leverage system added)

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

4. **🚀 NEW: Leverage Limit System** - Advanced leverage management:
   - **Basic Leverage Checking**: `check_leverage_limit()` - detects portfolio leverage violations
   - **Total Leverage Calculation**: `_calculate_total_leverage()` - computes portfolio-wide leverage
   - **Safe Leverage Calculation**: `calculate_safe_leverage_limit()` - liquidation distance-based limits
   - **Volatility Adjustment**: `calculate_volatility_adjusted_leverage()` - market regime-aware scaling
   - **Multi-layered Safety**: 3-sigma safety margins, regime-based adjustments, minimum 1x leverage

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

# 5. 🚀 NEW: Volatility-Adjusted Leverage
adjusted_leverage = risk_controller.calculate_volatility_adjusted_leverage(
    base_leverage=5.0,
    market_state={'daily_volatility': 0.08, 'regime': 'VOLATILE'}
)
# Returns: float - leverage adjusted for market conditions
```

#### **Integration Points:**

- **Portfolio State Dictionary**: Established structure for `check_var_limit()`
- **Return Format**: Violations returned as tuples for consistent error handling
- **Regime Awareness**: Market regime parameter for context-aware risk management
- **USDT Calculations**: Direct integration with position sizing systems

## 🧪 Test Suite

**Location**: `tests/test_risk_management.py`
**Status**: ✅ All tests passing (15/15) 🚀 **UPDATED**

### Test Coverage:
1. **Initialization Tests** (3 tests):
   - `test_should_initialize_with_correct_usdt_capital`
   - `test_should_set_default_risk_limits_based_on_capital`
   - `test_should_allow_custom_risk_parameters`

2. **VaR Limit Tests** (2 tests):
   - `test_should_detect_var_limit_violation`
   - `test_should_pass_when_var_within_limit`

3. **Kelly Criterion Tests** (4 tests):
   - `test_should_calculate_kelly_fraction_long_only_default`
   - `test_should_return_zero_for_negative_returns_long_only`
   - `test_should_allow_short_positions_when_enabled`
   - `test_should_return_zero_for_insufficient_data`

4. **🚀 NEW: Leverage Management Tests** (6 tests):
   - `test_should_detect_leverage_limit_violation`
   - `test_should_pass_when_leverage_within_limit`
   - `test_should_calculate_total_leverage_correctly`
   - `test_should_handle_empty_portfolio_leverage`
   - `test_should_calculate_safe_leverage_for_liquidation_distance`
   - `test_should_adjust_leverage_for_high_volatility`

### Test Execution Commands:
```bash
# Run all risk management tests
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_risk_management/ -v

# Run specific test
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_risk_management/test_risk_controller.py::TestRiskController::test_should_calculate_kelly_fraction_long_only_default -v
```

## 🚀 Next Implementation Priorities

### Still To Implement (Phase 1.2 completion):

1. ~~**Leverage Limit Checking**~~ ✅ **COMPLETED (2025-09-14)**:
```python
def check_leverage_limit(self, portfolio_state: Dict) -> List[Tuple[str, float]]:
    """✅ IMPLEMENTED: Check if total leverage exceeds limits"""

def calculate_safe_leverage_limit(self, portfolio_state: Dict) -> float:
    """✅ IMPLEMENTED: Calculate safe leverage based on liquidation distance"""

def calculate_volatility_adjusted_leverage(self, base_leverage: float, market_state: Dict) -> float:
    """✅ IMPLEMENTED: Adjust leverage for market volatility and regime"""
```

2. **Drawdown Monitoring** - **NEXT PRIORITY**:
```python
def update_drawdown(self, current_equity: float) -> float:
    """Update and return current drawdown percentage"""
```

3. **Position Sizing Engine**:
```python
def calculate_position_size(self, signal: Dict, market_state: Dict, portfolio_state: Dict) -> float:
    """Calculate optimal position size combining Kelly + ATR + liquidation safety"""
```

## 📚 Related Documentation

- **Design Reference**: `@docs/project-system-design/4-risk-management.md`
- **Financial Models**: `@docs/project-system-design/2-financial-engineering.md`
- **TDD Methodology**: `@docs/augmented-coding.md`
- **Main Progress**: `@IMPLEMENTATION_PROGRESS.md`

## ⚠️ Critical Dependencies

- **numpy**: 2.2.5 (for array operations)
- **pandas**: 2.3.2 (for time series handling)
- **scipy**: 1.15.3 (for statistical functions)
- **pytest**: For testing framework
- **Python Environment**: `autotrading` (Anaconda, Python 3.10.18)

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
**Last Implementation**: RiskController class (2025-09-14)
**Next Priority**: Leverage checking and drawdown monitoring