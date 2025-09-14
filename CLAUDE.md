# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Korean cryptocurrency futures automated trading system (코인 선물 자동매매 시스템) implementing advanced quantitative trading strategies with sophisticated risk management and portfolio optimization.

## 📚 **Document Navigation Map** ⭐ **SINGLE SOURCE OF TRUTH**

### **📋 Core Documentation**
- **📊 Progress & Status**: `@IMPLEMENTATION_PROGRESS.md` - Current phase, completed work, next priorities
- **🏗️ Structure & Environment**: `@PROJECT_STRUCTURE.md` - Complete structure, tech stack, commands
- **🗺️ Implementation Roadmap**: `@docs/AGREED_IMPLEMENTATION_PLAN.md` - 5-phase development plan

### **📂 Module-Specific Implementation Details**
- **⚠️ Risk Management**: `@src/risk_management/CLAUDE.md` - Complete risk framework (RiskController, PositionSizer, PositionManager)
- **📈 Strategy Engine**: `@src/strategy_engine/CLAUDE.md` - (planned)
- **⚡ Order Execution**: `@src/execution/CLAUDE.md` - (planned)
- **💼 Portfolio Management**: `@src/portfolio/CLAUDE.md` - (planned)

### **📖 Technical Specifications**
- **🏛️ System Architecture**: `@docs/project-system-architecture.md` - C4 model, components
- **💰 Financial Engineering**: `@docs/project-system-design/2-financial-engineering.md` - Kelly Criterion, VaR models
- **🧪 TDD Methodology**: `@docs/augmented-coding.md` - Development discipline
- **🔧 Engineering Guide**: `@docs/software-engineering-guide.md` - Best practices

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

**CRITICAL**: Always follow the structure defined in `@PROJECT_STRUCTURE.md`

### ⚠️ **IMPORTANT RULES**:
1. **NO files directly in `src/core/`** - Only configuration, logging, exceptions
2. **Each module gets its own directory** under `src/`
3. **Test structure mirrors source structure** under `tests/unit/`
4. **Always check `@PROJECT_STRUCTURE.md` before creating new files**

**For complete project structure details**: 📋 `@PROJECT_STRUCTURE.md`

## Architecture Overview

The system follows a microservices architecture with event-driven patterns, CQRS, and hexagonal architecture principles.

📋 **Complete Architecture Details**: `@docs/project-system-architecture.md`
📋 **Technology Stack & Environment Setup**: `@PROJECT_STRUCTURE.md`

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

**For current phase status and next priorities**: 📋 `@IMPLEMENTATION_PROGRESS.md`

## Development Commands

**For complete environment setup and commands**: 📋 `@PROJECT_STRUCTURE.md`

**Critical Environment Info**:
- **Environment Name**: `autotrading` (Anaconda, Python 3.10.18)
- **⚠️ MANDATORY**: Use direct paths (conda activation fails in this environment)
- **Test Command**: `"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/ -v`

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

## 📋 **Documentation Guidelines** ⭐ **DUPLICATION PREVENTION**

### **🚨 CRITICAL: When Creating New Documents**

**Before creating any new document, check these rules to prevent duplication:**

#### **1. Information Hierarchy Rules**
- **Level 1 (Main CLAUDE.md)**: Only concepts, principles, and navigation
- **Level 2 (Specialized docs)**: Complete details for specific domains
- **Level 3 (Module CLAUDE.md)**: Implementation specifics only

#### **2. Single Source of Truth Assignments**
- **Environment & Commands**: ➡️ `PROJECT_STRUCTURE.md` ONLY
- **Progress & Status**: ➡️ `IMPLEMENTATION_PROGRESS.md` ONLY
- **Tech Stack & Dependencies**: ➡️ `PROJECT_STRUCTURE.md` ONLY
- **Module Implementation**: ➡️ `src/[module]/CLAUDE.md` ONLY

#### **3. New Module Documentation Process**
1. **Use Template**: Copy `@MODULE_CLAUDE_TEMPLATE.md`
2. **Fill Module-Specific Info**: Only implementation details
3. **Add Navigation**: Update main CLAUDE.md navigation map
4. **NO Duplication**: Don't repeat environment, tech stack, or general info

#### **4. Documentation Update Rules**
- **Environment changes** ➡️ Update `PROJECT_STRUCTURE.md` only
- **Progress changes** ➡️ Update `IMPLEMENTATION_PROGRESS.md` only
- **Implementation details** ➡️ Update respective module CLAUDE.md only

#### **5. Duplication Check Checklist**
Before adding information to any document, ask:
- [ ] Is this information already in another document?
- [ ] Which document is the Single Source of Truth for this type of info?
- [ ] Am I adding navigation/reference instead of duplicating content?
- [ ] Does this follow the 3-level hierarchy rule?

### **🎯 Document Quality Standards**
- **Always add references** to related documents
- **Use navigation links** instead of copying information
- **Keep modules focused** on implementation specifics
- **Update navigation maps** when adding new documents