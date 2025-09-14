# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Korean cryptocurrency futures automated trading system (코인 선물 자동매매 시스템) implementing advanced quantitative trading strategies with sophisticated risk management and portfolio optimization.

## 📚 **Document Navigation Map** ⭐ **SINGLE SOURCE OF TRUTH**

### **📋 Core Documentation**
- **📊 Project Roadmap & Status**: `@PROJECT_ROADMAP_AND_STATUS.md` - Complete project overview, progress, roadmap, next priorities
- **🏗️ Structure & Environment**: `@PROJECT_STRUCTURE.md` - Complete structure, environment setup, commands
- **🔧 Technology Stack**: `@TECHNOLOGY_STACK.md` - All technical specifications, architecture patterns, dependencies

### **📂 Module-Specific Implementation Details**
- **⚠️ Risk Management**: `@src/risk_management/CLAUDE.md` - ✅ **PHASE 1 COMPLETED** (RiskController, PositionSizer, PositionManager)
- **📈 Strategy Engine**: `@src/strategy_engine/CLAUDE.md` - ✅ **PHASE 3.1-3.2 COMPLETED** (4 strategies + regime detection + portfolio integration)
- **💼 Portfolio Management**: `@src/portfolio/CLAUDE.md` - ✅ **PHASE 3.3 COMPLETED** (Markowitz optimization + performance attribution)
- **🏗️ Core Infrastructure**: `@src/core/CLAUDE.md` - ✅ **PHASE 2.1-2.2 COMPLETED** (Database + Configuration + Utilities)
- **⚡ Order Execution**: `@src/execution/CLAUDE.md` - (Phase 4.1 - Ready to start)

### **📖 Technical Specifications**
- **🏛️ System Architecture**: `@docs/project-system-architecture.md` - C4 model, components
- **💰 Financial Engineering**: `@docs/project-system-design/2-financial-engineering.md` - Kelly Criterion, VaR models
- **🧪 TDD Methodology**: `@docs/augmented-coding.md` - Development discipline
- **🔧 Engineering Guide**: `@docs/software-engineering-guide.md` - Best practices

## 📁 **Modular Context Management Strategy** ⭐

**IMPORTANT**: This project uses **modular CLAUDE.md files** for better context management:

### 📂 Module-Specific Context Files:
- **`@src/risk_management/CLAUDE.md`** ✅ **COMPLETED** - Risk management implementation (Phase 1)
- **`@src/strategy_engine/CLAUDE.md`** ✅ **COMPLETED** - 4-strategy system + regime detection (Phase 3.1-3.2)
- **`@src/portfolio/CLAUDE.md`** ✅ **COMPLETED** - Portfolio optimization system (Phase 3.3)
- **`@src/core/CLAUDE.md`** ✅ **COMPLETED** - Database + configuration infrastructure (Phase 2.1-2.2)
- **`@src/backtesting/CLAUDE.md`** ✅ **COMPLETED** - Backtesting framework (Phase 2.1)
- **`@src/utils/CLAUDE.md`** ✅ **COMPLETED** - Utility functions and logging (Phase 2.1)
- **`@src/execution/CLAUDE.md`** (Phase 4.1 - Ready to start) - Order execution context
- **`@src/data/CLAUDE.md`** (Future) - Data pipeline context

### 🎯 **Context Storage Pattern**:
- **Main CLAUDE.md**: Overall project guidance, architecture, environment setup
- **Module CLAUDE.md**: Specific implementation details, completed work, API interfaces
- **When working on a module**: Always check both main + module CLAUDE.md files

### ✅ **Already Implemented (70% Complete)**:
- **Risk Management Module**: Complete with TDD (Phase 1 ✅)
- **Backtesting Framework**: Complete with data validation (Phase 2.1 ✅)
- **Database Infrastructure**: Complete with migrations (Phase 2.2 ✅)
- **Strategy Engine**: 4 strategies + regime detection (Phase 3.1-3.2 ✅)
- **Portfolio Optimization**: Complete Markowitz system (Phase 3.3 ✅)
- **Core Infrastructure**: Configuration + utilities + logging (Phase 2.1-2.2 ✅)

## 🏆 **Implementation Status Overview** ⭐

### ✅ **COMPLETED PHASES (70% of project)**
- **Phase 1**: Risk Management Framework ✅ (RiskController + PositionSizer + PositionManager)
- **Phase 2.1**: Backtesting & Infrastructure ✅ (DataLoader + DataValidator + BacktestEngine + Utilities)
- **Phase 2.2**: Database Migration System ✅ (Alembic + PostgreSQL + 7 core tables)
- **Phase 3.1**: Core Strategy Engine ✅ (2 strategies + regime detection + signal aggregation)
- **Phase 3.2**: Enhanced Strategy Engine ✅ (4 strategies + enhanced StrategyMatrix)
- **Phase 3.3**: Complete Portfolio Optimization ✅ (Markowitz + Attribution + Correlation + Adaptive)

### 🚀 **NEXT PHASE (Ready to start)**
- **Phase 4.1**: Order Execution Engine (Smart routing + execution algorithms)
- **Phase 4.2**: API Integration (Binance Futures + WebSocket feeds)
- **Phase 5**: System Integration & Production (Complete system + 30-day validation)

### 📊 **Key Metrics**
- **222 tests**: 100% passing across all implemented modules
- **4 trading strategies**: Fully implemented and portfolio-integrated
- **Complete pipeline**: Strategy → Portfolio Optimization → Risk Management → Position Sizing
- **Production-ready**: Database migrations + configuration management + structured logging
- **Real-time performance**: <100ms complete trading workflow processing

### 🎯 **Revenue Generation Timeline**
- **Phase 4.2 completion (85%)**: First revenue generation possible (2 weeks from now)
- **Phase 5.1 completion (90%)**: Stable revenue with paper trading validation (4-5 weeks)
- **Phase 5.2 completion (100%)**: Optimized revenue with full system validation (8-10 weeks)

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
**MUST FOLLOW**: `@docs/augmented-coding.md` - Complete TDD methodology and discipline

**Core Development Cycle**: Red → Green → Refactor
**Critical Rules**: Test-first development, separate structural/behavioral changes, meaningful test names

### Engineering Best Practices
**Reference**: `@docs/software-engineering-guide.md` - Comprehensive engineering guidelines

**Key Principles**:
- **Separation of Concerns**: Each component has single responsibility
- **KISS & YAGNI**: Simple solutions, implement only when needed
- **Dependency Inversion**: High-level modules depend on abstractions
- **Clean Code**: Meaningful names, small functions, clear control flow

## Financial Engineering Components

**The system implements sophisticated quantitative finance concepts** - See module-specific CLAUDE.md files for detailed implementation:

- **Risk Management**: Kelly Optimization + VaR models + drawdown monitoring → `@src/risk_management/CLAUDE.md`
- **Strategy Engine**: HMM/GARCH regime detection + 4-strategy system → `@src/strategy_engine/CLAUDE.md`
- **Portfolio Optimization**: Markowitz optimization + performance attribution → `@src/portfolio/CLAUDE.md`
- **Backtesting**: Walk-forward validation + bias prevention → `@src/backtesting/CLAUDE.md`

📋 **Technical Specifications**: `@docs/project-system-design/` - Complete financial engineering models

## Development Guidelines

### Code Organization
- Follow domain-driven design principles
- Implement clean architecture with clear boundaries
- Use dependency injection for testability
- Separate pure business logic from infrastructure concerns

### Code Generation Standards
- **CRITICAL**: Never use emojis or Unicode characters in generated code, including string literals and outputs
- Use only ASCII characters for all code elements: variables, functions, comments, and string content
- Ensure all code remains compatible with standard text editors and version control systems
- Documentation files (.md) may use Unicode characters as needed

📋 **Reference**: `@docs/software-engineering-guide.md` (섹션 1-4)

### TDD Workflow for Financial Components
**Given the critical nature of financial calculations**: Comprehensive testing with known outputs, extensive edge case coverage, benchmark validation

📋 **Complete TDD Guidelines**: `@docs/augmented-coding.md` - Strict TDD discipline for financial systems

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

## 🎯 Current Development Status & Next Phase

**Overall Progress**: 70% complete (Phase 1-3.3 ✅ COMPLETED)
**Current Phase**: Phase 4.1 - Order Execution Engine (Ready to start)
**Last Milestone**: Phase 3.3 Portfolio Optimization completed (2025-09-15)

**📊 For detailed progress tracking**: 📋 `@IMPLEMENTATION_PROGRESS.md`
**🚀 Ready for Phase 4**: Order execution engine development can begin immediately

## Development Environment

**📋 Complete Environment Setup**: `@ENVIRONMENT.md` - Python environment, all commands, troubleshooting
**Quick Reference**: Anaconda `autotrading` (Python 3.10.18) - Use direct paths for all commands

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
- **Progress changes** ➡️ Update `PROJECT_ROADMAP_AND_STATUS.md` only
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