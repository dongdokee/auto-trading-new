# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

**Single Source of Truth for**: Development guidance, document navigation, TDD principles, documentation management rules

**Last Updated**: 2025-09-19 (Refactored: Removed duplicated content, focused on navigation and principles)

## Project Overview

This is a Korean cryptocurrency futures automated trading system (코인 선물 자동매매 시스템) implementing advanced quantitative trading strategies with sophisticated risk management and portfolio optimization.

**Current Status**: 100% complete (All phases ✅ COMPLETED + Enhanced Logging System)
**Latest Achievement**: Phase 6.1 Production Optimization + Paper Trading Validation System

## 📚 Document Navigation Map ⭐ SINGLE SOURCE OF TRUTH

### Core Documentation
- **📊 Project Status & Roadmap**: `@PROJECT_STATUS.md` - Complete project progress, roadmap, milestones, business value
- **🏗️ Technical Foundation**: `@PROJECT_STRUCTURE.md` - Complete structure, technology stack, architecture, environment setup
- **🚀 Quick Start Guide**: `@QUICK_START.md` - Essential commands and immediate productivity
- **📋 Document Management**: `@DOCUMENT_MANAGEMENT_GUIDE.md` - Documentation rules and maintenance

### Module-Specific Implementation Details (All 11 Modules Complete)
- **⚠️ Risk Management**: `@src/risk_management/CLAUDE.md` - ✅ COMPLETED (RiskController, PositionSizer, PositionManager)
- **📈 Strategy Engine**: `@src/strategy_engine/CLAUDE.md` - ✅ COMPLETED (4 strategies + regime detection + portfolio integration)
- **💼 Portfolio Management**: `@src/portfolio/CLAUDE.md` - ✅ COMPLETED (Markowitz optimization + performance attribution)
- **🏗️ Core Infrastructure**: `@src/core/CLAUDE.md` - ✅ COMPLETED (Database + Configuration + Utilities)
- **🧪 Backtesting**: `@src/backtesting/CLAUDE.md` - ✅ COMPLETED (Walk-forward validation + data quality)
- **🛠️ Utilities**: `@src/utils/CLAUDE.md` - ✅ COMPLETED (Enhanced logging + financial math + time utilities)
- **⚡ Order Execution**: `@src/execution/CLAUDE.md` - ✅ COMPLETED (Smart routing + execution algorithms)
- **🔗 API Integration**: `@src/api/CLAUDE.md` - ✅ COMPLETED (Binance integration + WebSocket + Paper trading)
- **🎯 System Integration**: `@src/integration/CLAUDE.md` - ✅ COMPLETED (Event-driven architecture + orchestration)
- **📊 Market Data**: `@src/market_data/CLAUDE.md` - ✅ COMPLETED (Real-time analytics + microstructure analysis)
- **🚀 Production Optimization**: `@src/optimization/CLAUDE.md` - ✅ COMPLETED (Performance optimization + deployment tools)

### Technical Specifications
- **🏛️ System Architecture**: `@docs/project-system-architecture.md` - C4 model, components
- **💰 Financial Engineering**: `@docs/project-system-design/2-financial-engineering.md` - Kelly Criterion, VaR models
- **🧪 TDD Methodology**: `@docs/augmented-coding.md` - Development discipline
- **🔧 Engineering Guide**: `@docs/software-engineering-guide.md` - Best practices
- **🔧 Architecture Decisions**: `@docs/ARCHITECTURE_DECISIONS.md` - Technical decision records

## Core Development Principles

### Test-Driven Development (TDD)
**MUST FOLLOW**: `@docs/augmented-coding.md` - Complete TDD methodology and discipline

**Core Development Cycle**: Red → Green → Refactor
- Write failing test first
- Implement minimum code to pass
- Refactor only when tests are passing
- Separate structural from behavioral changes

### Engineering Best Practices
**Reference**: `@docs/software-engineering-guide.md` - Comprehensive engineering guidelines

**Key Principles**:
- **Separation of Concerns**: Each component has single responsibility
- **KISS & YAGNI**: Simple solutions, implement only when needed
- **Dependency Inversion**: High-level modules depend on abstractions
- **Clean Code**: Meaningful names, small functions, clear control flow

### Code Quality Standards
- **CRITICAL**: Never use emojis or Unicode characters in generated code
- Use only ASCII characters for all code elements
- Ensure compatibility with standard text editors and version control
- Documentation files (.md) may use Unicode characters as needed

### Financial Engineering Discipline
**Given the critical nature of financial calculations**:
- Comprehensive testing with known outputs
- Extensive edge case coverage
- Benchmark validation against existing models
- Real-time risk monitoring and control

## Development Workflow

### Context Management Strategy
This project uses **modular CLAUDE.md files** for better context management:

- **Main CLAUDE.md**: Overall project guidance, navigation, principles
- **Module CLAUDE.md**: Specific implementation details, completed work, API interfaces
- **When working on a module**: Always check both main + module CLAUDE.md files

### Problem-Solving Approach
When debugging complex issues:

**Systematic Debugging Process** (`@docs/software-engineering-guide.md`):
1. **Reproduce the Issue**: Create reliable test case
2. **Gather Information**: Logs, traces, system state
3. **Form & Test Hypotheses**: One change at a time
4. **Document Findings**: Build institutional knowledge

### Quality Assurance Workflow
**STRICTLY FOLLOW**: `@docs/augmented-coding.md` commit discipline

**Before Every Commit**:
1. ✅ All tests passing
2. ✅ No compiler/linter warnings
3. ✅ Single logical unit of work
4. ✅ Clear commit message indicating structural vs. behavioral change

## Development Environment & Commands

**For complete environment setup**: `@PROJECT_STRUCTURE.md` - All commands, troubleshooting, package management

**Environment Strategy**: Two separate conda environments for development and production
```bash
# Development environment (testing, linting, etc.)
conda activate autotrading-dev
python -m pytest tests/ -v

# Production environment (paper trading, backtesting, live trading)
conda activate autotrading
python scripts/paper_trading.py
```

## Documentation Guidelines ⭐ DUPLICATION PREVENTION

### Critical: When Creating New Documents

**Before creating any new document, check these rules to prevent duplication:**

#### 1. Information Hierarchy Rules
- **Level 1 (Main CLAUDE.md)**: Only concepts, principles, and navigation
- **Level 2 (Specialized docs)**: Complete details for specific domains
- **Level 3 (Module CLAUDE.md)**: Implementation specifics only

#### 2. Single Source of Truth Assignments
- **Environment & Commands**: ➡️ `PROJECT_STRUCTURE.md` ONLY
- **Progress & Status**: ➡️ `PROJECT_STATUS.md` ONLY
- **Tech Stack & Dependencies**: ➡️ `PROJECT_STRUCTURE.md` ONLY
- **Module Implementation**: ➡️ `src/[module]/CLAUDE.md` ONLY

#### 3. New Module Documentation Process
1. **Use Template**: Copy `@MODULE_CLAUDE_TEMPLATE.md`
2. **Fill Module-Specific Info**: Only implementation details
3. **Add Navigation**: Update main CLAUDE.md navigation map
4. **NO Duplication**: Don't repeat environment, tech stack, or general info

#### 4. Documentation Update Rules
- **Environment changes** ➡️ Update `PROJECT_STRUCTURE.md` only
- **Progress changes** ➡️ Update `PROJECT_STATUS.md` only
- **Implementation details** ➡️ Update respective module CLAUDE.md only

#### 5. Duplication Check Checklist
Before adding information to any document, ask:
- [ ] Is this information already in another document?
- [ ] Which document is the Single Source of Truth for this type of info?
- [ ] Am I adding navigation/reference instead of duplicating content?
- [ ] Does this follow the 3-level hierarchy rule?

### Document Quality Standards
- **Always add references** to related documents
- **Use navigation links** instead of copying information
- **Keep modules focused** on implementation specifics
- **Update navigation maps** when adding new documents

## Key System Features (All Complete)

The system now includes all planned features and optimizations:

### 🎯 Production Ready System
- **924+ Tests**: All modules thoroughly tested with 100% pass rate
- **Paper Trading Validation**: Complete testing environment with enhanced logging
- **ROI Infrastructure**: 15-35% monthly ROI support with production optimization
- **Enterprise Features**: Real-time monitoring, deployment automation, advanced analytics

### 📊 Core Achievements
- **Complete Trading Pipeline**: Strategy signals → Portfolio optimization → Risk management → Order execution → Live trading
- **Advanced Financial Models**: Kelly Criterion, HMM/GARCH regime detection, Markowitz optimization, ML-based market impact
- **Production Infrastructure**: Event-driven architecture, real-time analytics, enterprise deployment tools

### 🔄 Operational Modes
- **Paper Trading**: Safe validation environment with comprehensive logging
- **Live Trading**: Production-ready system with full risk controls
- **Backtesting**: Historical validation with walk-forward analysis

## Related Documentation

### Essential References
- **📊 Current Progress**: `@PROJECT_STATUS.md` - Overall project status and next steps
- **🏗️ Technical Details**: `@PROJECT_STRUCTURE.md` - Complete technical foundation
- **🚀 Quick Commands**: `@QUICK_START.md` - Essential development commands
- **📋 Documentation Rules**: `@DOCUMENT_MANAGEMENT_GUIDE.md` - Documentation management

### Implementation-Specific
- **🧪 TDD Methodology**: `@docs/augmented-coding.md` - Complete TDD discipline
- **🏛️ System Architecture**: `@docs/project-system-architecture.md` - Complete architecture
- **🔧 Engineering Guide**: `@docs/software-engineering-guide.md` - Best practices

---

**CRITICAL SUCCESS METRIC**: Zero duplication across all documentation = Successful context delivery to Claude

**Last Updated**: 2025-01-04 (Updated: All 11 modules complete, 924+ tests, Paper Trading Validation System)