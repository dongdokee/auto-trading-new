# [Module Name] Module - CLAUDE.md

This file provides specific guidance for Claude Code when working on the [module_name] module.

## Module Overview

**Location**: `src/[module_name]/`
**Purpose**: [Brief description of module purpose]
**Status**: [Implementation status]
**Last Updated**: [Date]

## ⭐ IMPLEMENTATION CONTEXT ⭐

### 🚀 Successfully Completed: [List completed features]

#### **[Class/Component Name]** ✅
**File**: `src/[module_name]/[file_name].py`
**Tests**: `tests/unit/test_[module_name]/test_[file_name].py` ([N] test cases, all passing)
**Implementation Date**: [Date]

#### **Key Architecture Decisions:**

1. **[Decision 1]** - [Rationale]:
```python
# Code example
```

2. **[Decision 2]** - [Rationale]:
   - [Detail 1]
   - [Detail 2]

#### **Critical Technical Patterns:**

- **TDD Methodology**: Red → Green → Refactor cycles applied
- **Test Naming**: `test_should_[behavior]_when_[condition]`
- **Edge Case Handling**: [List key edge cases handled]
- **Type Safety**: Full type annotations throughout
- **Documentation**: Comprehensive docstrings with Args/Returns

#### **API Interface:**

```python
# Main API usage examples
[module_class] = [ModuleClass]([parameters])

# Key methods
result = [module_class].[key_method]([parameters])
# Returns: [return_type] - [description]
```

#### **Integration Points:**

- **[Integration 1]**: [Description]
- **[Integration 2]**: [Description]

## 🧪 Comprehensive Test Suite

**Total Tests**: ✅ [N] tests passing ([unit] unit + [integration] integration)

### **Unit Tests** ([N] tests)
**Location**: `tests/unit/test_[module_name]/`

#### **[TestClass] Tests** ([N] tests) - `test_[file_name].py`
- **[Category] Tests** ([N] tests): [Description]
- **[Category] Tests** ([N] tests): [Description]

### **Integration Tests** ([N] tests)
**Location**: `tests/integration/test_[module_name]_integration.py`

- **[Integration Test 1]** (1 test): [Description]
- **[Integration Test 2]** (1 test): [Description]

### Test Execution Commands:
**For complete environment commands**: 📋 `@PROJECT_STRUCTURE.md`

```bash
# [Module Name] specific tests
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/unit/test_[module_name]/ -v
"/c/Users/dongd/anaconda3/envs/autotrading/python.exe" -m pytest tests/integration/test_[module_name]_integration.py -v
```

## 🎉 **PHASE [X.Y] COMPLETED** - [Module Title] 🚀

### ✅ **ALL IMPLEMENTATIONS COMPLETED ([Date])**

#### 1. ~~**[Component 1]**~~ ✅ **FULLY COMPLETED**:
```python
# Usage example
```

**Key Features**:
- **[Feature 1]**: [Description]
- **[Feature 2]**: [Description]

## 🚀 **READY FOR NEXT PHASE: [Next Phase]**

The [module_name] module is **PRODUCTION-READY** and provides a complete foundation for:

### **Phase [X.Y]: [Next Phase] Integration** 🎯 **NEXT PRIORITY**

**Integration Points**:
```python
# Integration example code
```

### **Future Integration Phases**:
- **[Phase 1]** ([Phase X.Y]): [Description]
- **[Phase 2]** ([Phase X.Y]): [Description]

### **[Module] API Ready For**:
- ✅ **[Capability 1]**: [Description]
- ✅ **[Capability 2]**: [Description]

## 📚 **Related Documentation**

### **📋 Main Claude Code References**
- **🎯 Development Guide**: `@CLAUDE.md` - Core development guidance and document navigation
- **📊 Progress Status**: `@IMPLEMENTATION_PROGRESS.md` - Overall project progress and next steps
- **🏗️ Project Structure**: `@PROJECT_STRUCTURE.md` - Complete environment setup and commands

### **📖 Technical Specifications**
- **[Spec 1]**: `@docs/project-system-design/[N]-[module].md` - [Description]
- **[Spec 2]**: `@docs/project-system-design/[N]-[related].md` - [Description]
- **🧪 TDD Methodology**: `@docs/augmented-coding.md` - Development discipline and practices

## ⚠️ Critical Dependencies

**For complete dependency information**: 📋 `@PROJECT_STRUCTURE.md`
**Key Requirements**: [list key dependencies]

## 🔧 Development Patterns for This Module

When extending this module:

1. **Always TDD**: Write failing test first
2. **Type Everything**: Full type annotations required
3. **Document Edge Cases**: Handle [specific edge cases]
4. **[Module-Specific Pattern]**: [Description]
5. **Configuration First**: Make parameters configurable
6. **Test Integration**: Test with related modules

## 🎯 Performance Considerations

- **[Performance Requirement 1]**: [Description]
- **[Performance Requirement 2]**: [Description]

---
**Module Maintainer**: [Team/Person]
**Last Implementation**: [Feature] ([Date])
**Next Priority**: [Next feature/phase]