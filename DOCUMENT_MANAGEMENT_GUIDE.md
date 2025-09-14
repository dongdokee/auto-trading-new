# 📋 Document Management Guide - Duplication Prevention

**Created**: 2025-09-14
**Purpose**: Prevent documentation duplication and maintain Single Source of Truth
**For**: Claude Code and future developers

## 🎯 **Single Source of Truth Map**

### **📋 Core Documents & Their Exclusive Domains**

| Document | **Single Source of Truth For** | **Never Duplicate In** |
|----------|--------------------------------|-------------------------|
| `CLAUDE.md` | Development principles, navigation, TDD guidelines | Any other document |
| `PROJECT_STRUCTURE.md` | Tech stack, environment, commands, project structure | Any other document |
| `IMPLEMENTATION_PROGRESS.md` | Progress status, phase tracking, next priorities | Any other document |
| `src/[module]/CLAUDE.md` | Module implementation details, APIs, tests | Main documents |

### **🚨 CRITICAL RULES**

#### **1. Before Adding ANY Information:**
```
❓ Ask: "Is this information type already covered somewhere?"
👀 Check: Single Source of Truth Map above
✅ Action: Add reference link instead of duplicating
```

#### **2. Information Hierarchy:**
- **Level 1** (CLAUDE.md): Principles, concepts, navigation only
- **Level 2** (Specialized): Complete domain-specific info
- **Level 3** (Module): Implementation specifics only

#### **3. Mandatory Reference Format:**
```markdown
**For [information type]**: 📋 `@[SOURCE_DOCUMENT].md`
```

## 🔧 **New Document Creation Process**

### **Step 1: Determine Document Type**
- **New Module?** → Use `MODULE_CLAUDE_TEMPLATE.md`
- **New Specialized Domain?** → Create with clear Single Source boundaries
- **General Info?** → **STOP!** Add to existing document or reference it

### **Step 2: Duplication Check**
```
[ ] Information type not covered elsewhere?
[ ] Follows 3-level hierarchy rule?
[ ] Will be Single Source for specific domain?
[ ] Adds references to related documents?
```

### **Step 3: Navigation Update**
- **Update `CLAUDE.md`**: Add to Document Navigation Map
- **Update related documents**: Add cross-references
- **Update this guide**: Add new Single Source assignment

## 📊 **Current Document Roles (Post-Refactoring)**

### **CLAUDE.md** 🎯
**Role**: Development guide hub
**Contains**: Core principles, TDD, navigation map, documentation guidelines
**References**: All other documents
**Never Contains**: Environment details, progress status, tech stack

### **PROJECT_STRUCTURE.md** 🏗️
**Role**: Structure, tech, environment authority
**Contains**: Complete project structure, technology stack, all commands, environment setup
**References**: Main documents for general guidance
**Never Contains**: Progress details, module implementation specifics

### **IMPLEMENTATION_PROGRESS.md** 📊
**Role**: Progress tracking authority
**Contains**: Phase status, completed work, next priorities, revenue analysis
**References**: Main documents for structure/guidance
**Never Contains**: Environment commands, tech stack details

### **src/[module]/CLAUDE.md** 📂
**Role**: Module implementation details
**Contains**: APIs, implementation specifics, tests, module-specific patterns
**References**: Main documents for environment/guidance
**Never Contains**: General project info, environment setup, other modules' details

## ⚠️ **Common Duplication Traps to AVOID**

### **❌ DON'T:**
- Copy environment commands to module docs
- Repeat tech stack in multiple places
- Duplicate progress status across documents
- Copy project structure details
- Repeat TDD methodology in every document

### **✅ DO:**
- Add reference links to Single Source documents
- Keep module docs focused on implementation only
- Update navigation maps when adding documents
- Follow the 3-level hierarchy strictly
- Use the template for new modules

## 🔄 **Document Update Workflow**

### **Environment/Commands Change:**
1. Update `PROJECT_STRUCTURE.md` ONLY
2. Verify other documents reference (don't duplicate)

### **Progress Update:**
1. Update `IMPLEMENTATION_PROGRESS.md` ONLY
2. Update module CLAUDE.md with implementation details ONLY

### **New Module Implementation:**
1. Copy `MODULE_CLAUDE_TEMPLATE.md`
2. Fill implementation-specific details only
3. Add navigation link to main `CLAUDE.md`
4. Reference main documents, don't duplicate

## 🎯 **Quality Assurance Checklist**

Before committing any document changes:

```
[ ] No information duplicated from other documents?
[ ] Added appropriate references to related documents?
[ ] Follows Single Source of Truth assignments?
[ ] Updated navigation maps if new document?
[ ] Used template for module documentation?
[ ] Checked this guide for compliance?
```

## 📚 **Templates Available**

- **New Module**: `MODULE_CLAUDE_TEMPLATE.md` - Use for all new modules
- **Navigation Format**: See `CLAUDE.md` Document Navigation Map section
- **Reference Format**: `📋 @[DOCUMENT].md - [brief description]`

---

**⚠️ CRITICAL SUCCESS METRIC:**
**Zero duplication** across all documentation = **Successful document management**

**Last Updated**: 2025-09-14 (Post-refactoring)
**Next Review**: When next module is added