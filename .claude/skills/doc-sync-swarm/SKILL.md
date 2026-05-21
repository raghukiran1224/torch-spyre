---
name: doc-sync-swarm
description: "Multi-agent documentation sync that detects code-to-docs drift, updates the supported ops table, syncs API docs, and verifies compiler/user guide accuracy. Run after op additions or API changes."
---

# Doc Sync Swarm

Detects documentation drift against the current codebase and
generates fixes. Uses 4 parallel agents to audit different
documentation domains, then produces PRs or patches for each
area that has drifted.

Use after merging op additions, API changes, or compiler
modifications. Can also be run periodically to catch accumulated
drift.

---

## Invocation

```
/doc-sync-swarm
/doc-sync-swarm --ops-only
/doc-sync-swarm --since "3 days ago"
```

Without arguments, performs a full audit. With `--ops-only`, only
checks the supported operations table. With `--since`, limits the
scope to changes since the given date.

---

## Execution Strategy

### Phase 1: Detect Changes (sequential, fast)

Identify what code changed that could cause doc drift:

```bash
# If --since provided, scope to recent commits
git log --oneline --since="<date>" -- \
  torch_spyre/_inductor/spyre_kernel.py \
  torch_spyre/_inductor/decompositions.py \
  torch_spyre/_inductor/customops.py \
  torch_spyre/_inductor/lowering.py \
  torch_spyre/ops/eager.py \
  torch_spyre/__init__.py \
  torch_spyre/_inductor/constants.py

# Otherwise, compare docs against current code state
```

### Phase 2: Fan Out (4 parallel agents)

| Agent | Domain | Primary Doc File |
|-------|--------|------------------|
| **ops-table-agent** | Supported operations table | `docs/source/user_guide/supported_operations.md` |
| **api-docs-agent** | Public API reference | `docs/source/api/torch_spyre.rst` |
| **compiler-docs-agent** | Compiler architecture docs | `docs/source/compiler/*.md` |
| **user-guide-agent** | User-facing guides | `docs/source/user_guide/*.md` |

### Phase 3: Apply Fixes

For each agent that found drift, apply the fix directly to the
doc files. Then run Sphinx build to verify no broken links.

### Phase 4: Report

Present a summary of what was updated and offer to commit.

---

## Agent A: ops-table-agent

**Goal:** Synchronize `docs/source/user_guide/supported_operations.md`
with the actual code state.

### Data Sources

| Column | Source of Truth |
|--------|----------------|
| Eager | `torch_spyre/ops/eager.py` (register_torch_compile_kernel list) + `torch_spyre/_inductor/decompositions.py` (aten ops with PrivateUse1 dispatch) |
| Compiled | `torch_spyre/_inductor/spyre_kernel.py` (SpyreOpFuncs methods) + `torch_spyre/_inductor/customops.py` + `torch_spyre/_inductor/lowering.py` |
| Execution | "Spyre" if in SpyreOpFuncs or has lowering; "CPU fallback" if only in `fallbacks.py` |
| Notes | Decomposition details from `decompositions.py`; special handling notes |

### Steps

1. **Extract compiled ops from code:**

```bash
# SpyreOpFuncs methods (direct mappings)
grep -E "def [a-z]" torch_spyre/_inductor/spyre_kernel.py \
  | grep -v "__" | awk '{print $2}' | cut -d'(' -f1 | sort

# Custom ops with lowerings
grep "register_spyre_lowering" torch_spyre/_inductor/lowering.py \
  | grep -oE "torch\.ops\.(spyre|aten)\.[a-z_]+" | sort

# Decompositions (compiled path)
grep "register_spyre_decomposition" torch_spyre/_inductor/decompositions.py \
  | grep -oE "aten\.[a-z_]+\.[a-z_]+" | sort
```

2. **Extract eager ops from code:**

```bash
# Eager registration list
grep "aten\." torch_spyre/ops/eager.py | grep -oE "aten\.[a-z_]+" | sort

# Fallback ops (CPU execution)
grep -E "def |aten\." torch_spyre/ops/fallbacks.py \
  | grep -oE "aten\.[a-z_]+" | sort
```

3. **Extract current table from docs:**

```bash
grep "^|" docs/source/user_guide/supported_operations.md \
  | grep -v "^|---" | grep -v "^| Operation"
```

4. **Compare and generate diff:**

For each op in code but NOT in table → add row.
For each op in table but NOT in code → flag as stale.
For each op with wrong Eager/Compiled/Execution status → fix.

### Output

Edit `docs/source/user_guide/supported_operations.md` directly
to add missing ops or correct statuses. Maintain the existing
table format and category groupings.

Report:
```markdown
### Ops Table Sync

- **Added:** <N> ops (<list>)
- **Removed (stale):** <N> ops (<list>)
- **Corrected:** <N> ops (<list with what changed>)
- **No changes needed:** (if table is current)
```

---

## Agent B: api-docs-agent

**Goal:** Synchronize `docs/source/api/torch_spyre.rst` with the
actual public API exported by `torch_spyre/__init__.py`.

### Steps

1. **Extract public API from code:**

```bash
# Functions and classes exported via make_spyre_module
grep -E "^\s+(def |class )" torch_spyre/__init__.py \
  | grep -v "^.*_" | awk '{print $2}' | cut -d'(' -f1

# Stream API
grep -E "^\s+(def |class )" torch_spyre/streams.py \
  | grep -v "^.*_" | awk '{print $2}' | cut -d'(' -f1

# Constants / env vars
grep -E "^[A-Z_]+ =" torch_spyre/_inductor/constants.py
```

2. **Extract documented API from RST:**

```bash
grep -E "^\.\. (function|class|attribute|data)::" \
  docs/source/api/torch_spyre.rst
```

3. **Compare:**

- Functions in code but not in docs → add autodoc entry
- Functions in docs but removed from code → remove entry
- Signature changes → flag for manual review

### Output

Edit `docs/source/api/torch_spyre.rst` if needed, or report
what needs manual attention (signature changes require prose
updates).

```markdown
### API Docs Sync

- **Added:** <N> entries (<list>)
- **Removed (stale):** <N> entries (<list>)
- **Signature changes:** <list — needs manual review>
- **No changes needed:** (if current)
```

---

## Agent C: compiler-docs-agent

**Goal:** Verify compiler documentation matches current code
structure.

### Docs to Check

| Doc | Validates Against |
|-----|-------------------|
| `docs/source/compiler/architecture.md` | Module structure in `torch_spyre/_inductor/` |
| `docs/source/compiler/adding_operations.md` | Patterns in `spyre_kernel.py`, `decompositions.py`, `customops.py`, `lowering.py` |
| `docs/source/compiler/work_division_planning.md` | `torch_spyre/_inductor/codegen/` structure |

### Steps

1. **Check module references are valid:**

```bash
# Extract file/module references from docs
grep -oE "torch_spyre/[a-z_/]+\.py" docs/source/compiler/*.md | sort -u

# Verify each exists
for f in $(grep -ohE "torch_spyre/[a-z_/]+\.py" \
  docs/source/compiler/*.md | sort -u); do
  [ -f "$f" ] || echo "MISSING: $f"
done
```

2. **Check class/function references are valid:**

```bash
# Extract code references from docs
grep -oE "(SpyreOpFuncs|register_spyre_decomposition|register_spyre_lowering|PointwiseOp|ReductionOp)" \
  docs/source/compiler/*.md
```

Verify each referenced symbol still exists at the stated location.

3. **Check pattern examples are current:**

If `adding_operations.md` shows example code, verify the patterns
match current code style (e.g., decorator signatures, import paths).

4. **Check for new modules not yet documented:**

```bash
# Find Python modules in _inductor not mentioned in any doc
find torch_spyre/_inductor -name "*.py" -not -name "__*" \
  | while read f; do
    base=$(basename "$f" .py)
    grep -rl "$base" docs/source/compiler/ > /dev/null || echo "UNDOC: $f"
  done
```

### Output

```markdown
### Compiler Docs Sync

- **Stale references:** <list of dead file/symbol references>
- **New undocumented modules:** <list>
- **Pattern drift:** <examples that no longer match code>
- **No changes needed:** (if current)
```

---

## Agent D: user-guide-agent

**Goal:** Verify user-facing documentation is accurate.

### Docs to Check

| Doc | What to Verify |
|-----|----------------|
| `docs/source/user_guide/running_models.md` | Import patterns, device usage, compile invocation |
| `docs/source/user_guide/tensors_and_layouts.md` | Layout types, stick sizes, dtype support |
| `docs/source/user_guide/profiling.md` | Profiler API matches `torch_spyre/profiler/` |
| `docs/source/user_guide/debugging.md` | Env vars match `constants.py` and CLAUDE.md |
| `docs/source/getting_started/installation.md` | Dependencies match `pyproject.toml` |
| `docs/source/getting_started/quickstart.md` | Code examples are runnable |

### Steps

1. **Environment variables audit:**

```bash
# Extract env vars from code
grep -rhoE "os\.(environ|getenv)\(['\"]([A-Z_]+)" torch_spyre/ \
  --include="*.py" | sort -u

# Extract env vars documented
grep -oE "[A-Z_]{4,}" docs/source/user_guide/debugging.md | sort -u
```

Flag env vars in code but not documented, or documented but
no longer in code.

2. **Import pattern verification:**

Check that quickstart/running_models examples use current
patterns:
- `import torch_spyre` (not `from torch_spyre import ...`)
- `torch.compile()` without backend arg (Spyre is an Inductor
  device backend, dispatched automatically)
- Device tensors: `torch.randn(..., device="spyre")`

3. **Dependency version check:**

```bash
# pyproject.toml torch version
grep "torch" pyproject.toml | head -3

# installation.md mentioned versions
grep -iE "torch|pytorch|python" \
  docs/source/getting_started/installation.md | head -10
```

4. **Profiler API sync:**

```bash
# Public profiler functions
grep -E "^(def |class )" torch_spyre/profiler/*.py \
  | grep -v "^.*_"

# Documented profiler usage
grep -E "profiler\." docs/source/user_guide/profiling.md
```

### Output

```markdown
### User Guide Sync

- **Env var drift:** <undocumented vars or stale docs>
- **Version mismatch:** <pyproject vs docs>
- **API pattern drift:** <outdated examples>
- **No changes needed:** (if current)
```

---

## Final Output Format

```markdown
# Documentation Sync Report

**Date:** <today>
**Scope:** Full audit / Ops only / Since <date>

## Summary

| Area | Status | Changes |
|------|--------|---------|
| Ops Table | SYNCED/DRIFTED | +N added, -N stale |
| API Docs | SYNCED/DRIFTED | +N added, -N removed |
| Compiler Docs | SYNCED/DRIFTED | N stale refs, N undocumented |
| User Guide | SYNCED/DRIFTED | N issues |

## Changes Made

<list of files edited with description of change>

## Manual Action Required

<items that need human judgment — e.g., prose descriptions
for new features, architecture diagram updates>

## Verification

```bash
# Build docs to verify no broken links
cd docs && make html SPHINXOPTS="-W" 2>&1 | tail -20
```
```

---

## When NOT to Edit Docs

- Do not invent documentation for features you haven't verified
  exist in the code
- Do not remove table entries without confirming the op was
  actually removed (it may have been moved to a different
  implementation path)
- Do not change prose descriptions of architecture without
  explicit user approval — only fix factual references (file
  paths, function names, env vars)
- Factual corrections (wrong file path, renamed function) can
  be applied directly
- Subjective changes (rewording, restructuring) should be
  reported but not applied

---

## Integration with PR Review Swarm

The doc-impact-agent in the PR Review Swarm detects WHAT needs
updating. This Doc Sync Swarm actually MAKES the updates. They
work together:

1. PR merged with new op → PR Review Swarm flagged "ops table
   needs update"
2. Run `/doc-sync-swarm --ops-only` → ops-table-agent updates
   the table
3. Commit the doc fix as a follow-up PR

Or run `/doc-sync-swarm` periodically (weekly) to catch any
accumulated drift across all documentation.
