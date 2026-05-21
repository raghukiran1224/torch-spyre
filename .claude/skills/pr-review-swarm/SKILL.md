---
name: pr-review-swarm
description: "Multi-agent PR review that parallelizes compliance, code quality, test coverage, and documentation checks. Produces a unified review comment with severity-ranked findings."
---

# PR Review Swarm

Performs a comprehensive pull request review using 4 parallel agents,
each specializing in a different review dimension. Synthesizes results
into a single structured review comment.

Use when reviewing incoming PRs to torch-spyre. Accepts a PR number
or URL as argument.

---

## Invocation

```
/pr-review-swarm 2190
/pr-review-swarm https://github.com/torch-spyre/torch-spyre/pull/2190
```

---

## Execution Strategy

### Phase 1: Fetch PR Context (sequential, fast)

Before launching agents, gather PR metadata and diff:

```bash
gh pr view <number> --repo torch-spyre/torch-spyre \
  --json title,body,author,labels,files,additions,deletions,commits,reviewDecision

gh pr diff <number> --repo torch-spyre/torch-spyre
```

Determine which files are touched and categorize them:

| Category | Pattern |
|----------|---------|
| Compiler | `torch_spyre/_inductor/` |
| Runtime/C++ | `torch_spyre/_C/`, `*.cpp`, `*.hpp` |
| Eager ops | `torch_spyre/ops/` |
| Tests | `tests/` |
| Docs | `docs/` |
| Config/CI | `.github/`, `pyproject.toml`, `*.yaml` |

### Phase 2: Fan Out (4 parallel agents)

Launch ALL simultaneously:

| Agent | Focus | Relevant when |
|-------|-------|---------------|
| **compliance-agent** | License, DCO, imports, formatting | Always |
| **code-quality-agent** | Architecture, tensor layout, error handling | Compiler/Runtime/Ops changes |
| **test-coverage-agent** | Test completeness, shape variety, parameterization | Any code change |
| **doc-impact-agent** | Ops table, API docs, user guide impact | Op additions, API changes |

### Phase 3: Synthesis

Merge all agent findings into a single review, deduplicate, and
rank by severity (BLOCKER > WARNING > SUGGESTION > NOTE).

---

## Agent A: compliance-agent

**Goal:** Verify mechanical correctness that gates merge-readiness.

### Checks

1. **License Headers**

Every new `.py` file must have the 14-line Apache 2.0 header:

```python
# Copyright <year> The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 ...
```

Every new `.cpp`/`.hpp` file must have the `/* ... */` equivalent.

```bash
# Find new files in the diff
gh pr diff <number> --repo torch-spyre/torch-spyre \
  | grep "^diff --git" | grep -E '\.(py|cpp|hpp)$'
```

For each new file (shows `new file mode` in diff), verify header.

2. **Signed Commits (DCO)**

```bash
gh pr view <number> --repo torch-spyre/torch-spyre \
  --json commits --jq '.commits[].messageBody' \
  | grep -c "Signed-off-by:"
```

Every commit must have a `Signed-off-by:` line.

3. **Import Enforcement**

Check for `import re` (must use `import regex`):

```bash
gh pr diff <number> --repo torch-spyre/torch-spyre \
  | grep "^+" | grep -E "^\\+import re$|^\\+from re import"
```

4. **Line Length (88 chars)**

Check for obvious violations in Python files (ruff enforces this,
but catch it early):

```bash
gh pr diff <number> --repo torch-spyre/torch-spyre \
  | grep "^+" | awk 'length > 92'
```

5. **CI Status**

```bash
gh pr checks <number> --repo torch-spyre/torch-spyre
```

### Report Format

```markdown
### Compliance

| Check | Status | Details |
|-------|--------|---------|
| License headers | PASS/FAIL | <files missing headers> |
| DCO sign-off | PASS/FAIL | <unsigned commits> |
| Import regex | PASS/FAIL | <files using import re> |
| CI checks | PASS/FAIL/PENDING | <failing jobs> |
```

---

## Agent B: code-quality-agent

**Goal:** Review architectural correctness and Spyre-specific
patterns. This is the domain-expert agent.

### Checks

1. **Tensor Layout Correctness**

If diff touches layout-related code, verify:
- `SpyreTensorLayout` fields are used correctly
- Stick dimension alignment (multiples of 64 for fp16)
- `FixedTiledLayout` device_layout parameters
- No assumptions of contiguous memory where tiled applies

```bash
gh pr diff <number> --repo torch-spyre/torch-spyre \
  | grep -E "SpyreTensorLayout|FixedTiledLayout|stick|dim_map"
```

2. **Dual Path Impact**

Determine if changes affect eager, compiled, or both paths:

| File pattern | Path |
|---|---|
| `torch_spyre/ops/` | Eager |
| `torch_spyre/_inductor/` | Compiled |
| `torch_spyre/_C/` | Both |

Flag if a code change only covers one path but the op is listed
as supporting both in `supported_operations.md`.

3. **Error Handling**

New error raises should use `Unsupported()` from
`torch_spyre/_inductor/errors.py`, NOT generic `RuntimeError` or
`NotImplementedError`:

```bash
gh pr diff <number> --repo torch-spyre/torch-spyre \
  | grep "^+" | grep -E "raise (RuntimeError|NotImplementedError)" \
  | grep -v "test"
```

4. **SpyreOpFuncs Ordering**

If `spyre_kernel.py` is modified, verify methods remain
alphabetically sorted:

```bash
gh pr diff <number> --repo torch-spyre/torch-spyre \
  | grep -A2 "@staticmethod" | grep "def "
```

5. **FP32 Ops Consistency**

If a new op is added to `SpyreOpFuncs`, check whether it should
also be in `SPYRE_FP32_OPS` (ops that accumulate, use division,
or need precision).

6. **Constants and Magic Numbers**

Flag hardcoded stick sizes (64, 128) that should reference
constants. Flag hardcoded device strings that should use
`DEVICE_NAME`.

7. **Cross-File Consistency**

If an op is added/modified in one file, verify it's consistently
updated across all relevant files:
- `spyre_kernel.py` ↔ `constants.py` ↔ `eager.py`
- `decompositions.py` ↔ `customops.py` ↔ `lowering.py`

### Report Format

```markdown
### Code Quality

**Severity: BLOCKER / WARNING / SUGGESTION**

| Issue | File | Line | Severity | Details |
|-------|------|------|----------|---------|
| ... | ... | ... | ... | ... |
```

---

## Agent C: test-coverage-agent

**Goal:** Assess whether the PR has adequate test coverage for
the changes made.

### Checks

1. **New Op Has Tests**

If `spyre_kernel.py`, `decompositions.py`, `customops.py`, or
`lowering.py` are modified to add a new op, verify that
`test_inductor_ops.py` is also modified.

```bash
# Check which implementation files changed
gh pr diff <number> --repo torch-spyre/torch-spyre \
  --name-only | grep -E "spyre_kernel|decompositions|customops|lowering"

# Check if test file changed
gh pr diff <number> --repo torch-spyre/torch-spyre \
  --name-only | grep "test_inductor_ops"
```

2. **Shape Variety**

If new test cases are added, verify they include:
- At least one stick-aligned shape (dim multiple of 64)
- At least one non-aligned shape (to test padding paths)
- 2D minimum, ideally 3D or 4D coverage

```bash
gh pr diff <number> --repo torch-spyre/torch-spyre \
  | grep "^+" | grep -E "randn|zeros|ones|cached_randn" \
  | grep -oE "\([0-9, ]+\)"
```

3. **ParameterizedTestMeta Usage**

For new op tests, prefer the cross-product framework over
hand-written test methods. Check if the op was added to the
appropriate dict:

- `POINTWISE_UNARY_OPS_DICT`
- `POINTWISE_BINARY_OPS_DICT`
- `CORE_REDUCTION_OPS_DICT`

4. **Dtype Coverage**

Default should be `torch.float16`. If the op is in
`SPYRE_FP32_OPS`, verify an FP32 test path exists.

5. **Missing Tests for Changed Code**

If implementation files changed but no test files changed,
flag as WARNING (not all changes need new tests, but it
should be conscious).

6. **Test Config Coverage**

Every new `test_*.py` file must have a corresponding config YAML
in `tests/configs/` and be registered in the CI workflow matrix.

### Report Format

```markdown
### Test Coverage

| Metric | Status | Details |
|--------|--------|---------|
| New ops tested | YES/NO/N/A | <ops without tests> |
| Shape variety | GOOD/PARTIAL/MISSING | <what's missing> |
| Parameterized | YES/NO | <hand-written vs framework> |
| Dtype coverage | GOOD/PARTIAL | <missing dtypes> |
| Implementation without tests | YES/NO | <untested files> |
```

---

## Agent D: doc-impact-agent

**Goal:** Determine if the PR requires documentation updates and
whether those updates are included.

### Checks

1. **Supported Operations Table**

If a new op is added (detected via `spyre_kernel.py`,
`decompositions.py`, `customops.py`, `lowering.py`, or `eager.py`
changes), check if `docs/source/user_guide/supported_operations.md`
is also modified.

Cross-reference:
- Op in `SpyreOpFuncs` → should appear in "Compiled" column
- Op in `eager.py` registration → should appear in "Eager" column
- Op in `fallbacks.py` → should show "CPU fallback" in Execution

2. **API Documentation**

If `torch_spyre/__init__.py` exports change, verify
`docs/source/api/torch_spyre.rst` is updated.

If `torch_spyre/streams.py` public methods change, verify
stream docs are updated.

3. **Compiler Documentation**

If new passes are added to `torch_spyre/_inductor/`, check if
`docs/source/compiler/` docs reference them:
- New FX pass → `inductor_frontend.md`
- New codegen pattern → `backend.md`
- New lowering → `adding_operations.md`

4. **User Guide Impact**

If behavior visible to users changes (new env var, new device
capability, changed error message), flag that user guide may
need updating.

5. **Adding Operations Doc**

If a new op pattern is introduced (not just a new op using
existing patterns), flag that `adding_operations.md` may need
a new example.

### Report Format

```markdown
### Documentation Impact

| Area | Needs Update | Included in PR | Action |
|------|:------------:|:--------------:|--------|
| Ops table | YES/NO | YES/NO | <what to add> |
| API docs | YES/NO | YES/NO | <what changed> |
| Compiler docs | YES/NO | YES/NO | <which doc> |
| User guide | YES/NO | YES/NO | <what behavior> |
```

---

## Synthesis: Final Review Output

Combine all agent reports into a single review comment:

```markdown
## PR Review: #<number> — <title>

**Author:** @<author> | **Files:** <N> | **+<add>/-<del>**

### Summary

<1-2 sentence assessment: merge-ready, needs minor fixes, or
has blockers>

### Blockers (must fix before merge)

- [ ] <blocker 1 — file:line — explanation>
- [ ] <blocker 2>

### Warnings (should fix, not blocking)

- [ ] <warning 1>
- [ ] <warning 2>

### Suggestions (optional improvements)

- <suggestion 1>
- <suggestion 2>

### Notes (informational)

- <note 1>

---

<details>
<summary>Detailed Findings</summary>

#### Compliance
<compliance-agent output>

#### Code Quality
<code-quality-agent output>

#### Test Coverage
<test-coverage-agent output>

#### Documentation Impact
<doc-impact-agent output>

</details>
```

### Severity Definitions

| Level | Meaning | Action |
|-------|---------|--------|
| BLOCKER | Prevents merge; violates project policy | Must fix |
| WARNING | Likely bug or architectural issue | Should fix |
| SUGGESTION | Improvement opportunity | Author decides |
| NOTE | Informational, no action needed | FYI |

### What Constitutes a BLOCKER

- Missing license header on new file
- Unsigned commit (no DCO)
- `import re` instead of `import regex`
- New op with zero test coverage
- CI failing due to this PR's changes
- Security vulnerability (injection, unchecked input)

### What Does NOT Block

- Missing doc updates (WARNING, not BLOCKER)
- Suboptimal but correct test shapes
- Style preferences beyond ruff enforcement
- Missing parameterization (SUGGESTION)
