---
name: cross-repo-adapt
description: "Detect upstream flex/deeptools API changes that break torch-spyre, identify impacted call sites, generate adaptation code, build-verify on pod, and create a fix PR. Run after pulling latest upstream or when build fails."
---

# Cross-Repo API Adaptation Swarm

Detects upstream API changes in flex and deeptools that break
torch-spyre, then generates and validates the adaptation. Uses
4 agents to detect changes, assess impact, write fixes, and
verify the build.

Use when:
- The daily health check reports a build failure
- After pulling latest flex/deeptools on the pod
- When a cross-repo PR is blocked on adaptation (e.g., #2190)
- Proactively after seeing HIGH risk in the version-agent drift report

---

## Invocation

```
/cross-repo-adapt
/cross-repo-adapt --component flex
/cross-repo-adapt --since "3 days ago"
```

Without arguments, checks both flex and deeptools. With
`--component`, limits to one upstream repo. With `--since`,
only looks at commits since the given date.

---

## Execution Strategy

### Phase 1: Detect (1 agent, foreground)

The detect-agent identifies what changed upstream. Its output
determines whether adaptation is needed and scopes the remaining
agents.

### Phase 2: Impact + Adapt (2 agents, parallel)

If breaking changes are detected:
- impact-agent identifies which torch-spyre files are affected
- adapt-agent writes the fix (informed by detect-agent's output)

### Phase 3: Verify (1 agent, foreground)

Build on pod to confirm the fix works.

---

## Pod Connection

All pod commands use:

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "<cmd>"
```

Workspace layout:

```
$HOME/main-workspace/
├── deeptools/          # upstream, branch: master
├── flex/               # upstream, branch: main
├── torch-spyre/       # upstream, branch: main
├── build/             # cmake build dirs
├── install/           # installed headers/libs
│   ├── deeptools/
│   └── runtime/       # flex install
└── activate.sh        # venv + PATH setup
```

---

## Agent A: detect-agent

**Goal:** Identify API-breaking changes in upstream repos.

### Steps

1. **Pull latest and check recent commits:**

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace

echo '=== FLEX: API-relevant commits ==='
cd \$WS/flex && git fetch origin && git log --oneline \
  origin/main --since='7 days ago' -- '*.hpp' '*.h' 'CMakeLists.txt' \
  'include/' | head -20

echo '=== DEEPTOOLS: API-relevant commits ==='
cd \$WS/deeptools && git fetch origin && git log --oneline \
  origin/master --since='7 days ago' -- '*.hpp' '*.h' 'CMakeLists.txt' \
  'include/' | head -20
"
```

2. **Extract changed function signatures and types:**

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace

echo '=== FLEX: Changed headers (diff) ==='
cd \$WS/flex && git diff origin/main@{7.days.ago}..origin/main \
  -- 'include/' '*.hpp' '*.h' 2>/dev/null \
  | grep -E '^[+-].*(class |struct |enum |void |int |bool |auto |template|namespace|typedef|using )' \
  | head -40

echo '=== DEEPTOOLS: Changed headers (diff) ==='
cd \$WS/deeptools && git diff origin/master@{7.days.ago}..origin/master \
  -- 'include/' '*.hpp' '*.h' 2>/dev/null \
  | grep -E '^[+-].*(class |struct |enum |void |int |bool |auto |template|namespace|typedef|using )' \
  | head -40
"
```

3. **Identify specific breaking patterns:**

Look for these high-risk changes:
- Function signature changes (added/removed parameters)
- Renamed or removed classes/structs/enums
- Moved headers (new include paths)
- New required parameters in constructors
- Removed or renamed enum values
- Namespace changes
- C++ standard version changes (e.g., C++17 → C++20)

### Report Format

```markdown
## Upstream Changes Detected

### flex (N commits, M API-touching)

| Commit | Change Type | Symbol | Details |
|--------|-------------|--------|---------|
| abc1234 | New param | AllocationDirective() | Added MemoryType 4th arg |
| def5678 | Renamed | FlexAllocator | Added allocate() overload |

### deeptools (N commits, M API-touching)

| Commit | Change Type | Symbol | Details |
|--------|-------------|--------|---------|

### Risk Assessment

- **Breaking:** <list of changes that will break torch-spyre>
- **Non-breaking:** <list of additive changes, new APIs>
- **Unknown:** <changes that need manual inspection>
```

---

## Agent B: impact-agent

**Goal:** Identify which torch-spyre files use the changed APIs.

### torch-spyre C++ Sources

All C++ code lives in `torch_spyre/csrc/`:

| File | flex/deeptools APIs Used |
|------|-------------------------|
| `module.cpp` | `flex::initializeRuntime`, `flex.hpp` |
| `spyre_allocator.cpp` | `AllocationDirective`, `FlexAllocator` |
| `spyre_kernel.cpp` | `AllocationDirective`, binary loading |
| `job_plan.cpp` | `flex::RuntimeOperation*`, `flex::CompositeAddress`, `alloc_address.hpp` |
| `spyre_stream.cpp` | `flex::RuntimeStream` |
| `spyre_mem.cpp` | Memory management APIs |

### Steps

1. **For each breaking change, grep torch-spyre sources:**

```bash
# On pod or locally
grep -rn "<changed_symbol>" torch_spyre/csrc/ --include="*.cpp" --include="*.hpp" --include="*.h"
```

2. **For header moves, check includes:**

```bash
grep -rn "#include.*<old_path>" torch_spyre/csrc/
```

3. **For constructor signature changes, find call sites:**

```bash
grep -rn "<ClassName>(" torch_spyre/csrc/ --include="*.cpp"
```

4. **Check setup.py build configuration:**

```bash
grep -n "flex\|sendnn\|deeptools\|LIBRARIES\|INCLUDE" setup.py
```

### Report Format

```markdown
## Impact Assessment

| Changed Symbol | torch-spyre File | Line(s) | Impact |
|----------------|------------------|---------|--------|
| AllocationDirective() | spyre_allocator.cpp | 125, 142 | Needs 4th param |
| AllocationDirective() | spyre_kernel.cpp | 187 | Needs 4th param |
| flex::MemoryType | (new include needed) | — | Add #include |

### Files Requiring Changes

1. `torch_spyre/csrc/spyre_allocator.cpp` — lines 125, 142
2. `torch_spyre/csrc/spyre_kernel.cpp` — line 187

### Build Config Impact

- setup.py: NO CHANGE / NEEDS UPDATE
- New library deps: <if any>
- New include paths: <if any>
```

---

## Agent C: adapt-agent

**Goal:** Write the adaptation code.

### Guidelines

1. **Match the upstream API exactly** — don't add compatibility
   shims or `#ifdef` guards unless the change hasn't landed in
   the CI environment yet.

2. **Minimal changes** — only modify what's broken. Don't
   refactor surrounding code.

3. **Preserve existing style** — match indentation, naming
   conventions, and comment patterns of the surrounding code.

4. **Common adaptation patterns:**

**New constructor parameter:**
```cpp
// Before
AllocationDirective(size, alignment, segment);
// After
AllocationDirective(size, alignment, segment, flex::MemoryType::Tensor);
```

**Renamed header:**
```cpp
// Before
#include <flex/old_name.hpp>
// After
#include <flex/new_name.hpp>
```

**Renamed symbol:**
```cpp
// Before
flex::OldName obj;
// After
flex::NewName obj;
```

**New required method parameter:**
```cpp
// Before
allocator.allocate(size);
// After
allocator.allocate(size, flex::MemoryType::Tensor);
```

**Removed function (replaced by new API):**
```cpp
// Check the upstream commit for what replaced it
// Use the new API with equivalent semantics
```

5. **If unsure about semantics** — look at the upstream commit
   message and diff to understand the intent. If still unclear,
   report as "needs manual review" rather than guessing.

6. **License headers** — don't modify. Files already have them.

7. **`USE_SPYRE_CCL=0`** — always set this env var when building
   on single-card pods (spyre_comms not installed).

### Steps

1. Read the impact-agent report for affected files and lines
2. Read each affected file to understand the surrounding context
3. Apply the minimal fix to each call site
4. If a new `#include` is needed, add it in alphabetical order
   with existing includes

### Output

Provide the exact edits (file, old code, new code) for each
change needed.

---

## Agent D: verify-agent

**Goal:** Build on pod to confirm the adaptation works.

### Steps

1. **Apply changes on pod:**

Push the branch or apply patches via kubectl:

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
cd \$WS/torch-spyre
git stash
git checkout main
git pull origin main
# Apply the changes (either via cherry-pick or direct edit)
"
```

2. **Build torch-spyre with the fix:**

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
source \$WS/activate.sh
cd \$WS/torch-spyre
export CXX='/usr/lib64/ccache/c++'
export USE_SPYRE_CCL=0
uv sync --all-extras --active --inexact --no-build-isolation \
  --reinstall-package torch-spyre -v 2>&1 | tail -30
echo 'BUILD_EXIT_CODE='\$?
"
```

Use 600000ms timeout.

3. **Run quick import test:**

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
source \$WS/activate.sh
python3 -c '
import torch
import torch_spyre
print(f\"Import: PASS\")
print(f\"Spyre available: {torch.spyre.is_available()}\")
' 2>&1 | grep -v 'hf_adapters\|Remainder of file'
"
```

4. **If build fails, diagnose:**

Apply the Error Diagnosis Table:

| Error pattern | Likely cause |
|---|---|
| `no matching function for call to` | Signature still wrong — check param count/types |
| `undefined reference to` | Missing library or symbol not exported |
| `no member named 'X' in` | Wrong struct/class or missed rename |
| `fatal error: 'X.h' file not found` | Include path wrong |
| `use of undeclared identifier` | Missing include or wrong namespace |

### Report Format

```markdown
## Build Verification

- **Build status:** PASS / FAIL
- **Import test:** PASS / FAIL / SKIPPED
- **Errors remaining:** <if any>

### If FAIL — Diagnosis

| Error | File | Line | Probable Cause |
|-------|------|------|----------------|
| ... | ... | ... | ... |
```

---

## Final Output

```markdown
## Cross-Repo Adaptation Summary

**Date:** <today>
**Upstream changes:** flex (<N> API commits) / deeptools (<N>)
**Impact:** <N> torch-spyre files affected

### Changes Made

| File | Line(s) | Change |
|------|---------|--------|
| `torch_spyre/csrc/X.cpp` | 125 | Added MemoryType param |

### Build Verification

- Pod build: PASS/FAIL
- Import test: PASS/FAIL

### Next Steps

- [ ] Commit with `git commit -s`
- [ ] Push branch and create PR
- [ ] Reference upstream flex/deeptools commits in PR description
```

If no breaking changes detected:

```markdown
## Cross-Repo Adaptation Summary

**Date:** <today>
**Status:** No breaking changes detected.

Upstream repos have <N> commits in the last 7 days, but none
break torch-spyre's current API usage. Build verified on pod.
```

---

## When to STOP

Do not attempt adaptation if:

- The upstream change is still in-flight (PR not merged) — just
  report it as "upcoming breaking change"
- The fix requires understanding new business logic (not just
  API plumbing) — flag for manual review
- Multiple interdependent upstream PRs need to land together —
  report the dependency chain
- The change affects torch-spyre's Python code (not C++) — that's
  a different workflow (likely Inductor API changes from a PyTorch
  version bump)

---

## Integration with Other Skills

- **daily-health-check** → build-agent fails → run this skill
- **version-agent** reports HIGH drift → run this skill proactively
- After this skill fixes the build → run daily-health-check smoke
  test to confirm end-to-end
