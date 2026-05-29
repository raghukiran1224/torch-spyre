---
name: daily-health-check
description: "Build the full Spyre stack from source on a pod, check CI status, triage nightly failures, scan blocking PRs, check version alignment, and report project velocity. Use when asked for a health check, daily build, ecosystem status, or morning report."
---

# Daily Health Check

Produce a comprehensive health report for the torch-spyre ecosystem.
Fan out up to 6 agents in parallel, then synthesize into a single
structured report.

> **Looking for trends instead of a snapshot?** Use the
> `weekly-health-check` skill — it aggregates the daily HTML outputs
> from `tools/health_report_*.html` and surfaces week-over-week
> directionality, persistent vs flaky failures, bug backlog age, and
> cross-repo activity.

The ecosystem has three repos built sequentially:

- **deeptools** (`github.ibm.com:ai-chip-toolchain/deeptools.git`, branch: `master`)
- **flex** (`github.ibm.com:ai-chip-toolchain/flex.git`, branch: `main`)
- **torch-spyre** (`github.com/torch-spyre/torch-spyre`, branch: `main`)

---

## Execution Strategy

### Phase 1: Fan Out (6 parallel agents)

Launch ALL of these agents simultaneously:

| Agent | Task | Deps |
|-------|------|------|
| **build-agent** | Pull latest, build full stack on pod | None |
| **ci-agent** | Check CI workflow status on main | None |
| **nightly-triage-agent** | Download and triage nightly test failures | None |
| **pr-agent** | Scan blocking PRs and project velocity | None |
| **version-agent** | Check PyTorch version alignment and repo drift | None |
| **issues-agent** | Check open bugs and recent issue activity | None |

### Phase 2: Synthesis

Once all agents return, produce the structured report. The build-agent
result may inform interpretation of other results (e.g., if build
fails, the nightly failure may share the same root cause).

### Phase 3: Smoke Test (conditional)

If the build passed, run the smoke test on the pod.

---

## Agent A: build-agent

**Goal:** Pull latest on all repos and build the full stack.

### Pod Connection

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "<cmd>"
```

### Step 1: Pull Latest Sources

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
cd \$WS/deeptools && git fetch origin && git checkout master && git reset --hard origin/master
cd \$WS/flex && git fetch origin && git checkout main && git reset --hard origin/main
cd \$WS/torch-spyre && git fetch origin && git checkout main && git reset --hard origin/main
echo '=== Commits ==='
echo 'deeptools:' && cd \$WS/deeptools && git log --oneline -1
echo 'flex:' && cd \$WS/flex && git log --oneline -1
echo 'torch-spyre:' && cd \$WS/torch-spyre && git log --oneline -1
"
```

### Step 2: Build deeptools

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
mkdir -p \$WS/build/deeptools && cd \$WS/build/deeptools
cmake \$WS/deeptools \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_INSTALL_PREFIX=\$WS/install/deeptools \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDT_USE_DCC_DDC=on \
  -DLLVM_PROJ_SRC=/home/rganti/dt-inductor/llvm-project \
  -DLLVM_PROJ_BUILD=/home/rganti/dt-inductor/build/llvm \
  -DMANAGE_LLVM=0 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 2>&1 | tail -5
make -j 16 install 2>&1 | tail -10
echo 'DEEPTOOLS_EXIT_CODE='\$?
"
```

Use a 600000ms timeout.

### Step 3: Build flex

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
rm -rf \$WS/install/runtime
cd \$WS/flex && git submodule update --init --recursive 2>&1 | tail -3
mkdir -p \$WS/build/flex && cd \$WS/build/flex
cmake \$WS/flex \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_INSTALL_PREFIX=\$WS/install/runtime \
  -DCMAKE_BUILD_TYPE=Debug \
  -DDEEPTOOLS_PATH=\$WS/install/deeptools \
  -DCMAKE_PREFIX_PATH=\$WS/install/deeptools \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 2>&1 | tail -5
make -j 16 install 2>&1 | tail -10
echo 'FLEX_EXIT_CODE='\$?
"
```

Use a 600000ms timeout.

### Step 4: Build torch-spyre

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
source \$WS/activate.sh
cd \$WS/torch-spyre
export CXX='/usr/lib64/ccache/c++'
export USE_SPYRE_CCL=0
uv sync --all-extras --active --inexact --no-build-isolation \
  --reinstall-package torch-spyre -v 2>&1 | tail -30
echo 'TORCH_SPYRE_EXIT_CODE='\$?
"
```

Use a 600000ms timeout. If it fails, also run with grep for errors.

### Step 5: Diagnose Failures

If any step failed, apply the Error Diagnosis Table (at the bottom of
this document) to classify the root cause.

### Report back with

- Commit SHAs (short hash + subject) for each repo
- Build status per component (PASS/FAIL)
- If failed: component, error summary, cross-repo mismatch diagnosis

---

## Agent B: ci-agent

**Goal:** Check CI workflow status on main branch.

### Workflows to Check

| Workflow file | Display name |
|---|---|
| `torch_spyre_tests.yaml` | Spyre Hardware Tests |
| `linters.yaml` | Linters |
| `upstream_tests.yaml` | Upstream PyTorch Tests |
| `runtests_nightly.yaml` | Nightly Tests |

### Commands

For each workflow:

```bash
gh run list --repo torch-spyre/torch-spyre \
  --workflow=<filename> --branch main --limit 3 \
  --json status,conclusion,createdAt,headSha,url
```

For any failed run, get failing job details:

```bash
gh run view <run-id> --repo torch-spyre/torch-spyre \
  --json jobs --jq '.jobs[] | select(.conclusion=="failure") | .name'
```

### Report back with

- Last 3 run conclusions per workflow
- If most recent failed: which jobs failed
- Trend: "newly broken" / "persistently failing" / "healthy"

---

## Agent C: nightly-triage-agent

**Goal:** When nightly tests are failing, identify WHAT is failing and
WHY. This goes deeper than the ci-agent.

### Steps

1. Get the latest failed nightly run ID:

```bash
gh run list --repo torch-spyre/torch-spyre \
  --workflow=runtests_nightly.yaml --branch main --limit 1 \
  --json databaseId,conclusion \
  --jq '.[] | select(.conclusion=="failure") | .databaseId'
```

2. Get all failed jobs in that run:

```bash
gh run view <run-id> --repo torch-spyre/torch-spyre \
  --json jobs \
  --jq '.jobs[] | select(.conclusion=="failure") | {name, conclusion}'
```

3. For each failed job, try to get the log annotations (errors):

```bash
gh api repos/torch-spyre/torch-spyre/actions/runs/<run-id>/jobs \
  --jq '.jobs[] | select(.conclusion=="failure") |
    {name, steps: [.steps[] | select(.conclusion=="failure") | .name]}'
```

4. If available, download the failed job log and extract the last
   failure block (look for FAILED, ERROR, AssertionError):

```bash
gh run view <run-id> --repo torch-spyre/torch-spyre \
  --log-failed 2>&1 | tail -100
```

### Report back with

- Which test suite(s) failed
- Specific test names if identifiable
- Error type (assertion, timeout, import error, segfault, etc.)
- Whether the failure looks like: regression, flaky test, infra issue
- How many consecutive days it has been failing (check last 5 runs)

---

## Agent D: pr-agent

**Goal:** Scan for blocking PRs AND report project velocity.

### Part 1: Blocking PRs

1. Open PRs mentioning cross-repo terms:

```bash
gh pr list --repo torch-spyre/torch-spyre --state open --limit 30 \
  --search "flex OR deeptools OR AllocationDirective OR API OR mismatch OR compat" \
  --json number,title,url,author,createdAt,labels,reviewDecision
```

2. Open PRs with dependency labels:

```bash
gh pr list --repo torch-spyre/torch-spyre --state open \
  --label "dependencies" --json number,title,url
```

3. Recently merged cross-repo fixes (context):

```bash
gh pr list --repo torch-spyre/torch-spyre --state merged \
  --search "flex OR deeptools OR API" --limit 5 \
  --json number,title,mergedAt,url
```

### Part 2: Project Velocity (last 7 days)

4. PRs merged in the last 7 days:

```bash
gh pr list --repo torch-spyre/torch-spyre --state merged --limit 20 \
  --json number,title,mergedAt \
  --jq '[.[] | select(.mergedAt > (now - 7*24*3600 | strftime("%Y-%m-%dT%H:%M:%SZ")))] | length'
```

(If jq time filtering doesn't work, just list last 20 merged and
count those within the last 7 days manually.)

5. PRs currently in review (open, not draft):

```bash
gh pr list --repo torch-spyre/torch-spyre --state open \
  --json number,title,createdAt,isDraft,reviewDecision \
  --jq '[.[] | select(.isDraft==false)] | length'
```

6. Stalled PRs (open >7 days, no recent activity):

```bash
gh pr list --repo torch-spyre/torch-spyre --state open --limit 50 \
  --json number,title,createdAt,updatedAt,author \
  --jq '[.[] | select(.updatedAt < (now - 7*24*3600 | strftime("%Y-%m-%dT%H:%M:%SZ")))]'
```

(If jq filtering fails, list all open PRs with dates and identify
stalled ones manually — those with updatedAt > 7 days ago.)

### Report back with

- Blocking PRs (number, title, why blocking, review status)
- Velocity: PRs merged last 7 days (count)
- PRs in review (count)
- Stalled PRs >7 days (list with age)

---

## Agent E: version-agent

**Goal:** Check PyTorch version alignment and repo drift.

### Part 1: PyTorch Version Alignment

1. Check what torch-spyre pyproject.toml expects:

```bash
gh api repos/torch-spyre/torch-spyre/contents/pyproject.toml \
  --jq '.content' | base64 -d | grep 'torch'
```

Or read locally if available:
```bash
grep 'torch' pyproject.toml | head -5
```

2. Check what's installed in the workspace venv on pod:

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
source \$HOME/main-workspace/activate.sh
python3 -c 'import torch; print(f\"Installed: {torch.__version__}\")'
"
```

3. Flag if there's a mismatch (e.g., pyproject wants 2.11 but
   workspace has 2.10).

### Part 2: Repo Drift

Count commits since the last known-good build (use the commit SHAs
from the previous health check if available, otherwise count commits
in the last 7 days):

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
echo 'deeptools (last 7 days):' && cd \$WS/deeptools && git log --oneline --since='7 days ago' | wc -l
echo 'flex (last 7 days):' && cd \$WS/flex && git log --oneline --since='7 days ago' | wc -l
echo 'torch-spyre (last 7 days):' && cd \$WS/torch-spyre && git log --oneline --since='7 days ago' | wc -l
"
```

Also show the most impactful commits (those touching headers, APIs,
or CMakeLists):

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
echo '=== deeptools API-relevant commits (7d) ==='
cd \$WS/deeptools && git log --oneline --since='7 days ago' -- '*.hpp' '*.h' 'CMakeLists.txt' | head -10
echo '=== flex API-relevant commits (7d) ==='
cd \$WS/flex && git log --oneline --since='7 days ago' -- '*.hpp' '*.h' 'CMakeLists.txt' | head -10
"
```

### Report back with

- PyTorch version: expected vs installed (MATCH / MISMATCH)
- Repo drift: commits in last 7 days per repo
- High-risk commits (header/API/CMake changes) listed
- Risk assessment: "low" (few changes), "medium" (many changes,
  no API files), "high" (API-touching commits in upstream repos)

---

## Agent F: issues-agent

**Goal:** Check open bug count and recent issue activity.

### Commands

1. Total open bugs:

```bash
gh issue list --repo torch-spyre/torch-spyre --state open \
  --label "bug" --json number \
  --jq 'length'
```

2. Bugs filed in the last 24 hours:

```bash
gh issue list --repo torch-spyre/torch-spyre --state open \
  --label "bug" --limit 10 \
  --json number,title,createdAt,author \
  --jq '[.[] | select(.createdAt > "YESTERDAY_ISO")]'
```

Replace `YESTERDAY_ISO` with yesterday's date in ISO format
(e.g., `2026-05-18T00:00:00Z`). If jq filtering is unreliable,
list the 10 most recent bugs and filter by date manually.

3. Bugs filed in the last 7 days (for trend):

```bash
gh issue list --repo torch-spyre/torch-spyre --state open \
  --label "bug" --limit 30 \
  --json number,title,createdAt \
  --jq '[.[] | select(.createdAt > "WEEK_AGO_ISO")]'
```

4. Recently closed bugs (to show fix velocity):

```bash
gh issue list --repo torch-spyre/torch-spyre --state closed \
  --label "bug" --limit 10 \
  --json number,title,closedAt
```

### Report back with

- Total open bugs (count)
- New bugs (last 24h): count + titles
- Bug trend (last 7 days): filing rate vs close rate
- Any critical/high-priority bugs (if labeled)

---

## Phase 3: Smoke Test

Run only if ALL three components built successfully. The workspace
uses a single venv at `$HOME/main-workspace/venv` — the activate.sh
sources it and sets PATH for deeptools/runtime binaries only.

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
source \$WS/activate.sh
cd \$WS/torch-spyre
python3 -c '
import torch
import torch_spyre
print(f\"Spyre available: {torch.spyre.is_available()}\")
print(f\"Device count: {torch.spyre.device_count()}\")
print(f\"PyTorch: {torch.__version__}\")
print(f\"torch_spyre: {torch_spyre.__file__}\")
x = torch.randn(4, 4, dtype=torch.float16, device=\"spyre\")
y = torch.randn(4, 4, dtype=torch.float16, device=\"spyre\")
z = x + y
print(f\"Eager add: PASS, shape={z.shape}\")
' 2>&1 | grep -v 'hf_adapters\|Remainder of file'
"
```

If eager passes, also test compiled execution. torch-spyre registers
as an Inductor device backend, so use `torch.compile()` without
specifying a backend (inductor is the default and it dispatches to
Spyre when tensors are on the spyre device):

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
source \$WS/activate.sh
cd \$WS/torch-spyre
python3 -c '
import torch
import torch_spyre

@torch.compile()
def f(a, b):
    return a + b

a = torch.randn(128, 128, dtype=torch.float16, device=\"spyre\")
b = torch.randn(128, 128, dtype=torch.float16, device=\"spyre\")
c = f(a, b)
print(f\"Compiled add: PASS, shape={c.shape}\")
' 2>&1 | grep -v 'hf_adapters\|Remainder of file'
"
```

If compiled passes, run the full model E2E smoke test using
hf\_adapters with Qwen3 0.6B. This exercises the full compilation
pipeline (model load, weight stickification, torch.compile,
multi-token generation).

**IMPORTANT:** The `uv sync` build step for torch-spyre may downgrade
`transformers` to a version pinned in its lockfile (e.g., 4.57.x).
The `hf_adapters` package requires `transformers>=5.0`. Before running
the E2E smoke test, re-install the correct transformers version:

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
source \$WS/activate.sh
pip install --upgrade 'transformers>=5.0' 2>&1 | tail -5
"
```

Then run the smoke test:

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
source \$WS/activate.sh
cd \$HOME/hf_adapters
python3 tests/test_e2e_smoke_spyre.py qwen3 2>&1 \
  | grep -E 'PASS|FAIL|ERROR|Load time|Generate time|Output:|E2E Smoke'
"
```

Use a 300000ms timeout. The script loads `Qwen/Qwen3-0.6B` via
`AutoSpyreModelForCausalLM.from_pretrained`, generates 5 tokens,
and validates they are non-empty and diverse. Model weights should
be cached on the pod after the first run.

---

## Output Format

Present the final report in this structure:

```markdown
# torch-spyre Ecosystem Health Report

**Date:** <today>
**Pod:** rganti-spyre-main

## Build Status

| Component | Commit | Status | Notes |
|-----------|--------|--------|-------|
| deeptools | `<sha>` | PASS/FAIL | |
| flex | `<sha>` | PASS/FAIL | |
| torch-spyre | `<sha>` | PASS/FAIL | |

## CI Status

| Workflow | Last Run | Status | Details |
|----------|----------|--------|---------|
| Spyre Hardware Tests | <date> | PASS/FAIL | |
| Linters | <date> | PASS/FAIL | |
| Upstream PyTorch | <date> | PASS/FAIL | |
| Nightly Tests | <date> | PASS/FAIL | |

## Nightly Failure Triage

- Failing job(s): <name>
- Failing test(s): <test names if identifiable>
- Error type: <assertion / timeout / segfault / import / etc.>
- Duration: <N consecutive days>
- Assessment: <regression / flaky / infra issue>

(or "Nightly tests passing" if healthy)

## Version Alignment

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| PyTorch | <from pyproject> | <installed> | MATCH/MISMATCH |

## Repo Drift (last 7 days)

| Repo | Commits | API-touching | Risk |
|------|---------|--------------|------|
| deeptools | N | N | low/med/high |
| flex | N | N | low/med/high |
| torch-spyre | N | — | — |

## Blocking PRs

| PR | Title | Why Blocking | Review Status |
|----|-------|--------------|---------------|
| #N | title | reason | Approved/Pending |

(or "None identified")

## Cross-Repo API Mismatches

(details if build failed, or "None detected")

## Project Velocity

- PRs merged (7 days): N
- PRs in review: N
- Stalled PRs (>7 days): N (list if any)

## Open Bugs

- Total open: N
- New (24h): N — <titles if any>
- Trend (7d): +N filed, -N closed

## Smoke Test

- Import: PASS/FAIL
- Eager execution: PASS/FAIL/SKIPPED
- Compiled execution: PASS/FAIL/SKIPPED
- Model E2E (Qwen3 0.6B): PASS/FAIL/SKIPPED — <load_time>s load, <gen_time>s gen

## Action Items

1. <Highest priority — blocking build or persistent CI failure>
2. <Medium priority — version mismatch, stalled PRs>
3. <Low priority — new bugs to triage, velocity concerns>
```

---

## Error Diagnosis Table

When a build fails, use these patterns to classify the root cause:

| Error pattern | Likely cause | Action |
|---|---|---|
| `no matching function for call to` | Function signature changed upstream | Compare flex/deeptools headers with torch-spyre call sites |
| `undefined reference to` | Symbol removed or renamed upstream | Check recent flex/deeptools commits |
| `no member named 'X' in` | Struct/class field removed | Check upstream header changes |
| `fatal error: 'X.h' file not found` | Header moved or renamed | Check upstream file renames |
| `use of undeclared identifier` | Enum/constant renamed | Search upstream for the rename |
| `ImportError` / `ModuleNotFoundError` | Python dep missing or renamed | Check if new dep added upstream |
| `RuntimeError: Error compiling objects` | C++ ext build failed | Look at the ninja error above this line |

When a cross-repo mismatch is detected:

1. Identify which upstream repo changed (check `git log --oneline -5`)
2. Search torch-spyre open PRs for a fix (Agent D handles this)
3. If no fix PR exists, flag as high-priority action item

---

## Phase 4: Generate PDF Report

After synthesizing the report, generate a corporate-styled PDF for
executive consumption.

### Step 1: Generate HTML

Write an HTML file to `tools/health_report_YYYY-MM-DD.html` with:

- Embedded CSS (no external dependencies)
- Corporate styling: navy (#1F3A5F) section headers, light gray
  (#F5F7FA) alternating table rows, green/red/amber status badges
- Summary cards at the top (build pass count, CI status, PRs merged,
  open bugs)
- `@media print` rules for clean page breaks
- All report data from the synthesis phase populated into the template

Use the format from `tools/health_report_2026-05-20.html` as the
reference template for structure and styling.

### Step 2: Convert to PDF

Use Chrome headless to convert:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="tools/health_report_YYYY-MM-DD.pdf" \
  "tools/health_report_YYYY-MM-DD.html"
```

On Linux (pod or CI):

```bash
google-chrome --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="tools/health_report_YYYY-MM-DD.pdf" \
  "tools/health_report_YYYY-MM-DD.html"
```

### Step 3: Open the PDF

```bash
open "tools/health_report_YYYY-MM-DD.pdf"
```

Report the PDF path to the user.
