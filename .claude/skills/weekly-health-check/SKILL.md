---
name: weekly-health-check
description: "Generate a weekly trend report for the torch-spyre ecosystem. Reads the past 7 daily reports for week-over-week trends and aggregates fresh CI / bug / cross-repo data. Use when asked for a weekly status, weekly briefing, week-over-week trends, or Monday review."
---

# Weekly Health Check

Produce a 7-day **trend** report for the torch-spyre ecosystem.

The weekly is NOT a longer daily. It answers a different question:

| Daily answers                              | Weekly answers                             |
|--------------------------------------------|--------------------------------------------|
| "Is the ecosystem healthy right now?"      | "Is it getting better or worse?"           |
| "What's failing today?"                    | "What's persistently broken vs flaky?"     |
| "What's the current bug count?"            | "What's the bug backlog *shape* and age?"  |
| "What did each component build/CI report?" | "What changed in the past 7 days?"         |

## Anti-patterns to avoid

The weekly should NOT include:

- Per-component build/CI status tables ("Spyre Hardware Tests: PASS,
  Linters: PASS, ..."). The daily covers that. Weekly mentions
  *changes* in CI status, not the matrix of statuses.
- A blocking-PR list (the daily has it). Weekly only flags PRs whose
  review status *changed* this week, or PRs that have aged past a
  threshold.
- A "build commits" table. Weekly is downstream of the build — assumes
  the daily has already verified the stack builds. Weekly only cares
  about whether build pass *rate* changed.
- Long prose anywhere. Weekly is read by people scanning for deltas.
  Short sentences, lots of arrows (↑/↓/→), bullets.

## What the weekly DOES emphasize

1. **Headline** (1 sentence): net direction of the week.
2. **Trend deltas**: numbers WITH arrows + percentages, not status grids.
3. **What changed**: the "diff" against last week — new persistent
   failures, resolved failures, PRs landed, bugs filed/closed.
4. **Action items** sorted by *what changed in priority*, not what's
   on fire today.
5. **Backlog shape** (age histogram), not backlog count.
6. **Upstream-driven risk** (cross-repo API touches that may surface as
   future build breaks).

## Data sources (hybrid)

- **Fresh from GitHub:** today's CI status, today's open bug count,
  today's blocking PRs.
- **Historical from `tools/health_report_YYYY-MM-DD.html`:** PR merge
  history, repo-drift commits, prior nightly triages, bug filing/closing
  trend.

If fewer than 5 daily reports exist in the last 7 days, surface this
clearly in the output ("trends based on N days, not 7") rather than
silently producing thin numbers. Do NOT fabricate missing days.

The three repos are the same as the daily skill:

- **deeptools** (`github.ibm.com:ai-chip-toolchain/deeptools.git`, branch: `master`)
- **flex** (`github.ibm.com:ai-chip-toolchain/flex.git`, branch: `main`)
- **torch-spyre** (`github.com/torch-spyre/torch-spyre`, branch: `main`)

---

## Execution Strategy

### Phase 1: Locate daily history

```bash
WORKTREE_ROOT="$(git rev-parse --show-toplevel)"
TOOLS_DIR="$WORKTREE_ROOT/tools"

# Find daily reports from the last 7 days
TODAY=$(date +%Y-%m-%d)
SEVEN_DAYS_AGO=$(date -v-7d +%Y-%m-%d 2>/dev/null || date -d '7 days ago' +%Y-%m-%d)

ls "$TOOLS_DIR"/health_report_*.html 2>/dev/null | \
  awk -F'health_report_|\\.html' -v start="$SEVEN_DAYS_AGO" \
    '$2 >= start { print }' | sort
```

If the count is < 5, prepend the report with a banner: "⚠ Trend
data based on N daily reports, not the full 7-day window."

### Phase 2: Fan out (5 parallel agents)

Launch ALL of these agents simultaneously. They are independent.

| Agent | Task | Source |
|-------|------|--------|
| **trends-agent** | Compute week-over-week deltas for build / CI / velocity / bugs | Daily HTMLs |
| **persistence-agent** | Identify persistent vs flaky test failures across nightlies | Daily HTMLs + GitHub Actions API |
| **backlog-agent** | Bucket open bugs by age | GitHub fresh |
| **crossrepo-agent** | Summarize 7-day deeptools / flex / torch-spyre activity | Pod git logs |
| **today-agent** | Today's CI + bugs + blocking PRs (fresh snapshot) | GitHub fresh |

### Phase 3: Synthesize

Combine into the report structure below.

### Phase 4: Generate PDF

Same Chrome-headless conversion as the daily skill, output to
`tools/weekly_report_YYYY-MM-DD.pdf`.

---

## Agent A: trends-agent

**Goal:** Week-over-week trends with directionality.

### Step 1: Read the daily HTML files

For each daily report from the last 7 days, extract:

- Total build pass count (e.g., "3/3" or "2/3")
- Nightly CI status (PASS/FAIL)
- PRs merged that day (from "Project Velocity" section)
- Open bug count
- New bugs (last 24h)

You can `grep` for the key markers in the HTML:

```bash
# Build status — look for the .summary-card "Build Pass" value
grep -A 1 'Build Pass' health_report_*.html | grep -oE '[0-9]+/[0-9]+'

# Nightly status
grep -A 1 'Nightly CI' health_report_*.html

# Bug count
grep -A 1 'Open Bugs' health_report_*.html | grep -oE '<div class="value">[^<]*'
```

### Step 2: Compare to prior week

Compute the same metrics for the 7 days BEFORE the current 7-day window
(if those daily reports exist). For each metric, report:

- Current week: N
- Prior week: M (or "no data" if fewer than 3 daily reports exist for the prior week)
- Delta: ↑ / ↓ / → with absolute number AND percentage

### Step 3: Build pass-rate computation

If a daily report shows `3/3` build pass, that's a 100% build day. If
`2/3`, that's a 67% build day. Average across the 7 days.

### Report back (structured)

```
## Week-over-week trends

| Metric                | Current week | Prior week | Δ          |
|-----------------------|--------------|------------|------------|
| Build pass rate       | 6/7 days     | 5/7 days   | ↑ +14%     |
| Nightly CI pass rate  | 4/7 days     | 6/7 days   | ↓ −29%     |
| PRs merged            | 47           | 32         | ↑ +47%     |
| New bugs filed        | 5            | 8          | ↓ −38%     |
| Bugs closed           | 18           | 11         | ↑ +64%     |
| Net bug change        | −13          | −3         | ↑ better   |
| Open bugs (today)     | 28           | 32         | ↓ −12%     |
```

Keep under 350 words.

---

## Agent B: persistence-agent

**Goal:** Identify which test failures are **persistent** (recurring
across multiple days) vs **flaky** (one-off). Distinguishing these
matters because flaky tests waste triage time; persistent failures
need ownership.

### Steps

1. From each daily report's "Nightly Failure Triage" section, extract
   the failing test names (or "passing" if the day was clean).

   ```bash
   # The triage block lists tests in <code> tags inside .triage-block
   grep -oP '<code>test_[^<]+</code>' health_report_*.html | sort -u
   ```

2. Count occurrences per test across the 7 days:
   - **5+ days failed**: PERSISTENT (real regression with no owner)
   - **2-4 days failed**: INTERMITTENT (flaky or partially fixed)
   - **1 day failed**: ONE-OFF (likely flaky)

3. Cross-reference with the GitHub Actions API for the *current*
   nightly result, in case daily reports are stale:

   ```bash
   gh run list --repo torch-spyre/torch-spyre \
     --workflow=runtests_nightly.yaml --branch main --limit 7 \
     --json databaseId,conclusion,createdAt
   ```

### Report back

```
## Persistent vs flaky failures

### Persistent (act on these)
- test_X (failed 5/7 days; first seen 2026-05-21; no fix PR)
- test_Y (failed 7/7 days; first seen 2026-05-15; PR #N pending)

### Intermittent (watch)
- test_Z (failed 3/7 days)

### One-off (likely flaky)
- (none) | (3 tests, listed below)
```

Under 300 words. Persistent ones go to "Action Items" in the synthesis.

---

## Agent C: backlog-agent

**Goal:** Open-bug backlog age distribution. Catches issues that quietly
age out without triage.

### Steps

```bash
gh issue list --repo torch-spyre/torch-spyre --state open \
  --label "bug" --limit 200 \
  --json number,title,createdAt,updatedAt,labels
```

Bucket by age:

| Bucket | Definition |
|--------|------------|
| Fresh | createdAt within 24h |
| Active | created within 7d AND updated within 7d |
| Stalled-recent | created within 30d, no update in 14d |
| Aging | created 30-90d ago, no update in 30d |
| Stale | created >90d ago, no update in 60d |

### Report back

```
## Bug backlog (N total open)

| Bucket          | Count | Examples                  |
|-----------------|-------|---------------------------|
| Fresh (<24h)    | 0     | —                         |
| Active          | 4     | #2247, #2246, #2222, #2207|
| Stalled-recent  | 7     | #2195, #2191, ...         |
| Aging           | 12    | #1849, #1880, ...         |
| Stale           | 5     | #1485, #1488, ...         |
```

If "Stale" count is rising week-over-week, flag it. Under 250 words.

---

## Agent D: crossrepo-agent

**Goal:** 7-day activity summary across deeptools / flex / torch-spyre.
Catches upstream-driven risk before it surfaces as a build break.

### Pod commands

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
for repo in deeptools flex torch-spyre; do
  echo \"=== \$repo (last 7 days) ===\"
  cd \$WS/\$repo
  echo 'Total commits:' \$(git log --oneline --since='7 days ago' | wc -l)
  echo 'API-touching commits:'
  git log --oneline --since='7 days ago' -- '*.hpp' '*.h' 'CMakeLists.txt' '*.cmake' \
    | head -10
  echo 'Authors:'
  git log --since='7 days ago' --format='%an' | sort | uniq -c | sort -rn | head -5
done
"
```

### Report back

```
## Cross-repo activity (7 days)

| Repo        | Commits | API-touching | Top contributors        | Risk |
|-------------|---------|--------------|-------------------------|------|
| deeptools   | 82      | 12 (15%)     | A (24), B (15), C (10)  | Med  |
| flex        | 11      | 7 (64%)      | D (5), E (3)            | Med  |
| torch-spyre | 63      | —            | F (12), G (8), H (7)    | —    |

Notable upstream commits worth tracking:
- deeptools cdec8614 (DDB cstdint) — header-only
- flex a233ca5 (FlexAllocator API) — affects torch-spyre allocator
```

Risk scale:
- **High**: API-touching ≥10% AND any commit changes a struct field
  / function signature / public enum.
- **Medium**: API-touching ≥10% but only adds members or comments.
- **Low**: API-touching <10%.

Under 250 words.

---

## Agent E: today-agent

**Goal:** A *one-line* anchor for "where things stand at week-end" —
NOT a status matrix. The weekly assumes the daily already did the
component-by-component snapshot.

### Commands

```bash
# Just the headline numbers
gh run list --repo torch-spyre/torch-spyre --workflow=runtests_nightly.yaml \
  --branch main --limit 1 --json conclusion
gh issue list --repo torch-spyre/torch-spyre --state open --label bug \
  --limit 200 --json number --jq 'length'
# IMPORTANT: --limit 200 (the default page caps at 30 and silently
# truncates — the daily skill historically had this bug; do NOT
# inherit it).
```

### Report back (≤ 50 words)

One sentence stating the week-end status, plus a single bullet line for
*PRs whose status materially changed this week*. Examples:

- "Week ends with all CI green and 54 open bugs (PR #2218 went from
  CONFLICTING to fast-forward-ready after fixes landed via
  ani300/torch-spyre#1)."

Do NOT list every workflow status, every blocking PR, or every recent
PR opened. The daily has those.

---

## Synthesis

Combine into one structured report. Order matters — trends first,
snapshot last (and small).

### Required ordering

1. **Banner** — only if data is thin (<5 daily reports current week, or
   <3 prior week). Skip if data is sufficient.
2. **Headline** — exactly ONE sentence with direction language ("net
   positive," "regression week," "flat with backlog growth"). No two-
   sentence headlines.
3. **Summary cards** — 4 cards, each with a value AND a delta arrow.
   Suggested set: today's nightly status (with ↑/↓ if direction
   changed), PRs merged this week (with ↑/↓ %), open bugs (with
   absolute Δ), bugs >30d old (with absolute Δ if computable).
4. **What needs attention** (action items, sorted by *change in
   priority*). 4-6 items max. Lead each with a delta-language verb
   ("Bug backlog grew by N," "Nightly recovered after N days,"
   "Upstream API risk rose to High"). Avoid "investigate X" phrasing —
   say what changed.
5. **Week-over-week trends** (the deltas table from trends-agent).
6. **Persistent vs flaky failures**. Lead with "no persistent failures
   active" if true; otherwise list them.
7. **Bug backlog shape** (age histogram).
8. **Cross-repo activity** (notable upstream commits + risk).
9. **Today's anchor** — ONE LINE at the bottom from today-agent.

### What MUST NOT be in the synthesis

- A "Today's Snapshot" component-status table. Use the one-line anchor
  instead. If a reader wants the matrix they should run `daily-health-
  check`.
- A "Blocking PRs" table copied from the daily. Only include a PR if
  its status changed this week or it crossed an aging threshold.
- Per-component build/CI tables.
- "Generated by daily-health-check" or similar daily-style footers.
- The PyTorch version in the subtitle — irrelevant for trend reports.

### Required template

```markdown
# torch-spyre Weekly Health Report

**Week of:** 2026-05-22 to 2026-05-28
**Generated:** 2026-05-28
**Data source:** N daily reports + fresh GitHub state

[BANNER if data thin]

## Headline
[ONE sentence]

[SUMMARY CARDS — 4]

## What changed this week
[3-6 action items, each leading with delta language]

## Week-over-week trends
[deltas table from trends-agent]

## Persistent vs flaky failures
[from persistence-agent — be brief; "no persistent failures active" is a
valid one-line answer]

## Bug backlog shape
[age histogram from backlog-agent — emphasize age distribution change,
not absolute counts]

## Cross-repo activity
[from crossrepo-agent — focus on commits with API impact, not headcounts]

---

**Week ends with:** [one line from today-agent]
```

The "Action items" / "What needs attention" section moves to the TOP
(item 4 in the required ordering above), not the bottom. Readers
scanning the PDF should see the action items before the trend tables.

---

## Output

### Step 1: Generate HTML

Write `tools/weekly_report_YYYY-MM-DD.html` with the same corporate
styling as the daily report (`tools/health_report_*.html` is the template
reference). Add **trend arrows** (↑/↓/→) with green/red/gray coloring.

For trend arrows specifically:

```html
<span class="trend-up">↑ +14%</span>     <!-- green -->
<span class="trend-down">↓ −29%</span>   <!-- red -->
<span class="trend-flat">→ ±0</span>     <!-- gray -->
```

### Step 2: Convert to PDF

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="tools/weekly_report_YYYY-MM-DD.pdf" \
  "tools/weekly_report_YYYY-MM-DD.html"
```

### Step 3: Open

```bash
open "tools/weekly_report_YYYY-MM-DD.pdf"
```

---

## When to use this skill

- User asks for "weekly", "week-over-week", "Monday briefing", "weekly
  status", "trends report"
- Friday afternoon / Monday morning context
- After a daily run, when the user wants context on whether today's
  state is improving or degrading

## When NOT to use this skill

- "Daily" or "morning report" → use `daily-health-check` instead
- One-off question about a specific PR / test → just answer it directly
- Building or running the stack → `daily-health-check` covers that

If fewer than 3 daily reports exist in the last 7 days, suggest running
`daily-health-check` first to build up trend data, but proceed anyway
with whatever data is available, marking the report as thin.
