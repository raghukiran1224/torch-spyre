---
name: spyre-ops
description: "Meta-orchestrator for the torch-spyre development workflow. Assesses ecosystem state, identifies what needs attention, proposes an action plan, and executes approved steps using sub-skills. Supervised mode: proposes before acting."
---

# Spyre Ops — Meta Orchestrator

Top-level entry point that assesses the current state of the
torch-spyre ecosystem and proposes a prioritized action plan.
Operates in **supervised mode**: proposes actions and waits for
user approval before executing each phase.

---

## Invocation

```
/spyre-ops                → Full assessment + propose actions
/spyre-ops review         → Focus on PR review queue
/spyre-ops fix            → Focus on broken things (build, nightly, bugs)
/spyre-ops velocity       → Focus on shipping (merge approved, implement ops)
/spyre-ops <model-name>   → Model onboarding pipeline
```

---

## Core Loop

```
┌─────────────────────────────────────────────────┐
│  1. ASSESS — What's the current state?          │
│     (health check data, or quick refresh)       │
├─────────────────────────────────────────────────┤
│  2. PRIORITIZE — What needs attention most?     │
│     (blockers > broken > stalled > new work)    │
├─────────────────────────────────────────────────┤
│  3. PROPOSE — Present action plan to user       │
│     (numbered list with expected outcomes)      │
├─────────────────────────────────────────────────┤
│  4. EXECUTE — Run approved actions              │
│     (invoke sub-skills, report results)         │
├─────────────────────────────────────────────────┤
│  5. REPORT — Summary of what changed            │
│     (PRs created, reviews posted, fixes made)   │
└─────────────────────────────────────────────────┘
```

---

## Phase 1: ASSESS

Gather current state quickly. If a health check was run today,
use those results. Otherwise, run a lightweight assessment:

### Quick State Check (< 2 minutes)

```bash
# CI status (are we green?)
gh run list --repo torch-spyre/torch-spyre \
  --workflow=runtests_nightly.yaml --branch main --limit 1 \
  --json conclusion --jq '.[0].conclusion'

# PRs awaiting review
gh pr list --repo torch-spyre/torch-spyre --state open \
  --json number,title,reviewDecision,isDraft \
  --jq '[.[] | select(.isDraft==false and .reviewDecision=="REVIEW_REQUIRED")] | length'

# PRs approved (ready to merge)
gh pr list --repo torch-spyre/torch-spyre --state open \
  --json number,title,reviewDecision \
  --jq '[.[] | select(.reviewDecision=="APPROVED")] | length'

# Open bugs
gh issue list --repo torch-spyre/torch-spyre --state open \
  --label "bug" --json number --jq 'length'
```

### Full State Check (if no recent health check)

Invoke `/daily-health-check` as a sub-skill to get comprehensive
data including build status, repo drift, and nightly triage.

---

## Phase 2: PRIORITIZE

Rank issues by this priority ladder:

| Priority | Category | Example | Sub-Skill |
|----------|----------|---------|-----------|
| P0 | **Build broken** | flex API mismatch, C++ compile error | `/cross-repo-adapt` |
| P1 | **Nightly red** | Test failures blocking CI | Manual fix or `/implement-op` |
| P2 | **PRs blocked** | Waiting on infra, stalled reviews | Comment / unblock |
| P3 | **PRs awaiting review** | Open PRs with no review | `/pr-review-swarm` |
| P4 | **Approved PRs not merged** | Ready to land | `gh pr merge` |
| P5 | **Docs drifted** | Ops table stale, API docs wrong | `/doc-sync-swarm` |
| P6 | **Missing ops** | Bugs about unsupported ops | `/implement-op` |
| P7 | **Model onboarding** | New model needs enabling | Research + implement |

---

## Phase 3: PROPOSE

Present the action plan to the user. Format:

```markdown
## Spyre Ops — Proposed Actions

**State:** Build PASS | Nightly FAIL (2 tests) | 5 PRs need review | 3 approved

### Recommended Actions (in priority order)

1. **[P1] Fix nightly** — mish tolerance + MRO test
   - Skill: manual edit (bump atol) + merge PR #2161
   - Time: ~5 min
   - Outcome: Nightly goes green

2. **[P3] Review PRs** — #2184, #2179, #2190, #2218
   - Skill: `/pr-review-swarm`
   - Time: ~3 min per PR
   - Outcome: Actionable review comments posted

3. **[P4] Merge approved** — #2210, #2189, #2128
   - Action: `gh pr merge --squash`
   - Time: ~1 min
   - Outcome: 3 PRs landed

4. **[P5] Sync docs** — 3 new ops added this week without table updates
   - Skill: `/doc-sync-swarm --ops-only`
   - Time: ~2 min
   - Outcome: supported_operations.md updated

### Skip (no action needed)
- Build: PASS (all 3 components)
- Version alignment: MATCH (PT 2.11.0)

---

Which actions should I execute? (all / 1,2,3 / none)
```

**Wait for user response before proceeding.**

---

## Phase 4: EXECUTE

For each approved action, invoke the appropriate sub-skill:

| Action Type | How to Execute |
|-------------|----------------|
| Fix nightly | Edit files directly, create branch, commit, push PR |
| Review PRs | Invoke `/pr-review-swarm` for each PR |
| Merge approved | `gh pr merge <N> --repo torch-spyre/torch-spyre --squash` |
| Sync docs | Invoke `/doc-sync-swarm` |
| Fix build | Invoke `/cross-repo-adapt` |
| Implement op | Invoke `/implement-op <op-name>` |
| Model onboard | Research → identify missing ops → implement each |

### Execution Rules (supervised mode)

- **Before merging:** Always confirm with user ("Merge #2210?")
- **Before posting reviews:** Show the review text, get approval
- **Before creating PRs:** Show the diff summary first
- **Before editing files:** Proceed (reversible, local)
- **Before running builds on pod:** Proceed (non-destructive)

### Autonomous Actions (no confirmation needed)

- Reading files, running grep, checking git status
- Running CI status checks
- Fetching PR metadata
- Building on pod (non-destructive)
- Running tests on pod

### Actions Requiring Confirmation

- Posting GitHub comments or reviews
- Merging PRs
- Creating branches and pushing
- Modifying code in torch-spyre source files
- Any action visible to other team members

---

## Phase 5: REPORT

After executing all approved actions, present a summary:

```markdown
## Spyre Ops — Execution Report

### Actions Completed

| # | Action | Result |
|---|--------|--------|
| 1 | Fix mish tolerance | PR #XXXX created |
| 2 | Review PRs | 4 reviews posted (1 approved, 2 changes requested, 1 commented) |
| 3 | Merge approved | 3 PRs merged (#2210, #2189, #2128) |

### Remaining Work (for next session)

- Nightly: will go green after #XXXX merges
- #2190: still blocked on flex infra
- #2218: needs code adaptations for PT 2.12

### Metrics

- PRs moved forward: 7
- Issues addressed: 2
- Build status: PASS
```

---

## Subcommand Details

### `/spyre-ops review`

Focused on the PR review queue:

1. List all PRs with `REVIEW_REQUIRED`
2. Exclude drafts and bot PRs (dependabot)
3. Sort by age (oldest first) and impact (compiler > tests > docs)
4. Propose reviewing top 3-5
5. On approval, run `/pr-review-swarm` for each

### `/spyre-ops fix`

Focused on broken things:

1. Check: Is the build broken? → `/cross-repo-adapt`
2. Check: Is nightly red? → Triage and fix
3. Check: Are there critical bugs? → Assess and propose fix
4. Propose fixes in priority order

### `/spyre-ops velocity`

Focused on shipping:

1. Merge all approved PRs (with confirmation)
2. Identify stalled PRs and propose unblocking actions
3. Find highest-priority missing ops from open bugs
4. Propose implementing top 1-2 ops via `/implement-op`

### `/spyre-ops <model-name>`

Model onboarding pipeline:

1. Research: Check if model has an hf_adapters adapter
2. Analyze: Identify model architecture (head_dim, attention type)
3. Check constraints: head_dim >= 128 required for Spyre
4. Attempt: Run smoke test if adapter exists
5. Identify: List missing ops blocking compilation
6. Propose: Implementation plan for missing ops
7. Execute: `/implement-op` for each (with approval)

---

## State Persistence

The orchestrator does NOT persist state between sessions. Each
invocation starts fresh by querying current state. This avoids
stale assumptions.

If a health check was run earlier in the same session, its
results are reused (they're in conversation context). Otherwise,
a new assessment is performed.

---

## Escalation Paths

Some situations require human judgment. The orchestrator stops
and reports:

| Situation | Escalation |
|-----------|------------|
| Merge conflict on approved PR | "PR #N has conflicts — author needs to rebase" |
| Nightly failure in code I don't understand | "Failure in X — needs domain expert" |
| Cross-repo fix affects semantics | "API change alters behavior — needs design review" |
| Model needs ops requiring backend changes | "Model X needs OpFunc Y — file issue against deeptools" |
| PR has fundamental design concerns | "PR #N has architectural issues — flag for team discussion" |
