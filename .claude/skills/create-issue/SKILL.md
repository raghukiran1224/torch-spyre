---
name: create-issue
description: "Guide for creating well-structured GitHub issues in torch-spyre. Covers bug reports, feature requests, epics, and engineering tasks with correct labels, templates, and conventions."
---

# Creating a GitHub Issue

This skill guides issue creation on the **torch-spyre/torch-spyre** upstream
repo. Issues use GitHub's YAML form templates (`.github/ISSUE_TEMPLATE/`).
When creating via `gh issue create`, the body must follow the same structure
the template would produce.

---

## Issue Types

| Type | Title convention | Label | Template |
|---|---|---|---|
| Bug | `[BUG] <description>` or plain | `type/bug` | `bug-report.yml` |
| Feature | Plain descriptive sentence | `type/feature` | `feature-request.yml` |
| Epic | `[EPIC] <description>` | `epic` | None (freeform) |
| Engineering task | Plain imperative sentence | Component label(s) | None (freeform) |

---

## Decision Tree

```text
Is it a defect / unexpected behavior?
├── Yes → Bug Report
└── No
    ├── Is it a large initiative with sub-tasks? → Epic
    ├── Is it a user-visible capability? → Feature Request
    └── Is it an internal refactor / implementation task? → Engineering Task
```

---

## Bug Report

Use the bug report template. Required sections:

```markdown
### What happened?

<Clear description of the failure. Include the error message or traceback.>

### What did you expect to happen?

<Expected behavior.>

### How can we reproduce it?

<Minimal reproducer — a short Python script that triggers the bug.>

```python
import torch
import torch_spyre  # noqa: F401

device = torch.device("spyre")
# ... minimal code that triggers the bug
```

### Any environmental details we need to know?

- PyTorch version:
- torch-spyre version/commit:
- Python version:
- OS:
- Spyre firmware version (if relevant):
- Number of cores (`SENCORES`):

### Anything else we need to know?

<Optional — related issues, workarounds, frequency.>

### Relevant log output

```shell
<paste log output here>
```
```

**Labels:** `type/bug` + component label(s) from the list below.

**PyTorch convention:** If the bug is in a specific subsystem, prefix the
title with the component in brackets: `[Inductor] ...`, `[Runtime] ...`.

---

## Feature Request

Use the feature request template. Required sections:

```markdown
### What feature would you like to be added?

<Describe the feature. Include API sketches or code examples if helpful.>

### Why is this feature needed?

<Motivation — what use case does this enable? What pain point does it solve?>

### Any alternative solution that you may want to share?

<Optional — other approaches considered and why they were rejected.>

### Anything else we need to know?

<Optional — links to related work, upstream PyTorch issues, RFCs.>
```

**Labels:** `type/feature` + component label(s).

---

## Epic

Epics track large initiatives with multiple sub-tasks. Use freeform markdown.

```markdown
## Context

<Why this epic exists — the business or technical driver.>

## Scope

<What is in and out of scope.>

## Tasks

- [ ] Sub-task 1 (#issue or description)
- [ ] Sub-task 2
- [ ] Sub-task 3

## Status

| Item | Status | Owner | Notes |
|---|---|---|---|
| Sub-task 1 | In progress | @user | PR #123 |
| Sub-task 2 | Not started | — | Blocked on sub-task 1 |
```

**Labels:** `epic` + component label(s).

---

## Engineering Task

Internal refactors, implementation work, or infrastructure changes that are
not user-facing features. Use freeform markdown.

```markdown
**Context:** <Why this work is needed.>

**Changes:**
- Bullet list of planned changes

**Blocked:** <Any blockers or dependencies — omit if none.>

**Related:** #issue_number, PR #number
```

**Labels:** Component label(s) only (no type prefix needed).

---

## Available Labels

### Type labels

`type/bug`, `type/feature`, `epic`, `rfc`, `documentation`, `duplicate`,
`question`, `wontfix`, `invalid`, `help wanted`, `good first issue`

### Component labels

`inductor`, `backend-compiler`, `driver-runtime`, `torch-runtime`, `testing`,
`ci-cd`, `torch-profiler`, `model-enablement`, `fallback-impl`

### Team labels

`llmd`, `comms`, `vllm`, `card-firmware`, `openshift-operator`, `card-mgmt`

### Operation tracking

`torch-ops (simple)`, `torch-ops (complex)`, `torch-ops (models)`

---

## Creating via `gh` CLI

Use the companion `issue-templates/` directory for body templates, or
construct inline with a heredoc:

```bash
# Bug report
gh issue create -R torch-spyre/torch-spyre \
  --title "[BUG] Description of the bug" \
  --label "type/bug,inductor" \
  --body "$(cat <<'EOF'
### What happened?

<description>

### What did you expect to happen?

<expected>

### How can we reproduce it?

<reproducer>

### Any environmental details we need to know?

<environment>

### Relevant log output

```shell
<logs>
```
EOF
)"

# Feature request
gh issue create -R torch-spyre/torch-spyre \
  --title "Add support for XYZ" \
  --label "type/feature,inductor" \
  --body "$(cat <<'EOF'
### What feature would you like to be added?

<description>

### Why is this feature needed?

<motivation>
EOF
)"
```

---

## Best Practices

1. **Reproducers matter.** For bugs, always include a minimal Python script
   that reproduces the issue. Follow the PyTorch convention of including the
   full traceback.

2. **One issue, one concern.** Don't bundle unrelated bugs or features. If
   you find multiple problems, file separate issues and cross-reference.

3. **Link related work.** Reference related issues (`#123`), PRs, upstream
   PyTorch issues (`pytorch/pytorch#12345`), or RFCs
   (`torch-spyre/rfcs#N`).

4. **Mention stakeholders.** Use `@username` at the end to cc relevant
   people — especially component owners.

5. **Environment details.** For bugs, always include PyTorch version,
   torch-spyre commit, Python version, and Spyre firmware version if
   hardware-related.

6. **Use component prefixes for bugs.** Follow PyTorch's `[Module]` prefix
   convention when the bug is clearly in one subsystem:
   `[Inductor]`, `[Runtime]`, `[Eager]`, `[Driver]`.

7. **Epics need checklists.** Use `- [ ]` task lists so progress is
   trackable directly in the GitHub UI.

8. **RFCs are separate.** Design proposals go to
   <https://github.com/torch-spyre/rfcs>, not as issues. You may create a
   tracking issue that links to the RFC.
