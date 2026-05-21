---
name: implement-op
description: "Swarm-based implementation of new Spyre operations. Researches the op, determines pattern, writes code and tests, validates. Scoped to torch-spyre-only changes (no backend compiler modifications)."
---

# Implement Op Swarm

Implements a new operation on the Spyre backend using a multi-agent
swarm. Accepts an op name (ATen, torch, or functional) and
orchestrates research, implementation, testing, and validation.

**Scope:** Only ops implementable within torch-spyre. If the op
requires new backend compiler support (deeptools/flex), the swarm
stops and reports the blocker.

---

## Invocation

The user provides an op name as the argument:

- `torch.neg`
- `aten.maximum`
- `torch.nn.functional.silu`
- `torch.pow`

---

## Execution Strategy

### Phase 1: Research (1 agent, foreground)

Launch the research-agent first. Its output determines whether to
proceed and which pattern to use.

### Phase 2: Implement + Test (2 agents, parallel)

If research-agent approves, launch implement-agent and test-agent
in parallel.

### Phase 3: Validate (1 agent, foreground)

After code and tests are written, launch validate-agent to run
pre-commit and (optionally) pod tests.

---

## The Three Patterns

### Pattern 1: Direct ATen to OpFunc Mapping

**When:** The op maps directly to a backend OpFunc that already
exists (the op name is already supported by the Spyre compiler).

**Files to modify:**

| File | Change |
|------|--------|
| `torch_spyre/_inductor/spyre_kernel.py` | Add `@staticmethod` to `SpyreOpFuncs` class |
| `torch_spyre/_inductor/constants.py` | Add to `SPYRE_FP32_OPS` if op runs in FP32 |
| `torch_spyre/ops/eager.py` | Add to `register_torch_compile_kernel()` list |
| `tests/inductor/test_inductor_ops.py` | Add test cases |

**Example (maximum/minimum):**

```python
# spyre_kernel.py — add to SpyreOpFuncs in alphabetical order
@staticmethod
def maximum(a, b):
    return PointwiseOp("maximum", [a, b])

# constants.py — add to SPYRE_FP32_OPS list
SPYRE_FP32_OPS = [
    ...,
    "maximum",
    ...
]

# eager.py — add to register_torch_compile_kernel() list
register_torch_compile_kernel(
    [
        ...,
        aten.maximum,
        ...
    ]
)

# test_inductor_ops.py — add to appropriate dict
POINTWISE_BINARY_OPS_DICT = {
    ...,
    "maximum": torch.maximum,
}
```

### Pattern 2: Spyre-Specific Decomposition

**When:** The op can be rewritten using ops Spyre already supports.
The decomposition runs at the FX graph level before lowering.

**Files to modify:**

| File | Change |
|------|--------|
| `torch_spyre/_inductor/decompositions.py` | Add `@register_spyre_decomposition` function |
| `tests/inductor/test_inductor_ops.py` | Add test cases |

**Example (logical\_not):**

```python
# decompositions.py
@register_spyre_decomposition([torch.ops.aten.logical_not.default])
def spyre_logical_not(input):
    return torch.ops.aten.bitwise_not(input.to(torch.bool))
```

### Pattern 3: Custom Op + Lowering

**When:** The op needs a new `torch.ops.spyre.X` custom op with
explicit Inductor lowering and SpyreOpFuncs entry.

**Files to modify:**

| File | Change |
|------|--------|
| `torch_spyre/_inductor/customops.py` | Define `@torch.library.custom_op` + `@register_fake` |
| `torch_spyre/_inductor/lowering.py` | Add `@register_spyre_lowering` function |
| `torch_spyre/_inductor/spyre_kernel.py` | Add `@staticmethod` to `SpyreOpFuncs` |
| `tests/inductor/test_inductor_ops.py` | Add test cases |

**Example (softplus):**

```python
# customops.py
@torch.library.custom_op(
    "spyre::softplus", mutates_args=(), device_types="spyre"
)
def softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    pass

@softplus.register_fake
def _(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
):
    return input.new_empty(input.size())

# lowering.py
@register_spyre_lowering(torch.ops.spyre.softplus)
def lower_softplus(x, beta=1.0, threshold=20.0):
    fn = lowering.ops_wrapper(torch.ops.spyre.softplus.__name__)
    def inner_fn(index):
        return fn(x.make_loader()(index), beta, threshold)
    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
    )
    pw.realize()
    return pw

# spyre_kernel.py
@staticmethod
def softplus(x, beta, threshold):
    op_info = {
        "constants": {
            "softplusBeta": beta,
            "softplusThresh": threshold,
        }
    }
    return PointwiseOp("softplus", [x], op_info)
```

---

## Agent A: research-agent

**Goal:** Determine which pattern applies and whether the op is
implementable within torch-spyre.

### Steps

1. Resolve the op name to its ATen operator:

```bash
grep -r "<op_name>" torch_spyre/_inductor/ --include="*.py" | head -20
```

2. Check if already implemented in SpyreOpFuncs:

```bash
grep -n "<op_name>" torch_spyre/_inductor/spyre_kernel.py
```

3. Check if already decomposed:

```bash
grep -n "<op_name>" torch_spyre/_inductor/decompositions.py
```

4. Check if already a custom op:

```bash
grep -n "<op_name>" torch_spyre/_inductor/customops.py
```

5. Check if the op is in the eager registration list:

```bash
grep -n "<op_name>" torch_spyre/ops/eager.py
```

6. Check if the backend OpFunc name exists (search codegen):

```bash
grep -rn "<op_name>" torch_spyre/_inductor/codegen/ --include="*.py"
```

7. Look at how similar ops are implemented (find the closest
   existing op in SpyreOpFuncs and use it as a template).

8. Check if it needs FP32 whitelisting (ops that accumulate or
   need precision should be in SPYRE_FP32_OPS).

### Decision Logic

```
Is the op already fully implemented?
├─ YES → Report "already implemented" and STOP
│
└─ NO → Does an OpFunc with this name exist in the backend?
   ├─ YES → Pattern 1 (direct mapping)
   ├─ MAYBE (similar name exists) → Pattern 1 with name mapping
   │
   └─ NO → Can the op be decomposed into existing supported ops?
      ├─ YES → Pattern 2 (decomposition)
      │
      └─ NO → Is there a backend OpFunc that does the computation
              under a different name or with parameters?
         ├─ YES → Pattern 3 (custom op wrapping existing OpFunc)
         │
         └─ NO → STOP: "Requires backend compiler support"
```

### When to STOP

Report "requires backend compiler changes" if:

- No existing OpFunc can implement the computation
- The op needs new SuperDSC descriptor patterns
- The op needs new memory/data movement operations
- The op requires modifications to `codegen/superdsc.py` or
  `codegen/compute_ops.py`

### Report Format

```markdown
## Research Result

**Op:** <full aten name>
**Signature:** <arg types and return type>
**Pattern:** 1 / 2 / 3 / BLOCKED
**Reason:** <why this pattern>
**Template:** <name of similar existing op to use as reference>
**FP32 required:** yes/no
**Files to modify:** <list>
**Blockers:** none / <description>
```

---

## Agent B: implement-agent

**Goal:** Write the implementation code based on research-agent
output.

### Instructions

- Read the research-agent report for pattern choice and template op
- Read the template op's implementation as reference
- Write the new op following the EXACT same style
- Keep methods in `SpyreOpFuncs` sorted alphabetically
- Keep entries in `SPYRE_FP32_OPS` sorted alphabetically
- Keep entries in `register_torch_compile_kernel()` sorted by name
- Include the Apache 2.0 header only if creating new files
- Do NOT add comments explaining what the code does
- Do NOT add type annotations beyond what existing code uses

### Validation Checklist (before reporting done)

- [ ] SpyreOpFuncs method is alphabetically placed
- [ ] Op name string matches what backend expects
- [ ] Eager registration uses correct `aten.<name>` reference
- [ ] FP32 whitelist updated if needed
- [ ] No `import re` (use `import regex`)

---

## Agent C: test-agent

**Goal:** Write compiled-path tests for the new op.

### Test Location

All tests go in `tests/inductor/test_inductor_ops.py`.

### For Pointwise Unary Ops

Add to `POINTWISE_UNARY_OPS_DICT`:

```python
POINTWISE_UNARY_OPS_DICT = {
    ...,
    "<op_name>": torch.<op_name>,
}
```

The existing `TestOps` class with `ParameterizedTestMeta` will
auto-generate test cases covering multiple shapes.

### For Pointwise Binary Ops

Add to `POINTWISE_BINARY_OPS_DICT`:

```python
POINTWISE_BINARY_OPS_DICT = {
    ...,
    "<op_name>": torch.<op_name>,
}
```

### For Reduction Ops

Add to `CORE_REDUCTION_OPS_DICT`:

```python
CORE_REDUCTION_OPS_DICT = {
    ...,
    "<op_name>": torch.<op_name>,
}
```

### For Complex or Custom Ops

Write a dedicated test method using `compare_with_cpu()`:

```python
def test_<op_name>_fp16(self):
    def fn(x):
        return torch.<op_name>(x)

    x = torch.randn(128, 256, dtype=torch.float16, device="spyre")
    self.compare_with_cpu(fn, (x,), atol=0.1, rtol=0.1)
```

### Shape Requirements

- Stick-aligned: dimensions that are multiples of 64 (e.g., 128,
  256, 512)
- Non-aligned: dimensions that are NOT multiples of 64 (e.g., 67,
  71, 129) to test SDSC padding paths
- Cover 2D minimum; include 3D and 4D if the op handles
  multi-dimensional inputs

### Defaults

- dtype: `torch.float16` (Spyre default)
- atol: `0.1`, rtol: `0.1` (standard fp16 tolerance)
- Use `cached_randn()` where available for reproducibility

---

## Agent D: validate-agent

**Goal:** Verify implementation is correct and PR-ready.

### Steps

1. Run pre-commit on modified files:

```bash
pre-commit run --files <list of modified files>
```

Or if pre-commit is not available locally:

```bash
python3 -m ruff check torch_spyre/_inductor/spyre_kernel.py
python3 -m ruff format --check torch_spyre/_inductor/spyre_kernel.py
```

2. Check license headers on any new files:

```bash
head -14 <new_file> | grep "Apache License"
```

3. Check no `import re`:

```bash
grep -rn "^import re$" torch_spyre/ tests/ --include="*.py"
```

4. If pod is available, run the specific test:

```bash
kubectl exec -n a5-deepview rganti-spyre-main -- bash -lc "
WS=\$HOME/main-workspace
source \$WS/activate.sh
cd \$WS/torch-spyre
python3 -m pytest tests/inductor/test_inductor_ops.py \
  -k '<op_name>' -x --tb=short 2>&1 | tail -30
"
```

5. If pod is NOT available, report test as SKIPPED.

### Report Format

```markdown
## Validation Result

- Pre-commit: PASS / FAIL (<details>)
- License headers: PASS / N/A (no new files)
- Import check: PASS / FAIL
- Pod test: PASS / FAIL / SKIPPED
- **Overall: READY / NEEDS FIXES**

### Issues Found (if any)
- <issue 1>
- <issue 2>
```

---

## Output Format

Present the final result to the user:

```markdown
## Op Implementation Summary

**Op:** torch.<name> (aten.<name>)
**Pattern:** 1 (direct mapping) / 2 (decomposition) / 3 (custom op)
**Status:** COMPLETE / BLOCKED (requires backend changes)

### Files Modified

| File | Change |
|------|--------|
| `torch_spyre/_inductor/spyre_kernel.py` | Added <name> method |
| `torch_spyre/_inductor/constants.py` | Added to SPYRE_FP32_OPS |
| `torch_spyre/ops/eager.py` | Added to eager registration |
| `tests/inductor/test_inductor_ops.py` | Added test cases |

### Validation

- Pre-commit: PASS/FAIL
- Pod test: PASS/FAIL/SKIPPED

### Next Steps

- [ ] Review the diff
- [ ] Create branch and commit with `git commit -s`
- [ ] Push and create PR
```

If BLOCKED:

```markdown
## Op Implementation Summary

**Op:** torch.<name>
**Status:** BLOCKED — requires backend compiler changes

### Reason

<explanation of why the op cannot be implemented in torch-spyre alone>

### Recommendation

File an issue against deeptools/flex requesting OpFunc support for
"<op_name>". Include the ATen signature and expected behavior.
```

---

## Key Reference Files

| File | What to Find |
|------|--------------|
| `torch_spyre/_inductor/spyre_kernel.py` | SpyreOpFuncs class (all direct mappings) |
| `torch_spyre/_inductor/constants.py` | SPYRE_FP32_OPS whitelist |
| `torch_spyre/ops/eager.py` | Eager mode registration list |
| `torch_spyre/_inductor/decompositions.py` | Decomposition decorator and examples |
| `torch_spyre/_inductor/customops.py` | Custom op definitions |
| `torch_spyre/_inductor/lowering.py` | Lowering registrations |
| `torch_spyre/_inductor/codegen/superdsc.py` | Backend codegen (DO NOT MODIFY) |
| `torch_spyre/_inductor/codegen/compute_ops.py` | Compute op dispatch (DO NOT MODIFY) |
| `tests/inductor/test_inductor_ops.py` | Test framework and op dicts |
| `tests/_inductor/utils_inductor.py` | Test utilities (compare_with_cpu, etc.) |
