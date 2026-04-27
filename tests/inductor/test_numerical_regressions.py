# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Regression tests for numerical accuracy bugs found during LLM inference work.

Each test guards against a specific torch-spyre bug that caused numerical
divergence in multi-token generation on Spyre. Tests are self-contained
(no model downloads, no external dependencies beyond torch and torch_spyre).

Issue references:
  - #1760: torch.cat size-1 concat dim (fixed, regression guard)
  - #1765: spyre.overwrite fill-mode incorrect results
  - #1781: RMSNorm on non-contiguous inputs
  - #1783: Compiled view+transpose head-layout transform
"""

import pytest
import unittest

import torch
import torch.nn.functional as F

from utils_inductor import (
    DEVICE,
    ParameterizedTestMeta,
    _assert_results_close,
    _compile_and_run,
    cached_randn,
    compare_with_cpu,
)


# ---------------------------------------------------------------------------
# A. RMSNorm on non-contiguous inputs (#1781)
# ---------------------------------------------------------------------------


class TestRMSNormNonContiguous(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        # Non-contiguous input from transpose (#1781, fixed)
        ("test_rmsnorm_noncontig", "_base_rmsnorm_noncontig"): {
            "param_sets": {
                "bhsd_8h_seq64": (cached_randn((1, 64, 8, 128)),),
                "bhsd_16h_seq64": (cached_randn((1, 64, 16, 128), differentiation=1),),
            },
        },
        # Control cases: contiguous input (should pass)
        ("test_rmsnorm_contig_control", "_base_rmsnorm_contig"): {
            "param_sets": {
                "bhsd_8h_contig": (cached_randn((1, 8, 64, 128)),),
                "single_token": (cached_randn((1, 8, 1, 128)),),
            },
        },
    }

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def _base_rmsnorm_noncontig(self, x):
        x_noncontig = x.transpose(1, 2)
        assert not x_noncontig.is_contiguous()

        def fn(t):
            return F.rms_norm(t, [t.shape[-1]], eps=1e-6)

        compare_with_cpu(fn, x_noncontig, run_eager=False)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def _base_rmsnorm_contig(self, x):
        assert x.is_contiguous()

        def fn(t):
            return F.rms_norm(t, [t.shape[-1]], eps=1e-6)

        compare_with_cpu(fn, x, run_eager=False)


# ---------------------------------------------------------------------------
# B. Compiled view+transpose head-layout transform (#1783)
# ---------------------------------------------------------------------------


class TestHeadLayoutTransform(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        # Multi-token sequences (#1783, fixed)
        ("test_head_layout_multitok", "_base_head_layout"): {
            "param_sets": {
                "qwen3_q_seq64": (1, 64, 16, 128, cached_randn((1, 64, 2048))),
                "qwen3_k_seq64": (1, 64, 8, 128, cached_randn((1, 64, 1024))),
                "qwen3_q_seq192": (
                    1,
                    192,
                    16,
                    128,
                    cached_randn((1, 192, 2048), differentiation=1),
                ),
            },
        },
        # Single-token control
        ("test_head_layout_singletok", "_base_head_layout"): {
            "param_sets": {
                "qwen3_q_seq1": (1, 1, 16, 128, cached_randn((1, 1, 2048))),
                "qwen3_k_seq1": (
                    1,
                    1,
                    8,
                    128,
                    cached_randn((1, 1, 1024), differentiation=2),
                ),
            },
        },
    }

    def _base_head_layout(self, B, S, H, D, x):
        def fn(x):
            return x.view(B, S, H, D).transpose(1, 2)

        compare_with_cpu(fn, x, run_eager=False)


# ---------------------------------------------------------------------------
# C. spyre.overwrite fill-mode (#1765)
# ---------------------------------------------------------------------------


class TestOverwriteFill(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        # Fill mode: single-row write into populated cache (#1765, xfail)
        ("test_overwrite_fill", "_base_overwrite_fill"): {
            "param_sets": {
                "kv_pos0": (
                    cached_randn((1, 8, 128, 128)),
                    cached_randn((1, 8, 1, 128), differentiation="fill_0"),
                    0,
                ),
                "kv_pos65": (
                    cached_randn((1, 8, 128, 128), differentiation=1),
                    cached_randn((1, 8, 1, 128), differentiation="fill_65"),
                    65,
                ),
                "kv_pos_last": (
                    cached_randn((1, 8, 256, 128), differentiation=2),
                    cached_randn((1, 8, 1, 128), differentiation="fill_last"),
                    255,
                ),
            },
        },
        # Control: expansion (append into fresh buffer) — should pass
        ("test_overwrite_expand", "_base_overwrite_expand"): {
            "param_sets": {
                "expand_64_plus_64": (
                    cached_randn((1, 8, 64, 128), differentiation="exp_a"),
                    cached_randn((1, 8, 64, 128), differentiation="exp_b"),
                ),
            },
        },
    }

    @pytest.mark.xfail(
        reason="torch-spyre#1765: overwrite fill-mode produces wrong values",
        strict=True,
    )
    def _base_overwrite_fill(self, cache, new_val, position):
        # CPU reference: slice assignment
        cpu_ref = cache.clone()
        cpu_ref[:, :, position : position + 1, :] = new_val

        # Compiled Spyre: use spyre.overwrite
        def fn(cache, new_val):
            out = cache.clone()
            result = torch.ops.spyre.overwrite(
                input=new_val,
                output=out,
                dims=[2],
                offsets=[position],
            )
            if result is not None:
                out = result
            return out

        spyre_result = _compile_and_run(fn, (cache, new_val), DEVICE)
        _assert_results_close(
            spyre_result,
            cpu_ref,
            atol=0.1,
            rtol=0.1,
            comparison_name="overwrite fill vs cpu",
        )

    def _base_overwrite_expand(self, old_cache, new_block):
        old_len = old_cache.shape[2]
        new_len = old_len + new_block.shape[2]

        # CPU reference: torch.cat
        cpu_ref = torch.cat([old_cache, new_block], dim=2)

        # Compiled Spyre: two overwrites into fresh buffer
        def fn(old_cache, new_block):
            out = torch.zeros(
                old_cache.shape[0],
                old_cache.shape[1],
                new_len,
                old_cache.shape[3],
                dtype=old_cache.dtype,
                device=old_cache.device,
            )
            result = torch.ops.spyre.overwrite(
                input=old_cache,
                output=out,
                dims=[2],
                offsets=[0],
            )
            if result is not None:
                out = result
            result = torch.ops.spyre.overwrite(
                input=new_block,
                output=out,
                dims=[2],
                offsets=[old_len],
            )
            if result is not None:
                out = result
            return out

        spyre_result = _compile_and_run(fn, (old_cache, new_block), DEVICE)
        _assert_results_close(
            spyre_result,
            cpu_ref,
            atol=0.1,
            rtol=0.1,
            comparison_name="overwrite expand vs cpu cat",
        )


# ---------------------------------------------------------------------------
# D. torch.cat size-1 concat (#1760 — FIXED, regression guard)
# ---------------------------------------------------------------------------


class TestCatSize1Regression(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        ("test_cat_size1_kv", "_base_cat_size1_kv"): {
            "param_sets": {
                "dim2_64": (
                    cached_randn((1, 8, 64, 128), differentiation="cat_a"),
                    cached_randn((1, 8, 1, 128), differentiation="cat_b"),
                ),
                "dim2_192": (
                    cached_randn((1, 8, 192, 128), differentiation="cat_c"),
                    cached_randn((1, 8, 1, 128), differentiation="cat_d"),
                ),
            },
        },
    }

    def _base_cat_size1_kv(self, cache, new_kv):
        def fn(cache, new_kv):
            return torch.cat([cache, new_kv], dim=2)

        compare_with_cpu(fn, cache, new_kv)


# ---------------------------------------------------------------------------
# E. Softmax -inf masking residual
# ---------------------------------------------------------------------------


class TestSoftmaxMasking(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        ("test_softmax_neginf", "_base_softmax_neginf"): {
            "param_sets": {
                "attn_64x64": (cached_randn((1, 16, 64, 64)),),
                "attn_64x128": (cached_randn((1, 16, 64, 128), differentiation=1),),
            },
        },
    }

    def _base_softmax_neginf(self, scores):
        Q, K = scores.shape[-2], scores.shape[-1]
        mask = torch.full((Q, K), float("-inf"), dtype=scores.dtype)
        mask = torch.triu(mask, diagonal=1)
        masked_scores = scores + mask

        def fn(x):
            return torch.softmax(x, dim=-1)

        spyre_result = _compile_and_run(fn, (masked_scores,), DEVICE)

        inf_mask = masked_scores.isinf() & (masked_scores < 0)
        inf_mask_expanded = inf_mask.expand_as(spyre_result)
        if inf_mask_expanded.any():
            residual = spyre_result[inf_mask_expanded]
            max_residual = residual.abs().max().item()
            assert max_residual < 1e-4, (
                f"Softmax -inf masking leak: max residual = {max_residual} "
                f"(expected < 1e-4)"
            )


# ---------------------------------------------------------------------------
# F. Transpose + contiguous in compiled mode
# ---------------------------------------------------------------------------


class TestTransposeContiguousRegression(
    unittest.TestCase, metaclass=ParameterizedTestMeta
):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        ("test_transpose_contig_kvcache", "_base_transpose_contig_kvcache"): {
            "param_sets": {
                "dim_1_2_16h": (
                    1,
                    2,
                    cached_randn((1, 16, 64, 128)),
                ),
                "dim_1_2_8h": (
                    1,
                    2,
                    cached_randn((1, 8, 192, 128), differentiation=1),
                ),
            },
        },
    }

    def _base_transpose_contig_kvcache(self, dim0, dim1, x):
        compare_with_cpu(
            lambda x: torch.transpose(x, dim0, dim1).contiguous(),
            x,
            run_eager=False,
        )


if __name__ == "__main__":
    unittest.main()
