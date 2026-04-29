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
HF Transformers model CI gate tests for Spyre.

Validates that small HF models (Qwen3 0.6B, Granite 2B) produce correct
greedy tokens on Spyre by comparing against pre-computed golden reference
values from stock HuggingFace Transformers (fp16, CPU/GPU).

Uses a prompt long enough (~30 tokens) that all decode steps stay in fill
mode (no expansion), which produces GPU-identical accuracy on Spyre.

Usage (on Spyre pod):
    python3 -m pytest test_hf_models.py -v
    python3 test_hf_models.py
"""

import math
import unittest

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401

from transformers import AutoTokenizer

from hf_model_helpers import (
    BLOCK_SIZE,
    DEVICE,
    build_expansion_mask,
    build_prefill_mask,
    load_model,
    prepare_granite,
    prepare_qwen3,
    granite_run_forward,
    qwen3_run_forward,
)

PROMPT = (
    "Question: Marin and his neighbor Nancy each eat 4 apples a day. "
    "How many apples do they eat in 30 days?\nAnswer:"
)
NUM_DECODE = 4
MIN_TOP5_OVERLAP = 3
MAX_TOP5_LOGIT_DIFF = 1.0

# Golden reference: greedy tokens and top-5 token IDs from stock HF
# Transformers (fp16) on the prompt above.  Regenerate with:
#   python3 -c "from transformers import ...; ..."  (see bottom of file)
GOLDEN = {
    "Qwen/Qwen3-0.6B": {
        "golden_tokens": [220, 17, 19, 15, 198],
        "golden_top5": [
            [220, 6771, 1124, 17607, 576],
            [17, 16, 18, 19, 24],
            [19, 15, 23, 17, 20],
            [15, 271, 198, 40676, 21],
            [198, 271, 15, 40676, 382],
        ],
    },
    "ibm-granite/granite-3.3-2b-instruct": {
        "golden_tokens": [8359, 266, 484, 1966, 225],
        "golden_top5": [
            [8359, 2614, 203, 11228, 11072],
            [266, 5225, 482, 5930, 13762],
            [484, 461, 1182, 225, 16831],
            [1966, 1741, 1196, 17332, 4177],
            [225, 1123, 44, 461, 12074],
        ],
    },
}


def _run_spyre_inference(model_path, prepare_fn, run_forward_fn, num_decode=NUM_DECODE):
    """Load model, run adapter on Spyre, return per-step logits."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_model(model_path, prepare_fn)

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = (
        getattr(model, "_spyre_head_dim", None)
        or getattr(model.config, "head_dim", None)
        or model.config.hidden_size // model.config.num_attention_heads
    )
    v_head_dim = getattr(model, "_spyre_v_head_dim", head_dim)
    vocab_size = model.config.vocab_size

    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    prompt_offset = padded_len - seq_len
    max_cache_len = (
        padded_len + math.ceil(num_decode / BLOCK_SIZE) * BLOCK_SIZE + BLOCK_SIZE
    )

    if prompt_offset > 0:
        pad = input_ids.new_zeros((batch_size, prompt_offset))
        padded_ids = torch.cat([pad, input_ids], dim=1)
    else:
        padded_ids = input_ids

    position_ids = torch.zeros((batch_size, padded_len), dtype=torch.long)
    position_ids[:, prompt_offset:] = torch.arange(seq_len)

    key_caches = [
        torch.zeros(
            batch_size,
            num_kv_heads,
            max_cache_len,
            head_dim,
            dtype=torch.float16,
            device=DEVICE,
        )
        for _ in range(num_layers)
    ]
    value_caches = [
        torch.zeros(
            batch_size,
            num_kv_heads,
            max_cache_len,
            v_head_dim,
            dtype=torch.float16,
            device=DEVICE,
        )
        for _ in range(num_layers)
    ]

    spyre_results = []

    # Prefill
    prefill_mask = build_prefill_mask(
        batch_size, padded_len, max_cache_len, prompt_offset
    )
    with torch.no_grad():
        logits = run_forward_fn(
            model,
            padded_ids.to(DEVICE),
            position_ids.to(DEVICE),
            prefill_mask.to(DEVICE),
            key_caches,
            value_caches,
            is_filling=False,
            token_index=0,
            cache_position=0,
        )
    logits_cpu = logits.to("cpu")[0, -1, :].float()[:vocab_size]
    token = logits_cpu.argmax().item()
    spyre_results.append({"token": token, "logits": logits_cpu})

    # Decode (fill + expand mirroring generate())
    result = padded_ids.clone()
    current_cache_len = padded_len
    tokens_in_block = BLOCK_SIZE - 1
    decode_pos = torch.zeros((batch_size, BLOCK_SIZE), dtype=torch.long)
    for j in range(BLOCK_SIZE):
        decode_pos[:, j] = seq_len + j - BLOCK_SIZE
    fill_mask_device = None

    if tokens_in_block == BLOCK_SIZE - 1:
        result = F.pad(result, (0, BLOCK_SIZE))
    tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
    grab_idx = BLOCK_SIZE if tokens_in_block == 0 else BLOCK_SIZE - tokens_in_block
    result[:, -grab_idx] = token

    for step in range(1, num_decode + 1):
        is_filling = tokens_in_block > 0
        next_input = result[:, -BLOCK_SIZE:].to(DEVICE)

        if is_filling:
            fill_pos = current_cache_len - BLOCK_SIZE + tokens_in_block
            with torch.no_grad():
                logits = run_forward_fn(
                    model,
                    next_input,
                    decode_pos.to(DEVICE),
                    fill_mask_device,
                    key_caches,
                    value_caches,
                    is_filling=True,
                    token_index=tokens_in_block,
                    cache_position=fill_pos,
                )
            logits_cpu = logits.to("cpu")
            grab_logit = BLOCK_SIZE - tokens_in_block
            last_logits = logits_cpu[0, -grab_logit, :].float()[:vocab_size]
        else:
            current_cache_len += BLOCK_SIZE
            decode_pos = decode_pos + BLOCK_SIZE
            exp_mask = build_expansion_mask(
                batch_size,
                BLOCK_SIZE,
                max_cache_len,
                current_cache_len,
                prompt_offset,
            )
            with torch.no_grad():
                logits = run_forward_fn(
                    model,
                    next_input,
                    decode_pos.to(DEVICE),
                    exp_mask.to(DEVICE),
                    key_caches,
                    value_caches,
                    is_filling=False,
                    token_index=0,
                    cache_position=current_cache_len - BLOCK_SIZE,
                )
            logits_cpu = logits.to("cpu")
            last_logits = logits_cpu[0, -BLOCK_SIZE, :].float()[:vocab_size]
            fill_mask_device = exp_mask.to(DEVICE)

        token = last_logits.argmax().item()
        spyre_results.append({"token": token, "logits": last_logits})

        if tokens_in_block == BLOCK_SIZE - 1:
            result = F.pad(result, (0, BLOCK_SIZE))
        tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
        grab_idx = BLOCK_SIZE if tokens_in_block == 0 else BLOCK_SIZE - tokens_in_block
        result[:, -grab_idx] = token

    return spyre_results


class TestHFModels(unittest.TestCase):
    """CI gate: HF model accuracy on Spyre vs golden reference."""

    def _assert_against_golden(self, model_path, spyre_results):
        golden = GOLDEN[model_path]
        golden_tokens = golden["golden_tokens"]
        golden_top5 = golden["golden_top5"]

        for i, sp in enumerate(spyre_results):
            step = "prefill" if i == 0 else f"decode-{i}"
            sp_logits = sp["logits"]

            sp_top5_set = set(sp_logits.topk(5).indices.tolist())
            golden_top5_set = set(golden_top5[i])
            overlap = len(sp_top5_set & golden_top5_set)

            golden_top5_ids = torch.tensor(golden_top5[i])
            top5_diffs = (
                sp_logits[golden_top5_ids] - sp_logits[golden_tokens[i]]
            ).abs()
            max_top5_diff = top5_diffs.max().item()

            self.assertGreaterEqual(
                overlap,
                MIN_TOP5_OVERLAP,
                f"{model_path} {step}: top-5 overlap {overlap}/5 "
                f"< {MIN_TOP5_OVERLAP}. "
                f"golden={golden_top5[i]}, spyre={list(sp_top5_set)}",
            )
            self.assertLess(
                max_top5_diff,
                MAX_TOP5_LOGIT_DIFF,
                f"{model_path} {step}: max logit diff at golden top-5 "
                f"positions {max_top5_diff:.4f} exceeds {MAX_TOP5_LOGIT_DIFF}",
            )

    def test_qwen3_generation(self):
        """Qwen3 0.6B: Q/K RMSNorm, head_dim=128, 28 layers."""
        spyre_results = _run_spyre_inference(
            "Qwen/Qwen3-0.6B", prepare_qwen3, qwen3_run_forward
        )
        self._assert_against_golden("Qwen/Qwen3-0.6B", spyre_results)

    def test_granite_generation(self):
        """Granite 2B: head_dim padding 64->128, residual multipliers."""
        spyre_results = _run_spyre_inference(
            "ibm-granite/granite-3.3-2b-instruct",
            prepare_granite,
            granite_run_forward,
        )
        self._assert_against_golden(
            "ibm-granite/granite-3.3-2b-instruct", spyre_results
        )


if __name__ == "__main__":
    unittest.main()

# To regenerate golden values:
#
#   from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
#   import torch
#   prompt = "Question: Marin and his neighbor Nancy each eat 4 apples ..."
#   tokenizer = AutoTokenizer.from_pretrained(path)
#   model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16)
#   # run greedy decode with DynamicCache for 5 steps, collect argmax + topk(5)
