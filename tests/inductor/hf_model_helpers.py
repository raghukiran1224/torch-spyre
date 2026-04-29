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
Self-contained HF Transformers adapter helpers for Spyre CI model tests.

Provides everything needed to load, compile, and run HF models on Spyre:
RoPE precomputation, RMSNorm patching, head-dim padding, mask construction,
KV cache update, generation loop, and model-specific prepare/forward functions
for Qwen3 and Granite.

Copied from github.ibm.com:msrivats/hf_adapters (hf_common.py, hf_qwen3.py,
hf_granite.py) to keep torch-spyre CI self-contained.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "spyre"
BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class PrecomputedRotaryEmbedding(nn.Module):
    def __init__(self, original_rope: nn.Module, padded_head_dim: Optional[int] = None):
        super().__init__()
        self.original = original_rope
        self.padded_head_dim = padded_head_dim
        self._freq_cache: Optional[torch.Tensor] = None
        self._cached_len = 0

    def _extend_cache(self, max_len: int):
        if max_len <= self._cached_len:
            return
        target_len = max(max_len, self._cached_len * 2, 2048)
        inv_freq = self.original.inv_freq.to("cpu").float()
        rope_half = inv_freq.shape[0]
        t = torch.arange(target_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq).float()
        scaling = getattr(self.original, "attention_scaling", 1.0)
        rot = torch.stack(
            [
                torch.cos(freqs) * scaling,
                -torch.sin(freqs) * scaling,
                torch.sin(freqs) * scaling,
                torch.cos(freqs) * scaling,
            ],
            dim=1,
        ).view(target_len, 2, 2, rope_half)

        if self.padded_head_dim is not None:
            padded_half = self.padded_head_dim // 2
            if padded_half > rope_half:
                pad_half = padded_half - rope_half
                ident = torch.zeros(target_len, 2, 2, pad_half)
                ident[:, 0, 0, :] = 1.0
                ident[:, 1, 1, :] = 1.0
                rot = torch.cat([rot, ident], dim=-1)

        self._freq_cache = rot.contiguous().to(torch.float16)
        self._cached_len = target_len

    def forward(self, hidden_states, position_ids):
        pos_cpu = position_ids.to("cpu")
        max_pos = int(pos_cpu.max().item()) + 1
        self._extend_cache(max_pos)
        selected = self._freq_cache[pos_cpu]
        return selected.to(DEVICE)


def apply_rope_matmul(x, selected_freqs):
    B, H, L, D = x.shape
    half = D // 2
    x_ = x.transpose(1, 2).reshape(B, L, H, 2, half)
    sf = selected_freqs[:, :, None, :, :, :]
    out = sf.mul(x_.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# KV cache update
# ---------------------------------------------------------------------------


def kv_cache_update(
    k, v, key_cache, value_cache, is_filling, token_index, cache_position
):
    if is_filling:
        k_write = k[:, :, token_index : token_index + 1, :]
        v_write = v[:, :, token_index : token_index + 1, :]
    else:
        k_write = k
        v_write = v

    if key_cache.device.type == "spyre":
        torch.ops.spyre.overwrite(
            input=k_write,
            output=key_cache,
            dims=[2],
            offsets=[cache_position],
        )
        torch.ops.spyre.overwrite(
            input=v_write,
            output=value_cache,
            dims=[2],
            offsets=[cache_position],
        )
    else:
        seq_len = k_write.shape[2]
        key_cache[:, :, cache_position : cache_position + seq_len, :] = k_write
        value_cache[:, :, cache_position : cache_position + seq_len, :] = v_write

    return key_cache, value_cache


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


def pad_attention_heads(
    model, layers, orig_head_dim, padded_head_dim, num_heads, num_kv_heads
):
    assert padded_head_dim > orig_head_dim
    orig_half = orig_head_dim // 2
    padded_half = padded_head_dim // 2

    def _pad_qk_rope(proj, n_heads):
        w = proj.weight
        hidden = w.shape[1]
        new_w = torch.zeros(n_heads * padded_head_dim, hidden, dtype=w.dtype)
        for h in range(n_heads):
            s = h * orig_head_dim
            d = h * padded_head_dim
            new_w[d : d + orig_half, :] = w[s : s + orig_half, :]
            new_w[d + padded_half : d + padded_half + orig_half, :] = w[
                s + orig_half : s + orig_head_dim, :
            ]
        new_proj = nn.Linear(
            hidden, n_heads * padded_head_dim, bias=proj.bias is not None
        )
        new_proj.weight = nn.Parameter(new_w, requires_grad=False)
        if proj.bias is not None:
            new_b = torch.zeros(n_heads * padded_head_dim, dtype=proj.bias.dtype)
            for h in range(n_heads):
                s = h * orig_head_dim
                d = h * padded_head_dim
                new_b[d : d + orig_half] = proj.bias[s : s + orig_half]
                new_b[d + padded_half : d + padded_half + orig_half] = proj.bias[
                    s + orig_half : s + orig_head_dim
                ]
            new_proj.bias = nn.Parameter(new_b, requires_grad=False)
        return new_proj

    def _pad_v_simple(proj, n_heads):
        w = proj.weight
        hidden = w.shape[1]
        new_w = torch.zeros(n_heads * padded_head_dim, hidden, dtype=w.dtype)
        for h in range(n_heads):
            s = h * orig_head_dim
            d = h * padded_head_dim
            new_w[d : d + orig_head_dim, :] = w[s : s + orig_head_dim, :]
        new_proj = nn.Linear(
            hidden, n_heads * padded_head_dim, bias=proj.bias is not None
        )
        new_proj.weight = nn.Parameter(new_w, requires_grad=False)
        if proj.bias is not None:
            new_b = torch.zeros(n_heads * padded_head_dim, dtype=proj.bias.dtype)
            for h in range(n_heads):
                s = h * orig_head_dim
                d = h * padded_head_dim
                new_b[d : d + orig_head_dim] = proj.bias[s : s + orig_head_dim]
            new_proj.bias = nn.Parameter(new_b, requires_grad=False)
        return new_proj

    def _pad_o(proj, n_heads):
        w = proj.weight
        hidden = w.shape[0]
        new_w = torch.zeros(hidden, n_heads * padded_head_dim, dtype=w.dtype)
        for h in range(n_heads):
            s = h * orig_head_dim
            d = h * padded_head_dim
            new_w[:, d : d + orig_head_dim] = w[:, s : s + orig_head_dim]
        new_proj = nn.Linear(
            n_heads * padded_head_dim, hidden, bias=proj.bias is not None
        )
        new_proj.weight = nn.Parameter(new_w, requires_grad=False)
        if proj.bias is not None:
            new_proj.bias = nn.Parameter(proj.bias.clone(), requires_grad=False)
        return new_proj

    for layer in layers:
        attn = layer.self_attn
        orig_scaling = attn.scaling
        attn.q_proj = _pad_qk_rope(attn.q_proj, num_heads)
        attn.k_proj = _pad_qk_rope(attn.k_proj, num_kv_heads)
        attn.v_proj = _pad_v_simple(attn.v_proj, num_kv_heads)
        attn.o_proj = _pad_o(attn.o_proj, num_heads)
        attn.head_dim = padded_head_dim
        attn.scaling = orig_scaling

    model._spyre_head_dim = padded_head_dim


def patch_rmsnorm(rmsnorm_cls):
    def _forward_fp16(self, hidden_states):
        if hidden_states.device.type == "spyre":
            variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
            eps = torch.ops.spyre.full(
                (1,),
                self.variance_epsilon,
                hidden_states.device,
                torch.float16,
            )
            return self.weight * (hidden_states * torch.rsqrt(variance + eps))
        else:
            xf = hidden_states.float()
            variance = (xf * xf).mean(-1, keepdim=True)
            xf = xf * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * xf.to(hidden_states.dtype)

    rmsnorm_cls.forward = _forward_fp16


def pad_lm_head(model):
    w = model.lm_head.weight
    vocab = w.shape[0]
    padded = ((vocab + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) + BLOCK_SIZE
    if padded != vocab:
        model.lm_head.weight = nn.Parameter(
            F.pad(w, (0, 0, 0, padded - vocab)), requires_grad=False
        )


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def build_prefill_mask(batch_size, padded_len, max_cache_len, prompt_offset):
    mask = torch.zeros((batch_size, 1, padded_len, max_cache_len), dtype=torch.float16)
    mask[:, :, :, :prompt_offset] = -torch.inf
    for i in range(padded_len):
        mask[:, :, i, i + 1 :] = -torch.inf
    return mask


def build_expansion_mask(
    batch_size, block_size, max_cache_len, used_cache_len, prompt_offset
):
    mask = torch.zeros((batch_size, 1, block_size, max_cache_len), dtype=torch.float16)
    mask[:, :, :, :prompt_offset] = -torch.inf
    for j in range(block_size):
        attend_up_to = used_cache_len - block_size + j + 1
        mask[:, :, j, attend_up_to:] = -torch.inf
    return mask


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def generate(
    run_forward_fn, model, tokenizer, prompts, max_new_tokens=128, do_sample=False
):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    batch_size = input_ids.shape[0]
    prompt_length = input_ids.shape[1]

    padded_len = math.ceil(prompt_length / BLOCK_SIZE) * BLOCK_SIZE
    prompt_offset = padded_len - prompt_length
    max_cache_len = padded_len + math.ceil(max_new_tokens / BLOCK_SIZE) * BLOCK_SIZE
    if prompt_offset > 0:
        pad = input_ids.new_zeros((batch_size, prompt_offset))
        input_ids = torch.cat([pad, input_ids], dim=1)

    position_ids = torch.zeros((batch_size, padded_len), dtype=torch.long)
    position_ids[:, prompt_offset:] = torch.arange(prompt_length)

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = (
        getattr(model, "_spyre_head_dim", None)
        or getattr(model.config, "head_dim", None)
        or model.config.hidden_size // model.config.num_attention_heads
    )
    v_head_dim = getattr(model, "_spyre_v_head_dim", head_dim)
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

    result = input_ids.clone()
    current_cache_len = padded_len
    tokens_in_block = BLOCK_SIZE - 1
    decode_pos = None
    fill_mask_device = None
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    num_generated = 0

    for i in range(max_new_tokens):
        if i == 0:
            prefill_mask = build_prefill_mask(
                batch_size, padded_len, max_cache_len, prompt_offset
            )
            logits = run_forward_fn(
                model,
                input_ids.to(DEVICE),
                position_ids.to(DEVICE),
                prefill_mask.to(DEVICE),
                key_caches,
                value_caches,
                is_filling=False,
                token_index=0,
                cache_position=0,
            )
            logits_cpu = logits.to("cpu")
            next_logits = logits_cpu[0, -1, :]
            current_cache_len = padded_len
            decode_pos = torch.zeros((batch_size, BLOCK_SIZE), dtype=torch.long)
            for j in range(BLOCK_SIZE):
                decode_pos[:, j] = prompt_length + j - BLOCK_SIZE
        else:
            is_filling = tokens_in_block > 0
            next_input = result[:, -BLOCK_SIZE:].to(DEVICE)

            if is_filling:
                fill_pos = current_cache_len - BLOCK_SIZE + tokens_in_block
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
                grab_idx = BLOCK_SIZE - tokens_in_block
                next_logits = logits_cpu[0, -grab_idx, :]
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
                next_logits = logits_cpu[0, -BLOCK_SIZE, :]
                fill_mask_device = exp_mask.to(DEVICE)

        next_val = torch.argmax(next_logits).unsqueeze(0).unsqueeze(0)

        if tokens_in_block == BLOCK_SIZE - 1:
            result = F.pad(result, (0, BLOCK_SIZE))
        tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
        grab_idx = (BLOCK_SIZE - tokens_in_block) if tokens_in_block > 0 else BLOCK_SIZE
        result[:, -grab_idx] = next_val.squeeze()
        num_generated += 1

        if eos_token_id is not None and next_val.item() == eos_token_id:
            break

    all_gen_ids = []
    block_start = padded_len
    remaining = num_generated
    while remaining > 0:
        take = min(remaining, BLOCK_SIZE)
        for j in range(take):
            all_gen_ids.append(result[0, block_start + j].item())
        remaining -= take
        block_start += BLOCK_SIZE

    gen_ids = torch.tensor(all_gen_ids)
    if eos_token_id is not None:
        eos_pos = (gen_ids == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            gen_ids = gen_ids[: eos_pos[0].item()]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Qwen3 adapter
# ---------------------------------------------------------------------------


def _qwen3_make_compiled_block(layer):
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    q_norm = attn.q_norm
    k_norm = attn.k_norm

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        is_filling,
        token_index,
        cache_position,
    ):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape
        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)

        q = q_norm(q)
        k = k_norm(k)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            is_filling,
            token_index,
            cache_position,
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=attn.scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out
        residual = h
        h = post_attn_ln(h)
        h = mlp(h)
        h = residual + h

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def qwen3_run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    h = model.model.embed_tokens(input_ids)
    selected_freqs = model._spyre_rope(h, position_ids)
    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            is_filling,
            token_index,
            cache_position,
        )
    h = model.model.norm(h)
    logits = model.lm_head(h)
    return logits


def prepare_qwen3(model):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    model._spyre_rope = PrecomputedRotaryEmbedding(model.model.rotary_emb)
    patch_rmsnorm(Qwen3RMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        _qwen3_make_compiled_block(layer) for layer in model.model.layers
    ]


# ---------------------------------------------------------------------------
# Granite adapter
# ---------------------------------------------------------------------------


def _granite_make_compiled_block(layer):
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    res_mult = layer.residual_multiplier
    v_head_dim = getattr(attn, "v_head_dim", attn.head_dim)

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        is_filling,
        token_index,
        cache_position,
    ):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape
        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, -1, v_head_dim).transpose(1, 2)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            is_filling,
            token_index,
            cache_position,
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=attn.scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out * res_mult
        residual = h
        h = post_attn_ln(h)
        h = mlp(h)
        h = residual + h * res_mult

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def granite_run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    h = model.model.embed_tokens(input_ids)
    h = h * model.model.embedding_multiplier
    selected_freqs = model._spyre_rope(h, position_ids)
    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            is_filling,
            token_index,
            cache_position,
        )
    h = model.model.norm(h)
    logits = model.lm_head(h)
    logits = logits / model.config.logits_scaling
    return logits


def prepare_granite(model):
    from transformers.models.granite.modeling_granite import GraniteRMSNorm

    cfg = model.config
    orig_head_dim = getattr(
        cfg,
        "head_dim",
        cfg.hidden_size // cfg.num_attention_heads,
    )

    padded_head_dim = None
    stick_aligned_head_dim = (
        (orig_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)
    ) * (2 * BLOCK_SIZE)
    if stick_aligned_head_dim > orig_head_dim:
        padded_head_dim = stick_aligned_head_dim
        pad_attention_heads(
            model,
            model.model.layers,
            orig_head_dim,
            padded_head_dim,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        )

    model._spyre_rope = PrecomputedRotaryEmbedding(
        model.model.rotary_emb,
        padded_head_dim=padded_head_dim,
    )
    patch_rmsnorm(GraniteRMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        _granite_make_compiled_block(layer) for layer in model.model.layers
    ]


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------


def _patch_torch_empty():
    _orig = torch.empty

    def _patched(*args, size=None, **kwargs):
        if size is not None:
            return _orig(size, **kwargs)
        return _orig(*args, **kwargs)

    if getattr(torch.empty, "_hf_adapters_patched", False):
        return
    torch.empty = _patched
    torch.empty._hf_adapters_patched = True


def load_model(model_path, prepare_fn, dtype=torch.float16):
    from transformers import AutoModelForCausalLM

    _patch_torch_empty()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cpu",
    )
    model.eval()
    model.requires_grad_(False)
    prepare_fn(model)
    model.to(DEVICE)
    return model
