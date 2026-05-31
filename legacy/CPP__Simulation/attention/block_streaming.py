from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from attention.utils import safe_exp_difference


def block_streaming_mqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    causal: bool = True,
    exp_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Stream K/V in blocks while maintaining online-softmax state across blocks.

    Q shape: [batch, q_len, num_heads, d_k]
    K shape: [batch, kv_len, num_kv_heads, d_k]
    V shape: [batch, kv_len, num_kv_heads, d_k]
    Output:  [batch, q_len, num_heads, d_k]

    Unlike the fully materialized baseline, this kernel only forms scores for a
    single KV block at a time:
      block_scores shape: [batch, q_len, num_heads, block_len]

    The online state is carried across blocks:
      m: running max score
      l: running shifted softmax denominator
      o: running weighted value accumulator
    """
    if exp_fn is None:
        exp_fn = torch.exp
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v to be rank-4 tensors.")
    if k.shape != v.shape:
        raise ValueError("k and v must have matching shapes.")

    batch_size, q_len, num_heads, d_k = q.shape
    _, kv_len, num_kv_heads, kv_d = k.shape
    if kv_d != d_k:
        raise ValueError("d_k must match for q, k, v.")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads.")

    group_size = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(d_k)

    # Online softmax state, maintained per query token and per query head.
    # m, l: [batch, q_len, num_heads]
    # o:    [batch, q_len, num_heads, d_k]
    m = torch.full((batch_size, q_len, num_heads), float("-inf"), dtype=q.dtype, device=q.device)
    l = torch.zeros((batch_size, q_len, num_heads), dtype=q.dtype, device=q.device)
    o = torch.zeros((batch_size, q_len, num_heads, d_k), dtype=q.dtype, device=q.device)

    for block_start in range(0, kv_len, block_size):
        block_end = min(block_start + block_size, kv_len)
        block_len = block_end - block_start

        # K/V block shapes before MQA head expansion:
        # k_block, v_block: [batch, block_len, num_kv_heads, d_k]
        k_block = k[:, block_start:block_end, :, :]
        v_block = v[:, block_start:block_end, :, :]

        # Expand shared KV heads so each query head sees its aligned KV head.
        # Shapes after expansion:
        # k_block_expanded, v_block_expanded: [batch, block_len, num_heads, d_k]
        k_block_expanded = k_block.repeat_interleave(group_size, dim=2)
        v_block_expanded = v_block.repeat_interleave(group_size, dim=2)

        # Materialize only one score block at a time, never the full [B, H, N, N].
        # q:                [batch, q_len,   num_heads, d_k]
        # k_block_expanded: [batch, block_len, num_heads, d_k]
        # block_scores:     [batch, q_len,   num_heads, block_len]
        block_scores = torch.einsum("bqhd,bkhd->bqhk", q, k_block_expanded) * scale

        if causal:
            # Query t can only see keys with absolute positions <= t.
            q_idx = torch.arange(q_len, device=q.device).view(1, q_len, 1, 1)
            k_idx = torch.arange(block_start, block_end, device=q.device).view(1, 1, 1, block_len)
            valid = k_idx <= q_idx
            block_scores = block_scores.masked_fill(~valid, float("-inf"))

        # Block-local softmax summary:
        # block_m: [batch, q_len, num_heads]
        # block_l: [batch, q_len, num_heads]
        # block_o: [batch, q_len, num_heads, d_k]
        block_m = block_scores.max(dim=-1).values
        block_weights = safe_exp_difference(block_scores, block_m.unsqueeze(-1), exp_fn)
        block_l = block_weights.sum(dim=-1)
        block_o = torch.einsum("bqhk,bkhd->bqhd", block_weights, v_block_expanded)

        # Merge the block summary into the running online-softmax state.
        m_new = torch.maximum(m, block_m)
        old_scale = safe_exp_difference(m, m_new, exp_fn)
        block_scale = safe_exp_difference(block_m, m_new, exp_fn)

        l = l * old_scale + block_l * block_scale
        o = o * old_scale.unsqueeze(-1) + block_o * block_scale.unsqueeze(-1)
        m = m_new

    return o / l.clamp_min(torch.finfo(o.dtype).tiny).unsqueeze(-1)
