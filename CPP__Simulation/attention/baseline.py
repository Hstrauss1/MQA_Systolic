from __future__ import annotations

import math

import torch

from attention.utils import repeat_kv_for_mqa


def baseline_mqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Baseline materialized Multi-Query Attention.

    Q shape: [batch, q_len, num_heads, d_k]
    K shape: [batch, kv_len, num_kv_heads, d_k]
    V shape: [batch, kv_len, num_kv_heads, d_k]

    Output shape: [batch, q_len, num_heads, d_k]

    This version explicitly materializes the score tensor:
    scores shape: [batch, num_heads, q_len, kv_len]
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v to be rank-4 tensors.")
    if k.shape != v.shape:
        raise ValueError("k and v must have matching shapes.")

    batch_size, q_len, num_heads, d_k = q.shape
    k_batch, kv_len, _, k_dim = k.shape
    if batch_size != k_batch or d_k != k_dim:
        raise ValueError("Batch size and d_k must match between q and k/v.")

    # MQA dataflow: many Q heads consume a smaller number of shared KV heads.
    # We expand the KV heads so the batched matmul can operate head-by-head.
    k_expanded = repeat_kv_for_mqa(k, num_heads)
    v_expanded = repeat_kv_for_mqa(v, num_heads)

    # Rearranged for batched matrix multiplication:
    # q_t: [batch, num_heads, q_len, d_k]
    # k_t: [batch, num_heads, kv_len, d_k]
    q_t = q.permute(0, 2, 1, 3)
    k_t = k_expanded.permute(0, 2, 1, 3)
    v_t = v_expanded.permute(0, 2, 1, 3)

    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

    if causal:
        # Causal masking blocks future keys for every query position.
        # mask shape: [q_len, kv_len]
        mask = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v_t)
    return output.permute(0, 2, 1, 3).contiguous()
