from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from attention.utils import safe_exp_difference


def streaming_mqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    exp_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Streaming Multi-Query Attention with online softmax propagation.

    Q shape: [batch, q_len, num_heads, d_k]
    K shape: [batch, kv_len, num_kv_heads, d_k]
    V shape: [batch, kv_len, num_kv_heads, d_k]
    Output:  [batch, q_len, num_heads, d_k]

    Online softmax state for each (batch, query_position, query_head):
    - m: running maximum score
    - l: running denominator in the shifted softmax domain
    - o: running numerator-vector accumulation

    At key step i:
      m_out = max(m_in, s_i)
      l_out = l_in * exp(m_in - m_out) + exp(s_i - m_out)
      o_out = o_in * exp(m_in - m_out) + v_i * exp(s_i - m_out)

    Final attention output is o / l.
    """
    if exp_fn is None:
        exp_fn = torch.exp

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

    # State is maintained per query token and per query head.
    m = torch.full(
        (batch_size, q_len, num_heads),
        fill_value=float("-inf"),
        dtype=q.dtype,
        device=q.device,
    )
    l = torch.zeros((batch_size, q_len, num_heads), dtype=q.dtype, device=q.device)
    o = torch.zeros((batch_size, q_len, num_heads, d_k), dtype=q.dtype, device=q.device)

    # Stream over keys one position at a time. The full q_len x kv_len score matrix
    # is never materialized. Each iteration only holds scores for one key position:
    # score_t shape: [batch, q_len, num_heads]
    for key_idx in range(kv_len):
        k_i = k[:, key_idx, :, :]  # [batch, num_kv_heads, d_k]
        v_i = v[:, key_idx, :, :]  # [batch, num_kv_heads, d_k]

        # Broadcast each shared KV head across its query-head group.
        k_i_expanded = k_i.repeat_interleave(group_size, dim=1)  # [batch, num_heads, d_k]
        v_i_expanded = v_i.repeat_interleave(group_size, dim=1)  # [batch, num_heads, d_k]

        # Compare the single streamed key against every active query token/head.
        score_t = (q * k_i_expanded.unsqueeze(1)).sum(dim=-1) * scale

        if causal:
            # Query position t may only see keys up to and including t.
            valid = torch.arange(q_len, device=q.device) >= key_idx
            score_t = score_t.masked_fill(~valid.view(1, q_len, 1), float("-inf"))

        m_new = torch.maximum(m, score_t)
        old_scale = safe_exp_difference(m, m_new, exp_fn)
        new_scale = safe_exp_difference(score_t, m_new, exp_fn)

        l = l * old_scale + new_scale
        o = o * old_scale.unsqueeze(-1) + v_i_expanded.unsqueeze(1) * new_scale.unsqueeze(-1)
        m = m_new

    return o / l.clamp_min(torch.finfo(o.dtype).tiny).unsqueeze(-1)
