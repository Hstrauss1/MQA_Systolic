"""Analytical baseline for MQA mapped as two GEMMs.

SCALE-Sim should be used to model the GEMM-style baseline via topology CSVs and
the `-i gemm` option. This file is a lightweight analytical companion for quick
comparisons and is not a replacement for SCALE-Sim reports.

Two variants are supported via the `fused` flag:

  fused=False  (default) — standard unfused GPU attention:
      Q·Kᵀ → write scores to HBM → softmax → read scores → scores·V
      Score matrix is H×B×Q_T×T elements at score_bytes_per_element each.
      This dominates DRAM traffic at large T (17 GB at T=8192, H=64, full prefill).

  fused=True  — FlashAttention-style fused kernel:
      No intermediate score tensor written to HBM; softmax is computed
      tile-by-tile on-chip. DRAM traffic = Q + K + V + output only.
      64× less DRAM than unfused at T=8192 full prefill.

Score precision note: GPUs often accumulate Q·Kᵀ scores in FP32 even when
inputs are BF16/FP16. Use score_bytes_per_element=4 to model this; default
matches bytes_per_element (BF16 in, BF16 scores).
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict


def baseline_mqa_metrics(
    H: int,
    T: int,
    d: int,
    array_rows: int,
    array_cols: int,
    bytes_per_element: int,
    memory_bandwidth_bytes_per_cycle: int,
    batch_size: int = 1,
    query_tokens: int = 1,
    fused: bool = False,
    score_bytes_per_element: int | None = None,
    generate_tokens: int = 0,
) -> Dict[str, float]:
    """Estimate the two-GEMM baseline costs for MQA.

    Parameters
    ----------
    H : number of query heads (K/V shared across all heads — MQA)
    T : KV sequence length (number of tokens in the KV cache)
    d : head dimension
    array_rows, array_cols : systolic array size; throughput = rows×cols MACs/cycle
    bytes_per_element : data width in bytes (2 = BF16/FP16, 4 = FP32)
    memory_bandwidth_bytes_per_cycle : peak DRAM bandwidth in bytes per clock cycle
    batch_size : B independent sequences decoded simultaneously
    query_tokens : Q_T query tokens per head per sequence (1 = decode, T = prefill)
    fused : if True, model FlashAttention-style fusion — no score matrix HBM traffic
    score_bytes_per_element : precision of intermediate score tensor (default = bytes_per_element).
        Set to 4 to model FP32 score accumulation on GPU even when inputs are BF16.

    Memory traffic
    --------------
    Unfused (fused=False):
        Q reads        = H × B × Q_T × d × bpe
        K reads        = B × T × d × bpe          (MQA: shared, no H factor)
        V reads        = B × T × d × bpe          (MQA: shared, no H factor)
        Score wr+rd    = H × B × Q_T × T × sbpe × 2
        Output writes  = H × B × Q_T × d × bpe
    Fused (fused=True):
        Score wr+rd    = 0  (computed tile-by-tile on-chip)
        All other terms identical.
    """
    sbpe = score_bytes_per_element if score_bytes_per_element is not None else bytes_per_element

    # ── Compute (MACs) ──────────────────────────────────────────────────────
    # GEMM1: [H×B×Q_T, d] × [d, T]  →  scores [H×B×Q_T, T]
    qk_macs = H * batch_size * query_tokens * T * d
    # GEMM2: [H×B×Q_T, T] × [T, d]  →  output [H×B×Q_T, d]
    av_macs = H * batch_size * query_tokens * T * d
    # Softmax: exp + sum + div ≈ 5 ops per score element (memory-bound, not GEMM)
    softmax_ops = 5 * H * batch_size * query_tokens * T
    total_macs  = qk_macs + av_macs          # GEMM MACs only (softmax excluded from roofline compute)

    # ── Memory traffic ──────────────────────────────────────────────────────
    q_reads          = H * batch_size * query_tokens * d * bytes_per_element
    k_reads          = batch_size * T * d * bytes_per_element      # MQA: no H
    v_reads          = batch_size * T * d * bytes_per_element      # MQA: no H
    score_write_read = 0 if fused else H * batch_size * query_tokens * T * sbpe * 2
    output_writes    = H * batch_size * query_tokens * d * bytes_per_element
    total_dram_bytes = q_reads + k_reads + v_reads + score_write_read + output_writes

    # ── Roofline ────────────────────────────────────────────────────────────
    ideal_compute_cycles = math.ceil(total_macs / (array_rows * array_cols))
    memory_cycles        = math.ceil(total_dram_bytes / memory_bandwidth_bytes_per_cycle)
    estimated_cycles     = max(ideal_compute_cycles, memory_cycles)
    arithmetic_intensity = total_macs / total_dram_bytes if total_dram_bytes else 0

    # ── Autoregressive generation metrics ──────────────────────────────────
    # When generate_tokens > 0, model a full prefill-then-decode run.
    # Baseline reloads K and V from DRAM every decode step (no on-chip KV cache).
    # Prefill cost = this pass; decode cost = G × single-token attention pass.
    gen_metrics: Dict[str, float] = {}
    if generate_tokens > 0:
        sbpe_d = sbpe
        # Per-step decode DRAM (query_tokens=1, same fused flag)
        dec_q   = H * batch_size * 1 * d * bytes_per_element
        dec_k   = batch_size * T * d * bytes_per_element
        dec_v   = batch_size * T * d * bytes_per_element
        dec_sc  = 0 if fused else H * batch_size * 1 * T * sbpe_d * 2
        dec_out = H * batch_size * 1 * d * bytes_per_element
        dec_dram = dec_q + dec_k + dec_v + dec_sc + dec_out
        # Per-step decode compute
        dec_macs = 2 * H * batch_size * 1 * T * d
        dec_compute = math.ceil(dec_macs / (array_rows * array_cols))
        dec_mem_cyc = math.ceil(dec_dram / memory_bandwidth_bytes_per_cycle)
        dec_cycles  = max(dec_compute, dec_mem_cyc)

        total_gen_dram  = total_dram_bytes + generate_tokens * dec_dram
        total_gen_cycles = estimated_cycles + generate_tokens * dec_cycles
        gen_metrics = {
            "generate_tokens":             generate_tokens,
            "decode_dram_bytes_per_step":  dec_dram,
            "decode_cycles_per_step":      dec_cycles,
            "total_generate_dram_bytes":   total_gen_dram,
            "total_generate_dram_mb":      total_gen_dram / (1024 * 1024),
            "total_generate_cycles":       total_gen_cycles,
            "dram_per_token_kb":           dec_dram / 1024,
        }

    return {
        "H": H,
        "T": T,
        "d": d,
        "array_rows": array_rows,
        "array_cols": array_cols,
        "bytes_per_element": bytes_per_element,
        "score_bytes_per_element": sbpe,
        "memory_bandwidth_bytes_per_cycle": memory_bandwidth_bytes_per_cycle,
        "fused": fused,
        # macs
        "qk_macs": qk_macs,
        "av_macs": av_macs,
        "softmax_ops": softmax_ops,
        "total_macs": total_macs,
        # memory
        "q_reads_bytes": q_reads,
        "k_reads_bytes": k_reads,
        "v_reads_bytes": v_reads,
        "score_write_read_bytes": score_write_read,
        "output_writes_bytes": output_writes,
        "total_dram_bytes": total_dram_bytes,
        # cycles
        "ideal_compute_cycles": ideal_compute_cycles,
        "memory_cycles": memory_cycles,
        "estimated_cycles": estimated_cycles,
        "arithmetic_intensity": arithmetic_intensity,
        **gen_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate analytical baseline MQA performance."
    )
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--array-rows", type=int, default=64)
    parser.add_argument("--array-cols", type=int, default=64)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--memory-bandwidth-bytes-per-cycle", type=int, default=512)
    parser.add_argument("--fused", action="store_true",
                        help="Model FlashAttention-style fusion (no score HBM traffic)")
    parser.add_argument("--score-bytes-per-element", type=int, default=None,
                        help="Score tensor precision in bytes (default matches --bytes-per-element). "
                             "Use 4 to model FP32 score accumulation on GPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = baseline_mqa_metrics(
        H=args.heads,
        T=args.seq_len,
        d=args.head_dim,
        array_rows=args.array_rows,
        array_cols=args.array_cols,
        bytes_per_element=args.bytes_per_element,
        memory_bandwidth_bytes_per_cycle=args.memory_bandwidth_bytes_per_cycle,
        fused=args.fused,
        score_bytes_per_element=args.score_bytes_per_element,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
