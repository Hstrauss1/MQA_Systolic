"""Analytical baseline for MQA mapped as two GEMMs.

SCALE-Sim should be used to model the GEMM-style baseline via topology CSVs and
the `-i gemm` option. This file is a lightweight analytical companion for quick
comparisons and is not a replacement for SCALE-Sim reports.
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
) -> Dict[str, float]:
    """Estimate the two-GEMM baseline costs for MQA.

    With batch_size B: B independent sequences are decoded simultaneously.
    Each sequence has its own KV cache (K/V shared across heads per sequence).
    With query_tokens Q_T: each head attends Q_T query tokens over T KV tokens.
    Q_T=1 is decode; Q_T=T is full prefill.
    """
    qk_macs = H * batch_size * query_tokens * T * d
    av_macs = H * batch_size * query_tokens * T * d
    total_macs = qk_macs + av_macs

    q_reads = H * batch_size * query_tokens * d * bytes_per_element
    k_reads = batch_size * T * d * bytes_per_element
    v_reads = batch_size * T * d * bytes_per_element
    score_write_read = H * batch_size * query_tokens * T * bytes_per_element * 2
    output_writes = H * batch_size * query_tokens * d * bytes_per_element
    total_dram_bytes = q_reads + k_reads + v_reads + score_write_read + output_writes

    ideal_compute_cycles = math.ceil(total_macs / (array_rows * array_cols))
    memory_cycles = math.ceil(total_dram_bytes / memory_bandwidth_bytes_per_cycle)
    estimated_cycles = max(ideal_compute_cycles, memory_cycles)
    arithmetic_intensity = total_macs / total_dram_bytes

    return {
        "H": H,
        "T": T,
        "d": d,
        "array_rows": array_rows,
        "array_cols": array_cols,
        "bytes_per_element": bytes_per_element,
        "memory_bandwidth_bytes_per_cycle": memory_bandwidth_bytes_per_cycle,
        "qk_macs": qk_macs,
        "av_macs": av_macs,
        "total_macs": total_macs,
        "q_reads_bytes": q_reads,
        "k_reads_bytes": k_reads,
        "v_reads_bytes": v_reads,
        "score_write_read_bytes": score_write_read,
        "output_writes_bytes": output_writes,
        "total_dram_bytes": total_dram_bytes,
        "ideal_compute_cycles": ideal_compute_cycles,
        "memory_cycles": memory_cycles,
        "estimated_cycles": estimated_cycles,
        "arithmetic_intensity": arithmetic_intensity,
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
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
