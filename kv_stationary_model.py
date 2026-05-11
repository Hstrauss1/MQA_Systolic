"""Analytical KV-stationary performance model for MQA inference.

This is an analytical extension for a custom architecture concept. SCALE-Sim
does not natively model KV-stationary execution, so this file intentionally
does not claim cycle accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict


def kv_stationary_metrics(
    H: int,
    T: int,
    d: int,
    array_rows: int,
    array_cols: int,
    bytes_per_element: int,
    memory_bandwidth_bytes_per_cycle: int,
) -> Dict[str, float]:
    """Estimate KV-stationary execution costs."""
    dot_product_macs = H * T * d
    value_macs = H * T * d
    total_macs = dot_product_macs + value_macs

    q_reads = H * d * bytes_per_element
    k_reads_initial_load = T * d * bytes_per_element
    v_reads_initial_load = T * d * bytes_per_element
    total_dram_bytes = q_reads + k_reads_initial_load + v_reads_initial_load

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
        "dot_product_macs": dot_product_macs,
        "value_macs": value_macs,
        "total_macs": total_macs,
        "q_reads_bytes": q_reads,
        "k_reads_initial_load_bytes": k_reads_initial_load,
        "v_reads_initial_load_bytes": v_reads_initial_load,
        "total_dram_bytes": total_dram_bytes,
        "ideal_compute_cycles": ideal_compute_cycles,
        "memory_cycles": memory_cycles,
        "estimated_cycles": estimated_cycles,
        "arithmetic_intensity": arithmetic_intensity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate analytical KV-stationary MQA performance."
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
    results = kv_stationary_metrics(
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
