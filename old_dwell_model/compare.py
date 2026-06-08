"""Compare baseline GEMM-style MQA against the KV-stationary analytical model.

The baseline models the standard two-GEMM decomposition (Q@K^T, then scores@V)
on a dense systolic array using a roofline bound: max(compute, memory).

The KV-stationary model uses actual pipeline timing (fill + vertical-K-fill +
steady + drain) so its cycle count is always >= its own roofline bound.

Speedup is computed roofline-vs-roofline so both architectures are on the same
footing. The pipeline_efficiency column shows how much the KV-stationary design
loses to drain overhead versus its own ideal.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

from baseline_mqa_model import baseline_mqa_metrics
from kv_stationary_model import kv_stationary_metrics


DEFAULT_SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096, 8192]
DEFAULT_H = 64
DEFAULT_D = 128
DEFAULT_ARRAY_ROWS = 64
DEFAULT_ARRAY_COLS = 64
DEFAULT_BYTES_PER_ELEMENT = 2
DEFAULT_MEMORY_BANDWIDTH_BYTES_PER_CYCLE = 512
DEFAULT_PE_MAC_WIDTH = 128   # fully parallel upper PE — reasonable hardware target
DEFAULT_LOWER_MAC_COUNT = 1  # interleaved MACs in lower dot-product unit
DEFAULT_EXP_LATENCY = 4
DEFAULT_BATCH_SIZE = 1
DEFAULT_HEAD_PARALLELISM = 1
DEFAULT_MERGE_EXTENSIONS = 0
RESULTS_CSV = "results.csv"


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def kv_roofline_cycles(kv: Dict, array_rows: int, array_cols: int,
                       pe_mac_width: int, memory_bandwidth: int) -> int:
    """Two-resource roofline bound for the KV-stationary architecture.

    The architecture has two distinct compute units:

    1. Lower MAC array: array_rows × array_cols × lower_mac_count MAC units —
       each unit handles one Q·K dot product serially (d cycles), so total
       throughput = array_rows * array_cols * lower_mac_count MACs/cycle.
    2. Upper Running Attention PE (one per row, pe_mac_width MACs each) —
       handles Oout = Oin*exp_old + exp_new*V (3*d ops per pair).

    The roofline is the max of the three bottlenecks:
        max(lower_mac_bound, upper_pe_bound, memory_bound)
    """
    lower_mac_count = kv.get("lower_mac_count", 1)
    lower_mac_bound = math.ceil(kv["dot_product_macs"] / (array_rows * array_cols * lower_mac_count))
    upper_pe_bound  = math.ceil(kv["value_macs"] / (array_rows * pe_mac_width))
    memory_bound    = math.ceil(kv["total_dram_bytes"] / memory_bandwidth)
    return max(lower_mac_bound, upper_pe_bound, memory_bound)


def build_result_row(
    T: int,
    pe_mac_width: int,
    lower_mac_count: int,
    exp_latency: int,
    batch_size: int,
    head_parallelism: int,
    query_tokens: int,
    merge_extensions: int = 0,
) -> Dict[str, float]:
    # KV-stationary array dimensions scale with merge_extensions:
    #   rows = H * 2^n  (one row per lane in each sub-array level)
    #   cols = T / 2^n  (each row handles T/2^n tokens — one tile total per section)
    # This ensures drain = T/2^n - 1 steps instead of array_cols - 1 repeated over many tiles.
    eff_rows = DEFAULT_H * (2 ** merge_extensions)
    eff_cols = max(1, T // (2 ** merge_extensions))

    # Baseline always uses the standard dense array (fixed reference hardware).
    baseline = baseline_mqa_metrics(
        H=DEFAULT_H,
        T=T,
        d=DEFAULT_D,
        array_rows=DEFAULT_ARRAY_ROWS,
        array_cols=DEFAULT_ARRAY_COLS,
        bytes_per_element=DEFAULT_BYTES_PER_ELEMENT,
        memory_bandwidth_bytes_per_cycle=DEFAULT_MEMORY_BANDWIDTH_BYTES_PER_CYCLE,
        batch_size=batch_size,
        query_tokens=query_tokens,
    )
    kv = kv_stationary_metrics(
        H=DEFAULT_H,
        T=T,
        d=DEFAULT_D,
        array_rows=eff_rows,
        array_cols=eff_cols,
        bytes_per_element=DEFAULT_BYTES_PER_ELEMENT,
        memory_bandwidth_bytes_per_cycle=DEFAULT_MEMORY_BANDWIDTH_BYTES_PER_CYCLE,
        pe_mac_width=pe_mac_width,
        lower_mac_count=lower_mac_count,
        exp_latency_cycles=exp_latency,
        batch_size=batch_size,
        head_parallelism=head_parallelism,
        merge_extensions=merge_extensions,
        query_tokens=query_tokens,
    )

    baseline_roofline  = baseline["estimated_cycles"]           # already max(compute,mem)
    kv_roofline        = kv_roofline_cycles(
        kv, eff_rows, eff_cols,
        pe_mac_width,
        DEFAULT_MEMORY_BANDWIDTH_BYTES_PER_CYCLE,
    )
    baseline_dram_mb   = bytes_to_mb(baseline["total_dram_bytes"])
    kv_dram_mb         = bytes_to_mb(kv["total_dram_bytes"])

    # latency_tbot    = pipeline_latency_cycles  (single-sequence TBOT)
    #                 = (active_rows-1)*row_stagger + active_cols*column_dwell
    # throughput_tbot = row_stagger              (sustained rate, pipeline full)
    # pipeline_depth  = active_cols + active_rows - 1  (steps to fill pipeline)
    throughput_tbot = kv["throughput_cycles_per_token"]
    latency_tbot    = kv["pipeline_latency_cycles"]
    pipeline_depth  = kv["pipeline_depth_steps"]

    return {
        "T": T,
        # --- cycles: two distinct TBOT metrics ---
        "baseline_roofline_cycles":    baseline_roofline,
        "kv_roofline_cycles":          kv_roofline,
        "kv_latency_tbot":             latency_tbot,    # single-sequence TBOT
        "kv_throughput_tbot":          throughput_tbot, # batched/continuous TBOT
        "pipeline_depth":              pipeline_depth,  # steps to fill pipeline
        # --- speedup (roofline vs roofline) ---
        "roofline_speedup":            baseline_roofline / kv_roofline,
        # --- memory ---
        "baseline_dram_MB":            baseline_dram_mb,
        "kv_dram_MB":                  kv_dram_mb,
        "memory_reduction_ratio":      baseline_dram_mb / kv_dram_mb,
        # --- arithmetic intensity ---
        "baseline_AI":                 baseline["arithmetic_intensity"],
        "kv_AI":                       kv["arithmetic_intensity"],
        # --- hardware params ---
        "column_dwell_cyc":            kv["column_dwell"],
        "packet_stagger_cyc":          kv["packet_stagger"],
        "lower_mac_cyc":               kv["lower_mac_throughput_cycles"],
        "upper_pe_cyc":                kv["upper_pe_cycles_per_stage"],
        # --- on-chip K buffer cost ---
        "k_buf_per_pe_KB":             kv["k_buffer_bytes_per_pe"] / 1024,
        "total_k_buf_MB":              kv["total_k_buffer_bytes"] / (1024 * 1024),
        # --- merge extension ---
        "merge_stage_cyc":             kv["merge_stage_cycles"],
        "skew_per_level":              kv["skew_per_level"],
        "sync_buffer_KB":              kv["sync_buffer_bytes"] / 1024,
    }


HEADERS = [
    "T",
    "baseline_roofline_cycles",
    "kv_roofline_cycles",
    "kv_latency_tbot",       # single-sequence TBOT  = pipeline depth × cps
    "kv_throughput_tbot",    # batched/continuous TBOT = cycles_per_stage
    "pipeline_depth",        # steps to fill pipeline = latency_tbot / cps
    "roofline_speedup",
    "baseline_dram_MB",
    "kv_dram_MB",
    "memory_reduction_ratio",
    "baseline_AI",
    "kv_AI",
    "column_dwell_cyc",
    "packet_stagger_cyc",
    "lower_mac_cyc",
    "upper_pe_cyc",
    "k_buf_per_pe_KB",
    "total_k_buf_MB",
    "merge_stage_cyc",
    "skew_per_level",
    "sync_buffer_KB",
]


def format_value(key: str, val: float) -> str:
    if key in ("T", "pipeline_depth") or key.endswith("_cycles") or key.endswith("_cyc") or key.endswith("_tbot"):
        return f"{int(val)}"
    if key.endswith("_MB"):
        return f"{val:.2f}"
    if key.endswith("_KB"):
        return f"{val:.2f}"
    return f"{val:.4f}"


def format_table(rows: List[Dict[str, float]]) -> str:
    formatted = [
        [format_value(h, row[h]) for h in HEADERS]
        for row in rows
    ]
    widths = [
        max(len(h), max(len(r[i]) for r in formatted))
        for i, h in enumerate(HEADERS)
    ]
    sep   = "  ".join("-" * w for w in widths)
    hdr   = "  ".join(h.ljust(widths[i]) for i, h in enumerate(HEADERS))
    lines = ["  ".join(r[i].ljust(widths[i]) for i in range(len(HEADERS))) for r in formatted]
    return "\n".join([hdr, sep, *lines])


def write_results_csv(rows: List[Dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline vs KV-stationary MQA.")
    p.add_argument("--pe-mac-width", type=int, default=DEFAULT_PE_MAC_WIDTH,
                   help=f"Parallel MACs in upper PE (default {DEFAULT_PE_MAC_WIDTH})")
    p.add_argument("--exp-latency", type=int, default=DEFAULT_EXP_LATENCY,
                   help=f"Exp lookup latency in cycles (default {DEFAULT_EXP_LATENCY})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Decode batch size (default {DEFAULT_BATCH_SIZE})")
    p.add_argument("--lower-mac-count", type=int, default=DEFAULT_LOWER_MAC_COUNT,
                   help=f"Interleaved MACs in lower dot-product unit (default {DEFAULT_LOWER_MAC_COUNT})")
    p.add_argument("--head-parallelism", type=int, default=DEFAULT_HEAD_PARALLELISM,
                   help=f"Row lanes per head (default {DEFAULT_HEAD_PARALLELISM})")
    p.add_argument("--merge-extensions", type=int, default=DEFAULT_MERGE_EXTENSIONS,
                   help=f"Inline merge-tree levels (default {DEFAULT_MERGE_EXTENSIONS}). "
                        "Each level doubles array rows and halves array cols. "
                        "n=3 is optimal for H=64, T=8192.")
    p.add_argument("--query-tokens", type=int, default=1,
                   help="Q tokens per head per sequence: 1=decode, T=prefill (default 1)")
    p.add_argument("--output", type=str, default=RESULTS_CSV,
                   help=f"Output CSV path (default {RESULTS_CSV})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        build_result_row(T, args.pe_mac_width, args.lower_mac_count, args.exp_latency,
                         args.batch_size, args.head_parallelism, args.query_tokens,
                         args.merge_extensions)
        for T in DEFAULT_SEQUENCE_LENGTHS
    ]
    print(f"pe_mac_width={args.pe_mac_width}  lower_mac_count={args.lower_mac_count}  "
          f"exp_latency={args.exp_latency}  batch_size={args.batch_size}  "
          f"head_parallelism={args.head_parallelism}  merge_extensions={args.merge_extensions}  "
          f"query_tokens={args.query_tokens}")
    print()
    print(format_table(rows))
    path = Path(args.output)
    write_results_csv(rows, path)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
