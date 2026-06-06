"""Multi-pass silicon re-use extension to the 2D KV-stationary MQA analytical model.

Extends kv_stationary_model.py with a num_passes parameter.  Instead of one pass
over T columns, makes P passes over T/P columns using the same physical array.

After each pass, the partial softmax state (m, l, O) — a (d+2)-element vector —
is saved and re-injected at the start of the next pass via a single merge PE.

    total_cycles = P × single_pass_cycles(T/P) + (P-1) × reinject_cycles
    reinject_cycles = 2·exp_latency + ceil(6·d / pe_mac_width)

SRAM scales as 2 × (T/P) × d × bpe (K+V for the physical column count).
PE count scales as array_rows × (T/P).

Causal prefill (causal=True, query_tokens=T)
────────────────────────────────────────────
With a lower-triangular causal mask and P passes over C = T/P columns each, query
token i attends to KV columns 0..i.  Pass k covers columns [(k-1)·C .. k·C - 1].

  • Queries i < (k-1)·C  : entirely future columns → skip this pass (done already).
  • Queries i ∈ [(k-1)·C, k·C - 1] : the "diagonal" — complete their final pass here.
  • Queries i ≥ k·C      : full pass, carry state forward via reinject.

Active query count in pass k: Q_k = T − (k−1)·C  (decreasing each pass).
Queries that FINISH in pass k: C  (the diagonal slice — stream output immediately).

Per-pass DRAM:
  Q reads    = H · Q_k · d · bpe      (only active queries need Q vectors)
  KV reads   = 2 · C · d · bpe        (always load this pass's slice)
  Output     = H · C · d · bpe        (C queries write final O each pass)

Total causal cycles = Σ_k max(compute_k, memory_k) + (P−1) · reinject_cycles
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Sequence, Tuple

# ── Re-export unchanged helpers from the base model ──────────────────────────
from kv_stationary_model import (
    _queries_per_row,
    _compute_cycles_per_stage,
    _pipeline_phase_steps,
    _format_cycle_trace,
    simulate_2d_kv_stationary_array,
)


def kv_stationary_metrics(
    H: int,
    T: int,
    d: int,
    array_rows: int,
    array_cols: int,
    bytes_per_element: int,
    memory_bandwidth_bytes_per_cycle: int,
    exp_latency_cycles: int = 4,
    pe_mac_width: int = 1,
    lower_mac_count: int = 1,
    batch_size: int = 1,
    head_parallelism: int = 1,
    merge_extensions: int = 0,
    query_tokens: int = 1,
    generate_tokens: int = 0,
    num_passes: int = 1,
    causal: bool = False,
) -> Dict[str, object]:
    """Wrapper for the 2D KV-stationary model with optional multi-pass re-use.

    num_passes == 1
        Identical to the base model.  causal=True adds causal metrics but does
        not change cycle count (single-pass causal is a hardware masking concern,
        not a scheduling one).

    num_passes > 1, causal=False  (non-causal / decode)
        physical_cols = max(1, T // num_passes)
        One pass is simulated (T = physical_cols, one tile, no internal tiling).
        total_cycles_multipass = num_passes × single_pass_cycles + (num_passes−1) × reinject

    num_passes > 1, causal=True  (causal prefill)
        Per-pass computation with Q_k = T − (k−1)·C active queries.
        Pipeline depth term (H + C − 1)·column_dwell is unchanged; stagger term
        and DRAM both shrink as early queries complete and exit.
        Outputs are streamed: C queries write final O after EACH pass (not just last).
        total_cycles_causal = Σ_k max(compute_k, memory_k) + (num_passes−1) × reinject
    """
    reinject_cycles = 2 * exp_latency_cycles + math.ceil(6 * d / pe_mac_width)

    if num_passes <= 1:
        result = simulate_2d_kv_stationary_array(
            H=H,
            T=T,
            d=d,
            array_rows=array_rows,
            array_cols=array_cols,
            bytes_per_element=bytes_per_element,
            memory_bandwidth_bytes_per_cycle=memory_bandwidth_bytes_per_cycle,
            exp_latency_cycles=exp_latency_cycles,
            pe_mac_width=pe_mac_width,
            lower_mac_count=lower_mac_count,
            batch_size=batch_size,
            head_parallelism=head_parallelism,
            merge_extensions=merge_extensions,
            query_tokens=query_tokens,
            generate_tokens=generate_tokens,
        )
        result.update({
            "num_passes": 1,
            "physical_cols": array_cols,
            "reinject_cycles": reinject_cycles,
            "total_cycles_multipass": result["total_cycles"],
            "total_cycles_causal": result["total_cycles"],   # no benefit at P=1
            "causal": causal,
            "causal_cycles_per_pass": [result["total_cycles"]],
            "pe_count": array_rows * array_cols,
            "sram_bytes": 2 * array_cols * d * bytes_per_element,
        })
        return result

    physical_cols = max(1, T // num_passes)
    C = physical_cols

    # ── Non-causal single-pass baseline ──────────────────────────────────────
    # Called with T = physical_cols so the model runs exactly one tile (no
    # internal tiling).  query_tokens is the caller's value — for decode this
    # is 1; for prefill this is T (original).
    single_pass = simulate_2d_kv_stationary_array(
        H=H,
        T=C,
        d=d,
        array_rows=array_rows,
        array_cols=C,
        bytes_per_element=bytes_per_element,
        memory_bandwidth_bytes_per_cycle=memory_bandwidth_bytes_per_cycle,
        exp_latency_cycles=exp_latency_cycles,
        pe_mac_width=pe_mac_width,
        lower_mac_count=lower_mac_count,
        batch_size=batch_size,
        head_parallelism=head_parallelism,
        merge_extensions=merge_extensions,
        query_tokens=query_tokens,
        generate_tokens=0,
    )

    single_pass_cycles = single_pass["total_cycles"]
    total_cycles_multipass = (
        num_passes * single_pass_cycles + (num_passes - 1) * reinject_cycles
    )

    result = dict(single_pass)
    result.update({
        "num_passes": num_passes,
        "physical_cols": C,
        "reinject_cycles": reinject_cycles,
        "total_cycles_multipass": total_cycles_multipass,
        "causal": causal,
        "pe_count": array_rows * C,
        "sram_bytes": 2 * C * d * bytes_per_element,
    })

    # ── Causal per-pass computation ───────────────────────────────────────────
    # Recompute timing pass-by-pass with Q_k = T − (k−1)·C active queries and
    # per-pass DRAM sized to those active queries.
    col_dwell = single_pass["column_dwell"]
    upper_pe  = single_pass["upper_pe_cycles_per_stage"]
    act_rows  = single_pass["active_query_rows"]

    causal_pass_cycles: List[int] = []
    for k in range(1, num_passes + 1):
        Q_k = max(0, T - (k - 1) * C)

        # Compute: pipeline depth unchanged; stagger shrinks with fewer packets.
        eff_macs   = min(lower_mac_count, max(1, Q_k))
        stagger    = math.ceil(d / eff_macs)
        eff_stag   = max(stagger, upper_pe)
        compute_k  = (act_rows + C - 1) * col_dwell + max(0, Q_k - 1) * eff_stag

        if causal:
            # DRAM: only active queries re-read Q; C queries finish and write output.
            q_read_bytes = H * batch_size * Q_k * d * bytes_per_element
            kv_bytes     = 2 * C * d * bytes_per_element
            out_bytes    = H * batch_size * C * d * bytes_per_element
            memory_k = math.ceil(
                (q_read_bytes + kv_bytes + out_bytes) / memory_bandwidth_bytes_per_cycle
            )
            causal_pass_cycles.append(max(compute_k, memory_k))
        else:
            # Non-causal: store per-pass compute separately for reference.
            causal_pass_cycles.append(compute_k)

    total_cycles_causal = (
        sum(causal_pass_cycles) + (num_passes - 1) * reinject_cycles
        if causal
        else total_cycles_multipass
    )

    result.update({
        "causal_cycles_per_pass": causal_pass_cycles,
        "total_cycles_causal": total_cycles_causal,
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate analytical 2D KV-stationary MQA performance with multi-pass re-use."
    )
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--array-rows", type=int, default=64)
    parser.add_argument("--array-cols", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--memory-bandwidth-bytes-per-cycle", type=int, default=512)
    parser.add_argument("--exp-latency-cycles", type=int, default=4)
    parser.add_argument("--pe-mac-width", type=int, default=128)
    parser.add_argument("--lower-mac-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--head-parallelism", type=int, default=1)
    parser.add_argument("--merge-extensions", type=int, default=0)
    parser.add_argument("--query-tokens", type=int, default=1)
    parser.add_argument("--num-passes", type=int, default=1,
                        help="Number of re-use passes over T/P columns. Default 1 (no re-use).")
    parser.add_argument("--debug-cycles", type=int, default=3)
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
        exp_latency_cycles=args.exp_latency_cycles,
        pe_mac_width=args.pe_mac_width,
        lower_mac_count=args.lower_mac_count,
        batch_size=args.batch_size,
        head_parallelism=args.head_parallelism,
        merge_extensions=args.merge_extensions,
        query_tokens=args.query_tokens,
        num_passes=args.num_passes,
    )
    print(json.dumps({k: v for k, v in results.items() if k != "debug_trace"}, indent=2))


if __name__ == "__main__":
    main()
