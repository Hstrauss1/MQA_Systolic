"""Generate plot 11: prefill speedup normalized to same MAC count.

Three lines, all using KV-stat total_cycles (actual pipeline estimate):

  1. KV-stat 16MAC n=3 vs original 64×64 baseline (unfused)
     — reproduces the current misleading plot:
         baseline.estimated_cycles / kv.total_cycles = ~511× at T=8192
         because: baseline is compute-bound (64×64, 4 096 MACs/cycle),
                  KV-stat is memory-bound (512×1024×16 = 8 388 608 MACs/cycle)
         2048× MAC ratio + memory-bound KV-stat = inflated speedup

  2. KV-stat 16MAC n=3 vs MAC-normalized baseline (unfused)
     — fair MAC comparison:
         baseline array sized so baseline_MACs/cycle == kv_MACs/cycle
         array_rows = H × 2^n × lower_mac = 512 × 16 = 8192
         array_cols = T // 2^n  (= eff_cols, same as KV-stat at each T)
         ⇒ 8192 × (T//8) = 1024T = 512 × (T//8) × 16 MACs/cycle ✓

  3. KV-stat 16MAC n=3 vs MAC-normalized + fused baseline
     — fair MAC + FlashAttention-style fusion:
         same array as line 2, fused=True (no score-matrix HBM traffic)

At T=8192 the three speedups are approximately:
  Line 1: ~511×  (unfair: 2048× MAC ratio + KV-stat memory-bound)
  Line 2: ~65×   (MAC-fair: score-matrix traffic still dominates baseline)
  Line 3: ~1×    (MAC-fair + fused: KV-stat offers little advantage)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_mqa_model import baseline_mqa_metrics
from kv_stationary_model import kv_stationary_metrics

# ── Shared parameters ────────────────────────────────────────────────────────
H            = 64
D            = 128
BPE          = 2
BW           = 512    # bytes/cycle
PE_MAC_WIDTH = 128    # upper PE vector width (fully parallel)
LOWER_MAC    = 16     # interleaved MACs in lower dot-product unit per PE
MERGE_N      = 3      # merge-extension levels (optimal for H=64)
T_VALUES: List[int] = [128, 512, 1024, 2048, 4096, 8192]


def main() -> None:
    speedup_orig:       List[float] = []
    speedup_norm:       List[float] = []
    speedup_norm_fused: List[float] = []

    print(
        f"{'T':>6}  {'kv_cyc':>12}  {'b_orig':>12}  {'b_norm':>12}  {'b_fused':>12}"
        f"  {'su_orig':>8}  {'su_norm':>8}  {'su_fused':>8}"
    )
    print("-" * 105)

    for T in T_VALUES:
        # KV-stat array dimensions for merge_extensions=n
        eff_rows = H * (2 ** MERGE_N)          # 512  (constant across T)
        eff_cols = max(1, T // (2 ** MERGE_N))  # T//8 (grows with T)

        # KV-stat: 16 MACs/PE, n=3 merge extensions, full prefill (query_tokens=T)
        kv = kv_stationary_metrics(
            H=H, T=T, d=D,
            array_rows=eff_rows, array_cols=eff_cols,
            bytes_per_element=BPE,
            memory_bandwidth_bytes_per_cycle=BW,
            pe_mac_width=PE_MAC_WIDTH,
            lower_mac_count=LOWER_MAC,
            merge_extensions=MERGE_N,
            query_tokens=T,   # prefill: all T tokens attend over T KV entries
        )
        kv_cycles = int(kv["total_cycles"])  # actual pipeline estimate (not roofline)

        # Normalized baseline array: same MACs/cycle as KV-stat at each T
        #   norm_rows × norm_cols = eff_rows × eff_cols × LOWER_MAC
        #                         = 512 × (T//8) × 16
        #                         = 8192 × (T//8)
        norm_rows = eff_rows * LOWER_MAC   # 8192
        norm_cols = eff_cols               # T//8

        # Line 1: original 64×64, unfused — reproduces current (misleading) plot
        b_orig = baseline_mqa_metrics(
            H=H, T=T, d=D,
            array_rows=64, array_cols=64,
            bytes_per_element=BPE,
            memory_bandwidth_bytes_per_cycle=BW,
            query_tokens=T,
            fused=False,
        )

        # Line 2: MAC-normalized, unfused — fair comparison
        b_norm = baseline_mqa_metrics(
            H=H, T=T, d=D,
            array_rows=norm_rows, array_cols=norm_cols,
            bytes_per_element=BPE,
            memory_bandwidth_bytes_per_cycle=BW,
            query_tokens=T,
            fused=False,
        )

        # Line 3: MAC-normalized, fused — fair + FlashAttention-style fusion
        b_fused = baseline_mqa_metrics(
            H=H, T=T, d=D,
            array_rows=norm_rows, array_cols=norm_cols,
            bytes_per_element=BPE,
            memory_bandwidth_bytes_per_cycle=BW,
            query_tokens=T,
            fused=True,
        )

        su_orig  = b_orig["estimated_cycles"]  / kv_cycles
        su_norm  = b_norm["estimated_cycles"]  / kv_cycles
        su_fused = b_fused["estimated_cycles"] / kv_cycles

        speedup_orig.append(su_orig)
        speedup_norm.append(su_norm)
        speedup_norm_fused.append(su_fused)

        print(
            f"{T:>6}  {kv_cycles:>12,}  "
            f"{b_orig['estimated_cycles']:>12,}  "
            f"{b_norm['estimated_cycles']:>12,}  "
            f"{b_fused['estimated_cycles']:>12,}  "
            f"{su_orig:>8.1f}×  {su_norm:>8.1f}×  {su_fused:>8.2f}×"
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(T_VALUES, speedup_orig, marker="o", linewidth=2.2, color="tab:red",
            label=(
                "vs 64×64 baseline (unfused)  ← current plot\n"
                r"  unfair: KV-stat has $\leq$2048× more MACs/cycle"
            ))
    ax.plot(T_VALUES, speedup_norm, marker="s", linewidth=2.2, color="tab:blue",
            label=(
                "vs MAC-normalized baseline (unfused)\n"
                r"  fair: $8192\times(T/8)$ MACs/cycle = KV-stat MACs/cycle"
            ))
    ax.plot(T_VALUES, speedup_norm_fused, marker="^", linewidth=2.2, color="tab:green",
            label=(
                "vs MAC-normalized + fused baseline\n"
                "  fair + FlashAttention-style (no score HBM traffic)"
            ))

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7, label="1× (no advantage)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(T_VALUES)
    ax.set_xticklabels([str(t) for t in T_VALUES])
    ax.set_xlabel("Sequence Length T (tokens)", fontsize=11)
    ax.set_ylabel("Speedup  (baseline cycles / KV-stat pipeline cycles)", fontsize=11)
    ax.set_title(
        "Plot 11 — KV-stat 16MAC n=3 prefill speedup vs baseline\n"
        f"H={H}, d={D}, BW={BW} B/cyc, lower_mac={LOWER_MAC}, merge_ext={MERGE_N}  |  "
        f"KV-stat array: 512×(T/8), {LOWER_MAC} MACs/PE",
        fontsize=9.5,
    )
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    out = Path("plots/11_speedup_vs_baseline.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
