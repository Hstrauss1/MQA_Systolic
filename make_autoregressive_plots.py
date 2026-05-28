"""Generate plots 13 and 14: autoregressive generation comparison.

Plot 13 — DRAM and cycles vs G (generate tokens) at T=8192.
    Two subplots side-by-side:
      Left:  total DRAM (MB) over a G-token generation run
      Right: total cycles over a G-token generation run
    Both show KV-stat 16MAC n=3 vs MAC-normalized unfused baseline.

Plot 14 — Speedup vs G.
    baseline_total_generate_cycles / kv_total_generate_cycles across G.
    Also shows the DRAM savings ratio on a secondary y-axis.

Hardware constants match make_plot11.py:
    H=64, d=128, bpe=2, BW=512 B/cyc, pe_mac_width=128, lower_mac=16, n=3
    T=8192 (fixed — worst-case KV cache size for generation)
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
BW           = 512      # bytes/cycle
PE_MAC_WIDTH = 128
LOWER_MAC    = 16
MERGE_N      = 3
T            = 8192

G_VALUES: List[int] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# KV-stat array layout for n=3
EFF_ROWS = H * (2 ** MERGE_N)           # 512
EFF_COLS = max(1, T // (2 ** MERGE_N))  # 1024

# MAC-normalized baseline: same MACs/cycle as KV-stat
NORM_ROWS = EFF_ROWS * LOWER_MAC        # 8192
NORM_COLS = EFF_COLS                    # 1024


def compute_generate_row(G: int):
    """Return (kv_dram_mb, kv_cycles, base_dram_mb, base_cycles) for G tokens."""
    kv = kv_stationary_metrics(
        H=H, T=T, d=D,
        array_rows=EFF_ROWS, array_cols=EFF_COLS,
        bytes_per_element=BPE,
        memory_bandwidth_bytes_per_cycle=BW,
        pe_mac_width=PE_MAC_WIDTH,
        lower_mac_count=LOWER_MAC,
        merge_extensions=MERGE_N,
        query_tokens=T,         # prefill
        generate_tokens=G,
    )

    base = baseline_mqa_metrics(
        H=H, T=T, d=D,
        array_rows=NORM_ROWS, array_cols=NORM_COLS,
        bytes_per_element=BPE,
        memory_bandwidth_bytes_per_cycle=BW,
        query_tokens=T,         # prefill
        fused=False,
        generate_tokens=G,
    )

    return (
        float(kv["total_generate_dram_mb"]),
        int(kv["total_generate_cycles"]),
        float(base["total_generate_dram_mb"]),
        int(base["total_generate_cycles"]),
    )


def main() -> None:
    kv_dram:    List[float] = []
    kv_cycles:  List[int]   = []
    b_dram:     List[float] = []
    b_cycles:   List[int]   = []

    print(
        f"{'G':>6}  {'kv_dram_MB':>12}  {'b_dram_MB':>12}  "
        f"{'kv_cyc':>14}  {'b_cyc':>14}  "
        f"{'dram_ratio':>10}  {'cycle_speedup':>14}  {'kv_savings_MB':>14}"
    )
    print("-" * 120)

    for G in G_VALUES:
        kd, kc, bd, bc = compute_generate_row(G)
        kv_dram.append(kd)
        kv_cycles.append(kc)
        b_dram.append(bd)
        b_cycles.append(bc)
        savings = bd - kd
        print(
            f"{G:>6}  {kd:>12.2f}  {bd:>12.2f}  "
            f"{kc:>14,}  {bc:>14,}  "
            f"{bd/kd:>10.1f}×  {bc/kc:>14.1f}×  {savings:>14.2f}"
        )

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    # ── Plot 13: DRAM and cycles vs G ────────────────────────────────────────
    fig, (ax_dram, ax_cyc) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"Autoregressive generation cost vs tokens generated  |  T={T}, H={H}, d={D}\n"
        f"KV-stat: {EFF_ROWS}×{EFF_COLS} array, {LOWER_MAC} MACs/PE, n={MERGE_N} merge extensions  |  "
        f"Baseline: {NORM_ROWS}×{NORM_COLS} MAC-normalized, unfused",
        fontsize=9,
    )

    ax_dram.plot(G_VALUES, b_dram, marker="o", linewidth=2.2, color="tab:red",
                 label=f"MAC-normalized baseline\n(KV reloaded every step, unfused)")
    ax_dram.plot(G_VALUES, kv_dram, marker="s", linewidth=2.2, color="tab:blue",
                 label=f"KV-stationary\n(KV loaded once, stays in SRAM)")
    ax_dram.set_xscale("log")
    ax_dram.set_yscale("log")
    ax_dram.set_xlabel("Generate tokens (G)", fontsize=11)
    ax_dram.set_ylabel("Total DRAM  (prefill + G decode steps, MB)", fontsize=10)
    ax_dram.set_title("DRAM traffic vs G", fontsize=11)
    ax_dram.set_xticks(G_VALUES)
    ax_dram.set_xticklabels([str(g) for g in G_VALUES], fontsize=8)
    ax_dram.legend(fontsize=8.5)
    ax_dram.grid(True, which="both", alpha=0.25)

    ax_cyc.plot(G_VALUES, b_cycles, marker="o", linewidth=2.2, color="tab:red",
                label="MAC-normalized baseline")
    ax_cyc.plot(G_VALUES, kv_cycles, marker="s", linewidth=2.2, color="tab:blue",
                label="KV-stationary")
    ax_cyc.set_xscale("log")
    ax_cyc.set_yscale("log")
    ax_cyc.set_xlabel("Generate tokens (G)", fontsize=11)
    ax_cyc.set_ylabel("Total cycles  (prefill + G decode steps)", fontsize=10)
    ax_cyc.set_title("Cycles vs G", fontsize=11)
    ax_cyc.set_xticks(G_VALUES)
    ax_cyc.set_xticklabels([str(g) for g in G_VALUES], fontsize=8)
    ax_cyc.legend(fontsize=8.5)
    ax_cyc.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    out13 = out_dir / "13_dram_cycles_vs_generate_tokens.png"
    fig.savefig(out13, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out13}")
    plt.close(fig)

    # ── Plot 14: speedup and DRAM savings ratio vs G ─────────────────────────
    cycle_speedup = [bc / kc for bc, kc in zip(b_cycles, kv_cycles)]
    dram_ratio    = [bd / kd for bd, kd in zip(b_dram, kv_dram)]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(G_VALUES, cycle_speedup, marker="o", linewidth=2.2, color="tab:blue",
                   label="Cycle speedup  (baseline / KV-stat)")
    l2, = ax2.plot(G_VALUES, dram_ratio, marker="s", linewidth=2.2, color="tab:orange",
                   linestyle="--", label="DRAM ratio  (baseline / KV-stat)")

    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Generate tokens (G)", fontsize=11)
    ax1.set_ylabel("Cycle speedup  (baseline / KV-stat)", fontsize=11, color="tab:blue")
    ax2.set_ylabel("DRAM ratio  (baseline / KV-stat)", fontsize=11, color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_xticks(G_VALUES)
    ax1.set_xticklabels([str(g) for g in G_VALUES])

    # Annotate last G where cycle speedup >= 1 (crossover point)
    crossover_g, crossover_su = None, None
    for g, su in zip(G_VALUES, cycle_speedup):
        if su >= 1.0:
            crossover_g, crossover_su = g, su
        else:
            break
    if crossover_g is not None:
        ax1.annotate(
            f"Last breakeven\nG={crossover_g} ({crossover_su:.1f}×)",
            xy=(crossover_g, crossover_su),
            xytext=(crossover_g * 2, crossover_su * 3),
            fontsize=8, color="tab:blue",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.0),
        )

    ax1.set_title(
        f"Plot 14 — KV-stat speedup & DRAM savings vs generate tokens  |  T={T}, H={H}, d={D}\n"
        f"KV-stat: {EFF_ROWS}×{EFF_COLS}, {LOWER_MAC} MACs/PE, n={MERGE_N}  vs  "
        f"MAC-normalized baseline ({NORM_ROWS}×{NORM_COLS}, unfused)\n"
        f"KV-stat saves 64–78× DRAM at all G; wins cycles for small G (prefill-dominated), "
        f"baseline wins cycles for large G (decode-dominated with equal MAC budget)",
        fontsize=8.5,
    )
    ax1.legend(handles=[l1, l2], fontsize=9, loc="upper left")
    ax1.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    out14 = out_dir / "14_speedup_vs_generate_tokens.png"
    fig.savefig(out14, dpi=200, bbox_inches="tight")
    print(f"Saved → {out14}")
    plt.close(fig)


if __name__ == "__main__":
    main()
