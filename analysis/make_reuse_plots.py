"""Sweep the multi-pass re-use model and generate two diagnostic plots.

Plot A — Latency cost:  total_cycles vs P for each T.
Plot B — Area-normalised throughput:  T / (total_cycles × pe_count) vs P for each T.

Outputs saved to plots/.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from kv_reuse_model import kv_stationary_metrics

# ── Hardware constants (match the validated best configuration) ───────────────
H             = 64    # query heads
d             = 128   # head dimension
bpe           = 2     # bytes per element (FP16)
BW            = 512   # DRAM bandwidth bytes/cycle
pe_mac_width  = 128
lmc           = 16    # lower_mac_count
exp_latency   = 4
n_merge       = 0     # merge_extensions = 0 (flat array)

T_values = [1024, 2048, 4096, 8192]
P_values = [1, 2, 4, 8, 16]

# ── Run the sweep ─────────────────────────────────────────────────────────────
# results[(T, P)] = metrics dict
results: Dict[Tuple[int, int], dict] = {}

for T in T_values:
    for P in P_values:
        physical_cols = max(1, T // P)
        r = kv_stationary_metrics(
            H=H,
            T=T,
            d=d,
            array_rows=H,
            array_cols=physical_cols,   # used only for P=1 path
            bytes_per_element=bpe,
            memory_bandwidth_bytes_per_cycle=BW,
            exp_latency_cycles=exp_latency,
            pe_mac_width=pe_mac_width,
            lower_mac_count=lmc,
            merge_extensions=n_merge,
            query_tokens=1,
            num_passes=P,
        )
        results[(T, P)] = r

# ── Derived metrics ───────────────────────────────────────────────────────────
# baseline_cycles[T] = single-pass cycles for P=1 at T (full T columns)
baseline_cycles = {T: results[(T, 1)]["total_cycles_multipass"] for T in T_values}

table_header = f"{'T':>6}  {'P':>3}  {'phys_cols':>10}  {'total_cycles':>14}  "
table_header += f"{'pe_count':>10}  {'sram_mb':>8}  {'lat_ratio':>10}  {'area_norm_tput':>16}"
print(table_header)
print("-" * len(table_header))

sweep: Dict[Tuple[int, int], dict] = {}
for T in T_values:
    for P in P_values:
        r = results[(T, P)]
        total_cycles  = r["total_cycles_multipass"]
        pe_count      = r["pe_count"]                          # H × (T//P)
        sram_bytes    = r["sram_bytes"]                        # 2 × (T//P) × d × bpe
        sram_mb       = sram_bytes / (1024 ** 2)
        latency_ratio = total_cycles / baseline_cycles[T]      # vs P=1 same T
        # tokens processed per cycle per PE
        area_norm_tput = T / (total_cycles * pe_count)

        sweep[(T, P)] = {
            "total_cycles":       total_cycles,
            "pe_count":           pe_count,
            "sram_mb":            sram_mb,
            "latency_ratio":      latency_ratio,
            "area_norm_tput":     area_norm_tput,
        }

        print(
            f"{T:>6}  {P:>3}  {r['physical_cols']:>10}  {total_cycles:>14}  "
            f"{pe_count:>10}  {sram_mb:>8.2f}  {latency_ratio:>10.4f}  {area_norm_tput:>16.4e}"
        )
    print()

# ── Plot helpers ──────────────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MARKERS = ["o", "s", "^", "D"]

def _T_label(T: int) -> str:
    return f"T={T:,}"


# ── Plot A: Latency cost (total cycles vs P) ──────────────────────────────────
fig_a, ax_a = plt.subplots(figsize=(8, 5))

for idx, T in enumerate(T_values):
    xs = P_values
    ys = [sweep[(T, P)]["total_cycles"] for P in P_values]
    ax_a.plot(xs, ys, marker=MARKERS[idx], color=COLORS[idx],
              linewidth=2, markersize=7, label=_T_label(T))
    # Mark P=1 with an open circle
    ax_a.plot(1, sweep[(T, 1)]["total_cycles"],
              marker="o", markerfacecolor="white", markeredgecolor=COLORS[idx],
              markeredgewidth=2, markersize=11, zorder=5)

ax_a.set_xlabel("Pass count (P)", fontsize=12)
ax_a.set_ylabel("Total cycles", fontsize=12)
ax_a.set_title("Plot A — Latency cost: cycles vs pass count\n"
               "(open circle = P=1 single-pass baseline)", fontsize=11)
ax_a.set_xticks(P_values)
ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
ax_a.legend(framealpha=0.9)
ax_a.grid(True, linestyle="--", alpha=0.5)
fig_a.tight_layout()
fig_a.savefig("../results/figures/plot_A_latency_vs_passes.png", dpi=150)
plt.close(fig_a)
print("Saved plots/plot_A_latency_vs_passes.png")


# ── Plot B: Area-normalised throughput (T / (cycles × pe_count)) vs P ─────────
fig_b, ax_b = plt.subplots(figsize=(8, 5))

for idx, T in enumerate(T_values):
    xs = P_values
    ys = [sweep[(T, P)]["area_norm_tput"] for P in P_values]
    ax_b.plot(xs, ys, marker=MARKERS[idx], color=COLORS[idx],
              linewidth=2, markersize=7, label=_T_label(T))

ax_b.set_xlabel("Pass count (P)", fontsize=12)
ax_b.set_ylabel("Tokens / (cycle × PE)", fontsize=12)
ax_b.set_title("Plot B — Area-normalised throughput vs pass count\n"
               "(higher = more silicon-efficient; rising = re-use wins)", fontsize=11)
ax_b.set_xticks(P_values)
ax_b.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2e}"))
ax_b.legend(framealpha=0.9)
ax_b.grid(True, linestyle="--", alpha=0.5)
fig_b.tight_layout()
fig_b.savefig("../results/figures/plot_B_area_norm_throughput_vs_passes.png", dpi=150)
plt.close(fig_b)
print("Saved plots/plot_B_area_norm_throughput_vs_passes.png")
