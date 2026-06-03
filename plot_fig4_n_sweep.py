"""Figure 4 revised: cycle breakdown sweep across merge_n values at T=8192, lmc=16.

Uses the correct sub-array model for merge_n > 0:
  - Each merge level creates 2^n parallel sub-arrays
  - Each sub-array: H=64 rows × T/2^n cols (one sub-array, others identical in parallel)
  - Wall-clock = one sub-array's cycles (all 2^n run simultaneously)
  - Total silicon = H × T PEs in all cases (same area regardless of n)

tile_compute_cycles (one sub-array) =
    (H + eff_cols - 1) × column_dwell          [pipeline traversal]
  + (T - 1) × effective_stagger                [query packet scheduling]

column_dwell    = d + exp_latency + ceil(3d/pe_mac_width) = 128 + 4 + 3 = 135 cycles
effective_stagger (lmc=16) = max(ceil(d/lmc), upper_pe) = max(8, 7) = 8 cycles
"""

from __future__ import annotations
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT = Path("plots_prefill")
OUT.mkdir(exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
H           = 64
D           = 128
T           = 8192
LMC         = 16
EXP_LAT     = 4
PE_MAC_W    = 128
BPE         = 2
BW          = 512      # bytes/cycle
FLASH_TILE  = 64

# Fixed timing
column_dwell     = D + EXP_LAT + math.ceil(3 * D / PE_MAC_W)  # 128+4+3 = 135
packet_stagger   = math.ceil(D / LMC)                          # ceil(128/16) = 8
upper_pe_cycles  = EXP_LAT + math.ceil(3 * D / PE_MAC_W)      # 4+3 = 7
eff_stagger      = max(packet_stagger, upper_pe_cycles)         # max(8,7) = 8

# FlashAttention (SCALE-Sim validated tile × tile count)
ss_qk_tile  = 8571
ss_av_tile  = 8571
n_tiles     = (T // FLASH_TILE) ** 2
flash_total = n_tiles * (ss_qk_tile + ss_av_tile)
flash_area  = H * H   # 4,096 PEs


# ── Sub-array model for each merge_n ─────────────────────────────────────────
def subarray_cycles(merge_n: int) -> dict:
    eff_cols   = T // (2 ** merge_n)       # KV tokens per sub-array
    n_subarrays = 2 ** merge_n
    sub_area   = H * eff_cols              # PEs in one sub-array
    total_area = H * T                     # total PEs (constant for all n)

    # Pipeline: (H + eff_cols - 1) steps × column_dwell
    pipeline_steps = H + eff_cols - 1
    pipeline_cyc   = pipeline_steps * column_dwell

    # Stagger: (T - 1) query packets × effective_stagger
    # All T query tokens attend over this sub-array's eff_cols KV tokens
    stagger_cyc = (T - 1) * eff_stagger

    total_cyc  = pipeline_cyc + stagger_cyc

    # PE utilisation: useful work / total PE-steps
    # Each of H rows × T queries traverses eff_cols columns
    useful = H * T * eff_cols * column_dwell
    capacity = total_cyc * H * eff_cols
    pe_util = useful / capacity if capacity else 0

    return {
        "merge_n":        merge_n,
        "n_subarrays":    n_subarrays,
        "eff_cols":       eff_cols,
        "sub_area_pes":   sub_area,
        "total_area_pes": total_area,
        "pipeline_steps": pipeline_steps,
        "pipeline_cyc":   pipeline_cyc,
        "stagger_cyc":    stagger_cyc,
        "total_cyc":      total_cyc,
        "pe_util":        pe_util,
    }

configs = [subarray_cycles(n) for n in [0, 1, 2, 3]]

# Area-equal Flash reference (Flash given same H×T PEs → T/H parallel arrays)
flash_equal_area_cyc = flash_total / (H * T / (H * H))  # = flash_total / (T/H)
# i.e. divide total Flash cycles by how many H×H arrays fit in H×T PEs

print(f"column_dwell={column_dwell}  effective_stagger={eff_stagger}")
print(f"FlashAttention: {flash_total:,} cycles  ({n_tiles:,} tiles × {ss_qk_tile+ss_av_tile:,} cycles/tile)")
print(f"Flash equal-area reference: {flash_equal_area_cyc:,.0f} cycles")
print()
print(f"{'n':>3}  {'sub-arrays':>11}  {'eff_cols':>9}  {'pipeline':>10}  "
      f"{'stagger':>9}  {'total':>10}  {'speedup vs flash(=area)':>24}  {'PE_util':>8}")
for c in configs:
    su = flash_equal_area_cyc / c['total_cyc']
    print(f"  {c['merge_n']:>1}  {c['n_subarrays']:>11}  {c['eff_cols']:>9,}  "
          f"{c['pipeline_cyc']:>10,}  {c['stagger_cyc']:>9,}  {c['total_cyc']:>10,}  "
          f"{su:>24.2f}x  {c['pe_util']:>8.3f}")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(f"KV-stationary prefill cycle breakdown — sweep of merge levels\n"
             f"T={T:,}   lmc={LMC}   column_dwell={column_dwell} cycles   "
             f"(H={H}, d={D}, bpe={BPE})", fontsize=12)

labels      = [f"n={c['merge_n']}\n({c['n_subarrays']} sub-arrays\n{c['eff_cols']:,} cols each)"
               for c in configs]
pipeline_vals = [c['pipeline_cyc'] for c in configs]
stagger_vals  = [c['stagger_cyc']  for c in configs]
total_vals    = [c['total_cyc']    for c in configs]

C_PIPE    = "#55A868"
C_STAGGER = "#88cc99"
C_FLASH   = "#4C72B0"

# ── Left: stacked bar — cycle breakdown ──────────────────────────────────────
ax = axes[0]
x  = range(len(configs))
b1 = ax.bar(x, pipeline_vals, color=C_PIPE,    label="Pipeline traversal\n(H + eff_cols − 1) × column_dwell")
b2 = ax.bar(x, stagger_vals,  color=C_STAGGER, label="Query stagger\n(T−1) × eff_stagger",
            bottom=pipeline_vals)

# Flash equal-area reference line
ax.axhline(flash_equal_area_cyc, color=C_FLASH, lw=2, ls='--',
           label=f"FlashAttention (equal area)\n{flash_equal_area_cyc/1e6:.1f}M cycles")

# Annotate totals
for xi, total in zip(x, total_vals):
    ax.text(xi, total + flash_equal_area_cyc * 0.02,
            f"{total/1e6:.2f}M", ha='center', fontsize=9.5, fontweight='bold')

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Cycles (one sub-array, wall-clock)", fontsize=11)
ax.set_title("Cycle breakdown by merge level", fontsize=11)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', ls='--', alpha=0.3)

# Annotate pipeline bar content
for xi, pv in zip(x, pipeline_vals):
    c = configs[xi]
    ax.text(xi, pv / 2,
            f"({c['pipeline_steps']:,} steps\n× {column_dwell} cycles)",
            ha='center', va='center', fontsize=7.5, color='white')

# ── Right: speedup over equal-area FlashAttention ────────────────────────────
ax2 = axes[1]
speedups = [flash_equal_area_cyc / t for t in total_vals]
bars = ax2.bar(x, speedups, color=[C_PIPE if n < 3 else "#2d8a50" for n in range(4)],
               edgecolor='white', width=0.5)
ax2.axhline(1.0, color=C_FLASH, lw=1.5, ls='--', alpha=0.7, label="FlashAttention (1×)")

for bar, su in zip(bars, speedups):
    ax2.text(bar.get_x() + bar.get_width()/2, su + 0.1,
             f"{su:.2f}×", ha='center', fontsize=11, fontweight='bold')

ax2.set_xticks(list(x))
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Speedup over FlashAttention (equal area)", fontsize=11)
ax2.set_title("Speedup per unit silicon\n(same H×T total PEs for all KV-stat configs)", fontsize=11)
ax2.legend(fontsize=10)
ax2.set_ylim(0, max(speedups) * 1.2)
ax2.grid(axis='y', ls='--', alpha=0.3)

# Note: same area for all n
ax2.text(0.5, -0.15,
         f"All KV-stat configs: H×T = {H*T:,} PEs  |  "
         f"FlashAttention: H×H = {H*H:,} PEs  |  Area ratio = {H*T//(H*H)}×",
         transform=ax2.transAxes, ha='center', fontsize=8.5, color='gray')

plt.tight_layout()
path = OUT / "fig4_n_sweep_T8192.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {path}")
