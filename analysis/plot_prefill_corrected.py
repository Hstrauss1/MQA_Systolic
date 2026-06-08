"""Presentation figures using the corrected element-streaming model."""

from __future__ import annotations
import csv, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SRC = Path("../results/data/prefill_element_streaming.csv")
OUT = Path("../results/figures")
OUT.mkdir(exist_ok=True)

H, D, BPE = 64, 128, 2
EXP_LAT, PE_MAC_W = 4, 128
upper_pe = EXP_LAT + math.ceil(3 * D / PE_MAC_W)   # = 7

# ── Load ──────────────────────────────────────────────────────────────────────
rows = []
with SRC.open() as f:
    for r in csv.DictReader(f):
        rows.append({k: (int(v) if v.lstrip('-').isdigit() else
                         float(v) if v.replace('.','',1).lstrip('-').isdigit()
                         else v) for k, v in r.items()})

T_vals = sorted({r["T"] for r in rows})

def get(arch, key):
    return {r["T"]: r[key] for r in rows if r["arch"] == arch}

kv01_cyc  = get("kv_elem_n0_lmc1",  "total_cycles")
kv016_cyc = get("kv_elem_n0_lmc16", "total_cycles")
kv31_cyc  = get("kv_elem_n3_lmc1",  "total_cycles")
kv316_cyc = get("kv_elem_n3_lmc16", "total_cycles")

flash_single  = get("kv_elem_n0_lmc1", "flash_total")    # one H×H array
flash_eq_area = get("kv_elem_n0_lmc1", "flash_eq_area")  # T/H arrays = H×T PEs total

# Area-fair speedup: flash_eq_area already gives Flash the same H×T PEs as KV-stat.
# This IS the area-normalised comparison — no further division needed.
def speedup(kv_dict):
    return {T: flash_eq_area[T] / kv_dict[T] for T in T_vals}

su01  = speedup(kv01_cyc)
su016 = speedup(kv016_cyc)
su31  = speedup(kv31_cyc)
su316 = speedup(kv316_cyc)

C_FLASH   = "#4C72B0"
C01       = "#DD8452"
C016      = "#55A868"
C31       = "#9467BD"
C316      = "#C44E52"
C_MAC     = "#2d6a9f"
C_CHAIN   = "#55A868"
C_STAGGER = "#F0A500"
STYLE     = dict(linewidth=2.2, marker='o', markersize=6)


# ── Fig 1: Absolute cycle counts (log-log) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.loglog(T_vals, [flash_single[t]  for t in T_vals], color=C_FLASH,
          ls=':', lw=1.6, marker='s', markersize=5, alpha=0.55,
          label=f"Flash  single array  (H²={H*H:,} PEs)")
ax.loglog(T_vals, [flash_eq_area[t] for t in T_vals], color=C_FLASH,
          ls='--', **STYLE,
          label=f"Flash  equal-area   (H×T PEs,  T/H arrays)")
ax.loglog(T_vals, [kv01_cyc[t]  for t in T_vals], color=C01,  **STYLE,
          label="KV-stat  n=0  lmc=1")
ax.loglog(T_vals, [kv016_cyc[t] for t in T_vals], color=C016, **STYLE,
          label="KV-stat  n=0  lmc=16")
ax.loglog(T_vals, [kv31_cyc[t]  for t in T_vals], color=C31,  **STYLE, ls='--',
          label="KV-stat  n=3  lmc=1")
ax.loglog(T_vals, [kv316_cyc[t] for t in T_vals], color=C316, **STYLE, ls='--',
          label="KV-stat  n=3  lmc=16")

T_arr = np.array(T_vals, dtype=float)
ref = kv316_cyc[T_vals[0]] * (T_arr / T_vals[0])
ax.loglog(T_arr, ref, 'k:', lw=0.8, alpha=0.25)
ax.text(T_arr[-2] * 0.85, ref[-2] * 1.7, "∝ T", fontsize=9, color='gray')

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Cycles", fontsize=12)
ax.set_title(f"Absolute cycle counts  (H={H}, d={D})\nAll architectures scale O(T)", fontsize=11)
ax.legend(fontsize=8.5, loc='upper left')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, which='both', ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig1_cycles_scaling.png", dpi=150)
plt.close(); print("Saved fig1")


# ── Fig 2: Speedup over equal-area Flash (the area-fair result) ───────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

for label, su, color, ls in [
        ("n=0  lmc=1",  su01,  C01,  '-'),
        ("n=0  lmc=16", su016, C016, '-'),
        ("n=3  lmc=1",  su31,  C31,  '--'),
        ("n=3  lmc=16", su316, C316, '--'),
]:
    ax.semilogx(T_vals, [su[t] for t in T_vals],
                color=color, label=label, ls=ls, **STYLE)
    T_last = T_vals[-1]
    ax.annotate(f"  {su[T_last]:.1f}×", xy=(T_last, su[T_last]),
                fontsize=9, color=color, va='center', fontweight='bold')

ax.axhline(1, color='k', lw=0.9, ls='--', alpha=0.5, label="Break-even  (1×)")
ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Speedup vs equal-area FlashAttention", fontsize=12)
ax.set_title("KV-stationary vs FlashAttention — equal-area comparison\n"
             f"Both architectures use H×T total PEs  (H={H}, d={D})", fontsize=11)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlim(T_vals[0] * 0.75, T_vals[-1] * 3)
ax.grid(True, which='both', ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig2_speedup_equal_area.png", dpi=150)
plt.close(); print("Saved fig2")


# ── Fig 3: Cycle breakdown — all 4 configs at T=8192 ─────────────────────────
# Shows WHY lmc reduces total cycles (cuts stagger) while n reduces the PE chain.
T_bd = 8192

def breakdown(arch):
    return (get(arch, "mac_latency")[T_bd],
            get(arch, "upper_pe_chain")[T_bd],
            get(arch, "stagger_cost")[T_bd])

configs_bd = [
    ("n=0\nlmc=1",  *breakdown("kv_elem_n0_lmc1")),
    ("n=0\nlmc=16", *breakdown("kv_elem_n0_lmc16")),
    ("n=3\nlmc=1",  *breakdown("kv_elem_n3_lmc1")),
    ("n=3\nlmc=16", *breakdown("kv_elem_n3_lmc16")),
]
labels_bd = [c[0] for c in configs_bd]
macs      = [c[1] for c in configs_bd]
chains    = [c[2] for c in configs_bd]
staggers  = [c[3] for c in configs_bd]
totals_bd = [m + c + s for m, c, s in zip(macs, chains, staggers)]
x_bd      = np.arange(len(configs_bd))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(f"Cycle breakdown at T={T_bd:,}  (H={H}, d={D})\n"
             f"lmc cuts the stagger term (dominant); n cuts the upper PE chain",
             fontsize=12)

# Left: absolute stacked bars
ax = axes[0]
ax.bar(x_bd, macs,     color=C_MAC,     label=f"MAC latency (d={D}, constant)")
ax.bar(x_bd, chains,   color=C_CHAIN,   label=f"Upper PE chain  ({upper_pe}×eff_cols)",
       bottom=macs)
ax.bar(x_bd, staggers, color=C_STAGGER, label="Query stagger  ((T−1)×eff_stagger)",
       bottom=[m + c for m, c in zip(macs, chains)])
ax.axhline(flash_eq_area[T_bd], color=C_FLASH, lw=2, ls='--',
           label=f"Flash equal-area  ({flash_eq_area[T_bd]/1e6:.1f}M cycles)")

for xi, total in enumerate(totals_bd):
    ax.text(xi, total + flash_eq_area[T_bd] * 0.018, f"{total/1e3:.0f}K",
            ha='center', fontsize=11, fontweight='bold')

ax.set_xticks(x_bd); ax.set_xticklabels(labels_bd, fontsize=11)
ax.set_ylabel("Cycles", fontsize=11)
ax.set_title("Absolute cycles", fontsize=11)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(axis='y', ls='--', alpha=0.3)

# Right: percentage breakdown
ax2 = axes[1]
stagger_pct = [s / t * 100 for s, t in zip(staggers, totals_bd)]
chain_pct   = [c / t * 100 for c, t in zip(chains,   totals_bd)]
mac_pct     = [m / t * 100 for m, t in zip(macs,     totals_bd)]

ax2.bar(x_bd, mac_pct,     color=C_MAC,     label="MAC latency")
ax2.bar(x_bd, chain_pct,   color=C_CHAIN,   label="Upper PE chain",
        bottom=mac_pct)
ax2.bar(x_bd, stagger_pct, color=C_STAGGER, label="Query stagger",
        bottom=[m + c for m, c in zip(mac_pct, chain_pct)])

for xi, (sp, cp, mp) in enumerate(zip(stagger_pct, chain_pct, mac_pct)):
    # label stagger %
    ax2.text(xi, mp + cp + sp / 2, f"{sp:.0f}%",
             ha='center', va='center', fontsize=10, fontweight='bold', color='#222')
    # label chain % only if large enough to read
    if cp > 5:
        ax2.text(xi, mp + cp / 2, f"{cp:.0f}%",
                 ha='center', va='center', fontsize=9, color='white')

ax2.set_xticks(x_bd); ax2.set_xticklabels(labels_bd, fontsize=11)
ax2.set_ylabel("Fraction of total cycles (%)", fontsize=11)
ax2.set_title("% breakdown", fontsize=11)
ax2.set_ylim(0, 115)
ax2.legend(fontsize=8.5)
ax2.grid(axis='y', ls='--', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "fig3_cycle_breakdown.png", dpi=150)
plt.close(); print("Saved fig3")


# ── Fig 4: n sweep at T=8192, lmc=16 ─────────────────────────────────────────
T8          = 8192
lmc         = 16
eff_stagger = max(math.ceil(D / lmc), upper_pe)   # max(8, 7) = 8
mac_cost    = D
pe_chains   = [upper_pe * (T8 // 2**n) for n in range(4)]
stag_cost   = (T8 - 1) * eff_stagger
totals_n    = [mac_cost + pc + stag_cost for pc in pe_chains]
flash_eq_T8 = flash_eq_area[T8]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    f"Effect of merge level n  —  T={T8:,}, lmc={lmc}  (H={H}, d={D})\n"
    f"eff_stagger={eff_stagger} cycles/packet  ·  upper_pe={upper_pe} cycles/column",
    fontsize=12)

ax = axes[0]
n_labels = [f"n={n}\n({2**n} sub-array{'s' if n else ''}\neff_cols={T8//2**n:,})"
            for n in range(4)]
x = np.arange(4)

ax.bar(x, [mac_cost] * 4, color=C_MAC,
       label=f"MAC latency (d={D})")
ax.bar(x, pe_chains, color=C_CHAIN,
       label=f"Upper PE chain  ({upper_pe}×eff_cols)",
       bottom=[mac_cost] * 4)
ax.bar(x, [stag_cost] * 4, color=C_STAGGER,
       label=f"Query stagger  (T−1)×{eff_stagger} = {stag_cost:,}  (constant in n)",
       bottom=[mac_cost + pc for pc in pe_chains])
ax.axhline(flash_eq_T8, color=C_FLASH, lw=2, ls='--',
           label=f"Flash equal-area  ({flash_eq_T8/1e6:.1f}M cycles)")

for xi, total in zip(x, totals_n):
    ax.text(xi, total + flash_eq_T8 * 0.015, f"{total/1e6:.2f}M",
            ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(n_labels, fontsize=9)
ax.set_ylabel("Cycles (wall-clock, one sub-array)", fontsize=11)
ax.set_title("Cycle breakdown by merge level", fontsize=11)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.2f}M"))
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', ls='--', alpha=0.3)

ax2 = axes[1]
speedups_n = [flash_eq_T8 / t for t in totals_n]
colors_n   = [C_CHAIN, C_CHAIN, C_CHAIN, "#2d8a50"]
bars = ax2.bar(x, speedups_n, color=colors_n, edgecolor='white', width=0.5)
ax2.axhline(1.0, color=C_FLASH, lw=1.5, ls='--', alpha=0.7,
            label="Flash equal-area  (1×)")
for bar, su in zip(bars, speedups_n):
    ax2.text(bar.get_x() + bar.get_width() / 2, su + 0.3,
             f"{su:.2f}×", ha='center', fontsize=11, fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(n_labels, fontsize=9)
ax2.set_ylabel("Speedup over equal-area FlashAttention", fontsize=11)
ax2.set_title("Speedup vs equal-area Flash by merge level", fontsize=11)
ax2.legend(fontsize=10)
ax2.set_ylim(0, max(speedups_n) * 1.25)
ax2.grid(axis='y', ls='--', alpha=0.3)
ax2.text(0.5, -0.15,
         f"All KV-stat: H×T={H*T8:,} PEs  |  Flash: H×H={H*H:,} PEs  |  area ratio={T8//H}×",
         transform=ax2.transAxes, ha='center', fontsize=8.5, color='gray')

plt.tight_layout()
plt.savefig(OUT / "fig4_breakdown_n_sweep.png", dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4")


# ── Fig 5: Grouped bar — speedup at each T for all 4 configs ─────────────────
T_select = T_vals
configs_bar = [
    ("n=0  lmc=1",  su01,  C01),
    ("n=0  lmc=16", su016, C016),
    ("n=3  lmc=1",  su31,  C31),
    ("n=3  lmc=16", su316, C316),
]
x    = np.arange(len(T_select))
w    = 0.19
fig, ax = plt.subplots(figsize=(13, 5.5))

for i, (label, su, color) in enumerate(configs_bar):
    vals   = [su[T] for T in T_select]
    offset = (i - 1.5) * w
    bars   = ax.bar(x + offset, vals, w, label=label, color=color, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.25,
                f"{v:.1f}×", ha='center', va='bottom', fontsize=7.5, rotation=90)

ax.axhline(1, color='k', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Speedup vs equal-area FlashAttention", fontsize=12)
ax.set_title(f"KV-stationary speedup — all configs across sequence lengths  (H={H}, d={D})\n"
             f"Equal-area: both architectures use H×T total PEs", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([f"{T:,}" for T in T_select], fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', ls='--', alpha=0.3)
ax.set_ylim(0, max(su316[T] for T in T_select) * 1.35)
plt.tight_layout()
plt.savefig(OUT / "fig5_speedup_summary.png", dpi=150)
plt.close(); print("Saved fig5")

print(f"\nAll figures → {OUT}/")


# ── Fig 7: Raw (non-area-normalised) speedup vs single 64×64 Flash array ──────
# Baseline: one H×H = 4,096-PE Flash array running T tokens.
# KV-stat always uses H×T PEs.  No area equalization.
# Speedup grows roughly ∝ T because Flash is O(T²) compute while KV-stat is O(T).
raw_su = {
    ("n=0  lmc=1",  C01,  '-'):  {T: flash_single[T] / kv01_cyc[T]  for T in T_vals},
    ("n=0  lmc=16", C016, '-'):  {T: flash_single[T] / kv016_cyc[T] for T in T_vals},
    ("n=3  lmc=1",  C31,  '--'): {T: flash_single[T] / kv31_cyc[T]  for T in T_vals},
    ("n=3  lmc=16", C316, '--'): {T: flash_single[T] / kv316_cyc[T] for T in T_vals},
}

fig, ax = plt.subplots(figsize=(9, 5.5))
for (label, color, ls), su in raw_su.items():
    ax.semilogx(T_vals, [su[T] for T in T_vals],
                color=color, label=label, ls=ls, **STYLE)
    T_last = T_vals[-1]
    ax.annotate(f"  {su[T_last]:,.0f}×", xy=(T_last, su[T_last]),
                fontsize=9, color=color, va='center', fontweight='bold')

ax.axhline(1, color='k', lw=0.9, ls='--', alpha=0.5, label="Break-even  (1×)")
ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel(f"Speedup vs single {H}×{H} FlashAttention array", fontsize=12)
ax.set_title(
    "KV-stationary raw speedup  (non-area-normalised)\n"
    f"Baseline: one {H}×{H} Flash array ({H*H:,} PEs).  KV-stat: H×T PEs.  H={H}, d={D}",
    fontsize=11)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}×"))
ax.set_xlim(T_vals[0] * 0.75, T_vals[-1] * 3.5)
ax.grid(True, which='both', ls='--', alpha=0.3)
fig.text(0.5, 0.01,
         f"Area caveat: KV-stat uses H×T = {H}×T PEs vs {H*H:,} for single-array Flash. "
         f"Equal-area comparison (dividing by T/H) gives the ≈30× result in Fig. 6.",
         ha='center', fontsize=8, color='gray')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(OUT / "fig7_speedup_raw.png", dpi=150)
plt.close(); print("Saved fig7")


# ── Fig 6: GPU latency comparison (wall-clock µs) ─────────────────────────────
# KV-stat cycles → µs at 1 GHz (1 cycle = 1 ns = 0.001 µs  →  cycles / 1000 µs)
ASIC_GHZ = 1.0

# RTX 4090 specs (dense, no sparsity)
GPU_NAME     = "RTX 4090"
GPU_TFLOPS   = 165.2e12   # BF16 tensor core, dense
GPU_BW       = 1.008e12   # GDDR6X bytes/s
GPU_UTIL_TH  = 1.0        # theoretical peak
GPU_UTIL_PR  = 0.50       # practical single-sequence utilisation (~50% is generous)

def gpu_latency_us(T, util=1.0):
    """max(compute, memory) model — batch=1, causal prefill."""
    flops     = 4 * H * T * T * D          # QK^T + AV (2× for FWD+BWD skipped; FWD only)
    mem_bytes = 4 * H * T * D * BPE        # Q + K + V + O  (FA2: no materialised scores)
    return max(flops / (GPU_TFLOPS * util), mem_bytes / GPU_BW) * 1e6  # → µs

def kv_us(cyc_dict):
    return {T: cyc_dict[T] / (ASIC_GHZ * 1e3) for T in T_vals}  # cycles/(1e9 Hz)*1e6 = cyc/1e3

kv01_us  = kv_us(kv01_cyc)
kv016_us = kv_us(kv016_cyc)
kv31_us  = kv_us(kv31_cyc)
kv316_us = kv_us(kv316_cyc)

gpu_peak_us     = {T: gpu_latency_us(T, util=GPU_UTIL_TH) for T in T_vals}
gpu_prac_us     = {T: gpu_latency_us(T, util=GPU_UTIL_PR) for T in T_vals}

C_GPU_P = "#1f77b4"   # peak  (blue)
C_GPU_R = "#aec7e8"   # realistic (light blue)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    f"Wall-clock latency: KV-stationary vs {GPU_NAME} FlashAttention  (batch=1, H={H}, d={D})\n"
    f"KV-stat: {ASIC_GHZ:.0f} GHz ASIC.   GPU: max(compute, memory) model, "
    f"peak={GPU_TFLOPS/1e12:.0f} TFLOPS BF16, BW={GPU_BW/1e12:.0f} TB/s GDDR6X",
    fontsize=11)

# ── Left: absolute latency (log-log) ─────────────────────────────────────────
ax = axes[0]
ax.loglog(T_vals, [gpu_prac_us[t]  for t in T_vals], color=C_GPU_R,
          lw=2, ls='--', marker='D', markersize=5,
          label=f"{GPU_NAME} FA2  (50 % util, realistic)")
ax.loglog(T_vals, [gpu_peak_us[t]  for t in T_vals], color=C_GPU_P,
          lw=2, ls='--', marker='D', markersize=5,
          label=f"{GPU_NAME} FA2  (100 % util, theoretical peak)")
ax.loglog(T_vals, [kv01_us[t]  for t in T_vals],  color=C01,  **STYLE, label="KV-stat n=0 lmc=1")
ax.loglog(T_vals, [kv016_us[t] for t in T_vals],  color=C016, **STYLE, label="KV-stat n=0 lmc=16")
ax.loglog(T_vals, [kv31_us[t]  for t in T_vals],  color=C31,  **STYLE, ls='--', label="KV-stat n=3 lmc=1")
ax.loglog(T_vals, [kv316_us[t] for t in T_vals],  color=C316, **STYLE, ls='--', label="KV-stat n=3 lmc=16")

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Latency (µs)", fontsize=12)
ax.set_title("Absolute latency", fontsize=11)
ax.legend(fontsize=8.5, loc='upper left')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.1f}"))
ax.grid(True, which='both', ls='--', alpha=0.3)

# ── Right: GPU / KV-stat speedup (log scale y) for best config ────────────────
ax2 = axes[1]
su_gpu_peak = {T: gpu_peak_us[T] / kv316_us[T] for T in T_vals}
su_gpu_prac = {T: gpu_prac_us[T] / kv316_us[T] for T in T_vals}
su_gpu_peak_n01 = {T: gpu_peak_us[T] / kv01_us[T] for T in T_vals}

ax2.semilogy(T_vals, [su_gpu_prac[t]     for t in T_vals], color=C_GPU_R,
             lw=2, ls='--', marker='D', markersize=5,
             label=f"GPU (50% util) / KV-stat n=3 lmc=16")
ax2.semilogy(T_vals, [su_gpu_peak[t]     for t in T_vals], color=C_GPU_P,
             lw=2, ls='--', marker='D', markersize=5,
             label=f"GPU (peak) / KV-stat n=3 lmc=16")
ax2.semilogy(T_vals, [su_gpu_peak_n01[t] for t in T_vals], color=C01,
             lw=2, ls=':', marker='o', markersize=5,
             label=f"GPU (peak) / KV-stat n=0 lmc=1  (worst)")
ax2.axhline(1.0, color='k', lw=0.9, ls='--', alpha=0.5, label="Break-even (1×)")

# Annotate T=8192 values
for su_dict, color, label in [
        (su_gpu_peak, C_GPU_P, "peak"),
        (su_gpu_prac, C_GPU_R, "50%"),
]:
    v = su_dict[T_vals[-1]]
    ax2.annotate(f"  {v:.0f}×", xy=(T_vals[-1], v),
                 fontsize=9, color=color, va='center', fontweight='bold')

ax2.set_xlabel("Sequence length T", fontsize=12)
ax2.set_ylabel("GPU latency / KV-stat latency  (>1 = KV-stat faster)", fontsize=11)
ax2.set_title(f"How much faster is KV-stat than {GPU_NAME}?\n"
              f"(KV-stat @ {ASIC_GHZ:.0f} GHz,  raw latency, no area normalisation)", fontsize=10)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax2.grid(True, which='both', ls='--', alpha=0.3)

# footnote
fig.text(0.5, 0.01,
         f"⚠  Area caveat: KV-stat uses H×T={H}×T PEs; {GPU_NAME} has 16,384 CUDA cores.  "
         f"GPU utilisation at batch=1 single-sequence prefill is typically well below 50 %.",
         ha='center', fontsize=8, color='gray')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT / "fig6_gpu_comparison.png", dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig6")
