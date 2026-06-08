"""Generate presentation-ready figures from prefill_full_results.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SRC  = Path("prefill_full_results.csv")
OUT  = Path("plots_prefill")
OUT.mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
rows = []
with SRC.open() as f:
    for r in csv.DictReader(f):
        rows.append({k: (int(v) if v.lstrip('-').isdigit() else
                         float(v) if v.replace('.','',1).lstrip('-').isdigit() else v)
                     for k, v in r.items()})

T_vals = sorted({r["T"] for r in rows})

def get(arch, key):
    return {r["T"]: r[key] for r in rows if r["arch"] == arch}

# flash_total_cycles is a column on every KV row — read from any consistent arch
flash_cyc  = get("kv_stat_n0_lmc1",   "flash_total_cycles")
kv01_cyc   = get("kv_stat_n0_lmc1",   "kv_total_corrected")
kv016_cyc  = get("kv_stat_n0_lmc16",  "kv_total_corrected")
kv31_cyc   = get("kv_stat_n3_lmc1",   "kv_total_corrected")
kv316_cyc  = get("kv_stat_n3_lmc16",  "kv_total_corrected")

kv01_su    = get("kv_stat_n0_lmc1",   "speedup_kv_over_flash")
kv016_su   = get("kv_stat_n0_lmc16",  "speedup_kv_over_flash")
kv31_su    = get("kv_stat_n3_lmc1",   "speedup_kv_over_flash")
kv316_su   = get("kv_stat_n3_lmc16",  "speedup_kv_over_flash")

kv01_an    = get("kv_stat_n0_lmc1",   "area_norm_speedup")
kv016_an   = get("kv_stat_n0_lmc16",  "area_norm_speedup")
kv31_an    = get("kv_stat_n3_lmc1",   "area_norm_speedup")
kv316_an   = get("kv_stat_n3_lmc16",  "area_norm_speedup")

# Palette
C_FLASH   = "#4C72B0"
C_KV01    = "#DD8452"
C_KV016   = "#55A868"
C_KV316   = "#C44E52"

STYLE = dict(linewidth=2.2, marker='o', markersize=6)

# ── Fig 1: Cycles vs T (log-log) — scaling story ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(T_vals, [flash_cyc[t] for t in T_vals],  color=C_FLASH,  label="FlashAttention  (∝ T²)",  **STYLE)
ax.loglog(T_vals, [kv01_cyc[t]  for t in T_vals],  color=C_KV01,   label="KV-stat n=0 lmc=1  (∝ T)",  **STYLE)
ax.loglog(T_vals, [kv016_cyc[t] for t in T_vals],  color=C_KV016,  label="KV-stat n=0 lmc=16 (∝ T)", **STYLE)

# Annotate scaling guides
T_arr = np.array(T_vals, dtype=float)
ax.loglog(T_arr, flash_cyc[128]  * (T_arr/128)**2,  'k--', lw=0.8, alpha=0.4)
ax.loglog(T_arr, kv01_cyc[128]   * (T_arr/128)**1,  'k:',  lw=0.8, alpha=0.4)
ax.text(5000, 2e8, "slope = 2  (T²)", fontsize=8, color='gray', rotation=38)
ax.text(5000, 1.5e6, "slope = 1  (T)",  fontsize=8, color='gray', rotation=19)

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Total cycles", fontsize=12)
ax.set_title("Cycle scaling: FlashAttention vs KV-stationary (prefill)", fontsize=13)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, which='both', ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig1_cycles_scaling.png", dpi=150)
plt.close()
print("Saved fig1_cycles_scaling.png")

# ── Fig 2: Raw speedup vs T ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogx(T_vals, [kv01_su[t]  for t in T_vals], color=C_KV01,  label="n=0 lmc=1",  **STYLE)
ax.semilogx(T_vals, [kv016_su[t] for t in T_vals], color=C_KV016, label="n=0 lmc=16", **STYLE)
ax.semilogx(T_vals, [kv31_su[t]  for t in T_vals], color="purple", label="n=3 lmc=1",  **STYLE, ls='--')
ax.semilogx(T_vals, [kv316_su[t] for t in T_vals], color=C_KV316, label="n=3 lmc=16", **STYLE)
ax.axhline(1, color='k', lw=0.8, ls='--', alpha=0.5, label="1× (no gain)")

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Speedup vs FlashAttention", fontsize=12)
ax.set_title("Raw speedup: KV-stationary over FlashAttention (prefill)", fontsize=13)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(True, which='both', ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig2_raw_speedup.png", dpi=150)
plt.close()
print("Saved fig2_raw_speedup.png")

# ── Fig 3: Area-normalised speedup vs T ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogx(T_vals, [kv01_an[t]  for t in T_vals], color=C_KV01,  label="n=0 lmc=1   → ~1.0×",  **STYLE)
ax.semilogx(T_vals, [kv016_an[t] for t in T_vals], color=C_KV016, label="n=0 lmc=16  → ~1.85×", **STYLE)
ax.semilogx(T_vals, [kv31_an[t]  for t in T_vals], color="purple", label="n=3 lmc=1   → ~0.23×", **STYLE, ls='--')
ax.semilogx(T_vals, [kv316_an[t] for t in T_vals], color=C_KV316, label="n=3 lmc=16  → ~1.3×",  **STYLE)

ax.axhline(1.0,  color='k',      lw=1.0, ls='--', alpha=0.5, label="1× (break-even)")
ax.axhline(1.85, color=C_KV016,  lw=0.8, ls=':',  alpha=0.6)
ax.text(130, 1.87, "1.85×", color=C_KV016, fontsize=9)

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Speedup per unit silicon", fontsize=12)
ax.set_title("Area-normalised speedup: KV-stationary over FlashAttention\n"
             "(speedup ÷ PE area ratio — honest silicon comparison)", fontsize=12)
ax.legend(fontsize=9, loc='center left')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, 2.4)
ax.grid(True, which='both', ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig3_area_norm_speedup.png", dpi=150)
plt.close()
print("Saved fig3_area_norm_speedup.png")

# ── Fig 4: Cycle breakdown at T=8192 — what makes each architecture tick ─────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Cycle breakdown at T=8192 — where time is spent", fontsize=13)

# FlashAttention: tile count × per-tile cost
flash_tiles  = (8192 // 64) ** 2   # 16,384 tiles
flash_per_tile = 8571 + 8571
flash_total    = flash_tiles * flash_per_tile
ax = axes[0]
ax.bar(["QK tiles\n(16,384 × 8,571)", "AV tiles\n(16,384 × 8,571)"],
       [flash_tiles * 8571, flash_tiles * 8571],
       color=[C_FLASH, "#7fa8d8"], edgecolor='white')
ax.set_title("FlashAttention\n(64×64 array, SCALE-Sim validated)", fontsize=11)
ax.set_ylabel("Cycles", fontsize=11)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
ax.text(0, flash_tiles*8571*1.02, f"{flash_tiles*8571/1e6:.0f}M", ha='center', fontsize=9)
ax.text(1, flash_tiles*8571*1.02, f"{flash_tiles*8571/1e6:.0f}M", ha='center', fontsize=9)
ax.set_ylim(0, flash_total * 1.2)

# KV-stationary n=0 lmc=16 breakdown
T8 = 8192
kv_total  = kv016_cyc[T8]          # 1,179,953
lmc       = 16
lower_mac = T8 * 128 // lmc        # ceil(T×d/lmc) = T×d/lmc = 65,536
upper_pe  = T8 * (4 + 3) * 64      # H × T × upper_pe_cycles_per_step...

# From the model: total = (H + T - 1)×column_dwell + (T-1)×effective_stagger
# column_dwell = 135, effective_stagger = max(8,7)=8
col_dwell = 135
eff_stagger = 8
pipeline_steps = 64 + 8192 - 1    # H + T - 1
pipeline_cyc   = pipeline_steps * col_dwell
stagger_cyc    = (8192 - 1) * eff_stagger

ax = axes[1]
labels = ["Pipeline steps\n(column traversal)", "Query stagger\n(packet scheduling)"]
vals   = [pipeline_cyc, stagger_cyc]
colors = [C_KV016, "#88cc99"]
bars = ax.bar(labels, vals, color=colors, edgecolor='white')
ax.set_title("KV-stationary n=0 lmc=16\n(cycle-assumption model, lower MAC SCALE-Sim corrected)", fontsize=10)
ax.set_ylabel("Cycles", fontsize=11)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v*1.02, f"{v/1e6:.2f}M",
            ha='center', fontsize=9)
ax.set_ylim(0, max(vals) * 1.25)

# Annotate column_dwell inside first bar
ax.text(0, pipeline_cyc/2,
        f"column_dwell = {col_dwell} cycles\n= d({128}) + exp({4}) + V-accum({3})\nper KV step × {pipeline_steps} steps",
        ha='center', va='center', fontsize=8, color='white',
        bbox=dict(boxstyle='round', fc='none', ec='none'))

plt.tight_layout()
plt.savefig(OUT / "fig4_cycle_breakdown_T8192.png", dpi=150)
plt.close()
print("Saved fig4_cycle_breakdown_T8192.png")

# ── Fig 5: Summary bar chart — area-norm speedup at T=4096 ───────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

configs  = ["n=0\nlmc=1", "n=0\nlmc=16", "n=3\nlmc=1", "n=3\nlmc=16"]
an_vals  = [kv01_an[4096], kv016_an[4096], kv31_an[4096], kv316_an[4096]]
colors   = [C_KV01, C_KV016, "purple", C_KV316]
bars = ax.bar(configs, an_vals, color=colors, edgecolor='white', width=0.5)

ax.axhline(1.0, color='k', lw=1.2, ls='--', alpha=0.6, label="FlashAttention baseline (1×)")
for bar, v in zip(bars, an_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.2f}×",
            ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel("Area-normalised speedup over FlashAttention", fontsize=11)
ax.set_title("KV-stationary prefill speedup per unit silicon at T=4096\n"
             "(lower MAC SCALE-Sim validated, upper PE cycle-assumption model)", fontsize=11)
ax.set_ylim(0, 2.3)
ax.legend(fontsize=10)
ax.grid(axis='y', ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig5_summary_bar_T4096.png", dpi=150)
plt.close()
print("Saved fig5_summary_bar_T4096.png")

print(f"\nAll figures saved to {OUT}/")
