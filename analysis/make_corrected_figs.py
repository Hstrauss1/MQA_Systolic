"""Corrected-model (wavefront B) + SCALE-Sim-validation figure set.

Produces fig13–16 in plots_reuse/, styled to match the existing fig7–12 family:
  semilogx · linewidth=2.2 · marker='o' ms=6 · figsize≈(8,5) · dpi=150 · grid both/--/0.3

  fig13_scalesim_fill_validation.png  — measured per-column fill vs A(135) vs B  [SCALE-Sim]
  fig14_lmc_bw_codesign.png           — per-MAC speedup vs lmc, one line per BW (lmc≈BW/128)
  fig15_compute_vs_dram_floor.png     — compute vs flat DRAM floor vs lmc (the memory wall)
  fig16_speedup_vs_bandwidth.png      — total speedup vs BW with the compute-floor knee

Data sources:
  validate_sweep_scalesim.csv  (40 SCALE-Sim Q·K GEMM runs)
  reuse_full_sweep.csv         (750-row BW×config×P analytical sweep, B model)
  kv_stationary_model.simulate (fresh compute/memory split where needed)
"""

from __future__ import annotations
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from kv_stationary_model import simulate_2d_kv_stationary_array as sim

H, d, bpe = 64, 128, 2
COL_DWELL = 135
FLASH = 64 * 64

BW_C = {512: "#4C72B0", 1024: "#DD8452", 2048: "#55A868"}
LMC_C = {1: "#4C72B0", 2: "#DD8452", 4: "#55A868", 8: "#9467BD", 16: "#C44E52"}
STYLE = dict(linewidth=2.2, marker="o", markersize=6)

flash_cycles = {}
with open("prefill_full_results.csv") as f:
    for r in csv.DictReader(f):
        if r["arch"] == "kv_stat_n0_lmc1":
            flash_cycles[int(r["T"])] = float(r["flash_total_cycles"])


# ═════════════════════════════════════════════════════════════════════════════
# Fig 13 — SCALE-Sim fill validation: measured cyc/col vs A (135) vs B
# ═════════════════════════════════════════════════════════════════════════════
val = list(csv.DictReader(open("../results/data/validate_sweep_scalesim.csv")))

fig, ax = plt.subplots(figsize=(8, 5))
# measured per-column fill vs eff_cols (fill is lmc-independent → colour by T)
xs = [int(r["eff_cols"]) for r in val]
ys = [float(r["per_col_fill"]) for r in val]
ax.scatter(xs, ys, s=42, color="#55A868", zorder=3, edgecolor="k", linewidth=0.4,
           label="SCALE-Sim measured (40 Q·K GEMMs)")
ax.axhline(COL_DWELL, color="#C44E52", ls="--", lw=2.2,
           label="Dwell model (assumed): 135 cyc/col")
ax.axhline(2.5, color="#4C72B0", ls=":", lw=2.0,
           label="measured regime ≈ 2.5 cyc/col")
ax.annotate("~50× too high\n→ ruled out", xy=(800, 135), xytext=(180, 40),
            fontsize=9, color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.3))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Sub-array columns (eff_cols)", fontsize=12)
ax.set_ylabel("Pipeline fill  (cycles / column)", fontsize=12)
ax.set_title("SCALE-Sim validates the wavefront fill, rules out the dwell model\n"
             "Q·K dot-product stage on a cycle-accurate systolic array", fontsize=11)
ax.set_ylim(1, 300)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax.legend(fontsize=9, loc="center right")
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig13_scalesim_fill_validation.png", dpi=150)
plt.close()
print("Saved fig13")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 14 — lmc–BW co-design: per-MAC speedup vs lmc, one line per BW
# ═════════════════════════════════════════════════════════════════════════════
sweep = list(csv.DictReader(open("../results/data/reuse_full_sweep.csv")))
LMC = [1, 2, 4, 8, 16]
T_fix, P_fix, n_fix = 8192, 1, 0

fig, ax = plt.subplots(figsize=(8, 5))
for BW in [512, 1024, 2048]:
    ys, bounds = [], []
    for lmc in LMC:
        r = next(r for r in sweep if int(r["BW"]) == BW and int(r["merge_n"]) == n_fix
                 and int(r["lmc"]) == lmc and int(r["T"]) == T_fix and int(r["P"]) == P_fix)
        ys.append(float(r["speedup_per_mac"])); bounds.append(r["bound"])
    ax.plot(LMC, ys, color=BW_C[BW], label=f"BW={BW} B/cyc", **STYLE)
    # mark the design-rule balance point lmc≈BW/128 (the knee where compute=DRAM)
    bal = min(LMC, key=lambda x: abs(x - BW // 128))
    by = ys[LMC.index(bal)]
    ax.scatter([bal], [by], s=190, marker="*", color=BW_C[BW], zorder=5,
               edgecolor="k", linewidth=0.6)
ax.set_xscale("log", base=2)
ax.set_xticks(LMC); ax.set_xticklabels(LMC)
ax.set_xlabel("Lower MAC count  (lmc)", fontsize=12)
ax.set_ylabel("Speedup per physical MAC", fontsize=12)
ax.set_title("lmc–bandwidth co-design  (T=8192, single pass)\n"
             r"★ = balance point (lmc≈BW/128); past it, extra MACs idle → efficiency halves",
             fontsize=11)
ax.legend(fontsize=9, loc="lower left")
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig14_lmc_bw_codesign.png", dpi=150)
plt.close()
print("Saved fig14")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 15 — compute vs flat DRAM floor vs lmc  (the memory wall)
# ═════════════════════════════════════════════════════════════════════════════
T, BW = 8192, 512
comp, mem, idle = [], [], []
for lmc in LMC:
    m = sim(H=H, T=T, d=d, array_rows=H, array_cols=T, bytes_per_element=bpe,
            memory_bandwidth_bytes_per_cycle=BW, exp_latency_cycles=4, pe_mac_width=128,
            lower_mac_count=lmc, merge_extensions=0, query_tokens=T, wavefront_fill=True)
    c, mm = m["compute_cycles"], m["memory_service_cycles"]
    comp.append(c); mem.append(mm)
    idle.append(100 * (1 - c / mm) if mm >= c else 0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(LMC, [c / 1e3 for c in comp], color="#4C72B0", label="Compute (∝ 1/lmc)", **STYLE)
ax.plot(LMC, [mm / 1e3 for mm in mem], color="#C44E52", ls="--", lw=2.2,
        marker="s", markersize=6, label="DRAM floor (fixed, ∝ 1/BW)")
ax.fill_between(LMC, [mm / 1e3 for mm in mem], [c / 1e3 for c in comp],
                where=[c < mm for c, mm in zip(comp, mem)], color="#C44E52", alpha=0.12)
# balance point
bal = next(lmc for lmc, c, mm in zip(LMC, comp, mem) if c <= mm)
ax.axvline(bal, color="gray", ls=":", lw=1.5)
ax.annotate(f"balance  lmc={bal}\n(array fully fed)", xy=(bal, mem[LMC.index(bal)] / 1e3),
            xytext=(bal * 1.1, 1400), fontsize=9, color="gray")
ax.annotate(f"lmc=16: array {idle[-1]:.0f}% idle\n(memory-bound, wasted MACs)",
            xy=(16, comp[-1] / 1e3), xytext=(3.0, 250), fontsize=9, color="#4C72B0",
            arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.2))
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(LMC); ax.set_xticklabels(LMC)
ax.set_xlabel("Lower MAC count  (lmc)", fontsize=12)
ax.set_ylabel("Cycles  (×10³)", fontsize=12)
ax.set_title("Compute drops below the DRAM floor — the memory wall\n"
             f"T={T:,}, n=0, single pass, BW={BW} B/cyc", fontsize=11)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig15_compute_vs_dram_floor.png", dpi=150)
plt.close()
print("Saved fig15")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 16 — total speedup vs bandwidth, with the compute-floor knee
# ═════════════════════════════════════════════════════════════════════════════
T, lmc, n = 8192, 16, 0
BWs = [256, 512, 1024, 2048, 4096, 8192]
spd, totals = [], []
for BW in BWs:
    m = sim(H=H, T=T, d=d, array_rows=H, array_cols=T, bytes_per_element=bpe,
            memory_bandwidth_bytes_per_cycle=BW, exp_latency_cycles=4, pe_mac_width=128,
            lower_mac_count=lmc, merge_extensions=n, query_tokens=T, wavefront_fill=True)
    totals.append(m["total_cycles"]); spd.append(flash_cycles[T] / m["total_cycles"])
comp_floor = min(totals)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(BWs, spd, color="#9467BD", **STYLE)
knee = 2070
ax.axvline(knee, color="gray", ls=":", lw=1.5)
ax.annotate("compute-floor knee\n~2 TB/s  (≈2070 B/cyc)", xy=(knee, flash_cycles[T] / comp_floor),
            xytext=(560, flash_cycles[T] / comp_floor * 0.62), fontsize=9, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
ax.annotate("past the knee: more\nbandwidth wasted\n(compute-bound)",
            xy=(4096, spd[BWs.index(4096)]), xytext=(2600, spd[1] * 1.05), fontsize=9, color="#9467BD")
ax.set_xscale("log", base=2)
ax.set_xticks(BWs)
ax.set_xticklabels([f"{b}" for b in BWs], rotation=45, fontsize=8)
ax.set_xlabel("DRAM bandwidth  (B/cycle)", fontsize=12)
ax.set_ylabel("Total speedup vs FlashAttention", fontsize=12)
ax.set_title("Speedup saturates at the compute floor (lmc=16, T=8192)\n"
             "bandwidth helps up to ~4× (0.5→2 TB/s), then stops mattering", fontsize=11)
ax.legend(["lmc=16 KV-stat"], fontsize=9, loc="upper left")
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig16_speedup_vs_bandwidth.png", dpi=150)
plt.close()
print("Saved fig16")

print("\nfig13–16 → plots_reuse/")
