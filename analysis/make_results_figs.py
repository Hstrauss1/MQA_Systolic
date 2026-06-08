"""Core Results-section figures (wavefront model).

  fig17_prefill_raw_vs_permac.png  — Table 6 visualized: raw speedup (left) and
                                      per-physical-MAC speedup (right) vs T, 3 configs.
  fig18_dram_breakdown.png         — DRAM traffic: unfused baseline (score-matrix
                                      dominated) vs fused (KV-stat / FlashAttention).

Both at H=64, d=128, FP16, BW=512 B/cyc, prefill. Styled to match plots_reuse/.
"""

from __future__ import annotations
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from kv_stationary_model import simulate_2d_kv_stationary_array as sim

H, d, bpe, BW = 64, 128, 2, 512
FLASH_PE = 64 * 64
STYLE = dict(linewidth=2.2, marker="o", markersize=6)
CFG_C = {"n=0, lmc=1": "#4C72B0", "n=0, lmc=16": "#C44E52", "n=3, lmc=16": "#55A868"}

flash = {}
with open("../old_dwell_model/prefill_full_results.csv") as f:
    for r in csv.DictReader(f):
        if r["arch"] == "kv_stat_n0_lmc1":
            flash[int(r["T"])] = float(r["flash_total_cycles"])

T_vals = [512, 1024, 2048, 4096, 8192]
CONFIGS = [("n=0, lmc=1", 0, 1), ("n=0, lmc=16", 0, 16), ("n=3, lmc=16", 3, 16)]


def metrics(n, lmc, T):
    rows = H * (2 ** n); cols = T // (2 ** n)
    m = sim(H=H, T=T, d=d, array_rows=rows, array_cols=cols, bytes_per_element=bpe,
            memory_bandwidth_bytes_per_cycle=BW, exp_latency_cycles=4, pe_mac_width=128,
            lower_mac_count=lmc, merge_extensions=n, query_tokens=T, wavefront_fill=True)
    raw = flash[T] / m["total_cycles"]
    per_mac = raw / ((rows * cols * lmc) / FLASH_PE)
    return raw, per_mac


def metrics_compute(n, lmc, T):
    """Compute-only (DRAM floor removed): total = compute_cycles."""
    rows = H * (2 ** n); cols = T // (2 ** n)
    m = sim(H=H, T=T, d=d, array_rows=rows, array_cols=cols, bytes_per_element=bpe,
            memory_bandwidth_bytes_per_cycle=BW, exp_latency_cycles=4, pe_mac_width=128,
            lower_mac_count=lmc, merge_extensions=n, query_tokens=T, wavefront_fill=True)
    raw = flash[T] / m["compute_cycles"]
    per_mac = raw / ((rows * cols * lmc) / FLASH_PE)
    return raw, per_mac


# ═════════════════════════════════════════════════════════════════════════════
# Fig 17 — raw speedup (left) and per-MAC speedup (right) vs T
# ═════════════════════════════════════════════════════════════════════════════
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

for label, n, lmc in CONFIGS:
    raws = [metrics(n, lmc, T)[0] for T in T_vals]
    pms  = [metrics(n, lmc, T)[1] for T in T_vals]
    axL.semilogx(T_vals, raws, color=CFG_C[label], label=label, **STYLE)
    axR.semilogx(T_vals, pms,  color=CFG_C[label], label=label, **STYLE)
    axL.annotate(f" {raws[-1]:.0f}×", xy=(T_vals[-1], raws[-1]), fontsize=8.5,
                 color=CFG_C[label], va="center", fontweight="bold")
    axR.annotate(f" {pms[-1]:.2f}×", xy=(T_vals[-1], pms[-1]), fontsize=8.5,
                 color=CFG_C[label], va="center", fontweight="bold")

axL.set_yscale("log")
axL.set_xlabel("Sequence length T", fontsize=12)
axL.set_ylabel("Raw cycle speedup vs FlashAttention", fontsize=12)
axL.set_title("(a) Raw speedup — wall-clock", fontsize=11)
axL.legend(fontsize=9, loc="upper left")
axL.grid(True, which="both", ls="--", alpha=0.3)
axL.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axL.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}×"))

axR.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6, label="break-even (1×)")
axR.set_xlabel("Sequence length T", fontsize=12)
axR.set_ylabel("Speedup per physical MAC", fontsize=12)
axR.set_title("(b) Per physical MAC — silicon-normalized", fontsize=11)
axR.set_ylim(0, 1.25)
axR.legend(fontsize=9, loc="center right")
axR.grid(True, which="both", ls="--", alpha=0.3)
axR.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axR.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}×"))

fig.suptitle("Prefill vs FlashAttention: huge raw speedup, but below break-even per MAC "
             "(512 B/cyc)", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("../results/figures/fig17_prefill_raw_vs_permac.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig17")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 18 — DRAM traffic breakdown: unfused vs fused, T=8192
# ═════════════════════════════════════════════════════════════════════════════
T = 8192
GB = 1e9
q   = H * T * d * bpe / GB          # query reads
kv  = 2 * T * d * bpe / GB          # K + V (MQA: shared, no H)
out = H * T * d * bpe / GB          # output writes
score = H * T * T * bpe * 2 / GB    # score matrix write+read (unfused only)

unfused = [("Q reads", q), ("K+V", kv), ("Output", out), ("Score matrix", score)]
fused   = [("Q reads", q), ("K+V", kv), ("Output", out)]
COMP_C = {"Q reads": "#4C72B0", "K+V": "#DD8452", "Output": "#9467BD", "Score matrix": "#C44E52"}

fig, ax = plt.subplots(figsize=(7.5, 5))
for x, stack, name in [(0, unfused, "Unfused\nbaseline"),
                       (1, fused, "Fused\n(KV-stat / FlashAttn)")]:
    bottom = 0
    for comp, val in stack:
        ax.bar(x, val, 0.55, bottom=bottom, color=COMP_C[comp],
               label=comp if x == 0 else None, edgecolor="white", linewidth=0.5)
        bottom += val
    ax.annotate(f"{bottom:.2f} GB", xy=(x, bottom), ha="center", va="bottom",
                fontsize=10, fontweight="bold")

ax.annotate("Score matrix =\n98.4% of traffic\n(H·T² tensor)", xy=(0, score * 0.6),
            xytext=(0.42, 11), fontsize=9, color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))
ax.annotate("65× less\n(no score matrix)", xy=(1, 0.4), xytext=(1.05, 4.5),
            fontsize=9, color="#333",
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

ax.set_xticks([0, 1])
ax.set_xticklabels(["Unfused\nbaseline", "Fused\n(KV-stat / FlashAttn)"], fontsize=10)
ax.set_ylabel("DRAM traffic  (GB, prefill at T=8192)", fontsize=12)
ax.set_title("The 65× advantage is score-matrix elimination, not the dataflow\n"
             "both fused designs (KV-stat and FlashAttention) remove the H·T² tensor",
             fontsize=11)
ax.legend(fontsize=9, loc="upper right", title="DRAM component")
ax.grid(True, axis="y", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig18_dram_breakdown.png", dpi=150)
plt.close()
print("Saved fig18")
print(f"  (unfused total {q+kv+out+score:.2f} GB, fused {q+kv+out:.3f} GB, "
      f"ratio {(q+kv+out+score)/(q+kv+out):.0f}×, score {100*score/(q+kv+out+score):.1f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 19 — Table 7 (compute-only): raw + per-MAC vs T, DRAM floor removed
# ═════════════════════════════════════════════════════════════════════════════
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

for label, n, lmc in CONFIGS:
    raws = [metrics_compute(n, lmc, T)[0] for T in T_vals]
    pms  = [metrics_compute(n, lmc, T)[1] for T in T_vals]
    axL.semilogx(T_vals, raws, color=CFG_C[label], label=label, **STYLE)
    axR.semilogx(T_vals, pms,  color=CFG_C[label], label=label, **STYLE)
    axL.annotate(f" {raws[-1]:,.0f}×", xy=(T_vals[-1], raws[-1]), fontsize=8.5,
                 color=CFG_C[label], va="center", fontweight="bold")
    axR.annotate(f" {pms[-1]:.2f}×", xy=(T_vals[-1], pms[-1]), fontsize=8.5,
                 color=CFG_C[label], va="center", fontweight="bold")

axL.set_yscale("log")
axL.set_xlabel("Sequence length T", fontsize=12)
axL.set_ylabel("Raw cycle speedup vs FlashAttention", fontsize=12)
axL.set_title("(a) Raw speedup — compute-only (no DRAM floor)", fontsize=11)
axL.legend(fontsize=9, loc="upper left")
axL.grid(True, which="both", ls="--", alpha=0.3)
axL.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axL.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}×"))

axR.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6, label="break-even (1×)")
axR.set_xlabel("Sequence length T", fontsize=12)
axR.set_ylabel("Speedup per physical MAC", fontsize=12)
axR.set_title("(b) Per physical MAC — compute-only", fontsize=11)
axR.set_ylim(0, 2.0)
axR.legend(fontsize=9, loc="center right")
axR.grid(True, which="both", ls="--", alpha=0.3)
axR.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axR.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}×"))

fig.suptitle("Table 7 (compute-only): with the memory wall removed, lmc=16 reaches break-even "
             "and merge (n=3) exceeds it", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("../results/figures/fig19_prefill_compute_only.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig19")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 20 — combined 2x2: rows = {raw, per-MAC}, cols = {DRAM-bound, compute-only}
# (replaces fig17 + fig19 with one compact side-by-side float)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
(axRawM, axRawC), (axPmM, axPmC) = axes

def _plot(ax, fn, idx):
    for label, n, lmc in CONFIGS:
        ys = [fn(n, lmc, T)[idx] for T in T_vals]
        ax.semilogx(T_vals, ys, color=CFG_C[label], label=label, **STYLE)
        ax.annotate(f" {ys[-1]:,.0f}×" if idx == 0 else f" {ys[-1]:.2f}×",
                    xy=(T_vals[-1], ys[-1]), fontsize=8, color=CFG_C[label],
                    va="center", fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Row 1: raw speedup (log y)
for ax, fn, ttl in [(axRawM, metrics, "(a) Raw speedup — DRAM-bound (512 B/cyc)"),
                    (axRawC, metrics_compute, "(b) Raw speedup — compute-only")]:
    _plot(ax, fn, 0); ax.set_yscale("log"); ax.set_title(ttl, fontsize=10.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}×"))
axRawM.set_ylabel("Raw cycle speedup", fontsize=11)
axRawM.legend(fontsize=8.5, loc="upper left")

# Row 2: per-MAC (shared linear y for direct comparison)
for ax, fn, ttl in [(axPmM, metrics, "(c) Per physical MAC — DRAM-bound"),
                    (axPmC, metrics_compute, "(d) Per physical MAC — compute-only")]:
    _plot(ax, fn, 1); ax.set_ylim(0, 2.0); ax.set_title(ttl, fontsize=10.5)
    ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}×"))
    ax.set_xlabel("Sequence length T", fontsize=11)
axPmM.set_ylabel("Speedup per physical MAC", fontsize=11)

fig.suptitle("KV-stat vs FlashAttention, both bandwidth regimes "
             "(left: DRAM-bound at 512 B/cyc · right: compute-only / HBM limit)",
             fontsize=12, y=0.995)
plt.tight_layout()
plt.savefig("../results/figures/fig20_prefill_both_regimes.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig20")
