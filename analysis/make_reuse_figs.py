"""Publication-quality figures for the multi-pass silicon re-use model.

Visual style matches plots_prefill_corrected/fig3_area_norm_speedup.png:
  semilogx  ·  linewidth=2.2  ·  marker='o' markersize=6
  figsize=(8,5)  ·  dpi=150  ·  grid both/--/alpha=0.3

Outputs → plots_reuse/
  fig7_area_norm_throughput_decode.png   — area-norm throughput (×10⁻⁷) vs T, per P (decode)
  fig8_latency_overhead_normalised.png   — latency multiplier (P-pass / 1-pass) vs T, per P
  fig9_causal_savings_vs_T.png           — causal cycle savings % vs T, per P
  fig10_decode_vs_prefill_area_norm.png  — decode vs causal prefill area-norm speedup (norm. by decode P=1)
  fig11_area_norm_speedup_reuse.png      — causal prefill area-norm speedup over full array, per P
  fig12_speedup_per_mac.png              — speedup per physical MAC vs T, per P (is the silicon worth it?)
"""

from __future__ import annotations
import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from kv_reuse_model import kv_stationary_metrics

# ── Hardware constants (match CLAUDE.md §2) ───────────────────────────────────
H            = 64
d            = 128
bpe          = 2
BW           = 512
pe_mac_width = 128
lmc          = 16
exp_latency  = 4
n_merge      = 0

T_values = [128, 512, 1024, 2048, 4096, 8192]
P_values = [1, 2, 4, 8, 16]

# ── Style — matches plot_prefill_results.py exactly ───────────────────────────
STYLE = dict(linewidth=2.2, marker='o', markersize=6)
# One colour per P value
C = {
    1:  "#4C72B0",   # blue
    2:  "#DD8452",   # orange
    4:  "#55A868",   # green
    8:  "#9467BD",   # purple
    16: "#C44E52",   # red
}

def _fmt_T(x, _):
    return f"{int(x):,}"

# ── Sweep ─────────────────────────────────────────────────────────────────────
decode:           Dict[Tuple[int, int], dict] = {}
prefill_causal:   Dict[Tuple[int, int], dict] = {}
prefill_noncausal: Dict[Tuple[int, int], dict] = {}

for T in T_values:
    for P in P_values:
        cols = max(1, T // P)
        common = dict(
            H=H, T=T, d=d, array_rows=H, array_cols=cols,
            bytes_per_element=bpe,
            memory_bandwidth_bytes_per_cycle=BW,
            exp_latency_cycles=exp_latency,
            pe_mac_width=pe_mac_width,
            lower_mac_count=lmc,
            merge_extensions=n_merge,
            num_passes=P,
        )
        decode[(T, P)]            = kv_stationary_metrics(**common, query_tokens=1, causal=False)
        prefill_causal[(T, P)]    = kv_stationary_metrics(**common, query_tokens=T, causal=True)
        prefill_noncausal[(T, P)] = kv_stationary_metrics(**common, query_tokens=T, causal=False)

os.makedirs("plots_reuse", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Area-normalised throughput vs T — one line per P (decode)
#
# Raw metric: T / (total_cycles × pe_count)   [tok / (cycle × PE)]
# Scaled ×10⁷ so values sit in a 0–20 range — avoids scientific notation
# while preserving absolute comparability across P values.
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

SCALE7 = 1e7

for P in P_values:
    ys = [
        SCALE7 * T / (decode[(T, P)]["total_cycles_multipass"] * decode[(T, P)]["pe_count"])
        for T in T_values
    ]
    ax.semilogx(T_values, ys, color=C[P], label=f"P={P}", **STYLE)
    ax.annotate(f"  P={P}", xy=(T_values[-1], ys[-1]),
                fontsize=8.5, color=C[P], va="center", fontweight="bold")

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel(r"Throughput  (×10$^{-7}$ tok / cycle / PE)", fontsize=12)
ax.set_title(
    "Area-normalised throughput vs T — multi-pass decode\n"
    "(higher P = smaller chip; higher = more silicon-efficient)",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_T))
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(T_values[0] * 0.75, T_values[-1] * 3)
ax.set_ylim(bottom=0)
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig7_area_norm_throughput_decode.png", dpi=150)
plt.close()
print("Saved fig7")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Latency overhead vs T — cycle multiplier (P-pass / 1-pass), per P
#
# Y = total_cycles_P(T) / total_cycles_1(T)
# Shows that overhead is sub-linear in P and converges toward 1× as T grows
# because pipeline drain dominates and shrinks proportionally with column count.
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for P in [2, 4, 8, 16]:
    ys = [
        decode[(T, P)]["total_cycles_multipass"] / decode[(T, 1)]["total_cycles_multipass"]
        for T in T_values
    ]
    ax.semilogx(T_values, ys, color=C[P], label=f"P={P}", **STYLE)
    ax.annotate(f"  {ys[-1]:.2f}×", xy=(T_values[-1], ys[-1]),
                fontsize=8.5, color=C[P], va="center", fontweight="bold")

ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.5, label="No overhead  (1×)")

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Cycle multiplier vs P=1", fontsize=12)
ax.set_title(
    "Latency overhead of multi-pass re-use vs T  (decode)\n"
    "(overhead shrinks as T grows — drain dominates at large T)",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_T))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}×"))
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(T_values[0] * 0.75, T_values[-1] * 3)
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig8_latency_overhead_normalised.png", dpi=150)
plt.close()
print("Saved fig8")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: Causal masking cycle savings (%) vs T — one line per P
# savings = 1 - causal_cycles / noncausal_cycles
# Driven by: fewer Q reads per pass + streaming output per pass
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for P in [2, 4, 8, 16]:
    ys = [
        100.0 * (
            prefill_noncausal[(T, P)]["total_cycles_multipass"]
            - prefill_causal[(T, P)]["total_cycles_causal"]
        ) / prefill_noncausal[(T, P)]["total_cycles_multipass"]
        for T in T_values
    ]
    ax.semilogx(T_values, ys, color=C[P], label=f"P={P}", **STYLE)
    ax.annotate(f"  {ys[-1]:.0f}%", xy=(T_values[-1], ys[-1]),
                fontsize=8.5, color=C[P], va="center", fontweight="bold")

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Cycle savings vs non-causal (%)", fontsize=12)
ax.set_title(
    "Causal masking savings in multi-pass prefill vs T\n"
    "(savings grow with P: each pass discards more finished queries + Q reads)",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_T))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(T_values[0] * 0.75, T_values[-1] * 3)
ax.set_ylim(-5, 80)
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig9_causal_savings_vs_T.png", dpi=150)
plt.close()
print("Saved fig9")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: Decode vs causal prefill — area-norm speedup vs T, P=1 and P=16
#
# Y = P × cycles_decode1(T) / cycles_mode_P(T)
#   = [tput(mode,T,P)] / [tput(decode,T,1)]     (same pe_count normalisation)
#
# decode P=1 = 1.0× reference; decode P>1 > 1×; prefill P=1 ≤ 1×.
# Shows both the mode difference and the re-use benefit on a single dimensionless axis.
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for P, dec_ls, pf_ls, pf_marker in [(1, "-", ":", "s"), (16, "--", ":", "s")]:
    ref = {T: decode[(T, 1)]["total_cycles_multipass"] for T in T_values}

    dec_ys = [P * ref[T] / decode[(T, P)]["total_cycles_multipass"]         for T in T_values]
    pf_ys  = [P * ref[T] / prefill_causal[(T, P)]["total_cycles_causal"]    for T in T_values]

    ax.semilogx(T_values, dec_ys, color=C[P], ls=dec_ls,
                label=f"Decode  P={P}", **STYLE)
    ax.semilogx(T_values, pf_ys,  color=C[P], ls=pf_ls,
                label=f"Causal prefill  P={P}",
                linewidth=2.2, marker=pf_marker, markersize=6)

    ax.annotate(f"  {dec_ys[-1]:.1f}×", xy=(T_values[-1], dec_ys[-1]),
                fontsize=8.5, color=C[P], va="center", fontweight="bold")
    ax.annotate(f"  {pf_ys[-1]:.1f}×",  xy=(T_values[-1], pf_ys[-1]),
                fontsize=8.5, color=C[P], va="center")

ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.5, label="Decode P=1  (1×)")

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Speedup per unit silicon  (vs decode P=1)", fontsize=12)
ax.set_title(
    "Area-normalised speedup: decode vs causal prefill\n"
    "solid circle = decode  ·  dotted square = causal prefill  ·  blue = P=1, red = P=16",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_T))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(T_values[0] * 0.75, T_values[-1] * 3)
ax.set_ylim(bottom=0)
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig10_decode_vs_prefill_area_norm.png", dpi=150)
plt.close()
print("Saved fig10")


# ─────────────────────────────────────────────────────────────────────────────
# Load prefill_full_results.csv — needed for fig11 and fig12.
# flash_total_cycles is the single 64×64 FlashAttention prefill baseline.
# area_norm formula (same as fig3):
#   area_norm = flash_total × H × P / (causal_cycles × T)
# ─────────────────────────────────────────────────────────────────────────────
import csv as _csv

_src_rows = []
with open("../old_dwell_model/prefill_full_results.csv") as _f:
    for _r in _csv.DictReader(_f):
        _src_rows.append({k: (float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v)
                          for k, v in _r.items()})

def _get(arch, key):
    return {int(r["T"]): r[key] for r in _src_rows if r["arch"] == arch}

flash_total = _get("kv_stat_n0_lmc1", "flash_total_cycles")
kv316_an    = _get("kv_stat_n3_lmc16", "area_norm_speedup")
T_shared    = [T for T in T_values if T in flash_total]


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11: Area-normalised speedup of multi-pass re-use vs FlashAttention
#
# Same formula as fig3 and fig12's re-use lines:
#   area_norm = flash_total[T] × H × P / (causal_cycles × T)
#             = (flash_cycles / kv_cycles) / (kv_PEs / flash_PEs)
#
# kv_PEs = H × (T/P)  ·  flash_PEs = H² → PE ratio = T/(P×H)
# Direct drop-in equivalent of fig3 for the re-use model.
#
# Also overlays n=3 lmc=16 (single-pass, from prefill_full_results.csv) so the
# re-use lines can be compared directly against the merge-extension baseline
# from fig3.  n=3 lmc=16 at T=8192 uses 1,024 cols — same footprint as P=8.
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

C_P = {
    1:  "#4C72B0",
    2:  "#DD8452",
    4:  "#55A868",
    8:  "#9467BD",
    16: "#C44E52",
}

# n=3 lmc=16 single-pass from fig3 (CSV data, same area_norm formula)
n3_xs = [T for T in sorted(kv316_an) if T in T_shared]
n3_ys = [kv316_an[T] for T in n3_xs]
ax.semilogx(n3_xs, n3_ys, color="#8c564b", ls="--", marker="s", markersize=6,
            linewidth=2.2, label="n=3  lmc=16  (single-pass, merge ext.)")
ax.annotate(f"  {n3_ys[-1]:.2f}×", xy=(n3_xs[-1], n3_ys[-1]),
            fontsize=9, color="#8c564b", va="center")

for P in [1, 2, 4, 8, 16]:
    ys = [
        flash_total[T] * H * P / (prefill_causal[(T, P)]["total_cycles_causal"] * T)
        for T in T_shared
    ]
    label = f"P={P}  ({T_shared[-1]//P:,} cols)" if P > 1 else f"P=1  n=0  lmc=16  (single-pass)"
    ax.semilogx(T_shared, ys, color=C_P[P], label=label, **STYLE)
    ax.annotate(f"  {ys[-1]:.2f}×", xy=(T_shared[-1], ys[-1]),
                fontsize=9, color=C_P[P], va="center", fontweight="bold")

ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.5, label="Break-even  (1×)")

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Speedup per unit silicon", fontsize=12)
ax.set_title(
    "Area-normalised speedup: KV-stationary over FlashAttention  (causal prefill)\n"
    r"(flash_cycles / kv_cycles) / (kv_PEs / flash_PEs)  —  same metric as fig3",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_T))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
ax.set_xlim(T_shared[0] * 0.75, T_shared[-1] * 3)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig11_area_norm_speedup_reuse.png", dpi=150)
plt.close()
print("Saved fig11")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 12: Speedup per PHYSICAL MAC vs T — is the silicon scaling worth it?
#
# The earlier "area-norm" figures (fig3, fig11) normalise raw speedup by PE
# *count* only.  That hides the real cost: every KV-stationary PE packs
# lower_mac_count (=16) serial MACs, while every FlashAttention PE is a single
# MAC.  Normalising by PE count therefore overstates KV-stat efficiency by a
# flat ~lmc× factor.  This figure normalises by the actual physical MAC unit
# count so the y-axis is honest "speedup per multiplier of silicon".
#
#   kv_phys_macs    = PE_count × lmc          = (H × T/P) × lmc
#   flash_phys_macs = flash_PEs × 1           = 64 × 64 = 4096   (= H²)
#   mac_ratio       = kv_phys_macs / flash_phys_macs = (T × lmc) / (P × H)
#   raw_speedup     = flash_total / kv_cycles
#   speedup_per_mac = raw_speedup / mac_ratio
#                   = flash_total × P × H / (kv_cycles × T × lmc)
#
# Reading the plot:
#   • A line ABOVE 1×  → KV-stat returns more speedup per multiplier than flash.
#   • A line BELOW 1×  → the extra silicon (the lmc MACs) is NOT paying for
#                        itself versus a plain systolic array.
#   • Higher P = fewer columns = fewer physical MACs → climbs toward / past 1×,
#     directly answering "does shrinking the chip via re-use make it worth it?"
#
# Lines are the multi-pass re-use family P∈{1,2,4,8,16} (causal prefill, model),
# with the n=3 lmc=16 merge-extension design overlaid as an alternative scaling
# strategy (same PE/MAC count as n=0 at equal T → simply its fig11 value / lmc).
# ─────────────────────────────────────────────────────────────────────────────
kv016_an = _get("kv_stat_n0_lmc16", "area_norm_speedup")  # (kept for reference)

flash_phys_macs = 64 * 64        # 64×64 FlashAttention array, 1 MAC per PE

def _speedup_per_mac(T, P):
    """Raw speedup ÷ (kv physical MAC count / flash physical MAC count)."""
    kv_cycles = prefill_causal[(T, P)]["total_cycles_causal"]
    kv_pe     = prefill_causal[(T, P)]["pe_count"]      # H × (T/P)
    kv_phys_macs = kv_pe * lmc
    mac_ratio    = kv_phys_macs / flash_phys_macs
    raw_speedup  = flash_total[T] / kv_cycles
    return raw_speedup / mac_ratio

fig, ax = plt.subplots(figsize=(9, 5.5))

# ── Multi-pass re-use family (causal prefill) — one line per P ─────────────────
for P in P_values:                                   # [1, 2, 4, 8, 16]
    ys = [_speedup_per_mac(T, P) for T in T_shared]
    label = ("P=1  (full array, n=0 lmc=16)" if P == 1
             else f"P={P}  ({T_shared[-1]//P:,} cols @ T={T_shared[-1]:,})")
    ax.semilogx(T_shared, ys, color=C[P], label=label, **STYLE)
    ax.annotate(f"  {ys[-1]:.2f}×", xy=(T_shared[-1], ys[-1]),
                fontsize=8.5, color=C[P], va="center", fontweight="bold")

# ── n=3 lmc=16 merge-extension reference — computed from the B model ──────────
# Under the wavefront (B) fill, prefill is DRAM-bound, so n=3 (merge-invariant PE/MAC
# count) collapses onto the n=0 single-pass line: merge extensions buy nothing against
# the memory wall.  Computed here (not from the stale A-model CSV) so it stays B-consistent.
def _n3_spm(T):
    m = kv_stationary_metrics(
        H=H, T=T, d=d, array_rows=H * 8, array_cols=max(1, T // 8),
        bytes_per_element=bpe, memory_bandwidth_bytes_per_cycle=BW,
        exp_latency_cycles=exp_latency, pe_mac_width=pe_mac_width,
        lower_mac_count=lmc, merge_extensions=3, num_passes=1,
        query_tokens=T, causal=True,
    )
    return (flash_total[T] / m["total_cycles_causal"]) / ((m["pe_count"] * lmc) / flash_phys_macs)

n3_ys = [_n3_spm(T) for T in T_shared]
ax.semilogx(T_shared, n3_ys, color="#8c564b", ls="--", marker="s", markersize=6,
            linewidth=2.2, label="n=3  lmc=16  single-pass (collapses onto P=1, DRAM-bound)")
ax.annotate(f"  {n3_ys[-1]:.2f}×", xy=(T_shared[-1], n3_ys[-1]),
            fontsize=8.5, color="#8c564b", va="center")

ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6,
           label="Break-even vs FlashAttention  (1× per MAC)")

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel(f"Speedup per physical MAC\n(KV-stat MACs = PE×{lmc}, flash = PE×1)",
              fontsize=11)
ax.set_title(
    f"Is the silicon worth it?  Speedup per physical MAC vs FlashAttention\n"
    f"wavefront-fill model, BW={BW} B/cyc, DRAM-bound  —  higher P = smaller chip = more MAC-efficient",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_T))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}×"))
ax.set_xlim(T_shared[0] * 0.75, T_shared[-1] * 3.5)
ax.set_ylim(bottom=0)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/fig12_speedup_per_mac.png", dpi=150)
plt.close()
print("Saved fig12")

print("\nAll figures → plots_reuse/")
