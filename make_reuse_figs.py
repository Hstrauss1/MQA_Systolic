"""Publication-quality figures for the multi-pass silicon re-use model.

Visual style matches plots_prefill_corrected/fig3_area_norm_speedup.png:
  semilogx  ·  linewidth=2.2  ·  marker='o' markersize=6
  figsize=(8,5)  ·  dpi=150  ·  grid both/-- /alpha=0.3

Outputs → plots_reuse/
  fig7_area_norm_throughput_decode.png   — area-normalised throughput vs T, per P
  fig8_latency_overhead.png              — latency ratio (P passes vs 1 pass) vs T
  fig9_causal_savings_vs_T.png           — causal cycle savings % vs T, per P
  fig10_decode_vs_prefill_area_norm.png  — decode vs causal prefill area-norm tput
"""

from __future__ import annotations
import math, os
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

T_values = [512, 1024, 2048, 4096, 8192]
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
decode:          Dict[Tuple[int, int], dict] = {}
prefill_causal:  Dict[Tuple[int, int], dict] = {}
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
        decode[(T, P)]           = kv_stationary_metrics(**common, query_tokens=1, causal=False)
        prefill_causal[(T, P)]   = kv_stationary_metrics(**common, query_tokens=T, causal=True)
        prefill_noncausal[(T, P)]= kv_stationary_metrics(**common, query_tokens=T, causal=False)

os.makedirs("plots_reuse", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Area-normalised throughput vs T — one line per P (decode)
# area_norm_tput = T / (total_cycles × pe_count)   [tokens / (cycle × PE)]
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for P in P_values:
    ys = [
        T / (decode[(T, P)]["total_cycles_multipass"] * decode[(T, P)]["pe_count"])
        for T in T_values
    ]
    label = f"P={P}  ({T_values[0]//P}–{T_values[-1]//P} cols)"
    ax.semilogx(T_values, ys, color=C[P], label=label, **STYLE)
    # Annotate right endpoint
    ax.annotate(f"  P={P}", xy=(T_values[-1], ys[-1]),
                fontsize=8.5, color=C[P], va="center", fontweight="bold")

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Tokens / (cycle × PE)", fontsize=12)
ax.set_title(
    "Area-normalised throughput vs T — multi-pass decode\n"
    "(higher = more silicon-efficient; same metric as area-norm speedup)",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_T))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2e}"))
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(T_values[0] * 0.75, T_values[-1] * 3)
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plots_reuse/fig7_area_norm_throughput_decode.png", dpi=150)
plt.close()
print("Saved fig7")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Latency overhead — total_cycles(P) / total_cycles(P=1) vs T
# Shows the price paid in wall-clock cycles for each re-use level.
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

ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.5, label="P=1 baseline  (1×)")

# Ideal linear reference: at P passes, latency should be exactly P×
for P in [2, 4, 8, 16]:
    ax.axhline(P, color=C[P], lw=0.6, ls=":", alpha=0.35)

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Cycle multiplier vs single-pass (P=1)", fontsize=12)
ax.set_title(
    "Latency overhead of multi-pass re-use vs T\n"
    "(dotted reference = ideal P× cost; actual stays below due to smaller pipeline drain)",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_T))
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(T_values[0] * 0.75, T_values[-1] * 3)
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plots_reuse/fig8_latency_overhead.png", dpi=150)
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

# P=1 saves nothing — single pass, no queries to discard
ax.semilogx(T_values, [0.0] * len(T_values), color=C[1],
            label="P=1  (no savings — all queries active)", **STYLE, ls=":")

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Cycle savings vs non-causal (%)", fontsize=12)
ax.set_title(
    "Causal masking savings in multi-pass prefill vs T\n"
    "(savings grow with P: each extra pass discards more finished queries + Q reads)",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_T))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(T_values[0] * 0.75, T_values[-1] * 3)
ax.set_ylim(-5, 80)
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plots_reuse/fig9_causal_savings_vs_T.png", dpi=150)
plt.close()
print("Saved fig9")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: Decode vs causal-prefill area-norm throughput vs T
# Both modes plotted for P=1 and best P (P=16), to show mode × re-use interaction.
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for P, ls in [(1, "-"), (16, "--")]:
    dec_ys = [
        T / (decode[(T, P)]["total_cycles_multipass"] * decode[(T, P)]["pe_count"])
        for T in T_values
    ]
    pf_ys = [
        T / (prefill_causal[(T, P)]["total_cycles_causal"] * prefill_causal[(T, P)]["pe_count"])
        for T in T_values
    ]
    lbl_d = f"Decode    P={P}"
    lbl_p = f"Prefill (causal)  P={P}"
    ax.semilogx(T_values, dec_ys, color=C[P], ls=ls,   label=lbl_d, **STYLE)
    ax.semilogx(T_values, pf_ys,  color=C[P], ls=":",  label=lbl_p,
                linewidth=2.2, marker="s", markersize=6)

ax.set_xlabel("Sequence length T", fontsize=12)
ax.set_ylabel("Tokens / (cycle × PE)", fontsize=12)
ax.set_title(
    "Area-normalised throughput: decode vs causal prefill\n"
    "(solid/dashed = decode  ·  dotted+square = causal prefill)",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_T))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2e}"))
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(T_values[0] * 0.75, T_values[-1] * 3)
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plots_reuse/fig10_decode_vs_prefill_area_norm.png", dpi=150)
plt.close()
print("Saved fig10")

print("\nAll figures → plots_reuse/")
