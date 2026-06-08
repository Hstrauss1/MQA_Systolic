"""Causal-prefill sweep: compare decode vs causal prefill in the multi-pass re-use model.

Generates four plots saved to plots/:
  Plot C — Causal prefill cycles vs P for each T.
  Plot D — Area-normalised throughput: decode vs causal prefill vs P.
  Plot E — Per-pass breakdown for T=8192 (compute vs memory bottleneck per pass).
  Plot F — Causal prefill cycle savings vs non-causal prefill (fraction saved).
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

# ── Hardware constants ────────────────────────────────────────────────────────
H            = 64
d            = 128
bpe          = 2
BW           = 512
pe_mac_width = 128
lmc          = 16
exp_latency  = 4
n_merge      = 0

T_values = [1024, 2048, 4096, 8192]
P_values = [1, 2, 4, 8, 16]

COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MARKERS = ["o", "s", "^", "D"]

def _T_label(T: int) -> str:
    return f"T={T:,}"

# ── Sweep ─────────────────────────────────────────────────────────────────────
# decode[(T, P)]         — query_tokens=1, causal=False
# prefill_causal[(T, P)] — query_tokens=T, causal=True
decode:         Dict[Tuple[int, int], dict] = {}
prefill_causal: Dict[Tuple[int, int], dict] = {}
prefill_noncausal: Dict[Tuple[int, int], dict] = {}

for T in T_values:
    for P in P_values:
        C = max(1, T // P)
        common = dict(
            H=H, T=T, d=d, array_rows=H, array_cols=C,
            bytes_per_element=bpe,
            memory_bandwidth_bytes_per_cycle=BW,
            exp_latency_cycles=exp_latency,
            pe_mac_width=pe_mac_width,
            lower_mac_count=lmc,
            merge_extensions=n_merge,
            num_passes=P,
        )
        decode[(T, P)]          = kv_stationary_metrics(**common, query_tokens=1,  causal=False)
        prefill_causal[(T, P)]  = kv_stationary_metrics(**common, query_tokens=T,  causal=True)
        prefill_noncausal[(T,P)]= kv_stationary_metrics(**common, query_tokens=T,  causal=False)

# ── Print summary table ───────────────────────────────────────────────────────
hdr = (f"{'T':>6}  {'P':>3}  {'C':>6}  {'dec_cyc':>12}  "
       f"{'pfill_nc_cyc':>14}  {'pfill_c_cyc':>13}  {'causal_save%':>12}  "
       f"{'dec_tput':>12}  {'pfill_tput':>12}")
print(hdr)
print("-" * len(hdr))

for T in T_values:
    for P in P_values:
        dc = decode[(T, P)]["total_cycles_multipass"]
        nc = prefill_noncausal[(T, P)]["total_cycles_multipass"]
        cc = prefill_causal[(T, P)]["total_cycles_causal"]
        pe = decode[(T, P)]["pe_count"]
        save_pct = 100.0 * (nc - cc) / nc if nc else 0.0
        dec_tput = T / (dc * pe)
        pf_tput  = T / (cc * pe)
        print(
            f"{T:>6}  {P:>3}  {T//P:>6}  {dc:>12}  "
            f"{nc:>14}  {cc:>13}  {save_pct:>11.1f}%  "
            f"{dec_tput:>12.3e}  {pf_tput:>12.3e}"
        )
    print()

os.makedirs("plots", exist_ok=True)

# ── Plot C — Causal prefill cycles vs P ──────────────────────────────────────
fig_c, ax_c = plt.subplots(figsize=(8, 5))
for idx, T in enumerate(T_values):
    xs = P_values
    ys_c = [prefill_causal[(T, P)]["total_cycles_causal"]  for P in P_values]
    ys_n = [prefill_noncausal[(T, P)]["total_cycles_multipass"] for P in P_values]
    ax_c.plot(xs, ys_c, marker=MARKERS[idx], color=COLORS[idx],
              linewidth=2, markersize=7, label=f"{_T_label(T)} causal")
    ax_c.plot(xs, ys_n, marker=MARKERS[idx], color=COLORS[idx],
              linewidth=1, linestyle="--", markersize=5, alpha=0.5,
              label=f"{_T_label(T)} non-causal")

ax_c.set_xlabel("Pass count (P)", fontsize=12)
ax_c.set_ylabel("Total cycles", fontsize=12)
ax_c.set_title("Plot C — Causal prefill cycles vs pass count\n"
               "(solid = causal, dashed = non-causal)", fontsize=11)
ax_c.set_xticks(P_values)
ax_c.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
ax_c.legend(fontsize=7, ncol=2, framealpha=0.9)
ax_c.grid(True, linestyle="--", alpha=0.5)
fig_c.tight_layout()
fig_c.savefig("../results/figures/plot_C_prefill_cycles_vs_passes.png", dpi=150)
plt.close(fig_c)
print("Saved plots/plot_C_prefill_cycles_vs_passes.png")

# ── Plot D — Area-norm throughput: decode vs causal prefill ───────────────────
fig_d, ax_d = plt.subplots(figsize=(8, 5))
for idx, T in enumerate(T_values):
    pe = decode[(T, 1)]["pe_count"]   # pe_count varies by P; track via formula
    xs = P_values
    dec_ys = []
    pf_ys  = []
    for P in P_values:
        pe_p  = decode[(T, P)]["pe_count"]   # H * (T//P)
        dc    = decode[(T, P)]["total_cycles_multipass"]
        cc    = prefill_causal[(T, P)]["total_cycles_causal"]
        dec_ys.append(T / (dc * pe_p))
        pf_ys.append(T  / (cc * pe_p))

    ax_d.plot(xs, dec_ys, marker=MARKERS[idx], color=COLORS[idx],
              linewidth=2, markersize=7, linestyle="-",
              label=f"{_T_label(T)} decode")
    ax_d.plot(xs, pf_ys, marker=MARKERS[idx], color=COLORS[idx],
              linewidth=2, markersize=7, linestyle=":",
              label=f"{_T_label(T)} prefill (causal)")

ax_d.set_xlabel("Pass count (P)", fontsize=12)
ax_d.set_ylabel("Tokens / (cycle × PE)", fontsize=12)
ax_d.set_title("Plot D — Area-normalised throughput: decode vs causal prefill\n"
               "(solid = decode, dotted = causal prefill)", fontsize=11)
ax_d.set_xticks(P_values)
ax_d.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2e}"))
ax_d.legend(fontsize=7, ncol=2, framealpha=0.9)
ax_d.grid(True, linestyle="--", alpha=0.5)
fig_d.tight_layout()
fig_d.savefig("../results/figures/plot_D_decode_vs_prefill_throughput.png", dpi=150)
plt.close(fig_d)
print("Saved plots/plot_D_decode_vs_prefill_throughput.png")

# ── Plot E — Per-pass cycle breakdown for T=8192 ──────────────────────────────
# Show how each pass gets cheaper as causal masking discards finished queries.
T_show = 8192
P_show = 8
C_show = T_show // P_show

r = prefill_causal[(T_show, P_show)]
pass_labels = [f"Pass {k}\n(Q={T_show-(k-1)*C_show:,})" for k in range(1, P_show + 1)]

# Also compute per-pass compute (without DRAM) for reference
from kv_stationary_model import _compute_cycles_per_stage
col_dwell = r["column_dwell"]
upper_pe  = r["upper_pe_cycles_per_stage"]
act_rows  = r["active_query_rows"]

compute_per_pass = []
memory_per_pass  = []
for k in range(1, P_show + 1):
    Q_k = max(0, T_show - (k - 1) * C_show)
    eff_macs = min(lmc, max(1, Q_k))
    stagger  = math.ceil(d / eff_macs)
    eff_stag = max(stagger, upper_pe)
    compute_k = (act_rows + C_show - 1) * col_dwell + max(0, Q_k - 1) * eff_stag
    q_read    = H * Q_k * d * bpe
    kv_load   = 2 * C_show * d * bpe
    out_write = H * C_show * d * bpe
    memory_k  = math.ceil((q_read + kv_load + out_write) / BW)
    compute_per_pass.append(compute_k)
    memory_per_pass.append(memory_k)

fig_e, ax_e = plt.subplots(figsize=(9, 5))
xs = range(1, P_show + 1)
ax_e.bar(xs, compute_per_pass, width=0.4, align="center",
         color="#1f77b4", alpha=0.8, label="Compute cycles")
ax_e.bar([x + 0.4 for x in xs], memory_per_pass, width=0.4, align="center",
         color="#ff7f0e", alpha=0.8, label="Memory-service cycles")
ax_e.set_xlabel("Pass", fontsize=12)
ax_e.set_ylabel("Cycles", fontsize=12)
ax_e.set_title(f"Plot E — Per-pass compute vs memory cycles\n"
               f"T={T_show:,}, P={P_show}, C={C_show:,} (causal prefill)", fontsize=11)
ax_e.set_xticks([x + 0.2 for x in xs])
ax_e.set_xticklabels(pass_labels, fontsize=8)
ax_e.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax_e.legend(framealpha=0.9)
ax_e.grid(True, axis="y", linestyle="--", alpha=0.5)
fig_e.tight_layout()
fig_e.savefig("../results/figures/plot_E_per_pass_breakdown.png", dpi=150)
plt.close(fig_e)
print("Saved plots/plot_E_per_pass_breakdown.png")

# ── Plot F — Causal cycle savings fraction vs P ───────────────────────────────
fig_f, ax_f = plt.subplots(figsize=(8, 5))
for idx, T in enumerate(T_values):
    xs = P_values
    ys = []
    for P in P_values:
        nc = prefill_noncausal[(T, P)]["total_cycles_multipass"]
        cc = prefill_causal[(T, P)]["total_cycles_causal"]
        ys.append(100.0 * (nc - cc) / nc)
    ax_f.plot(xs, ys, marker=MARKERS[idx], color=COLORS[idx],
              linewidth=2, markersize=7, label=_T_label(T))

ax_f.set_xlabel("Pass count (P)", fontsize=12)
ax_f.set_ylabel("Cycle savings vs non-causal (%)", fontsize=12)
ax_f.set_title("Plot F — Causal masking cycle savings vs pass count\n"
               "(savings from reduced Q reads + streaming output per pass)", fontsize=11)
ax_f.set_xticks(P_values)
ax_f.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax_f.legend(framealpha=0.9)
ax_f.grid(True, linestyle="--", alpha=0.5)
fig_f.tight_layout()
fig_f.savefig("../results/figures/plot_F_causal_savings_vs_passes.png", dpi=150)
plt.close(fig_f)
print("Saved plots/plot_F_causal_savings_vs_passes.png")
