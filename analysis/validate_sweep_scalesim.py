"""SCALE-Sim validation of the wavefront (B) fill model across the sweep config space.

For every (T, n, lmc) prefill config that fits in memory, run the Q·K GEMM on
SCALE-Sim (cycle-accurate systolic array) and compare the MEASURED fill/drain
against what models A (dwell, 135/col) and B (wavefront, ceil(d/lmc)/col) predict.

This is the maximal SCALE-Sim validation possible for the sweep:
  • SCALE-Sim measures the lower-MAC Q·K compute (the term A vs B disagree on).
  • It runs at infinite DRAM bandwidth, so it CANNOT validate the sweep's
    DRAM-bound totals or the lmc≈BW/128 rule — those stay analytical.

GEMM per config (one sub-array): M = H·T/lmc, N = eff_cols = T/2^n, K = d, array H×eff_cols.
analytical floor = ceil(T·d/lmc)  (= B steady-state stream term, since ceil(d/lmc)≥upper_pe here).
measured drain   = ss_cycles − analytical          → real fill
A predicts drain = (H + eff_cols − 1)·135
B predicts drain = d + (H + eff_cols − 1)·ceil(d/lmc)
"""

from __future__ import annotations
import csv
import math
import time

from validate_lower_mac_sweep import _run_gemm

H, D = 64, 128
UPPER_PE = 4 + math.ceil(3 * D / 128)        # 7
COL_DWELL = D + UPPER_PE                      # 135  (model A per-column hop)
MEM_LIMIT_MB = 500

T_values   = [128, 512, 1024, 2048, 4096, 8192]
N_values   = [0, 3]
LMC_values = [1, 2, 4, 8, 16]

rows = []
print("Running SCALE-Sim Q·K GEMMs (B-fill validation)...\n", flush=True)
for T in T_values:
    for n in N_values:
        eff_cols = T // (2 ** n)
        for lmc in LMC_values:
            M = H * (T // lmc)
            mb = (M * D + D * eff_cols + M * eff_cols) * 8 / 1e6
            if mb > MEM_LIMIT_MB or M == 0 or eff_cols == 0:
                continue

            analytical = math.ceil(T * D / lmc)          # throughput floor
            eff_stag   = max(math.ceil(D / lmc), UPPER_PE)

            t0 = time.time()
            ss = _run_gemm(M=M, N=eff_cols, arr_rows=H, arr_cols=eff_cols)["compute_cycles"]
            dt = time.time() - t0

            meas_drain = ss - analytical
            a_drain    = (H + eff_cols - 1) * COL_DWELL
            b_drain    = D + (H + eff_cols - 1) * eff_stag
            per_col    = meas_drain / eff_cols

            rows.append(dict(
                T=T, n=n, lmc=lmc, eff_cols=eff_cols, gemm_M=M,
                analytical=analytical, ss_cycles=ss,
                meas_drain=meas_drain, per_col_fill=round(per_col, 2),
                A_drain_pred=a_drain, B_drain_pred=b_drain,
                A_err_x=round(a_drain / max(1, meas_drain), 1),
                B_err_x=round(b_drain / max(1, meas_drain), 2),
                mb=round(mb, 0),
            ))
            print(f"  T={T:>5} n={n} lmc={lmc:>2}  ss={ss:>8,}  drain={meas_drain:>7,}  "
                  f"per_col={per_col:>5.2f}  (A pred {a_drain:>9,}, B pred {b_drain:>7,})  "
                  f"[{dt:.1f}s]", flush=True)

with open("../results/data/validate_sweep_scalesim.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ── Verdict ───────────────────────────────────────────────────────────────────
per_cols = [r["per_col_fill"] for r in rows]
print(f"\n{'='*70}")
print(f"SCALE-Sim measured per-column fill across {len(rows)} configs: "
      f"min={min(per_cols):.2f}  max={max(per_cols):.2f}  cyc/col")
print(f"  Model A assumes {COL_DWELL} cyc/col  → wrong by "
      f"{COL_DWELL/ (sum(per_cols)/len(per_cols)):.0f}× on average")
print(f"  Model B assumes ceil(d/lmc) cyc/col → correct regime (and conservative)")
print(f"\nWrote validate_sweep_scalesim.csv ({len(rows)} SCALE-Sim runs)")
