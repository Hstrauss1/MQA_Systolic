"""Full multi-pass (P) × bandwidth (BW) sweep across every config — wavefront (B) fill.

Sweeps over all combinations of:
    DRAM bandwidth    BW  ∈ {512, 1024, 2048} B/cyc   (~0.5 / 1 / 2 TB/s @1GHz)
    merge extensions  n   ∈ {0, 3}
    lower MAC count   lmc ∈ {1, 2, 4, 8, 16}
    sequence length   T   ∈ {512, 1024, 2048, 4096, 8192}
    passes            P   ∈ {1, 2, 4, 8, 16}
mode = causal prefill (query_tokens = T).

The DRAM floor scales as 1/BW; compute scales as 1/lmc.  The compute/memory
balance point therefore moves as lmc_balance ≈ BW/128 (lmc=4 @512, 8 @1024,
16 @2048).  Below balance the design is DRAM-bound and extra lmc is wasted
silicon (per-MAC efficiency halves per doubling); at/above balance per-MAC
efficiency is lmc-invariant and lmc only buys lower wall-clock latency.

Compute model = interpretation B (wavefront fill): every column owns a dedicated
lower MAC + stationary K, so a single packet's per-column dot products compute in
parallel; the d-cycle dot-product latency is paid once and the pipeline advances
one step per effective_stagger.  Under B essentially all prefill is DRAM-bound.

Silicon normalisation (per user's "lower MACs only" choice):
    kv_phys_macs    = pe_count × lmc          (pe_count = H·T/P, merge-invariant)
    flash_phys_macs = 64×64 = 4096            (FlashAttention, 1 MAC/PE)
    speedup_per_mac = (flash_cycles / kv_cycles) / (kv_phys_macs / flash_phys_macs)

Outputs → reuse_full_sweep.csv  + a printed T=8192 summary.
"""

from __future__ import annotations
import csv
import math

from kv_reuse_model import kv_stationary_metrics

# ── Constants (CLAUDE.md §2) ──────────────────────────────────────────────────
H, d, bpe, BW = 64, 128, 2, 512
pe_mac_width, exp_latency = 128, 4
FLASH_PHYS_MACS = 64 * 64          # 64×64 FlashAttention array, 1 MAC/PE

T_values  = [512, 1024, 2048, 4096, 8192]
P_values  = [1, 2, 4, 8, 16]
BW_values = [512, 1024, 2048]                       # B/cyc  (~0.5 / 1 / 2 TB/s @1GHz)
CONFIGS   = [(n, lmc) for n in (0, 3) for lmc in (1, 2, 4, 8, 16)]   # (merge_n, lmc)

# ── FlashAttention baseline cycles (per T) from the validated CSV ─────────────
# Flash baseline is compute-bound (SCALE-Sim infinite-BW), so it is BW-invariant.
flash_cycles: dict[int, float] = {}
with open("../old_dwell_model/prefill_full_results.csv") as f:
    for r in csv.DictReader(f):
        if r["arch"] == "kv_stat_n0_lmc1":
            flash_cycles[int(r["T"])] = float(r["flash_total_cycles"])


def run(T: int, P: int, n: int, lmc: int, BW: int) -> dict:
    merge_rows = 2 ** n
    cols = max(1, (T // P) // merge_rows)        # physical columns per sub-array
    return kv_stationary_metrics(
        H=H, T=T, d=d,
        array_rows=H * merge_rows,
        array_cols=cols,
        bytes_per_element=bpe,
        memory_bandwidth_bytes_per_cycle=BW,
        exp_latency_cycles=exp_latency,
        pe_mac_width=pe_mac_width,
        lower_mac_count=lmc,
        merge_extensions=n,
        num_passes=P,
        query_tokens=T,
        causal=True,
        wavefront_fill=True,
    )


rows = []
for BW in BW_values:
    for n, lmc in CONFIGS:
        for T in T_values:
            for P in P_values:
                m = run(T, P, n, lmc, BW)
                kv_cycles = m["total_cycles_causal"]
                pe        = m["pe_count"]
                phys_macs = pe * lmc
                mac_ratio = phys_macs / FLASH_PHYS_MACS
                raw_spd   = flash_cycles[T] / kv_cycles
                comp, mem = m["causal_compute_total"], m["causal_memory_total"]
                rows.append({
                    "BW": BW,
                    "merge_n": n, "lmc": lmc, "T": T, "P": P,
                    "array_rows": H * (2 ** n),
                    "sub_cols": m["sub_array_cols"],
                    "pe_count": pe,
                    "phys_macs": phys_macs,
                    "kv_cycles": int(kv_cycles),
                    "compute_cycles": comp,
                    "memory_cycles": mem,
                    "bound": "mem" if mem >= comp else "compute",
                    "flash_cycles": int(flash_cycles[T]),
                    "raw_speedup": round(raw_spd, 2),
                    "mac_ratio": round(mac_ratio, 1),
                    "speedup_per_mac": round(raw_spd / mac_ratio, 4),
                    "sram_mb": round(m["sram_bytes"] / 1e6, 2),
                })

with open("../results/data/reuse_full_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"Wrote reuse_full_sweep.csv  ({len(rows)} rows: "
      f"{len(BW_values)} BW × {len(CONFIGS)} configs × {len(T_values)} T × {len(P_values)} P)\n")

# ── Summary: speedup-per-MAC grid @ T=8192, one block per BW ──────────────────
def cell(BW, n, lmc, P):
    r = next(r for r in rows if r["BW"] == BW and r["merge_n"] == n
             and r["lmc"] == lmc and r["T"] == 8192 and r["P"] == P)
    return f"{r['speedup_per_mac']:.3f}{r['bound'][0]}"

for BW in BW_values:
    print(f"speedup_per_mac @ T=8192, BW={BW}   (c=compute-bound, m=mem-bound)")
    print(f"{'config':>12} " + " ".join(f"P={P:<7}" for P in P_values))
    for n, lmc in CONFIGS:
        print(f"  n={n} lmc={lmc:<3} " + " ".join(f"{cell(BW,n,lmc,P):<9}" for P in P_values))
    print()
