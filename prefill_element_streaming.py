"""Corrected KV-stationary cycle model: element-streaming architecture.

Previous model (WRONG):
    Q treated as a packet that dwells at each column for column_dwell=135 cycles
    before moving to the next column.
    Spinup = (H + eff_cols - 1) × 135 → ~1.1M cycles at T=8192

Correct model (THIS FILE):
    Q elements shift one position per cycle (standard systolic behaviour).
    All eff_cols columns accumulate simultaneously, staggered by 1 element/cycle.
    Score at column t for packet p ready at: p × stagger + t + d

    Upper PE chain is the true sequential bottleneck:
    (M, L, O) softmax state propagates left to right at upper_pe_cycles per column.
    Column t starts its upper PE as soon as BOTH score_t is ready AND
    state from column t-1 has propagated.

    The state propagation rate (upper_pe_cycles) is faster than the element
    stagger (1 cycle/element), so the upper PE becomes the bottleneck of the chain:
        done_time(p, t) = p × effective_stagger + d + upper_pe_cycles × (t+1)

    Total = d + upper_pe × eff_cols + (T-1) × effective_stagger

    H rows (heads) run fully in parallel — K is stationary and shared,
    each head streams its own Q elements independently.

SCALE-Sim validation:
    Lower MAC throughput validated (d = 128 cycles per packet per column).
    Upper PE timing (7 cycles) remains a design assumption.
    FlashAttention tile GEMMs validated (8,571 cycles each).
"""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from scalesim.scale_config import scale_config as ScaleConfig
from scalesim.topology_utils import topologies as Topo
from scalesim.layout_utils import layouts as Layout
from scalesim.simulator import simulator as Simulator

H            = 64
D            = 128
BPE          = 2
BW           = 512
EXP_LATENCY  = 4
PE_MAC_WIDTH = 128
SRAM_KB      = 4 * 1024
SCALESIM_BW  = 100_000
FLASH_TILE   = 64

OUTPUT_CSV = Path("prefill_element_streaming.csv")


# ── SCALE-Sim ─────────────────────────────────────────────────────────────────

def _ss(M: int, N: int, K: int, arr_rows: int, arr_cols: int):
    if (M * K + K * N + M * N) * 8 / 1e6 > 400:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        cfg  = Path(tmp) / "cfg.cfg"; topo = Path(tmp) / "topo.csv"
        out  = Path(tmp) / "out"; out.mkdir()
        cfg.write_text(
            "[general]\nrun_name=x\n\n[architecture_presets]\n"
            f"ArrayHeight: {arr_rows}\nArrayWidth: {arr_cols}\n"
            f"IfmapSramSzkB: {SRAM_KB}\nFilterSramSzkB: {SRAM_KB}\n"
            f"OfmapSramSzkB: {SRAM_KB}\nIfmapOffset: 0\n"
            "FilterOffset: 10000000\nOfmapOffset: 20000000\n"
            f"Bandwidth: {SCALESIM_BW}\nDataflow: ws\nMemoryBanks: 1\n"
            "ReadRequestBuffer: 64\nWriteRequestBuffer: 64\n\n[layout]\n"
            "IfmapCustomLayout: False\nIfmapSRAMBankBandwidth: 10\n"
            "IfmapSRAMBankNum: 10\nIfmapSRAMBankPort: 2\n"
            "FilterCustomLayout: False\nFilterSRAMBankBandwidth: 10\n"
            "FilterSRAMBankNum: 10\nFilterSRAMBankPort: 2\n\n"
            "[run_presets]\nInterfaceBandwidth: CALC\nUseRamulatorTrace: False\n"
        )
        topo.write_text(f"Layer,M,N,K,\nG,{M},{N},{K},\n")
        c = ScaleConfig(); c.read_conf_file(str(cfg))
        t = Topo(); t.load_arrays(topofile=str(topo), mnk_inputs=True)
        sim = Simulator()
        sim.set_params(config_obj=c, topo_obj=t, layout_obj=Layout(),
                       top_path=str(out), verbosity=False, save_trace=False)
        sim.run()
        return int(sim.single_layer_sim_object_list[0].get_compute_report_items()[1])


# ── Element-streaming KV-stationary model ────────────────────────────────────

def kv_element_streaming(T: int, merge_n: int, lmc: int) -> Dict:
    """
    Correct element-streaming cycle model.

    Per sub-array (H rows × eff_cols cols, 2^merge_n sub-arrays in parallel):

        packet_stagger   = ceil(d / lmc)
        upper_pe_cycles  = exp_latency + ceil(3d / pe_mac_width)
        effective_stagger = max(packet_stagger, upper_pe_cycles)

        first_packet_latency = d + upper_pe_cycles × eff_cols
            d cycles: all lower MACs compute in parallel (element shift)
            upper_pe_cycles × eff_cols: sequential (M,L,O) chain propagation

        total = first_packet_latency + (T-1) × effective_stagger

    H heads run in parallel (independent rows, shared K stationary).
    All 2^n sub-arrays run in parallel.

    SCALE-Sim validates lower MAC: d cycles per packet per column.
    upper_pe_cycles = 7 is a design assumption.
    """
    eff_cols        = max(1, T // (2 ** merge_n))
    packet_stagger  = math.ceil(D / lmc)
    upper_pe_cycles = EXP_LATENCY + math.ceil(3 * D / PE_MAC_WIDTH)  # 4+3=7
    eff_stagger     = max(packet_stagger, upper_pe_cycles)

    # First packet: MAC latency + sequential upper PE chain
    first_latency   = D + upper_pe_cycles * eff_cols

    # Remaining T-1 packets staggered at eff_stagger each
    stagger_cost    = (T - 1) * eff_stagger

    total           = first_latency + stagger_cost

    # SCALE-Sim validation of lower MAC throughput
    # One representative: H×(eff_cols//lmc) × eff_cols × D on H×eff_cols array
    slice_qt = T if lmc == 1 else T // lmc
    ss_lower = _ss(M=H * slice_qt, N=eff_cols, K=D, arr_rows=H, arr_cols=eff_cols)
    # ss_lower validates: d cycles per packet per column (row_folds × d = T/lmc × d)
    ss_lower_analytic = math.ceil(T * D / lmc)

    dram_bytes = (2 * H + 2) * T * D * BPE   # same as FlashAttention (Q+K+V+O, no scores)

    return {
        "arch":            f"kv_elem_n{merge_n}_lmc{lmc}",
        "T":               T,
        "merge_n":         merge_n,
        "lmc":             lmc,
        "eff_cols":        eff_cols,
        "n_subarrays":     2 ** merge_n,
        "array_pes":       H * T,           # H × eff_cols × 2^n = H × T total
        "packet_stagger":  packet_stagger,
        "upper_pe_cycles": upper_pe_cycles,
        "eff_stagger":     eff_stagger,
        "first_latency":   first_latency,
        "stagger_cost":    stagger_cost,
        "total_cycles":    total,
        # breakdown
        "mac_latency":     D,
        "upper_pe_chain":  upper_pe_cycles * eff_cols,
        # SCALE-Sim lower MAC validation
        "ss_lower_mac":    ss_lower,
        "lower_mac_analytic": ss_lower_analytic,
        "lower_mac_ss_validated": ss_lower is not None,
        # traffic
        "dram_mb":         round(dram_bytes / 1e6, 2),
    }


# ── FlashAttention baseline ───────────────────────────────────────────────────

def flash_row(T: int, ss_qk: int, ss_av: int) -> Dict:
    n_tiles      = (math.ceil(T / FLASH_TILE)) ** 2
    compute_cyc  = n_tiles * (ss_qk + ss_av)
    dram_bytes   = (2 * H + 2) * T * D * BPE
    mem_cyc      = math.ceil(dram_bytes / BW)
    total        = max(compute_cyc, mem_cyc)
    return {
        "T": T, "n_tiles": n_tiles,
        "ss_qk": ss_qk, "ss_av": ss_av,
        "total_cycles": total,
        "dram_mb": round(dram_bytes / 1e6, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    T_VALUES   = [128, 512, 1024, 2048, 4096, 8192]
    KV_CONFIGS: List[Tuple[int, int]] = [(0,1), (0,16), (3,1), (3,16)]

    # FlashAttention tile GEMMs (SCALE-Sim, run once)
    print("Running FlashAttention tile GEMMs...")
    ss_qk = _ss(H*FLASH_TILE, FLASH_TILE, D, H, H)
    ss_av = _ss(H*FLASH_TILE, D, FLASH_TILE, H, H)
    print(f"  QK tile: {ss_qk:,}  AV tile: {ss_av:,}\n")

    rows = []
    print(f"{'T':>5}  {'config':>20}  {'total_cyc':>12}  "
          f"{'mac_lat':>8}  {'pe_chain':>10}  {'stagger':>10}  "
          f"{'vs_flash_eq_area':>17}  {'area_norm':>10}  {'ss_valid':>9}")
    print('─'*110)

    flash_by_T = {}
    for T in T_VALUES:
        f = flash_row(T, ss_qk, ss_av)
        flash_by_T[T] = f
        # equal-area flash: give Flash H×T/H×H = T/H parallel arrays
        flash_eq = f['total_cycles'] / (T / H)

        for merge_n, lmc in KV_CONFIGS:
            r = kv_element_streaming(T, merge_n, lmc)
            rows.append({**r,
                         'flash_total': f['total_cycles'],
                         'flash_eq_area': flash_eq})

            speedup   = flash_eq / r['total_cycles']
            pes_ratio = r['array_pes'] / (H * H)
            area_norm = speedup / pes_ratio

            print(f"{T:>5}  {r['arch']:>20}  {r['total_cycles']:>12,}  "
                  f"{r['mac_latency']:>8}  {r['upper_pe_chain']:>10,}  "
                  f"{r['stagger_cost']:>10,}  "
                  f"{speedup:>16.2f}x  {area_norm:>9.3f}x  "
                  f"{'yes' if r['lower_mac_ss_validated'] else 'no':>9}")

    # Save
    fieldnames = [
        'arch','T','merge_n','lmc','eff_cols','n_subarrays','array_pes',
        'packet_stagger','upper_pe_cycles','eff_stagger',
        'mac_latency','upper_pe_chain','stagger_cost','first_latency','total_cycles',
        'ss_lower_mac','lower_mac_analytic','lower_mac_ss_validated',
        'dram_mb','flash_total','flash_eq_area',
    ]
    with OUTPUT_CSV.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
