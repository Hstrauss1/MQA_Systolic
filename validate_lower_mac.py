"""Validate the KV-stationary lower MAC analytical bound against SCALE-Sim.

Array mapping
─────────────
    rows (Y) = H = 64        one row lane per attention head
    cols (X) = T             one column per KV token (token-stationary)
    Z-axis   = Q tokens      folded as M = H × Q_tokens in a single GEMM

Each Q-token attends to T KV tokens via a d-length dot product.
Stacking H × Q_tokens queries into one GEMM (M = H*Q_T, N = T, K = d):
    row folds = ceil(H*Q_T / H) = Q_T   (one fold per Q token)
    col folds = 1                        (N = T = array_cols, perfect fit)
    ⟹ SCALE-Sim total_cycles = Q_T × d = analytical lower_mac_bound  ✓

For lmc=16 (16 interleaved MACs per PE column):
    Each MAC handles Q_T//lmc Q tokens independently and in parallel.
    SCALE-Sim models one MAC's slice: M = H × (Q_T//lmc), same N and K.
    All 16 MACs run simultaneously ⟹ wall_clock = slice_cycles.
    slice_cycles = (Q_T//lmc) × d = analytical bound  ✓

    The stagger (packet_stagger = ceil(d/lmc) = 8 cycles) is the rate at
    which the upper PE receives scores from the lower MAC.  It is validated
    separately against the upper PE stage; it is NOT added here.

Safety: small T only, small SRAM, single GEMM per run.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, List

from scalesim.scale_config import scale_config as ScaleConfig
from scalesim.topology_utils import topologies as Topo
from scalesim.layout_utils import layouts as Layout
from scalesim.simulator import simulator as Simulator

from kv_stationary_model import kv_stationary_metrics

# ── Parameters ────────────────────────────────────────────────────────────────
H              = 64    # heads → array rows
D              = 128   # head dimension → GEMM K
BPE            = 2
BW_ANALYTICAL  = 512
MERGE_N        = 3
BATCH          = 1

# Small T only — large T creates huge numpy operand matrices and crashes
T_VALUES: List[int]       = [128, 512]
LOWER_MAC_CONFIGS: List[int] = [1, 16]

# SRAM: must hold the filter matrix K[d × T] = 128×512×2 = 128 KB at most.
# 4 MB per bank is safe for T ≤ 512 and leaves no risk of OOM.
SRAM_KB     = 4 * 1024
SCALESIM_BW = 100_000   # words/cycle — effectively infinite, no DRAM stalls


# ── SCALE-Sim helpers ─────────────────────────────────────────────────────────

def _write_config(path: Path, arr_cols: int) -> None:
    path.write_text(
        "[general]\n"
        "run_name = kv_lower_mac\n\n"
        "[architecture_presets]\n"
        f"ArrayHeight:    {H}\n"
        f"ArrayWidth:     {arr_cols}\n"
        f"IfmapSramSzkB:  {SRAM_KB}\n"
        f"FilterSramSzkB: {SRAM_KB}\n"
        f"OfmapSramSzkB:  {SRAM_KB}\n"
        "IfmapOffset:    0\n"
        "FilterOffset:   10000000\n"
        "OfmapOffset:    20000000\n"
        f"Bandwidth:      {SCALESIM_BW}\n"
        "Dataflow:       ws\n"
        "MemoryBanks:    1\n"
        "ReadRequestBuffer:  64\n"
        "WriteRequestBuffer: 64\n\n"
        "[layout]\n"
        "IfmapCustomLayout:       False\n"
        "IfmapSRAMBankBandwidth:  10\n"
        "IfmapSRAMBankNum:        10\n"
        "IfmapSRAMBankPort:       2\n"
        "FilterCustomLayout:      False\n"
        "FilterSRAMBankBandwidth: 10\n"
        "FilterSRAMBankNum:       10\n"
        "FilterSRAMBankPort:      2\n\n"
        "[run_presets]\n"
        "InterfaceBandwidth: CALC\n"
        "UseRamulatorTrace:  False\n"
    )


def _run_gemm(M: int, N: int) -> Dict[str, object]:
    """Run a single GEMM (M × N × D) on an H × N array. Returns compute cycles."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path  = Path(tmp) / "cfg.cfg"
        topo_path = Path(tmp) / "topo.csv"
        out_path  = Path(tmp) / "out"
        out_path.mkdir()

        _write_config(cfg_path, arr_cols=N)
        topo_path.write_text(f"Layer,M,N,K,\nQK,{M},{N},{D},\n")

        cfg    = ScaleConfig(); cfg.read_conf_file(str(cfg_path))
        topo   = Topo();        topo.load_arrays(topofile=str(topo_path), mnk_inputs=True)
        layout = Layout()

        sim = Simulator()
        sim.set_params(
            config_obj=cfg, topo_obj=topo, layout_obj=layout,
            top_path=str(out_path), verbosity=False, save_trace=False,
        )
        sim.run()

        items = sim.single_layer_sim_object_list[0].get_compute_report_items()
        # [overall_cycles, total_cycles, stall_cycles, overall_util, mapping_eff, compute_util]
        return {
            "compute_cycles": int(items[1]),
            "stall_cycles":   int(items[2]),
            "mapping_eff":    float(items[4]),
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("KV-stationary lower MAC validation: analytical bound vs SCALE-Sim")
    print("=" * 72)
    print(f"Array: rows=H={H}  cols=T  |  GEMM: M=H×Q_T  N=T  K=d={D}")
    print(f"Row folds = Q_T (one per Q token)  Col folds = 1 (perfect fit)")
    print()

    for lmc in LOWER_MAC_CONFIGS:
        packet_stagger = math.ceil(D / lmc)
        print(f"── lower_mac_count={lmc}  packet_stagger=ceil({D}/{lmc})={packet_stagger} cycles {'─'*30}")
        if lmc == 1:
            print(f"   Full GEMM: M=H×T  N=T  K={D}  on {H}×T array")
        else:
            print(f"   One-slice GEMM: M=H×(T//{lmc})  N=T  K={D}  on {H}×T array")
            print(f"   All {lmc} slices run in parallel → wall_clock = slice_cycles")
        print()

        hdr = (f"  {'T':>5}  {'M':>7}  {'N':>5}  "
               f"{'analytical':>12}  {'ss_cycles':>10}  "
               f"{'delta':>7}  {'row_folds':>10}  {'map_eff%':>9}  {'stall_cyc':>10}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for T in T_VALUES:
            Q_T        = T                          # prefill: Q tokens = T
            slice_Q_T  = Q_T if lmc == 1 else Q_T // lmc
            M          = H * slice_Q_T
            N          = T
            row_folds  = math.ceil(M / H)           # = slice_Q_T (should be exact)

            # Analytical lower_mac_bound = ceil(H × T × T × d / (H × T × lmc))
            #                            = ceil(T × d / lmc)
            analytical = math.ceil(T * D / lmc)

            r = _run_gemm(M=M, N=N)
            ss = r["compute_cycles"]
            delta_pct = 100.0 * (ss - analytical) / analytical

            print(
                f"  {T:>5}  {M:>7,}  {N:>5}  "
                f"{analytical:>12,}  {ss:>10,}  "
                f"{delta_pct:>+6.2f}%  {row_folds:>10,}  "
                f"{r['mapping_eff']*100:>8.1f}%  {r['stall_cycles']:>10,}"
            )
        print()

    # ── Cross-check: kv_stationary_metrics dot_product_macs ──────────────────
    print("── Cross-check: kv_stationary_metrics dot_product_macs vs H×T²×d ─────────")
    lmc = LOWER_MAC_CONFIGS[-1]
    print(f"  {'T':>5}  {'kv_dot_macs':>14}  {'H×T²×d':>12}  {'match':>6}")
    print("  " + "─" * 42)
    for T in T_VALUES:
        eff_rows = H * (2 ** MERGE_N)
        eff_cols = max(1, T // (2 ** MERGE_N))
        kv = kv_stationary_metrics(
            H=H, T=T, d=D,
            array_rows=eff_rows, array_cols=eff_cols,
            bytes_per_element=BPE,
            memory_bandwidth_bytes_per_cycle=BW_ANALYTICAL,
            lower_mac_count=lmc,
            merge_extensions=MERGE_N,
            query_tokens=T,
        )
        kv_dot   = int(kv["dot_product_macs"])
        hand_dot = H * BATCH * T * T * D
        ok = "✓" if kv_dot == hand_dot else f"✗ ({kv_dot:,} vs {hand_dot:,})"
        print(f"  {T:>5}  {kv_dot:>14,}  {hand_dot:>12,}  {ok}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
