"""Extended lower MAC validation: interleaved MACs + merge extensions vs SCALE-Sim.

Two sweeps:

1. LMC sweep (T=512, merge_n=0):
   Tests lmc=[1,2,4,8,16] to show how the fill/drain overhead gap grows as lmc
   increases (fewer query packets per slice → pipeline less utilised → bigger gap).

2. Merge extension sweep (lmc=1, T=512):
   Tests merge_n=[0,1,2,3]. Each level halves eff_cols and doubles eff_rows.
   Fewer columns → shorter pipeline → proportionally more fill/drain overhead.
   lmc=1 is used so the only variable is column count vs pipeline depth.

3. Combined sweep (lmc=16, merge_n=[0,1,2,3], T=512):
   The configuration closest to full_results.csv "optimistic" rows.
   Expected to show the largest discrepancy.

Array mapping (same as validate_lower_mac.py):
    rows (Y) = eff_rows = H * 2^merge_n
    cols (X) = eff_cols = T // 2^merge_n
    GEMM K   = d = 128

For lmc > 1: one slice = M = eff_rows × (eff_cols // lmc), all lmc slices parallel.
Analytical bound = ceil(eff_cols * d / lmc).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from scalesim.scale_config import scale_config as ScaleConfig
from scalesim.topology_utils import topologies as Topo
from scalesim.layout_utils import layouts as Layout
from scalesim.simulator import simulator as Simulator

H     = 64
D     = 128
T     = 512
SRAM_KB    = 4 * 1024
SCALESIM_BW = 100_000   # words/cycle — effectively infinite DRAM


def _write_config(path: Path, arr_rows: int, arr_cols: int) -> None:
    path.write_text(
        "[general]\n"
        "run_name = kv_lower_mac_ext\n\n"
        "[architecture_presets]\n"
        f"ArrayHeight:    {arr_rows}\n"
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


def _run_gemm(M: int, N: int, arr_rows: int, arr_cols: int) -> Dict[str, object]:
    """Run GEMM (M × N × D) on arr_rows × arr_cols array."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path  = Path(tmp) / "cfg.cfg"
        topo_path = Path(tmp) / "topo.csv"
        out_path  = Path(tmp) / "out"
        out_path.mkdir()

        _write_config(cfg_path, arr_rows=arr_rows, arr_cols=arr_cols)
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
        return {
            "compute_cycles": int(items[1]),
            "stall_cycles":   int(items[2]),
            "mapping_eff":    float(items[4]),
        }


def _run_config(merge_n: int, lmc: int) -> Dict[str, object]:
    """Run one (merge_n, lmc) configuration and return annotated result.

    With merge_n levels there are 2^merge_n parallel sub-arrays, each with
    H rows (one per query head) and T/2^n columns (its KV token range).
    SCALE-Sim models one representative sub-array; wall-clock = its cycles
    since all sub-arrays run in parallel with the same shape.

    The wrong approach (previous version) was to run a single H*2^n × T/2^n
    array — that serialised all sub-arrays into one tall GEMM and collapsed
    mapping efficiency once eff_cols < H.
    """
    sub_rows = H                           # one sub-array always has H query-head rows
    eff_cols = max(1, T // (2 ** merge_n)) # KV tokens this sub-array covers

    # Analytical lower MAC bound: Q_T query-token row-folds × d cycles / lmc
    # For prefill Q_T = eff_cols (all tokens in this lane); wall_clock = eff_cols*d/lmc
    analytical = math.ceil(eff_cols * D / lmc)

    # SCALE-Sim GEMM on one sub-array (sub_rows × eff_cols):
    #   lmc=1: M = sub_rows × eff_cols  (all row-folds in one shot)
    #   lmc>1: M = sub_rows × (eff_cols // lmc)  (one slice; all lmc run in parallel)
    slice_qt = eff_cols if lmc == 1 else eff_cols // lmc
    M = sub_rows * slice_qt
    N = eff_cols

    r = _run_gemm(M=M, N=N, arr_rows=sub_rows, arr_cols=eff_cols)
    ss = r["compute_cycles"]
    delta_pct = 100.0 * (ss - analytical) / analytical

    return {
        "merge_n": merge_n,
        "lmc": lmc,
        "sub_rows": sub_rows,
        "eff_cols": eff_cols,
        "M": M,
        "N": N,
        "analytical": analytical,
        "ss_cycles": ss,
        "delta_pct": delta_pct,
        "mapping_eff_pct": r["mapping_eff"] * 100,
        "stall_cycles": r["stall_cycles"],
        # fill/drain overhead = ss - analytical (absolute cycles wasted)
        "overhead_cycles": ss - analytical,
        # drain fraction = overhead / ss (fraction of time draining)
        "drain_frac": (ss - analytical) / ss if ss > 0 else 0.0,
    }


def _print_table(rows: List[Dict], title: str, vary_col: str) -> None:
    print(f"\n{'─' * 90}")
    print(f"  {title}")
    print(f"{'─' * 90}")
    print(f"  {'merge_n':>7}  {'lmc':>4}  {'sub_rows':>8}  {'eff_cols':>8}  "
          f"{'M':>7}  {'N':>5}  {'analytical':>12}  {'ss_cycles':>10}  "
          f"{'delta%':>7}  {'drain_frac':>10}  {'map_eff%':>9}")
    print(f"  {'─'*7}  {'─'*4}  {'─'*8}  {'─'*8}  "
          f"{'─'*7}  {'─'*5}  {'─'*12}  {'─'*10}  "
          f"{'─'*7}  {'─'*10}  {'─'*9}")
    for r in rows:
        print(f"  {r['merge_n']:>7}  {r['lmc']:>4}  {r['sub_rows']:>8}  {r['eff_cols']:>8}  "
              f"{r['M']:>7,}  {r['N']:>5}  {r['analytical']:>12,}  {r['ss_cycles']:>10,}  "
              f"{r['delta_pct']:>+6.1f}%  {r['drain_frac']:>10.3f}  "
              f"{r['mapping_eff_pct']:>8.1f}%")


def main() -> None:
    print("Extended lower MAC validation: analytical bound vs SCALE-Sim")
    print(f"H={H}  d={D}  T={T}")

    # ── Sweep 1: lmc sweep, no merge ────────────────────────────────────────
    lmc_rows = [_run_config(merge_n=0, lmc=lmc) for lmc in [1, 2, 4, 8, 16]]
    _print_table(lmc_rows, "Sweep 1: lmc=[1,2,4,8,16]  merge_n=0  (eff_rows=64, eff_cols=512)", "lmc")

    # ── Sweep 2: merge sweep, lmc=1 ─────────────────────────────────────────
    merge_rows_lmc1 = [_run_config(merge_n=n, lmc=1) for n in [0, 1, 2, 3]]
    _print_table(merge_rows_lmc1, "Sweep 2: merge_n=[0,1,2,3]  lmc=1  (eff_cols shrinks T/2^n)", "merge_n")

    # ── Sweep 3: combined lmc=16 + merge sweep ───────────────────────────────
    merge_rows_lmc16 = [_run_config(merge_n=n, lmc=16) for n in [0, 1, 2, 3]]
    _print_table(merge_rows_lmc16, "Sweep 3: merge_n=[0,1,2,3]  lmc=16  (full_results.csv 'optimistic' config)", "merge_n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Summary: drain_frac = (ss_cycles - analytical) / ss_cycles")
    print(f"  {'config':>25}  {'analytical':>12}  {'ss_cycles':>10}  {'delta%':>7}  {'drain_frac':>10}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*10}  {'─'*7}  {'─'*10}")
    for r in lmc_rows + merge_rows_lmc1[1:] + merge_rows_lmc16:
        label = f"lmc={r['lmc']:>2}, merge_n={r['merge_n']}"
        print(f"  {label:>25}  {r['analytical']:>12,}  {r['ss_cycles']:>10,}  "
              f"{r['delta_pct']:>+6.1f}%  {r['drain_frac']:>10.3f}")
    print()


if __name__ == "__main__":
    main()
