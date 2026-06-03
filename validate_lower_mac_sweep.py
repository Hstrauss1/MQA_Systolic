"""Full lower-MAC SCALE-Sim sweep matching full_results.csv parameter space.

Validates the analytical lower-MAC bound (ceil(Q_T * d / lmc)) against
SCALE-Sim for all (T, merge_n, lmc, mode) combinations that fit in memory.

Array mapping for one sub-array (merge_n levels → 2^merge_n parallel sub-arrays):
    rows = H  (one per query head, fixed regardless of merge_n)
    cols = T / 2^merge_n  (KV tokens assigned to this sub-array)

lmc note: for decode Q_T=1, effective_lmc=min(lmc,1)=1 always — interleaved MACs
only help when multiple packets share a row. decode+lmc>1 configs are skipped.

Memory limits (empirical from size check):
    prefill lmc=1  : T <= 512 (T=1024+ builds >134 MB operand matrices)
    prefill lmc=16 : T <= 2048 merge_n=0, T <= 4096 merge_n=3
    decode         : all T safe (M=H=64 always)
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from scalesim.scale_config import scale_config as ScaleConfig
from scalesim.topology_utils import topologies as Topo
from scalesim.layout_utils import layouts as Layout
from scalesim.simulator import simulator as Simulator

from kv_stationary_model import kv_stationary_metrics
from baseline_mqa_model import baseline_mqa_metrics

H          = 64
D          = 128
BW_ANALYTIC = 512      # bytes/cycle — matches full_results.csv
PE_MAC_WIDTH = 128
EXP_LATENCY  = 4
SRAM_KB    = 4 * 1024
SCALESIM_BW = 100_000  # infinite DRAM for SCALE-Sim (isolate compute)

OUTPUT_CSV = Path("validate_lower_mac_sweep.csv")


# ── SCALE-Sim helpers ─────────────────────────────────────────────────────────

def _write_config(path: Path, arr_rows: int, arr_cols: int) -> None:
    path.write_text(
        "[general]\nrun_name=sweep\n\n"
        "[architecture_presets]\n"
        f"ArrayHeight:    {arr_rows}\n"
        f"ArrayWidth:     {arr_cols}\n"
        f"IfmapSramSzkB:  {SRAM_KB}\n"
        f"FilterSramSzkB: {SRAM_KB}\n"
        f"OfmapSramSzkB:  {SRAM_KB}\n"
        "IfmapOffset:    0\nFilterOffset:   10000000\nOfmapOffset:    20000000\n"
        f"Bandwidth:      {SCALESIM_BW}\n"
        "Dataflow:       ws\nMemoryBanks:    1\n"
        "ReadRequestBuffer:  64\nWriteRequestBuffer: 64\n\n"
        "[layout]\n"
        "IfmapCustomLayout:       False\nIfmapSRAMBankBandwidth:  10\n"
        "IfmapSRAMBankNum:        10\nIfmapSRAMBankPort:       2\n"
        "FilterCustomLayout:      False\nFilterSRAMBankBandwidth: 10\n"
        "FilterSRAMBankNum:       10\nFilterSRAMBankPort:      2\n\n"
        "[run_presets]\nInterfaceBandwidth: CALC\nUseRamulatorTrace:  False\n"
    )


def _run_gemm(M: int, N: int, arr_rows: int, arr_cols: int) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path  = Path(tmp) / "cfg.cfg"
        topo_path = Path(tmp) / "topo.csv"
        out_path  = Path(tmp) / "out"
        out_path.mkdir()
        _write_config(cfg_path, arr_rows=arr_rows, arr_cols=arr_cols)
        topo_path.write_text(f"Layer,M,N,K,\nQK,{M},{N},{D},\n")
        cfg    = ScaleConfig(); cfg.read_conf_file(str(cfg_path))
        topo   = Topo();        topo.load_arrays(topofile=str(topo_path), mnk_inputs=True)
        sim    = Simulator()
        sim.set_params(config_obj=cfg, topo_obj=topo, layout_obj=Layout(),
                       top_path=str(out_path), verbosity=False, save_trace=False)
        sim.run()
        items = sim.single_layer_sim_object_list[0].get_compute_report_items()
        return {"compute_cycles": int(items[1]), "stall_cycles": int(items[2]),
                "mapping_eff": float(items[4])}


# ── Config builder ────────────────────────────────────────────────────────────

def _build_configs() -> List[Dict]:
    """Return all (T, merge_n, lmc, mode) configs that fit in memory."""
    configs = []
    for T in [128, 512, 1024, 2048, 4096, 8192]:
        for merge_n in [0, 3]:
            eff_cols = T // (2 ** merge_n)
            for lmc in [1, 16]:
                for mode in ['decode', 'prefill']:
                    Q_T = 1 if mode == 'decode' else T

                    # lmc>1 has no effect for decode (only 1 packet per row)
                    if mode == 'decode' and lmc > 1:
                        continue

                    # Memory guard: skip configs that would OOM
                    slice_qt = Q_T if lmc == 1 else Q_T // lmc
                    if slice_qt == 0:
                        continue
                    M = H * slice_qt
                    elems = M * D + D * eff_cols + M * eff_cols
                    mb = elems * 8 / 1e6
                    if mb > 500:
                        continue

                    configs.append(dict(T=T, merge_n=merge_n, lmc=lmc,
                                        mode=mode, Q_T=Q_T,
                                        eff_cols=eff_cols, M=M, mb=mb))
    return configs


# ── Per-config runner ─────────────────────────────────────────────────────────

def _run_config(cfg: Dict) -> Dict:
    T        = cfg['T']
    merge_n  = cfg['merge_n']
    lmc      = cfg['lmc']
    Q_T      = cfg['Q_T']
    eff_cols = cfg['eff_cols']
    M        = cfg['M']

    # Analytical lower-MAC bound
    analytical = math.ceil(Q_T * D / lmc)

    # SCALE-Sim: one sub-array (H rows × eff_cols cols)
    ss = _run_gemm(M=M, N=eff_cols, arr_rows=H, arr_cols=eff_cols)
    ss_cycles = ss['compute_cycles']
    delta_pct = 100.0 * (ss_cycles - analytical) / analytical

    # Full analytical pipeline from kv_stationary_model (for context)
    eff_rows_kv = H * (2 ** merge_n)
    eff_cols_kv = eff_cols
    kv = kv_stationary_metrics(
        H=H, T=T, d=D,
        array_rows=eff_rows_kv, array_cols=eff_cols_kv,
        bytes_per_element=2,
        memory_bandwidth_bytes_per_cycle=BW_ANALYTIC,
        pe_mac_width=PE_MAC_WIDTH, lower_mac_count=lmc,
        exp_latency_cycles=EXP_LATENCY,
        query_tokens=Q_T,
    )
    pipeline_cycles = int(kv['total_cycles'])

    # Baseline roofline for comparison
    baseline = baseline_mqa_metrics(
        H=H, T=T, d=D,
        array_rows=H, array_cols=H,
        bytes_per_element=2,
        memory_bandwidth_bytes_per_cycle=BW_ANALYTIC,
        query_tokens=Q_T,
    )
    baseline_cycles = int(baseline['estimated_cycles'])

    # How much of the full pipeline is the lower MAC?
    mac_frac_analytic = analytical / pipeline_cycles if pipeline_cycles > 0 else 0
    mac_frac_ss       = ss_cycles  / pipeline_cycles if pipeline_cycles > 0 else 0

    return {
        'T':               T,
        'merge_n':         merge_n,
        'lmc':             lmc,
        'mode':            cfg['mode'],
        'Q_T':             Q_T,
        'sub_rows':        H,
        'eff_cols':        eff_cols,
        'GEMM_M':          M,
        'GEMM_N':          eff_cols,
        # lower MAC
        'analytical_mac':  analytical,
        'ss_mac_cycles':   ss_cycles,
        'delta_pct':       round(delta_pct, 1),
        'drain_frac':      round((ss_cycles - analytical) / ss_cycles, 3) if ss_cycles else 0,
        'mapping_eff_pct': round(ss['mapping_eff'] * 100, 1),
        # full pipeline context
        'pipeline_cycles': pipeline_cycles,
        'baseline_cycles': baseline_cycles,
        'mac_frac_analytic': round(mac_frac_analytic, 3),
        'mac_frac_ss':       round(mac_frac_ss,       3),
        # corrected pipeline: replace analytical MAC with SCALE-Sim MAC
        'corrected_pipeline': pipeline_cycles - analytical + ss_cycles,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def _print_section(rows: List[Dict], title: str) -> None:
    if not rows:
        return
    print(f"\n{'─'*105}")
    print(f"  {title}")
    print(f"{'─'*105}")
    print(f"  {'T':>5}  {'mn':>2}  {'lmc':>3}  {'mode':>7}  "
          f"{'analytic_MAC':>13}  {'ss_MAC':>8}  {'delta%':>7}  "
          f"{'drain':>6}  {'pipeline':>10}  {'mac_frac_ss':>12}  "
          f"{'corrected_pipeline':>19}  {'vs_baseline':>12}")
    print(f"  {'─'*5}  {'─'*2}  {'─'*3}  {'─'*7}  "
          f"{'─'*13}  {'─'*8}  {'─'*7}  "
          f"{'─'*6}  {'─'*10}  {'─'*12}  "
          f"{'─'*19}  {'─'*12}")
    for r in rows:
        vs_base = r['corrected_pipeline'] / r['baseline_cycles'] if r['baseline_cycles'] else 0
        print(f"  {r['T']:>5}  {r['merge_n']:>2}  {r['lmc']:>3}  {r['mode']:>7}  "
              f"{r['analytical_mac']:>13,}  {r['ss_mac_cycles']:>8,}  {r['delta_pct']:>+6.1f}%  "
              f"{r['drain_frac']:>6.3f}  {r['pipeline_cycles']:>10,}  "
              f"{r['mac_frac_ss']:>11.3f}  "
              f"{r['corrected_pipeline']:>19,}  "
              f"{vs_base:>11.2f}x")


def main() -> None:
    configs = _build_configs()
    print(f"Running {len(configs)} SCALE-Sim configs...", flush=True)

    results: List[Dict] = []
    for i, cfg in enumerate(configs):
        label = f"T={cfg['T']:>5} merge_n={cfg['merge_n']} lmc={cfg['lmc']:>2} {cfg['mode']:>7}  ({cfg['mb']:.0f} MB)"
        print(f"  [{i+1:>2}/{len(configs)}] {label}", end='', flush=True)
        r = _run_config(cfg)
        results.append(r)
        print(f"  → ss={r['ss_mac_cycles']:,}  delta={r['delta_pct']:+.1f}%  "
              f"corrected_pipeline={r['corrected_pipeline']:,}", flush=True)

    # Print grouped tables
    decode_rows   = [r for r in results if r['mode'] == 'decode']
    prefill_lmc1  = [r for r in results if r['mode'] == 'prefill' and r['lmc'] == 1]
    prefill_lmc16 = [r for r in results if r['mode'] == 'prefill' and r['lmc'] == 16]

    _print_section(decode_rows,   "DECODE  (Q_T=1, lmc=1 — lmc>1 has no effect for single-token decode)")
    _print_section(prefill_lmc1,  "PREFILL lmc=1  (T≤512 only — larger configs OOM)")
    _print_section(prefill_lmc16, "PREFILL lmc=16 (T≤2048 merge_n=0, T≤4096 merge_n=3)")

    # Save CSV
    if results:
        fieldnames = list(results[0].keys())
        with OUTPUT_CSV.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} rows to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
