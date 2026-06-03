"""Prefill attention comparison: KV-stationary vs FlashAttention.

SCALE-Sim is used to validate the two GEMM-mappable components:
  [1] FlashAttention QK tile:   M=H×Br, N=Bc, K=d   on H×H array
  [2] FlashAttention AV tile:   M=H×Br, N=d,  K=Bc  on H×H array
  [3] KV-stationary lower MAC:  M=H×slice_qt, N=eff_cols, K=d on H×eff_cols sub-array

All other cycle estimates come from the analytical pipeline model (kv_stationary_metrics):
  - column_dwell  = d + exp_latency + ceil(3d/pe_mac_width)   [design assumption]
  - pipeline steps = (active_rows + active_cols - 1) × column_dwell + stagger terms
  - exp_latency, pe_mac_width: model parameters, not validated here

KV-stationary corrected total = analytical_pipeline - analytical_lower_mac + ss_lower_mac
"""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scalesim.scale_config import scale_config as ScaleConfig
from scalesim.topology_utils import topologies as Topo
from scalesim.layout_utils import layouts as Layout
from scalesim.simulator import simulator as Simulator

from kv_stationary_model import kv_stationary_metrics

H            = 64
D            = 128
BPE          = 2
BW           = 512
EXP_LATENCY  = 4
PE_MAC_WIDTH = 128
SRAM_KB      = 4 * 1024
SCALESIM_BW  = 100_000
FLASH_TILE   = min(H, (SRAM_KB * 1024) // (4 * D * BPE))  # 64

OUTPUT_CSV = Path("prefill_full_results.csv")


# ── SCALE-Sim helper ──────────────────────────────────────────────────────────

def _ss(M: int, N: int, K: int, arr_rows: int, arr_cols: int) -> Optional[int]:
    """Run one GEMM in SCALE-Sim. Returns None if operand matrices would OOM (>400 MB)."""
    if (M * K + K * N + M * N) * 8 / 1e6 > 400:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        cfg  = Path(tmp) / "cfg.cfg"
        topo = Path(tmp) / "topo.csv"
        out  = Path(tmp) / "out"; out.mkdir()
        cfg.write_text(
            "[general]\nrun_name=x\n\n[architecture_presets]\n"
            f"ArrayHeight: {arr_rows}\nArrayWidth:  {arr_cols}\n"
            f"IfmapSramSzkB: {SRAM_KB}\nFilterSramSzkB: {SRAM_KB}\nOfmapSramSzkB: {SRAM_KB}\n"
            "IfmapOffset: 0\nFilterOffset: 10000000\nOfmapOffset: 20000000\n"
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
        t = Topo();        t.load_arrays(topofile=str(topo), mnk_inputs=True)
        sim = Simulator()
        sim.set_params(config_obj=c, topo_obj=t, layout_obj=Layout(),
                       top_path=str(out), verbosity=False, save_trace=False)
        sim.run()
        return int(sim.single_layer_sim_object_list[0].get_compute_report_items()[1])


# ── FlashAttention ────────────────────────────────────────────────────────────

def flash_row(T: int, ss_qk: int, ss_av: int) -> Dict:
    """FlashAttention on H×H array. Both tile GEMMs SCALE-Sim validated."""
    Br = Bc      = FLASH_TILE
    n_tiles      = math.ceil(T / Br) * math.ceil(T / Bc)
    compute_cyc  = n_tiles * (ss_qk + ss_av)
    dram_bytes   = (2 * H + 2) * T * D * BPE   # Q + K(shared) + V(shared) + O, no score HBM
    mem_cyc      = math.ceil(dram_bytes / BW)
    total_cycles = max(compute_cyc, mem_cyc)
    total_macs   = 2 * H * T * T * D
    return {
        "arch": "flash_attention", "T": T, "merge_n": 0, "lmc": 1,
        "array_rows": H, "array_cols": H, "array_pes": H * H,
        "n_tiles": n_tiles, "tile_Br": Br, "tile_Bc": Bc,
        # SCALE-Sim validated
        "ss_qk_tile": ss_qk, "ss_av_tile": ss_av,
        "ss_lower_mac": None, "lower_mac_analytic": None,
        "lower_mac_validated": None,
        # cycles
        "compute_cycles": compute_cyc, "mem_cycles": mem_cyc,
        "total_cycles": total_cycles, "total_analytical": total_cycles,
        "bottleneck": "compute" if compute_cyc >= mem_cyc else "memory",
        # model params
        "column_dwell": None, "packet_stagger": None,
        "pe_utilization": None, "upper_pe_util_pct": None,
        "k_buf_mb": None,
        # traffic
        "dram_mb": round(dram_bytes / 1e6, 2),
        "total_macs": total_macs,
        "arithmetic_intensity": round(total_macs / dram_bytes, 2),
    }


# ── KV-stationary ─────────────────────────────────────────────────────────────

def kv_row(T: int, merge_n: int, lmc: int) -> Dict:
    """KV-stationary prefill (Q_T=T). Lower MAC SCALE-Sim corrected where feasible."""
    eff_rows = H * (2 ** merge_n)
    eff_cols = max(1, T // (2 ** merge_n))

    # Full analytical pipeline (column_dwell, fill/drain, pipeline steps)
    kv = kv_stationary_metrics(
        H=H, T=T, d=D,
        array_rows=eff_rows, array_cols=eff_cols,
        bytes_per_element=BPE,
        memory_bandwidth_bytes_per_cycle=BW,
        pe_mac_width=PE_MAC_WIDTH,
        lower_mac_count=lmc,
        exp_latency_cycles=EXP_LATENCY,
        query_tokens=T,
    )
    total_analytical = int(kv["total_cycles"])

    # Analytical lower MAC bound
    lower_mac_analytic = math.ceil(T * D / lmc)

    # SCALE-Sim lower MAC correction
    slice_qt  = T if lmc == 1 else T // lmc
    ss_lower  = _ss(M=H * slice_qt, N=eff_cols, K=D, arr_rows=H, arr_cols=eff_cols)
    ss_validated = ss_lower is not None

    lower_mac_validated = ss_lower if ss_validated else lower_mac_analytic
    total_corrected = total_analytical - lower_mac_analytic + lower_mac_validated

    dram_bytes = int(kv["total_dram_bytes"])
    total_macs = int(kv["total_macs"])

    return {
        "arch": f"kv_stat_n{merge_n}_lmc{lmc}", "T": T, "merge_n": merge_n, "lmc": lmc,
        "array_rows": eff_rows, "array_cols": eff_cols, "array_pes": eff_rows * eff_cols,
        "n_tiles": int(kv["token_tiles"]), "tile_Br": None, "tile_Bc": None,
        # SCALE-Sim validated
        "ss_qk_tile": None, "ss_av_tile": None,
        "ss_lower_mac": ss_lower,
        "lower_mac_analytic": lower_mac_analytic,
        "lower_mac_validated": lower_mac_validated,
        # cycles
        "compute_cycles": total_analytical,
        "mem_cycles": math.ceil(dram_bytes / BW),
        "total_analytical": total_analytical,
        "total_cycles": total_corrected,
        "bottleneck": "compute",
        # model params (design assumptions)
        "column_dwell": int(kv["column_dwell"]),
        "packet_stagger": int(kv["packet_stagger"]),
        "pe_utilization": round(float(kv["pe_utilization"]), 4),
        "upper_pe_util_pct": round(
            float(kv["upper_pe_cycles_per_stage"]) / max(1, float(kv["column_dwell"])) * 100, 1),
        "k_buf_mb": round(float(kv["total_k_buffer_bytes"]) / 1e6, 3),
        # traffic
        "dram_mb": round(dram_bytes / 1e6, 2),
        "total_macs": total_macs,
        "arithmetic_intensity": round(total_macs / dram_bytes, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    T_VALUES   = [128, 512, 1024, 2048, 4096, 8192]
    KV_CONFIGS: List[Tuple[int, int]] = [(0,1), (0,16), (3,1), (3,16)]

    # FlashAttention tile GEMMs — same shape for all T, run once
    print(f"Tile size Br=Bc={FLASH_TILE}  (H={H}, d={D}, SRAM={SRAM_KB}KB)")
    print("Running FlashAttention tile GEMMs in SCALE-Sim...")
    ss_qk = _ss(M=H*FLASH_TILE, N=FLASH_TILE, K=D,        arr_rows=H, arr_cols=H)
    ss_av = _ss(M=H*FLASH_TILE, N=D,          K=FLASH_TILE, arr_rows=H, arr_cols=H)
    print(f"  QK tile: {ss_qk:,} cycles  AV tile: {ss_av:,} cycles\n")

    rows: List[Dict] = []

    for T in T_VALUES:
        rows.append(flash_row(T, ss_qk, ss_av))
        for merge_n, lmc in KV_CONFIGS:
            rows.append(kv_row(T, merge_n, lmc))

    # ── Print comparison table ────────────────────────────────────────────────
    flash_by_T = {r["T"]: r for r in rows if r["arch"] == "flash_attention"}

    print(f"{'─'*115}")
    print(f"  {'T':>5}  {'config':>18}  {'flash_cyc':>12}  {'kv_cyc':>12}  "
          f"{'ss_corrected':>13}  {'speedup':>8}  {'pes_ratio':>10}  {'area_norm':>10}  {'dram=':>6}")
    print(f"  {'─'*5}  {'─'*18}  {'─'*12}  {'─'*12}  "
          f"{'─'*13}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*6}")

    cmp_rows: List[Dict] = []
    for r in rows:
        if r["arch"] == "flash_attention":
            continue
        f         = flash_by_T[r["T"]]
        speedup   = f["total_cycles"] / r["total_cycles"]
        pes_ratio = r["array_pes"] / f["array_pes"]
        area_norm = speedup / pes_ratio
        ss_flag   = "yes" if r["ss_lower_mac"] is not None else "no "
        dram_eq   = "=" if abs(r["dram_mb"] - f["dram_mb"]) < 0.1 else f"{r['dram_mb']:.1f}"
        print(f"  {r['T']:>5}  {r['arch']:>18}  {f['total_cycles']:>12,}  "
              f"{r['total_analytical']:>12,}  ss={ss_flag} {r['total_cycles']:>10,}  "
              f"{speedup:>7.2f}x  {pes_ratio:>9.1f}x  {area_norm:>9.3f}x  {dram_eq:>6}")
        cmp_rows.append({
            "T": r["T"], "merge_n": r["merge_n"], "lmc": r["lmc"],
            "arch": r["arch"],
            "array_pes": r["array_pes"],
            "flash_total_cycles": f["total_cycles"],
            "ss_qk_tile": f["ss_qk_tile"], "ss_av_tile": f["ss_av_tile"],
            "kv_total_analytical": r["total_analytical"],
            "kv_lower_mac_analytic": r["lower_mac_analytic"],
            "kv_ss_lower_mac": r["ss_lower_mac"],
            "kv_total_corrected": r["total_cycles"],
            "kv_lower_mac_ss_validated": r["ss_lower_mac"] is not None,
            "speedup_kv_over_flash": round(speedup, 3),
            "pe_area_ratio_kv_over_flash": round(pes_ratio, 1),
            "area_norm_speedup": round(area_norm, 3),
            "dram_equal": abs(r["dram_mb"] - f["dram_mb"]) < 0.1,
            "kv_dram_mb": r["dram_mb"],
            "flash_dram_mb": f["dram_mb"],
            "kv_total_macs": r["total_macs"],
            "kv_pe_utilization": r["pe_utilization"],
            "column_dwell": r["column_dwell"],
            "packet_stagger": r["packet_stagger"],
            "k_buf_mb": r["k_buf_mb"],
            "upper_pe_util_pct": r["upper_pe_util_pct"],
        })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    fieldnames = list(cmp_rows[0].keys())
    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cmp_rows)
    print(f"\nSaved {len(cmp_rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
