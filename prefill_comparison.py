"""Prefill-focused comparison: KV-stationary vs FlashAttention on standard array.

Both architectures perform the same attention computation at prefill:
    scores = softmax(Q [T×H×d] × K^T [T×d] / √d) × V [T×d]  → [T×H×d]

FlashAttention (standard H×H systolic array):
    Tiles the T×T score matrix into Br×Bc blocks.
    Per tile: QK GEMM [H×Br×d]×[d×Bc] + online softmax + AV GEMM [H×Br×Bc]×[Bc×d]
    Score matrix never written to DRAM.
    Both tile GEMMs validated with SCALE-Sim.

KV-stationary (2D streaming array, merge_n levels):
    K and V loaded once into array columns (T/2^n per sub-array).
    All T×H query packets stream horizontally, softmax inline.
    Lower MAC validated with SCALE-Sim (from validate_lower_mac_sweep.csv).
    Upper PE (exp + V accumulation) remains analytical.

Comparison is on equal silicon: same total MAC count, same DRAM bandwidth.
Array area noted explicitly — KV-stationary uses more PEs at merge_n>0.
"""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from scalesim.scale_config import scale_config as ScaleConfig
from scalesim.topology_utils import topologies as Topo
from scalesim.layout_utils import layouts as Layout
from scalesim.simulator import simulator as Simulator

from kv_stationary_model import kv_stationary_metrics

# ── Fixed parameters (match full_results.csv) ────────────────────────────────
H           = 64       # query heads
D           = 128      # head dimension
BPE         = 2        # bytes per element (fp16)
BW          = 512      # DRAM bytes/cycle
EXP_LATENCY = 4
PE_MAC_W    = 128      # upper PE MAC width (fully parallel)
SRAM_KB     = 4 * 1024
SCALESIM_BW = 100_000

# FlashAttention tile size: largest Br=Bc that fits in SRAM
# SRAM holds: Q tile (Br×d) + K tile (Bc×d) + V tile (Bc×d) + O acc (Br×d)
# = d×(2Br + 2Bc) elements. With Br=Bc=B: 4×d×B = SRAM_bytes
# B = SRAM_bytes / (4 × d × bpe)
FLASH_TILE = min(H, (SRAM_KB * 1024) // (4 * D * BPE))  # = 64

OUTPUT_CSV = Path("prefill_comparison.csv")


# ── SCALE-Sim runner ──────────────────────────────────────────────────────────

def _write_config(path: Path, arr_rows: int, arr_cols: int) -> None:
    path.write_text(
        "[general]\nrun_name=prefill\n\n"
        "[architecture_presets]\n"
        f"ArrayHeight:    {arr_rows}\n"
        f"ArrayWidth:     {arr_cols}\n"
        f"IfmapSramSzkB:  {SRAM_KB}\n"
        f"FilterSramSzkB: {SRAM_KB}\n"
        f"OfmapSramSzkB:  {SRAM_KB}\n"
        "IfmapOffset: 0\nFilterOffset: 10000000\nOfmapOffset: 20000000\n"
        f"Bandwidth:      {SCALESIM_BW}\n"
        "Dataflow: ws\nMemoryBanks: 1\n"
        "ReadRequestBuffer: 64\nWriteRequestBuffer: 64\n\n"
        "[layout]\n"
        "IfmapCustomLayout: False\nIfmapSRAMBankBandwidth: 10\n"
        "IfmapSRAMBankNum: 10\nIfmapSRAMBankPort: 2\n"
        "FilterCustomLayout: False\nFilterSRAMBankBandwidth: 10\n"
        "FilterSRAMBankNum: 10\nFilterSRAMBankPort: 2\n\n"
        "[run_presets]\nInterfaceBandwidth: CALC\nUseRamulatorTrace: False\n"
    )


def _run_gemm(M: int, N: int, K: int, arr_rows: int, arr_cols: int) -> Dict:
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path  = Path(tmp) / "cfg.cfg"
        topo_path = Path(tmp) / "topo.csv"
        out_path  = Path(tmp) / "out"; out_path.mkdir()
        _write_config(cfg_path, arr_rows, arr_cols)
        topo_path.write_text(f"Layer,M,N,K,\nGEMM,{M},{N},{K},\n")
        cfg  = ScaleConfig(); cfg.read_conf_file(str(cfg_path))
        topo = Topo();        topo.load_arrays(topofile=str(topo_path), mnk_inputs=True)
        sim  = Simulator()
        sim.set_params(config_obj=cfg, topo_obj=topo, layout_obj=Layout(),
                       top_path=str(out_path), verbosity=False, save_trace=False)
        sim.run()
        items = sim.single_layer_sim_object_list[0].get_compute_report_items()
        return {"cycles": int(items[1]), "mapping_eff": float(items[4])}


# ── FlashAttention baseline ───────────────────────────────────────────────────

def flash_attention_baseline(T: int) -> Dict:
    """
    Standard H×H array running tiled FlashAttention.

    Tile sizes Br = Bc = FLASH_TILE (= H = 64 here).
    Number of tiles = (T/Br) × (T/Bc) = (T/64)²

    Per tile:
      QK GEMM:  M = H×Br, N = Bc,  K = D   (query block × key block)
      AV GEMM:  M = H×Br, N = D,   K = Bc  (attn weights × value block)

    DRAM traffic (FlashAttention — no score matrix written):
      Q:  T × H × D × bpe
      K:  T × D × bpe        (MQA: 1 KV head, shared)
      V:  T × D × bpe
      O:  T × H × D × bpe
    """
    Br = Bc = FLASH_TILE
    n_tiles = math.ceil(T / Br) * math.ceil(T / Bc)   # total QK tile pairs

    # SCALE-Sim for one representative QK tile
    qk_M = H * Br
    qk = _run_gemm(M=qk_M, N=Bc, K=D, arr_rows=H, arr_cols=H)

    # SCALE-Sim for one representative AV tile
    av = _run_gemm(M=qk_M, N=D, K=Bc, arr_rows=H, arr_cols=H)

    # Total compute cycles (tiles run sequentially on H×H array)
    # Each tile pair = qk_cycles + av_cycles (softmax is memory-side, free at infinite BW)
    total_compute = n_tiles * (qk["cycles"] + av["cycles"])

    # DRAM traffic — FlashAttention tiles K/V in SRAM, no score HBM
    q_bytes  = T * H * D * BPE
    k_bytes  = T * D * BPE        # MQA: shared K
    v_bytes  = T * D * BPE        # MQA: shared V
    o_bytes  = T * H * D * BPE
    dram_bytes = q_bytes + k_bytes + v_bytes + o_bytes
    mem_cycles = math.ceil(dram_bytes / BW)

    total_cycles = max(total_compute, mem_cycles)
    bottleneck = "compute" if total_compute >= mem_cycles else "memory"

    # MACs: QK + AV for all heads, all query-key pairs
    total_macs = 2 * H * T * T * D
    ai = total_macs / dram_bytes

    return {
        "T": T,
        "arch": "flash_attention",
        "array_rows": H,
        "array_cols": H,
        "array_pes": H * H,
        "n_tiles": n_tiles,
        "tile_Br": Br,
        "tile_Bc": Bc,
        "ss_qk_tile_cycles": qk["cycles"],
        "ss_av_tile_cycles": av["cycles"],
        "ss_tile_pair_cycles": qk["cycles"] + av["cycles"],
        "total_compute_cycles": total_compute,
        "mem_cycles": mem_cycles,
        "total_cycles": total_cycles,
        "bottleneck": bottleneck,
        "dram_bytes": dram_bytes,
        "dram_mb": round(dram_bytes / 1e6, 2),
        "total_macs": total_macs,
        "arithmetic_intensity": round(ai, 2),
        "qk_mapping_eff_pct": round(qk["mapping_eff"] * 100, 1),
        "av_mapping_eff_pct": round(av["mapping_eff"] * 100, 1),
    }


# ── KV-stationary prefill ─────────────────────────────────────────────────────

def kv_stationary_prefill(T: int, merge_n: int, lmc: int,
                           ss_correction: Optional[Dict] = None) -> Dict:
    """
    KV-stationary array with merge_n levels and lmc interleaved MACs.
    Q_T = T (all query tokens stream through — prefill mode).

    sub-array: H rows × (T/2^n) cols
    2^n sub-arrays run in parallel.
    Total array PEs = H × T (fixed regardless of merge_n — same silicon).

    ss_correction: SCALE-Sim validated lower MAC cycles from sweep CSV.
    """
    eff_rows = H * (2 ** merge_n)
    eff_cols = max(1, T // (2 ** merge_n))

    kv = kv_stationary_metrics(
        H=H, T=T, d=D,
        array_rows=eff_rows, array_cols=eff_cols,
        bytes_per_element=BPE,
        memory_bandwidth_bytes_per_cycle=BW,
        pe_mac_width=PE_MAC_W,
        lower_mac_count=lmc,
        exp_latency_cycles=EXP_LATENCY,
        query_tokens=T,
    )

    analytical_mac = math.ceil(T * D / lmc)
    pipeline_cycles = int(kv["total_cycles"])

    # Apply SCALE-Sim correction for lower MAC if available
    ss_mac = None
    corrected_cycles = None
    if ss_correction:
        ss_mac = ss_correction["ss_mac_cycles"]
        corrected_cycles = pipeline_cycles - analytical_mac + ss_mac

    dram_bytes = int(kv["total_dram_bytes"])
    total_macs = int(kv["total_macs"])
    ai = total_macs / dram_bytes if dram_bytes else 0

    return {
        "T": T,
        "arch": f"kv_stationary_n{merge_n}_lmc{lmc}",
        "merge_n": merge_n,
        "lmc": lmc,
        "array_rows": eff_rows,
        "array_cols": eff_cols,
        "array_pes": eff_rows * eff_cols,
        "n_tiles": int(kv["token_tiles"]),
        "analytical_mac_cycles": analytical_mac,
        "ss_mac_cycles": ss_mac,
        "pipeline_cycles": pipeline_cycles,
        "corrected_cycles": corrected_cycles,
        "total_cycles": corrected_cycles if corrected_cycles else pipeline_cycles,
        "dram_bytes": dram_bytes,
        "dram_mb": round(dram_bytes / 1e6, 2),
        "total_macs": total_macs,
        "arithmetic_intensity": round(ai, 2),
        "pe_utilization": round(float(kv["pe_utilization"]), 4),
        "bottleneck": str(kv.get("bottleneck", "compute")),
        "upper_pe_util_pct": round(float(kv.get("upper_pe_cycles_per_stage", 0)) /
                                   max(1, float(kv["column_dwell"])) * 100, 1),
        "mac_frac_ss": round(ss_mac / pipeline_cycles, 3) if ss_mac else None,
    }


# ── Load SCALE-Sim corrections from sweep CSV ────────────────────────────────

def load_ss_corrections() -> Dict:
    """Load validated lower MAC cycles from validate_lower_mac_sweep.csv."""
    path = Path("validate_lower_mac_sweep.csv")
    if not path.exists():
        return {}
    corrections = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["mode"] != "prefill":
                continue
            key = (int(row["T"]), int(row["merge_n"]), int(row["lmc"]))
            corrections[key] = {
                "ss_mac_cycles": int(row["ss_mac_cycles"]),
                "delta_pct": float(row["delta_pct"]),
            }
    return corrections


# ── Comparison and output ─────────────────────────────────────────────────────

def compare(flash: Dict, kv: Dict) -> Dict:
    """Compute speedup and traffic ratios between the two architectures."""
    kv_cycles = kv["total_cycles"]
    fl_cycles  = flash["total_cycles"]
    return {
        "T":                   flash["T"],
        "merge_n":             kv.get("merge_n"),
        "lmc":                 kv.get("lmc"),
        "flash_cycles":        fl_cycles,
        "kv_cycles":           kv_cycles,
        "speedup_kv_over_flash": round(fl_cycles / kv_cycles, 3),
        "flash_dram_mb":       flash["dram_mb"],
        "kv_dram_mb":          kv["dram_mb"],
        "dram_ratio_flash_over_kv": round(flash["dram_mb"] / kv["dram_mb"], 3),
        "flash_pes":           flash["array_pes"],
        "kv_pes":              kv["array_pes"],
        "pe_area_ratio_kv_over_flash": round(kv["array_pes"] / flash["array_pes"], 1),
        # area-normalised speedup: speedup per unit silicon
        "area_norm_speedup":   round((fl_cycles / kv_cycles) /
                                     (kv["array_pes"] / flash["array_pes"]), 3),
        "kv_ss_corrected":     kv["corrected_cycles"] is not None,
        "kv_mac_frac_ss":      kv.get("mac_frac_ss"),
    }


def _print_results(flash_rows: List[Dict], kv_rows: List[Dict],
                   cmp_rows: List[Dict]) -> None:
    print(f"\n{'─'*85}")
    print("  FlashAttention baseline (H×H=64×64 array, SCALE-Sim validated tiles)")
    print(f"{'─'*85}")
    print(f"  {'T':>5}  {'tiles':>6}  {'qk_tile':>8}  {'av_tile':>8}  "
          f"{'compute':>10}  {'memory':>8}  {'total':>10}  {'bot':>7}  {'DRAM_MB':>8}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  "
          f"{'─'*10}  {'─'*8}  {'─'*10}  {'─'*7}  {'─'*8}")
    for r in flash_rows:
        print(f"  {r['T']:>5}  {r['n_tiles']:>6,}  {r['ss_qk_tile_cycles']:>8,}  "
              f"{r['ss_av_tile_cycles']:>8,}  "
              f"{r['total_compute_cycles']:>10,}  {r['mem_cycles']:>8,}  "
              f"{r['total_cycles']:>10,}  {r['bottleneck']:>7}  {r['dram_mb']:>8.1f}")

    print(f"\n{'─'*100}")
    print("  KV-stationary prefill (Q_T=T, SCALE-Sim corrected lower MAC where available)")
    print(f"{'─'*100}")
    print(f"  {'T':>5}  {'mn':>2}  {'lmc':>3}  {'pes':>6}  "
          f"{'analytic':>10}  {'ss_corr':>10}  {'total':>10}  "
          f"{'mac_frac':>9}  {'PE_util':>8}  {'DRAM_MB':>8}")
    print(f"  {'─'*5}  {'─'*2}  {'─'*3}  {'─'*6}  "
          f"{'─'*10}  {'─'*10}  {'─'*10}  "
          f"{'─'*9}  {'─'*8}  {'─'*8}")
    for r in kv_rows:
        ss_str  = f"{r['corrected_cycles']:>10,}" if r['corrected_cycles'] else "  (analytic)"
        mac_str = f"{r['mac_frac_ss']:.3f}" if r.get('mac_frac_ss') else "    n/a"
        print(f"  {r['T']:>5}  {r['merge_n']:>2}  {r['lmc']:>3}  "
              f"{r['array_pes']:>6,}  "
              f"{r['pipeline_cycles']:>10,}  {ss_str}  {r['total_cycles']:>10,}  "
              f"{mac_str:>9}  {r['pe_utilization']:>8.4f}  {r['dram_mb']:>8.1f}")

    print(f"\n{'─'*95}")
    print("  Comparison (speedup = flash_cycles / kv_cycles)")
    print(f"{'─'*95}")
    print(f"  {'T':>5}  {'mn':>2}  {'lmc':>3}  {'flash_cyc':>10}  {'kv_cyc':>10}  "
          f"{'speedup':>8}  {'area_ratio':>10}  {'area_norm':>10}  {'dram_ratio':>11}")
    print(f"  {'─'*5}  {'─'*2}  {'─'*3}  {'─'*10}  {'─'*10}  "
          f"{'─'*8}  {'─'*10}  {'─'*10}  {'─'*11}")
    for r in cmp_rows:
        print(f"  {r['T']:>5}  {r['merge_n']:>2}  {r['lmc']:>3}  "
              f"{r['flash_cycles']:>10,}  {r['kv_cycles']:>10,}  "
              f"{r['speedup_kv_over_flash']:>8.2f}x  "
              f"{r['pe_area_ratio_kv_over_flash']:>9.1f}x  "
              f"{r['area_norm_speedup']:>9.3f}x  "
              f"{r['dram_ratio_flash_over_kv']:>10.2f}x")


def main() -> None:
    T_VALUES   = [128, 512, 1024, 2048, 4096, 8192]
    KV_CONFIGS = [(0, 1), (0, 16), (3, 1), (3, 16)]  # (merge_n, lmc)

    print(f"Tile size Br=Bc={FLASH_TILE}  (SRAM={SRAM_KB}KB, d={D}, bpe={BPE})")
    ss_corrections = load_ss_corrections()
    print(f"Loaded {len(ss_corrections)} SCALE-Sim corrections from sweep CSV")

    # Run FlashAttention baseline (SCALE-Sim validated tiles, then multiply by tile count)
    print("\nRunning FlashAttention tile GEMMs in SCALE-Sim...")
    flash_rows: List[Dict] = []
    for T in T_VALUES:
        print(f"  T={T}", end='', flush=True)
        flash_rows.append(flash_attention_baseline(T))
        print(f"  → {flash_rows[-1]['total_cycles']:,} cycles", flush=True)

    # Run KV-stationary prefill
    print("\nComputing KV-stationary prefill...")
    kv_rows: List[Dict] = []
    for T in T_VALUES:
        for merge_n, lmc in KV_CONFIGS:
            corr = ss_corrections.get((T, merge_n, lmc))
            kv_rows.append(kv_stationary_prefill(T, merge_n, lmc, corr))

    # Comparison
    cmp_rows: List[Dict] = []
    flash_by_T = {r["T"]: r for r in flash_rows}
    for kv in kv_rows:
        cmp_rows.append(compare(flash_by_T[kv["T"]], kv))

    _print_results(flash_rows, kv_rows, cmp_rows)

    # Save
    all_rows = []
    for f, kv_list in zip(flash_rows, [kv_rows[i::len(T_VALUES)] for i in range(len(T_VALUES))]):
        all_rows.append(f)
    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cmp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cmp_rows)
    print(f"\nSaved comparison to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
