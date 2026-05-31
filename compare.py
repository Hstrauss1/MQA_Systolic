#!/usr/bin/env python3
"""Compare Phase 7 baseline and KV-stationary MQA sweep outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_OUTPUT_ROOT = Path('outputs')
DEFAULT_SWEEP_CSV = 'sweep_results.csv'
DEFAULT_CSV = 'comparison.csv'
DEFAULT_JSON = 'comparison.json'
JOIN_KEYS = [
    'experiment_id',
    'sequence_length',
    'batch_size',
    'query_heads',
    'kv_heads',
    'head_dim',
    'array_shape',
    'decode_tokens',
    'precision',
]


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def load_rows(path: Path) -> List[Dict[str, object]]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def normalize_numeric(row: Dict[str, object], key: str) -> float:
    value = row.get(key, 0)
    if value in ('', None):
        return 0.0
    return float(value)


def find_latest_run_dir(output_root: Path) -> Path:
    if not output_root.exists():
        raise FileNotFoundError(f'Output root does not exist: {output_root}')
    candidates = sorted(path for path in output_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f'No run directories found under {output_root}')
    return candidates[-1]


def maybe_progress(iterable, enabled: bool, total: int, desc: str):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def compare_rows(baseline: Dict[str, object], kv: Dict[str, object]) -> Dict[str, object]:
    baseline_total_cycles = normalize_numeric(baseline, 'total_cycles')
    kv_total_cycles = normalize_numeric(kv, 'total_cycles')
    baseline_dram_reads = normalize_numeric(baseline, 'dram_reads')
    kv_dram_reads = normalize_numeric(kv, 'dram_reads')
    baseline_dram_writes = normalize_numeric(baseline, 'dram_writes')
    kv_dram_writes = normalize_numeric(kv, 'dram_writes')
    baseline_sram_total = normalize_numeric(baseline, 'sram_reads') + normalize_numeric(baseline, 'sram_writes')
    kv_sram_total = normalize_numeric(kv, 'sram_reads') + normalize_numeric(kv, 'sram_writes')
    baseline_util = normalize_numeric(baseline, 'weighted_pe_utilization')
    kv_util = normalize_numeric(kv, 'weighted_pe_utilization')
    baseline_stall = normalize_numeric(baseline, 'memory_stall_cycles')
    kv_stall = normalize_numeric(kv, 'memory_stall_cycles')
    kv_preload_bw = normalize_numeric(kv, 'kv_preload_bandwidth_cycles')

    out = {key: baseline[key] for key in JOIN_KEYS}
    out.update({
        'baseline_total_cycles': baseline_total_cycles,
        'kv_total_cycles': kv_total_cycles,
        'baseline_over_kv_cycle_ratio': safe_ratio(baseline_total_cycles, kv_total_cycles),
        'kv_over_baseline_cycle_ratio': safe_ratio(kv_total_cycles, baseline_total_cycles),
        'baseline_dram_reads': baseline_dram_reads,
        'kv_dram_reads': kv_dram_reads,
        'dram_read_ratio_baseline_over_kv': safe_ratio(baseline_dram_reads, kv_dram_reads),
        'baseline_dram_writes': baseline_dram_writes,
        'kv_dram_writes': kv_dram_writes,
        'dram_write_ratio_baseline_over_kv': safe_ratio(baseline_dram_writes, kv_dram_writes),
        'baseline_total_dram': baseline_dram_reads + baseline_dram_writes,
        'kv_total_dram': kv_dram_reads + kv_dram_writes,
        'total_dram_ratio_baseline_over_kv': safe_ratio(baseline_dram_reads + baseline_dram_writes, kv_dram_reads + kv_dram_writes),
        'baseline_total_sram': baseline_sram_total,
        'kv_total_sram': kv_sram_total,
        'sram_ratio_baseline_over_kv': safe_ratio(baseline_sram_total, kv_sram_total),
        'baseline_weighted_pe_utilization': baseline_util,
        'kv_weighted_pe_utilization': kv_util,
        'weighted_pe_utilization_delta_kv_minus_baseline': kv_util - baseline_util,
        'baseline_memory_stall_cycles': baseline_stall,
        'kv_memory_stall_cycles': kv_stall,
        'memory_stall_cycles_delta_kv_minus_baseline': kv_stall - baseline_stall,
        'kv_preload_bandwidth_cycles': kv_preload_bw,
        'baseline_stage_names': baseline.get('stage_names', ''),
        'kv_stage_names': kv.get('stage_names', ''),
    })
    return out


def write_outputs(output_dir: Path, rows: List[Dict[str, object]]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / DEFAULT_CSV
    json_path = output_dir / DEFAULT_JSON

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)

    return {'csv': str(csv_path), 'json': str(json_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare Phase 7 baseline and KV-stationary MQA sweep outputs.')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--input', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(args.output_root)
    input_path = args.input or (run_dir / DEFAULT_SWEEP_CSV)
    output_dir = args.output_dir or run_dir

    log(f'Using run directory {run_dir}', args.verbose)
    log(f'Loading sweep results from {input_path}', args.verbose)
    rows = load_rows(input_path)
    log(f'Loaded {len(rows)} rows', args.verbose)

    baseline_rows = {tuple(row[k] for k in JOIN_KEYS): row for row in rows if row.get('mode') == 'baseline_mqa_decode'}
    kv_rows = {tuple(row[k] for k in JOIN_KEYS): row for row in rows if row.get('mode') == 'kv_stationary_mqa_decode'}
    common_keys = sorted(set(baseline_rows.keys()) & set(kv_rows.keys()))
    if not common_keys:
        raise ValueError('No matching baseline/KV rows found in input CSV')

    log(f'Found {len(common_keys)} matched baseline/KV workload pairs', args.verbose)
    comparison_rows: List[Dict[str, object]] = []
    iterator = maybe_progress(common_keys, enabled=args.progress, total=len(common_keys), desc='Phase 7 compare')
    for key in iterator:
        baseline = baseline_rows[key]
        kv = kv_rows[key]
        if args.verbose:
            log(
                f"[COMPARE] exp={baseline['experiment_id']} seq={baseline['sequence_length']} tok={baseline['decode_tokens']} array={baseline['array_shape']}",
                True,
            )
        comparison_rows.append(compare_rows(baseline, kv))

    outputs = write_outputs(output_dir, comparison_rows)
    print(f'Phase 7 comparison complete: {len(comparison_rows)} rows')
    print(f'RUN_DIR: {run_dir}')
    print(f"CSV: {outputs['csv']}")
    print(f"JSON: {outputs['json']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
