#!/usr/bin/env python3
"""Compare integrated baseline and KV-stationary MQA results from the shared Phase 5 backend."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from mqa_scalesim.baseline_decode import BaselineMQADecodeSimulator
from mqa_scalesim.kv_stationary_decode import KVStationaryMQADecodeSimulator
from mqa_scalesim.validation_bridge import result_to_experiment_row
from mqa_scalesim.workload import MQAWorkload


DEFAULT_OUTPUT_DIR = Path('phase5_outputs')
DEFAULT_CSV = 'phase5_compare_results.csv'
DEFAULT_JSON = 'phase5_compare_results.json'


def build_workload(sequence_length: int,
                   batch_size: int,
                   query_heads: int,
                   kv_heads: int,
                   head_dim: int,
                   array_rows: int,
                   array_cols: int,
                   decode_tokens: int,
                   mode: str) -> MQAWorkload:
    return MQAWorkload(
        mode=mode,
        sequence_length=sequence_length,
        batch_size=batch_size,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        precision='int8',
        array_rows=array_rows,
        array_cols=array_cols,
        ifmap_sram_kb=64,
        filter_sram_kb=64,
        ofmap_sram_kb=64,
        bandwidth_mode='calc',
        dram_bandwidth=None,
        decode_tokens=decode_tokens,
        decode_step=4,
        softmax_variant='online',
        exp_variant='lookup',
        reuse_kv_across_tokens=True,
        metadata={'array_shape': f'{array_rows}x{array_cols}'},
    )


def run_mode(workload: MQAWorkload) -> Dict[str, object]:
    if workload.mode == 'baseline_mqa_decode':
        result = BaselineMQADecodeSimulator(workload).simulate()
    elif workload.mode == 'kv_stationary_mqa_decode':
        result = KVStationaryMQADecodeSimulator(workload).simulate()
    else:
        raise ValueError(f'Unsupported mode: {workload.mode}')
    return result_to_experiment_row(workload, result)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def compare_rows(baseline: Dict[str, object], kv: Dict[str, object]) -> Dict[str, object]:
    return {
        'sequence_length': baseline['sequence_length'],
        'batch_size': baseline['batch_size'],
        'query_heads': baseline['query_heads'],
        'kv_heads': baseline['kv_heads'],
        'head_dim': baseline['head_dim'],
        'array_shape': baseline['array_shape'],
        'decode_tokens': baseline['decode_tokens'],
        'baseline_total_cycles': baseline['total_cycles'],
        'kv_total_cycles': kv['total_cycles'],
        'cycle_speedup_baseline_over_kv': safe_ratio(baseline['total_cycles'], kv['total_cycles']),
        'dram_read_ratio_baseline_over_kv': safe_ratio(baseline['dram_reads'], kv['dram_reads']),
        'dram_write_ratio_baseline_over_kv': safe_ratio(baseline['dram_writes'], kv['dram_writes']),
        'sram_traffic_ratio_baseline_over_kv': safe_ratio(
            baseline['sram_reads'] + baseline['sram_writes'],
            kv['sram_reads'] + kv['sram_writes'],
        ),
        'weighted_pe_util_delta_kv_minus_baseline': kv['weighted_pe_utilization'] - baseline['weighted_pe_utilization'],
        'baseline_amortized_preload_bytes_per_token': baseline['amortized_preload_bytes_per_token'],
        'kv_amortized_preload_bytes_per_token': kv['amortized_preload_bytes_per_token'],
        'amortized_preload_ratio_baseline_over_kv': safe_ratio(
            baseline['amortized_preload_bytes_per_token'],
            kv['amortized_preload_bytes_per_token'],
        ),
        'baseline_stage_names': baseline['stage_names'],
        'kv_stage_names': kv['stage_names'],
    }


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
    parser = argparse.ArgumentParser(description='Compare integrated baseline and KV-stationary MQA paths.')
    parser.add_argument('--sequence-length', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--query-heads', type=int, default=8)
    parser.add_argument('--kv-heads', type=int, default=2)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--array-rows', type=int, default=16)
    parser.add_argument('--array-cols', type=int, default=16)
    parser.add_argument('--decode-tokens', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    baseline_row = run_mode(build_workload(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        array_rows=args.array_rows,
        array_cols=args.array_cols,
        decode_tokens=args.decode_tokens,
        mode='baseline_mqa_decode',
    ))
    kv_row = run_mode(build_workload(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        array_rows=args.array_rows,
        array_cols=args.array_cols,
        decode_tokens=args.decode_tokens,
        mode='kv_stationary_mqa_decode',
    ))

    comparison = compare_rows(baseline_row, kv_row)
    outputs = write_outputs(args.output_dir, [comparison])
    print(json.dumps({'outputs': outputs, 'comparison': comparison}, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
