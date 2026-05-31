#!/usr/bin/env python3
"""Phase 7 canonical sweep runner for integrated MQA SCALE-Sim backends."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from mqa_scalesim.baseline_decode import BaselineMQADecodeSimulator
from mqa_scalesim.kv_stationary_decode import KVStationaryMQADecodeSimulator
from mqa_scalesim.validation_bridge import result_to_experiment_row
from mqa_scalesim.workload import MQAWorkload

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_OUTPUT_ROOT = Path('outputs')
DEFAULT_CSV = 'sweep_results.csv'
DEFAULT_JSON = 'phase7_sweep_results.json'
DEFAULT_SUMMARY = 'phase7_sweep_summary.json'
DEFAULT_SEQUENCE_LENGTHS = [64, 128, 512, 1024]
DEFAULT_DECODE_TOKENS = [1, 4, 8]
DEFAULT_ARRAY_SIZES = [(8, 8), (16, 16), (32, 32)]
DEFAULT_QUERY_HEADS = 8
DEFAULT_KV_HEADS = 2
DEFAULT_HEAD_DIM = 64
DEFAULT_BATCH_SIZE = 1
DEFAULT_PRECISION = 'int8'
STRESS_SEQUENCE_LENGTHS = [2048]
STRESS_ARRAY_SIZES = [(32, 32)]


def parse_int_list(raw: str) -> List[int]:
    return [int(token.strip()) for token in raw.split(',') if token.strip()]


def parse_array_sizes(raw: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for token in raw.split(','):
        token = token.strip().lower()
        if not token:
            continue
        if 'x' not in token:
            raise ValueError(f'Invalid array size token: {token}')
        rows, cols = token.split('x', 1)
        pairs.append((int(rows), int(cols)))
    return pairs


def make_run_directory(output_root: Path) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f'{timestamp}_{suffix:02d}'
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_workloads(sequence_lengths: Sequence[int],
                    decode_tokens_list: Sequence[int],
                    array_sizes: Sequence[Tuple[int, int]],
                    batch_size: int,
                    query_heads: int,
                    kv_heads: int,
                    head_dim: int,
                    precision: str,
                    include_stress: bool) -> List[MQAWorkload]:
    workloads: List[MQAWorkload] = []
    experiment_id = 0
    effective_sequence_lengths = list(sequence_lengths)
    effective_array_sizes = list(array_sizes)
    if include_stress:
        for seq in STRESS_SEQUENCE_LENGTHS:
            if seq not in effective_sequence_lengths:
                effective_sequence_lengths.append(seq)
        for shape in STRESS_ARRAY_SIZES:
            if shape not in effective_array_sizes:
                effective_array_sizes.append(shape)

    for sequence_length in sorted(effective_sequence_lengths):
        for decode_tokens in decode_tokens_list:
            for array_rows, array_cols in sorted(effective_array_sizes):
                experiment_id += 1
                sram_kb = 32 if array_rows <= 8 else 64 if array_rows <= 16 else 128
                for mode in ('baseline_mqa_decode', 'kv_stationary_mqa_decode'):
                    workloads.append(MQAWorkload(
                        mode=mode,
                        sequence_length=sequence_length,
                        batch_size=batch_size,
                        query_heads=query_heads,
                        kv_heads=kv_heads,
                        head_dim=head_dim,
                        precision=precision,
                        array_rows=array_rows,
                        array_cols=array_cols,
                        ifmap_sram_kb=sram_kb,
                        filter_sram_kb=sram_kb,
                        ofmap_sram_kb=sram_kb,
                        bandwidth_mode='calc',
                        dram_bandwidth=None,
                        decode_tokens=decode_tokens,
                        decode_step=4,
                        softmax_variant='online',
                        exp_variant='lookup',
                        reuse_kv_across_tokens=True,
                        metadata={
                            'experiment_id': experiment_id,
                            'array_shape': f'{array_rows}x{array_cols}',
                            'phase': 'phase7',
                            'preset': 'stress' if sequence_length in STRESS_SEQUENCE_LENGTHS else 'default',
                        },
                    ))
    return workloads


def run_workload(workload: MQAWorkload) -> Dict[str, object]:
    if workload.mode == 'baseline_mqa_decode':
        result = BaselineMQADecodeSimulator(workload).simulate(run_memory_model=True)
    elif workload.mode == 'kv_stationary_mqa_decode':
        result = KVStationaryMQADecodeSimulator(workload).simulate(run_memory_model=True)
    else:
        raise ValueError(f'Unsupported mode: {workload.mode}')
    return result_to_experiment_row(workload, result)


def maybe_progress(iterable, enabled: bool, total: int, desc: str):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError('No rows to write')
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    experiment_ids = sorted({row['experiment_id'] for row in rows})
    modes = sorted({row['mode'] for row in rows})
    return {
        'row_count': len(rows),
        'experiment_count': len(experiment_ids),
        'modes': modes,
        'sequence_lengths': sorted({row['sequence_length'] for row in rows}),
        'decode_tokens': sorted({row['decode_tokens'] for row in rows}),
        'array_shapes': sorted({row['array_shape'] for row in rows}),
        'memory_model_applied_values': sorted({row.get('memory_model_applied') for row in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Run canonical Phase 7 MQA sweeps.')
    parser.add_argument('--sequence-lengths', default=','.join(str(v) for v in DEFAULT_SEQUENCE_LENGTHS))
    parser.add_argument('--decode-tokens', default=','.join(str(v) for v in DEFAULT_DECODE_TOKENS))
    parser.add_argument('--array-sizes', default=','.join(f'{r}x{c}' for r, c in DEFAULT_ARRAY_SIZES))
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--query-heads', type=int, default=DEFAULT_QUERY_HEADS)
    parser.add_argument('--kv-heads', type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument('--head-dim', type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument('--precision', default=DEFAULT_PRECISION)
    parser.add_argument('--include-stress', action='store_true')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    sequence_lengths = parse_int_list(args.sequence_lengths)
    decode_tokens_list = parse_int_list(args.decode_tokens)
    array_sizes = parse_array_sizes(args.array_sizes)
    workloads = build_workloads(
        sequence_lengths=sequence_lengths,
        decode_tokens_list=decode_tokens_list,
        array_sizes=array_sizes,
        batch_size=args.batch_size,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        precision=args.precision,
        include_stress=args.include_stress,
    )

    run_dir = make_run_directory(args.output_root)
    log(f'Created output directory {run_dir}', args.verbose)
    log(f'Phase 7 sweep starting with {len(workloads)} workload runs', args.verbose)

    rows: List[Dict[str, object]] = []
    iterator = maybe_progress(workloads, enabled=args.progress, total=len(workloads), desc='Phase 7 sweep')
    for workload in iterator:
        exp_id = workload.metadata.get('experiment_id', 'unknown')
        label = f"exp={exp_id} mode={workload.mode} seq={workload.sequence_length} tok={workload.decode_tokens} array={workload.array_rows}x{workload.array_cols}"
        log(f'[RUN] {label}', args.verbose)
        row = run_workload(workload)
        rows.append(row)
        log(
            f"[DONE] {label} total_cycles={row.get('total_cycles')} dram_reads={row.get('dram_reads')} dram_writes={row.get('dram_writes')}",
            args.verbose,
        )

    csv_path = run_dir / DEFAULT_CSV
    json_path = run_dir / DEFAULT_JSON
    summary_path = run_dir / DEFAULT_SUMMARY
    summary = summarize(rows)
    summary['run_directory'] = str(run_dir)

    log(f'Writing CSV to {csv_path}', args.verbose)
    write_csv(csv_path, rows)
    log(f'Writing JSON to {json_path}', args.verbose)
    write_json(json_path, rows)
    log(f'Writing summary to {summary_path}', args.verbose)
    write_json(summary_path, summary)

    print(f'Phase 7 sweep complete: {len(rows)} rows')
    print(f'RUN_DIR: {run_dir}')
    print(f'CSV: {csv_path}')
    print(f'JSON: {json_path}')
    print(f'SUMMARY: {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
