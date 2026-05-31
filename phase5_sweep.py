#!/usr/bin/env python3
"""Phase 5 experiment sweep runner for integrated MQA SCALE-Sim backends."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from mqa_scalesim.baseline_decode import BaselineMQADecodeSimulator
from mqa_scalesim.kv_stationary_decode import KVStationaryMQADecodeSimulator
from mqa_scalesim.validation_bridge import result_to_experiment_row
from mqa_scalesim.workload import MQAWorkload


DEFAULT_SEQUENCE_LENGTHS = [128, 512, 1024, 2048]
DEFAULT_DECODE_TOKENS = [1, 4, 8]
DEFAULT_ARRAY_SIZES = [(16, 16), (32, 32)]
DEFAULT_QUERY_HEADS = 8
DEFAULT_KV_HEADS = 2
DEFAULT_HEAD_DIM = 64
DEFAULT_BATCH_SIZE = 1
DEFAULT_PRECISION = 'int8'
DEFAULT_OUTPUT_DIR = Path('phase5_outputs')
DEFAULT_CSV = 'phase5_sweep_results.csv'
DEFAULT_JSON = 'phase5_sweep_results.json'


def parse_int_list(raw: str) -> List[int]:
    return [int(token.strip()) for token in raw.split(',') if token.strip()]


def parse_array_sizes(raw: str) -> List[tuple[int, int]]:
    pairs: List[tuple[int, int]] = []
    for token in raw.split(','):
        token = token.strip().lower()
        if not token:
            continue
        if 'x' not in token:
            raise ValueError(f'Invalid array size token: {token}')
        rows, cols = token.split('x', 1)
        pairs.append((int(rows), int(cols)))
    return pairs


def build_workloads(sequence_lengths: Sequence[int],
                    decode_tokens_list: Sequence[int],
                    array_sizes: Sequence[tuple[int, int]],
                    batch_size: int,
                    query_heads: int,
                    kv_heads: int,
                    head_dim: int,
                    precision: str) -> Iterable[MQAWorkload]:
    experiment_id = 0
    for sequence_length in sequence_lengths:
        for decode_tokens in decode_tokens_list:
            for array_rows, array_cols in array_sizes:
                experiment_id += 1
                for mode in ('baseline_mqa_decode', 'kv_stationary_mqa_decode'):
                    yield MQAWorkload(
                        mode=mode,
                        sequence_length=sequence_length,
                        batch_size=batch_size,
                        query_heads=query_heads,
                        kv_heads=kv_heads,
                        head_dim=head_dim,
                        precision=precision,
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
                        metadata={
                            'experiment_id': experiment_id,
                            'array_shape': f'{array_rows}x{array_cols}',
                        },
                    )


def run_workload(workload: MQAWorkload) -> Dict[str, object]:
    if workload.mode == 'baseline_mqa_decode':
        result = BaselineMQADecodeSimulator(workload).simulate()
    elif workload.mode == 'kv_stationary_mqa_decode':
        result = KVStationaryMQADecodeSimulator(workload).simulate()
    else:
        raise ValueError(f'Unsupported mode: {workload.mode}')
    return result_to_experiment_row(workload, result)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError('No rows to write')
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Run integrated Phase 5 MQA sweeps.')
    parser.add_argument('--sequence-lengths', default=','.join(str(v) for v in DEFAULT_SEQUENCE_LENGTHS))
    parser.add_argument('--decode-tokens', default=','.join(str(v) for v in DEFAULT_DECODE_TOKENS))
    parser.add_argument('--array-sizes', default=','.join(f'{r}x{c}' for r, c in DEFAULT_ARRAY_SIZES))
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--query-heads', type=int, default=DEFAULT_QUERY_HEADS)
    parser.add_argument('--kv-heads', type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument('--head-dim', type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument('--precision', default=DEFAULT_PRECISION)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        run_workload(workload)
        for workload in build_workloads(
            sequence_lengths=parse_int_list(args.sequence_lengths),
            decode_tokens_list=parse_int_list(args.decode_tokens),
            array_sizes=parse_array_sizes(args.array_sizes),
            batch_size=args.batch_size,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            precision=args.precision,
        )
    ]

    csv_path = args.output_dir / DEFAULT_CSV
    json_path = args.output_dir / DEFAULT_JSON
    summary_path = args.output_dir / 'phase5_sweep_summary.json'

    write_csv(csv_path, rows)
    write_json(json_path, rows)
    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')

    print(json.dumps({
        'csv': str(csv_path),
        'json': str(json_path),
        'summary': str(summary_path),
        'details': summary,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
