#!/usr/bin/env python3
"""Compatibility helper that emits baseline GEMM topology metadata from the shared MQA workload spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mqa_scalesim.workload import MQAWorkload


def build_workload(args: argparse.Namespace) -> MQAWorkload:
    return MQAWorkload(
        mode='baseline_mqa_decode',
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        precision=args.precision,
        array_rows=args.array_rows,
        array_cols=args.array_cols,
        ifmap_sram_kb=args.ifmap_sram_kb,
        filter_sram_kb=args.filter_sram_kb,
        ofmap_sram_kb=args.ofmap_sram_kb,
        bandwidth_mode='calc',
        dram_bandwidth=None,
        decode_tokens=args.decode_tokens,
        decode_step=args.decode_step,
        softmax_variant='online',
        exp_variant='lookup',
        reuse_kv_across_tokens=True,
        metadata={'source': 'generate_scalesim_topology.py'},
    )


def build_topology_payload(workload: MQAWorkload) -> dict:
    score_rows = workload.batch_size * workload.query_heads * workload.decode_tokens
    score_cols = workload.sequence_length
    score_reduction = workload.head_dim

    value_rows = score_rows
    value_cols = workload.head_dim
    value_reduction = workload.sequence_length

    return {
        'workload': {
            'mode': workload.mode,
            'sequence_length': workload.sequence_length,
            'batch_size': workload.batch_size,
            'query_heads': workload.query_heads,
            'kv_heads': workload.kv_heads,
            'head_dim': workload.head_dim,
            'array_rows': workload.array_rows,
            'array_cols': workload.array_cols,
            'decode_tokens': workload.decode_tokens,
            'decode_step': workload.decode_step,
        },
        'baseline_topology': {
            'score_gemm': {
                'rows': score_rows,
                'cols': score_cols,
                'reduction_dim': score_reduction,
            },
            'value_gemm': {
                'rows': value_rows,
                'cols': value_cols,
                'reduction_dim': value_reduction,
            },
        },
        'note': 'Compatibility helper: topology dimensions are now derived from the shared MQAWorkload spec used by the integrated Phase 5 flow.',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate compatibility topology metadata from shared MQA workload inputs.')
    parser.add_argument('--sequence-length', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--query-heads', type=int, default=8)
    parser.add_argument('--kv-heads', type=int, default=2)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--array-rows', type=int, default=16)
    parser.add_argument('--array-cols', type=int, default=16)
    parser.add_argument('--decode-tokens', type=int, default=1)
    parser.add_argument('--decode-step', type=int, default=4)
    parser.add_argument('--precision', default='int8')
    parser.add_argument('--ifmap-sram-kb', type=int, default=64)
    parser.add_argument('--filter-sram-kb', type=int, default=64)
    parser.add_argument('--ofmap-sram-kb', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('phase5_outputs/topology_compat.json'))
    args = parser.parse_args()

    workload = build_workload(args)
    payload = build_topology_payload(workload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'payload': payload}, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
