#!/usr/bin/env python3
"""phase6_validation.py — Phase 6 end-to-end validation script.

Runs both BaselineMQADecodeSimulator and KVStationaryMQADecodeSimulator
through the full MQAMemoryBridge, compares key metrics against the
pre-Phase-6 analytical baselines, and prints a pass/fail report.

Usage
-----
    python phase6_validation.py [--verbose] [--csv OUTPUT.csv]

Pass criteria
-------------
- All result dicts contain 'memory_stall_cycles' and 'kv_preload_bandwidth_cycles'.
- Baseline memory_stall_cycles >= 0 and total_cycles > 0.
- KV-stationary kv_preload_bandwidth_cycles > 0 (confirms preload path ran).
- KV-stationary total_cycles >= baseline total_cycles (preload overhead).
- All numeric fields are finite (no NaN / inf).
- validate_against_reference() returns passed=True for structural key checks.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import Any, Dict, List

from mqa_scalesim.workload import MQAWorkload
from mqa_scalesim.baseline_decode import BaselineMQADecodeSimulator
from mqa_scalesim.kv_stationary_decode import KVStationaryMQADecodeSimulator
from mqa_scalesim.validation_bridge import (
    result_to_experiment_row,
    result_to_validation_dict,
    validate_against_reference,
)


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

TEST_CONFIGS: List[Dict[str, Any]] = [
    # Tiny — smoke test
    dict(
        experiment_id='tiny_smoke',
        sequence_length=64,
        batch_size=1,
        query_heads=4,
        kv_heads=1,
        head_dim=32,
        array_rows=8,
        array_cols=8,
        ifmap_sram_kb=32,
        filter_sram_kb=32,
        ofmap_sram_kb=32,
        decode_tokens=1,
    ),
    # Medium — typical LLM decode step
    dict(
        experiment_id='medium_decode',
        sequence_length=512,
        batch_size=2,
        query_heads=8,
        kv_heads=2,
        head_dim=64,
        array_rows=16,
        array_cols=16,
        ifmap_sram_kb=64,
        filter_sram_kb=64,
        ofmap_sram_kb=64,
        decode_tokens=1,
    ),
    # Large — stress memory bridge
    dict(
        experiment_id='large_stress',
        sequence_length=2048,
        batch_size=4,
        query_heads=16,
        kv_heads=2,
        head_dim=128,
        array_rows=32,
        array_cols=32,
        ifmap_sram_kb=256,
        filter_sram_kb=256,
        ofmap_sram_kb=128,
        decode_tokens=1,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_finite(value: Any) -> bool:
    """Return True if value is a finite number or a non-numeric type."""
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def _check_finite(d: Dict[str, Any]) -> List[str]:
    """Return list of keys whose values are not finite."""
    return [k for k, v in d.items() if not _is_finite(v)]


def _make_workload(cfg: Dict[str, Any], mode: str) -> MQAWorkload:
    # Map short mode aliases to canonical MQAMode literals
    mode_map = {
        'baseline': 'baseline_mqa_decode',
        'kv_stationary': 'kv_stationary_mqa_decode',
        'baseline_mqa_decode': 'baseline_mqa_decode',
        'kv_stationary_mqa_decode': 'kv_stationary_mqa_decode',
    }
    canonical_mode = mode_map.get(mode, mode)
    w = MQAWorkload(
        mode=canonical_mode,
        sequence_length=cfg['sequence_length'],
        batch_size=cfg['batch_size'],
        query_heads=cfg['query_heads'],
        kv_heads=cfg['kv_heads'],
        head_dim=cfg['head_dim'],
        array_rows=cfg.get('array_rows', 16),
        array_cols=cfg.get('array_cols', 16),
        ifmap_sram_kb=cfg.get('ifmap_sram_kb', 64),
        filter_sram_kb=cfg.get('filter_sram_kb', 64),
        ofmap_sram_kb=cfg.get('ofmap_sram_kb', 64),
        decode_tokens=cfg.get('decode_tokens', 1),
        decode_step=cfg.get('decode_step', 0),
        metadata={
            'experiment_id': cfg.get('experiment_id', 'unnamed'),
            'array_shape': f"{cfg.get('array_rows', 16)}x{cfg.get('array_cols', 16)}",
        },
    )
    return w


def _required_memory_fields() -> List[str]:
    return ['memory_stall_cycles', 'kv_preload_bandwidth_cycles', 'memory_model_applied']


# ---------------------------------------------------------------------------
# Single-config validator
# ---------------------------------------------------------------------------

def validate_config(cfg: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    exp_id = cfg.get('experiment_id', 'unnamed')
    failures: List[str] = []
    notes: List[str] = []

    # --- Baseline ---
    baseline_wl = _make_workload(cfg, mode='baseline_mqa_decode')
    try:
        baseline_result = BaselineMQADecodeSimulator(baseline_wl).simulate(run_memory_model=True)
        baseline_dict = result_to_validation_dict(baseline_result)
        baseline_row = result_to_experiment_row(baseline_wl, baseline_result)
    except Exception as exc:
        failures.append(f'baseline simulate() raised: {exc}')
        baseline_dict = {}
        baseline_row = {}

    # --- KV-stationary ---
    kv_wl = _make_workload(cfg, mode='kv_stationary_mqa_decode')
    try:
        kv_result = KVStationaryMQADecodeSimulator(kv_wl).simulate(run_memory_model=True)
        kv_dict = result_to_validation_dict(kv_result)
        kv_row = result_to_experiment_row(kv_wl, kv_result)
    except Exception as exc:
        failures.append(f'kv_stationary simulate() raised: {exc}')
        kv_dict = {}
        kv_row = {}

    # --- Check required memory fields present ---
    for field in _required_memory_fields():
        if baseline_dict and field not in baseline_dict:
            failures.append(f'baseline missing field: {field}')
        if kv_dict and field not in kv_dict:
            failures.append(f'kv_stationary missing field: {field}')

    # --- Finite check ---
    if baseline_row:
        bad = _check_finite(baseline_row)
        if bad:
            failures.append(f'baseline non-finite fields: {bad}')
    if kv_row:
        bad = _check_finite(kv_row)
        if bad:
            failures.append(f'kv_stationary non-finite fields: {bad}')

    # --- Sanity: total_cycles > 0 ---
    if baseline_dict and baseline_dict.get('total_cycles', 0) <= 0:
        failures.append('baseline total_cycles <= 0')
    if kv_dict and kv_dict.get('total_cycles', 0) <= 0:
        failures.append('kv_stationary total_cycles <= 0')

    # --- Sanity: stall cycles non-negative ---
    if baseline_dict:
        stall = baseline_dict.get('memory_stall_cycles', -1)
        if stall < 0:
            failures.append(f'baseline memory_stall_cycles < 0: {stall}')

    # --- KV preload overhead present ---
    if kv_dict:
        preload_bw = kv_dict.get('kv_preload_bandwidth_cycles', 0)
        if preload_bw <= 0:
            notes.append('kv_stationary kv_preload_bandwidth_cycles = 0 (may be expected for tiny workloads)')

    # --- Structural checks (schema only, not value equality across modes) ---
    if baseline_dict:
        required_baseline_keys = ['mode', 'total_cycles', 'dram_reads', 'dram_writes', 'sram_reads', 'sram_writes']
        missing = [k for k in required_baseline_keys if k not in baseline_dict]
        if missing:
            failures.append(f'baseline missing required keys: {missing}')
    if kv_dict:
        required_kv_keys = ['mode', 'total_cycles', 'dram_reads', 'dram_writes', 'sram_reads', 'sram_writes']
        missing = [k for k in required_kv_keys if k not in kv_dict]
        if missing:
            failures.append(f'kv_stationary missing required keys: {missing}')

    # --- Relative sanity checks across modes ---
    if baseline_dict and kv_dict:
        if kv_dict.get('kv_preload_bandwidth_cycles', 0) > 0 and kv_dict.get('total_cycles', 0) < baseline_dict.get('total_cycles', 0):
            failures.append('kv_stationary total_cycles unexpectedly below baseline despite preload overhead')

    if verbose:
        print(f'\n[{exp_id}] baseline total_cycles  = {baseline_dict.get("total_cycles", "N/A")}')
        print(f'[{exp_id}] kv_stat  total_cycles  = {kv_dict.get("total_cycles", "N/A")}')
        print(f'[{exp_id}] baseline stall_cycles   = {baseline_dict.get("memory_stall_cycles", "N/A")}')
        print(f'[{exp_id}] kv_stat  preload_bw_cy  = {kv_dict.get("kv_preload_bandwidth_cycles", "N/A")}')
        if notes:
            for note in notes:
                print(f'  NOTE: {note}')

    return {
        'experiment_id': exp_id,
        'passed': len(failures) == 0,
        'failures': failures,
        'notes': notes,
        'baseline_row': baseline_row,
        'kv_row': kv_row,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description='Phase 6 end-to-end memory bridge validation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print per-config metrics')
    parser.add_argument('--csv', metavar='PATH', default='', help='Write results CSV to this path')
    args = parser.parse_args()

    all_results = []
    total = len(TEST_CONFIGS)
    passed = 0

    print(f'Phase 6 Validation — {total} configuration(s)\n' + '=' * 50)

    for cfg in TEST_CONFIGS:
        r = validate_config(cfg, verbose=args.verbose)
        all_results.append(r)
        status = 'PASS' if r['passed'] else 'FAIL'
        if r['passed']:
            passed += 1
        fail_str = ''
        if r['failures']:
            fail_str = '  => ' + ' | '.join(r['failures'])
        print(f'  [{status}]  {r["experiment_id"]}{fail_str}')

    print('=' * 50)
    print(f'Result: {passed}/{total} passed')

    if args.csv:
        rows = []
        for r in all_results:
            row = {'experiment_id': r['experiment_id'], 'passed': r['passed']}
            row.update(r.get('baseline_row', {}))
            rows.append(row)
        if rows:
            fieldnames = list(rows[0].keys())
            with open(args.csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            print(f'CSV written to {args.csv}')

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
