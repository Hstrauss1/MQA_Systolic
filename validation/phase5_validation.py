#!/usr/bin/env python3
"""Phase 5 validation for the integrated sweep, compare, and plot workflow."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

VALIDATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VALIDATION_ROOT.parent
ARTIFACT_DIR = VALIDATION_ROOT / 'phase5_validation_artifacts'
PYTHON = sys.executable


class ResultCollector:
    def __init__(self):
        self.results = []

    def record(self, name, passed, details=None, error=None):
        payload = {'name': name, 'status': 'PASS' if passed else 'FAIL'}
        if details is not None:
            payload['details'] = details
        if error is not None:
            payload['error'] = error
        self.results.append(payload)

    def report(self):
        total = len(self.results)
        passed = sum(1 for row in self.results if row['status'] == 'PASS')
        failed = total - passed
        return {
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'artifact_dir': str(ARTIFACT_DIR),
            },
            'results': self.results,
        }


collector = ResultCollector()


def reset_artifacts():
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def run_check(name, fn):
    try:
        details = fn()
        collector.record(name, True, details=details)
    except Exception as exc:
        collector.record(name, False, error={
            'message': str(exc),
            'type': exc.__class__.__name__,
            'traceback': traceback.format_exc(),
        })


def run_command(args):
    completed = subprocess.run(args, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return {
        'args': args,
        'stdout': completed.stdout,
        'stderr': completed.stderr,
    }


def check_sweep_outputs():
    output_dir = ARTIFACT_DIR / 'sweep'
    run_command([
        PYTHON,
        'legacy/phase5_sweep.py',
        '--sequence-lengths', '128,256',
        '--decode-tokens', '1,2',
        '--array-sizes', '16x16',
        '--output-dir', str(output_dir),
    ])
    csv_path = output_dir / 'phase5_sweep_results.csv'
    json_path = output_dir / 'phase5_sweep_results.json'
    summary_path = output_dir / 'phase5_sweep_summary.json'
    if not csv_path.exists() or not json_path.exists() or not summary_path.exists():
        raise AssertionError('Expected sweep CSV, JSON, and summary outputs')

    with csv_path.open('r', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError('Expected non-empty sweep CSV')

    experiment_modes = {}
    for row in rows:
        experiment_modes.setdefault(row['experiment_id'], set()).add(row['mode'])
    if not all(modes == {'baseline_mqa_decode', 'kv_stationary_mqa_decode'} for modes in experiment_modes.values()):
        raise AssertionError('Each sweep experiment_id must contain both baseline and KV rows')

    return {
        'csv_rows': len(rows),
        'experiment_count': len(experiment_modes),
        'csv_path': str(csv_path),
        'json_path': str(json_path),
    }


def check_required_columns():
    csv_path = ARTIFACT_DIR / 'sweep' / 'phase5_sweep_results.csv'
    with csv_path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
    required = {
        'experiment_id', 'mode', 'sequence_length', 'batch_size', 'query_heads', 'kv_heads', 'head_dim',
        'array_shape', 'decode_tokens', 'total_cycles', 'dram_reads', 'dram_writes', 'sram_reads',
        'sram_writes', 'weighted_pe_utilization', 'kv_preload_bytes', 'amortized_preload_bytes_per_token',
    }
    missing = sorted(required - set(fieldnames))
    if missing:
        raise AssertionError(f'Missing required sweep columns: {missing}')
    return {'field_count': len(fieldnames), 'required_columns_checked': sorted(required)}


def check_compare_outputs():
    output_dir = ARTIFACT_DIR / 'compare'
    run_command([
        PYTHON,
        'compare.py',
        '--sequence-length', '256',
        '--decode-tokens', '2',
        '--output-dir', str(output_dir),
    ])
    csv_path = output_dir / 'phase5_compare_results.csv'
    json_path = output_dir / 'phase5_compare_results.json'
    if not csv_path.exists() or not json_path.exists():
        raise AssertionError('Expected compare CSV and JSON outputs')

    with csv_path.open('r', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise AssertionError('Expected exactly one comparison row')
    row = rows[0]
    required = {
        'baseline_total_cycles', 'kv_total_cycles', 'cycle_speedup_baseline_over_kv',
        'dram_read_ratio_baseline_over_kv', 'sram_traffic_ratio_baseline_over_kv',
        'weighted_pe_util_delta_kv_minus_baseline',
    }
    missing = sorted(required - set(row.keys()))
    if missing:
        raise AssertionError(f'Missing comparison columns: {missing}')
    return {'csv_path': str(csv_path), 'json_path': str(json_path), 'columns_checked': sorted(required)}


def check_plot_generation():
    sweep_csv = ARTIFACT_DIR / 'sweep' / 'phase5_sweep_results.csv'
    output_dir = ARTIFACT_DIR / 'plots'
    run_command([
        PYTHON,
        'plot_results.py',
        '--input', str(sweep_csv),
        '--output-dir', str(output_dir),
    ])
    expected = [
        'total_cycles_vs_sequence_length.png',
        'amortized_preload_vs_decode_tokens.png',
        'dram_reads_vs_sequence_length.png',
        'weighted_pe_util_vs_array_shape.png',
        'baseline_vs_kv_speedup_vs_sequence_length.png',
    ]
    missing = [name for name in expected if not (output_dir / name).exists()]
    if missing:
        raise AssertionError(f'Missing plot files: {missing}')
    return {'plot_dir': str(output_dir), 'plots_checked': expected}


def check_topology_wrapper():
    output_path = ARTIFACT_DIR / 'topology' / 'topology_compat.json'
    run_command([
        PYTHON,
        'generate_scalesim_topology.py',
        '--sequence-length', '256',
        '--decode-tokens', '2',
        '--output', str(output_path),
    ])
    if not output_path.exists():
        raise AssertionError('Expected compatibility topology output to exist')
    payload = json.loads(output_path.read_text(encoding='utf-8'))
    if 'baseline_topology' not in payload or 'workload' not in payload:
        raise AssertionError('Topology compatibility payload missing required sections')
    return {'output_path': str(output_path), 'keys': sorted(payload.keys())}


def check_schema_stability():
    sweep_path = ARTIFACT_DIR / 'sweep' / 'phase5_sweep_results.csv'
    with sweep_path.open('r', encoding='utf-8') as handle:
        first_fields = csv.DictReader(handle).fieldnames or []
    rerun_dir = ARTIFACT_DIR / 'sweep_rerun'
    run_command([
        PYTHON,
        'legacy/phase5_sweep.py',
        '--sequence-lengths', '128,256',
        '--decode-tokens', '1,2',
        '--array-sizes', '16x16',
        '--output-dir', str(rerun_dir),
    ])
    with (rerun_dir / 'phase5_sweep_results.csv').open('r', encoding='utf-8') as handle:
        second_fields = csv.DictReader(handle).fieldnames or []
    if first_fields != second_fields:
        raise AssertionError('Sweep CSV schema changed between repeated runs')
    return {'field_count': len(first_fields)}


def write_report(report):
    report_path = ARTIFACT_DIR / 'phase5_validation_report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return report_path


def main() -> int:
    reset_artifacts()
    run_check('sweep_outputs', check_sweep_outputs)
    run_check('required_columns', check_required_columns)
    run_check('compare_outputs', check_compare_outputs)
    run_check('plot_generation', check_plot_generation)
    run_check('topology_wrapper', check_topology_wrapper)
    run_check('schema_stability', check_schema_stability)

    report = collector.report()
    report_path = write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f'\nValidation report written to: {report_path}')
    return 1 if report['summary']['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
