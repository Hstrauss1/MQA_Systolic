#!/usr/bin/env python3
"""
Phase 3 validation for analytical MQA execution modeling.

This script assumes Phase 2 control-plane routing is already in place and validates:
1. Baseline MQA returns a rich structured result.
2. Baseline score/value stages carry non-zero MAC counts.
3. Larger sequence length increases baseline total cycles.
4. Larger array size reduces or preserves baseline total cycles.
5. Simulator route dispatch still works for baseline_mqa_decode with the real baseline backend.
6. KV-stationary route still returns a structurally valid rich result.
7. Validation bridge produces a normalized dictionary with stage summaries.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

VALIDATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VALIDATION_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mqa_scalesim.baseline_decode import BaselineMQADecodeSimulator
from mqa_scalesim.kv_stationary_decode import KVStationaryMQADecodeSimulator
from mqa_scalesim.validation_bridge import result_to_validation_dict
from mqa_scalesim.workload import MQAWorkload
from scalesim.scale_config import scale_config
from scalesim.simulator import simulator
from scalesim.topology_utils import topologies


ARTIFACT_DIR = VALIDATION_ROOT / "phase3_validation_artifacts"


class ResultCollector:
    def __init__(self):
        self.results = []

    def record(self, name, passed, details=None, error=None):
        payload = {"name": name, "status": "PASS" if passed else "FAIL"}
        if details is not None:
            payload["details"] = details
        if error is not None:
            payload["error"] = error
        self.results.append(payload)

    def report(self):
        total = len(self.results)
        passed = sum(1 for item in self.results if item["status"] == "PASS")
        failed = total - passed
        return {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "artifact_dir": str(ARTIFACT_DIR),
            },
            "results": self.results,
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
            "message": str(exc),
            "type": exc.__class__.__name__,
            "traceback": traceback.format_exc(),
        })


def build_workload(mode: str,
                   sequence_length: int = 128,
                   batch_size: int = 1,
                   query_heads: int = 8,
                   kv_heads: int = 2,
                   head_dim: int = 64,
                   array_rows: int = 16,
                   array_cols: int = 16,
                   decode_tokens: int = 1,
                   decode_step: int = 4) -> MQAWorkload:
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
        decode_step=decode_step,
        softmax_variant='online',
        exp_variant='lookup',
    )


def check_baseline_rich_result():
    workload = build_workload('baseline_mqa_decode')
    result = BaselineMQADecodeSimulator(workload).simulate()
    stage_names = [stage.name for stage in result.stages]
    expected = ['score_gemm', 'softmax_reduce', 'value_gemm', 'writeback']
    if stage_names != expected:
        raise AssertionError(f'Unexpected stage names: {stage_names}')
    if result.total_cycles <= 0 or result.total_macs <= 0:
        raise AssertionError('Expected positive total cycles and total MACs')
    return {
        'total_cycles': result.total_cycles,
        'total_macs': result.total_macs,
        'stage_names': stage_names,
    }


def check_baseline_stage_macs():
    workload = build_workload('baseline_mqa_decode', sequence_length=96, head_dim=48)
    result = BaselineMQADecodeSimulator(workload).simulate()
    stage_map = {stage.name: stage for stage in result.stages}
    if stage_map['score_gemm'].macs <= 0:
        raise AssertionError('score_gemm MACs must be positive')
    if stage_map['value_gemm'].macs <= 0:
        raise AssertionError('value_gemm MACs must be positive')
    return {
        'score_macs': stage_map['score_gemm'].macs,
        'value_macs': stage_map['value_gemm'].macs,
    }


def check_sequence_scaling():
    small = BaselineMQADecodeSimulator(build_workload('baseline_mqa_decode', sequence_length=64)).simulate()
    large = BaselineMQADecodeSimulator(build_workload('baseline_mqa_decode', sequence_length=256)).simulate()
    if large.total_cycles <= small.total_cycles:
        raise AssertionError('Expected larger sequence length to increase total cycles')
    return {
        'small_total_cycles': small.total_cycles,
        'large_total_cycles': large.total_cycles,
    }


def check_array_scaling():
    small_array = BaselineMQADecodeSimulator(build_workload('baseline_mqa_decode', array_rows=8, array_cols=8)).simulate()
    large_array = BaselineMQADecodeSimulator(build_workload('baseline_mqa_decode', array_rows=32, array_cols=32)).simulate()
    if large_array.total_cycles > small_array.total_cycles:
        raise AssertionError('Expected larger array to reduce or preserve total cycles')
    return {
        'small_array_cycles': small_array.total_cycles,
        'large_array_cycles': large_array.total_cycles,
    }


def build_phase3_cfg(mode: str) -> scale_config:
    conf = scale_config()
    values = scale_config.get_default_conf_as_list()
    values[0] = f'phase3_{mode}'
    values[14] = mode
    values[15] = '128'
    values[16] = '1'
    values[17] = '8'
    values[18] = '2'
    values[19] = '64'
    values[20] = 'int8'
    values[21] = '1'
    values[22] = '4'
    values[23] = 'online'
    values[24] = 'lookup'
    conf.update_from_list(values)
    return conf


def check_simulator_baseline_route():
    conf = build_phase3_cfg('baseline_mqa_decode')
    topo = topologies()
    sim = simulator()
    sim.set_params(config_obj=conf, topo_obj=topo, top_path=str(ARTIFACT_DIR / 'reports'), verbosity=False, save_trace=False)
    sim.run()
    result = sim.mqa_result
    if result is None or result.total_cycles <= 0:
        raise AssertionError('Expected simulator baseline route to produce a real result')
    return {
        'total_cycles': result.total_cycles,
        'stage_count': len(result.stages),
        'mode': result.mode,
    }


def check_kv_structural_validity():
    workload = build_workload('kv_stationary_mqa_decode')
    result = KVStationaryMQADecodeSimulator(workload).simulate()
    if result.total_cycles <= 0:
        raise AssertionError('Expected KV-stationary total_cycles > 0')
    if len(result.stages) != 5:
        raise AssertionError(f'Expected 5 KV stages, got {len(result.stages)}')
    return {
        'total_cycles': result.total_cycles,
        'stage_names': [stage.name for stage in result.stages],
        'kv_preload_bytes': result.kv_preload_bytes,
    }


def check_validation_bridge():
    workload = build_workload('baseline_mqa_decode', sequence_length=80, head_dim=40)
    result = BaselineMQADecodeSimulator(workload).simulate()
    payload = result_to_validation_dict(result)
    required_keys = {'mode', 'total_cycles', 'total_macs', 'stage_names', 'stage_cycles', 'stage_macs'}
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise AssertionError(f'Missing validation payload keys: {missing}')
    return {
        'stage_names': payload['stage_names'],
        'stage_cycles': payload['stage_cycles'],
    }


def write_report(report):
    report_path = ARTIFACT_DIR / 'phase3_validation_report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return report_path


def main():
    reset_artifacts()
    run_check('baseline_rich_result', check_baseline_rich_result)
    run_check('baseline_stage_macs', check_baseline_stage_macs)
    run_check('sequence_scaling', check_sequence_scaling)
    run_check('array_scaling', check_array_scaling)
    run_check('simulator_baseline_route', check_simulator_baseline_route)
    run_check('kv_structural_validity', check_kv_structural_validity)
    run_check('validation_bridge', check_validation_bridge)

    report = collector.report()
    report_path = write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f'\nValidation report written to: {report_path}')
    return 1 if report['summary']['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
