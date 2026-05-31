#!/usr/bin/env python3
"""
Phase 4 validation for KV-stationary streaming MQA execution modeling.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mqa_scalesim.baseline_decode import BaselineMQADecodeSimulator
from mqa_scalesim.kv_stationary_decode import KVStationaryMQADecodeSimulator
from mqa_scalesim.validation_bridge import result_to_validation_dict
from mqa_scalesim.workload import MQAWorkload
from scalesim.scale_config import scale_config
from scalesim.simulator import simulator
from scalesim.topology_utils import topologies


ARTIFACT_DIR = REPO_ROOT / "phase4_validation_artifacts"


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
                   decode_step: int = 4,
                   reuse_kv_across_tokens: bool = True) -> MQAWorkload:
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
        reuse_kv_across_tokens=reuse_kv_across_tokens,
    )


def check_baseline_still_structural():
    result = BaselineMQADecodeSimulator(build_workload('baseline_mqa_decode')).simulate()
    if len(result.stages) != 4:
        raise AssertionError(f'Expected 4 baseline stages, got {len(result.stages)}')
    return {
        'stage_names': [stage.name for stage in result.stages],
        'total_cycles': result.total_cycles,
    }


def check_kv_stage_names():
    result = KVStationaryMQADecodeSimulator(build_workload('kv_stationary_mqa_decode')).simulate()
    expected = ['kv_preload', 'query_stream', 'online_softmax_accum', 'final_normalize', 'writeback']
    actual = [stage.name for stage in result.stages]
    if actual != expected:
        raise AssertionError(f'Unexpected KV stage names: {actual}')
    return {'stage_names': actual}


def check_sequence_scaling():
    small = KVStationaryMQADecodeSimulator(build_workload('kv_stationary_mqa_decode', sequence_length=64)).simulate()
    large = KVStationaryMQADecodeSimulator(build_workload('kv_stationary_mqa_decode', sequence_length=256)).simulate()
    if large.total_cycles <= small.total_cycles:
        raise AssertionError('Expected larger sequence length to increase KV total cycles')
    if large.kv_preload_bytes <= small.kv_preload_bytes:
        raise AssertionError('Expected larger sequence length to increase KV preload bytes')
    return {
        'small_total_cycles': small.total_cycles,
        'large_total_cycles': large.total_cycles,
        'small_preload': small.kv_preload_bytes,
        'large_preload': large.kv_preload_bytes,
    }


def check_array_scaling():
    small_array = KVStationaryMQADecodeSimulator(build_workload('kv_stationary_mqa_decode', array_rows=8, array_cols=8)).simulate()
    large_array = KVStationaryMQADecodeSimulator(build_workload('kv_stationary_mqa_decode', array_rows=32, array_cols=32)).simulate()
    if large_array.total_cycles > small_array.total_cycles:
        raise AssertionError('Expected larger array to reduce or preserve KV total cycles')
    return {
        'small_array_cycles': small_array.total_cycles,
        'large_array_cycles': large_array.total_cycles,
    }


def check_decode_token_amortization():
    one_token = KVStationaryMQADecodeSimulator(build_workload('kv_stationary_mqa_decode', decode_tokens=1)).simulate()
    four_tokens = KVStationaryMQADecodeSimulator(build_workload('kv_stationary_mqa_decode', decode_tokens=4)).simulate()
    one_amortized = one_token.metadata['amortized_preload_bytes_per_token']
    four_amortized = four_tokens.metadata['amortized_preload_bytes_per_token']
    if four_tokens.total_cycles <= one_token.total_cycles:
        raise AssertionError('Expected more decode tokens to increase total cycles')
    if four_amortized >= one_amortized:
        raise AssertionError('Expected amortized preload bytes per token to decrease with more tokens')
    return {
        'one_token_total_cycles': one_token.total_cycles,
        'four_token_total_cycles': four_tokens.total_cycles,
        'one_amortized_preload': one_amortized,
        'four_amortized_preload': four_amortized,
    }


def check_kv_heads_impact():
    kv1 = KVStationaryMQADecodeSimulator(build_workload('kv_stationary_mqa_decode', kv_heads=1, query_heads=8)).simulate()
    kv4 = KVStationaryMQADecodeSimulator(build_workload('kv_stationary_mqa_decode', kv_heads=4, query_heads=8)).simulate()
    if kv4.metadata['kv_resident_bytes'] <= kv1.metadata['kv_resident_bytes']:
        raise AssertionError('Expected more kv_heads to increase resident KV bytes')
    return {
        'kv1_resident_bytes': kv1.metadata['kv_resident_bytes'],
        'kv4_resident_bytes': kv4.metadata['kv_resident_bytes'],
    }


def check_validation_bridge_stream_metrics():
    result = KVStationaryMQADecodeSimulator(build_workload('kv_stationary_mqa_decode', decode_tokens=2)).simulate()
    payload = result_to_validation_dict(result)
    stream_metrics = payload.get('stream_metrics', {})
    required = {'kv_preload_bytes', 'weighted_pe_utilization', 'pipeline_depth', 'stream_groups', 'amortized_preload_bytes_per_token'}
    missing = sorted(required - set(stream_metrics.keys()))
    if missing:
        raise AssertionError(f'Missing stream metrics: {missing}')
    return stream_metrics


def build_phase4_cfg(mode: str) -> scale_config:
    conf = scale_config()
    values = scale_config.get_default_conf_as_list()
    values[0] = f'phase4_{mode}'
    values[14] = mode
    values[15] = '128'
    values[16] = '1'
    values[17] = '8'
    values[18] = '2'
    values[19] = '64'
    values[20] = 'int8'
    values[21] = '2'
    values[22] = '4'
    values[23] = 'online'
    values[24] = 'lookup'
    conf.update_from_list(values)
    return conf


def check_simulator_kv_route():
    conf = build_phase4_cfg('kv_stationary_mqa_decode')
    topo = topologies()
    sim = simulator()
    sim.set_params(config_obj=conf, topo_obj=topo, top_path=str(ARTIFACT_DIR / 'reports'), verbosity=False, save_trace=False)
    sim.run()
    result = sim.mqa_result
    if result is None or result.total_cycles <= 0:
        raise AssertionError('Expected simulator KV route to produce a real result')
    return {
        'mode': result.mode,
        'stage_count': len(result.stages),
        'total_cycles': result.total_cycles,
    }


def check_directional_vs_baseline():
    decode_tokens = 4
    sequence_length = 128
    head_dim = 64
    kv_heads = 2

    baseline = BaselineMQADecodeSimulator(
        build_workload('baseline_mqa_decode', decode_tokens=decode_tokens, sequence_length=sequence_length, head_dim=head_dim, kv_heads=kv_heads)
    ).simulate()
    kv = KVStationaryMQADecodeSimulator(
        build_workload('kv_stationary_mqa_decode', decode_tokens=decode_tokens, sequence_length=sequence_length, head_dim=head_dim, kv_heads=kv_heads)
    ).simulate()

    if kv.kv_preload_bytes <= 0:
        raise AssertionError('Expected positive KV preload bytes')

    baseline_kv_bytes_per_token = sequence_length * head_dim * kv_heads * 2
    baseline_total_kv_bytes = baseline_kv_bytes_per_token * decode_tokens
    kv_amortized_preload = kv.metadata['amortized_preload_bytes_per_token']

    if kv_amortized_preload >= baseline_kv_bytes_per_token:
        raise AssertionError('Expected KV-stationary amortized KV preload per token to be lower than baseline repeated KV traffic')

    if kv.kv_preload_bytes >= baseline_total_kv_bytes:
        raise AssertionError('Expected one-time KV preload traffic to be lower than baseline repeated KV traffic across decode tokens')

    return {
        'baseline_dram_reads': baseline.dram_reads,
        'kv_dram_reads': kv.dram_reads,
        'baseline_kv_bytes_per_token': baseline_kv_bytes_per_token,
        'baseline_total_kv_bytes': baseline_total_kv_bytes,
        'kv_preload_bytes': kv.kv_preload_bytes,
        'kv_amortized_preload_bytes_per_token': kv_amortized_preload,
    }


def write_report(report):
    report_path = ARTIFACT_DIR / 'phase4_validation_report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return report_path


def main():
    reset_artifacts()
    run_check('baseline_still_structural', check_baseline_still_structural)
    run_check('kv_stage_names', check_kv_stage_names)
    run_check('sequence_scaling', check_sequence_scaling)
    run_check('array_scaling', check_array_scaling)
    run_check('decode_token_amortization', check_decode_token_amortization)
    run_check('kv_heads_impact', check_kv_heads_impact)
    run_check('validation_bridge_stream_metrics', check_validation_bridge_stream_metrics)
    run_check('simulator_kv_route', check_simulator_kv_route)
    run_check('directional_vs_baseline', check_directional_vs_baseline)

    report = collector.report()
    report_path = write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f'\nValidation report written to: {report_path}')
    return 1 if report['summary']['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
