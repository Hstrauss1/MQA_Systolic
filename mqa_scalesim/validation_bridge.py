from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .result_schema import MQASimulationResult
from .workload import MQAWorkload


@dataclass(slots=True)
class ValidationSummary:
    passed: bool
    message: str
    metadata: Dict[str, Any]


def result_to_validation_dict(result: MQASimulationResult) -> Dict[str, Any]:
    payload = result.to_dict()
    payload['stage_names'] = [stage['name'] for stage in payload['stages']]
    payload['stage_cycles'] = {stage['name']: stage['cycles'] for stage in payload['stages']}
    payload['stage_macs'] = {stage['name']: stage['macs'] for stage in payload['stages']}
    payload['stage_occupancies'] = {stage['name']: stage['occupancy'] for stage in payload['stages']}
    payload['stage_stall_cycles'] = {stage['name']: stage['stall_cycles'] for stage in payload['stages']}
    payload['stream_metrics'] = {
        'kv_preload_bytes': payload.get('kv_preload_bytes', 0),
        'weighted_pe_utilization': payload.get('weighted_pe_utilization', 0.0),
        'pipeline_depth': payload.get('metadata', {}).get('pipeline_depth'),
        'stream_groups': payload.get('metadata', {}).get('stream_groups'),
        'amortized_preload_bytes_per_token': payload.get('metadata', {}).get('amortized_preload_bytes_per_token'),
        'memory_stall_cycles': payload.get('memory_stall_cycles', 0),
        'kv_preload_bandwidth_cycles': payload.get('kv_preload_bandwidth_cycles', 0),
    }
    return payload


def result_to_experiment_row(workload: MQAWorkload, result: MQASimulationResult) -> Dict[str, Any]:
    payload = result.to_dict()
    stage_cycles = {stage['name']: stage['cycles'] for stage in payload['stages']}
    metadata = payload.get('metadata', {})
    row: Dict[str, Any] = {
        'experiment_id': workload.metadata.get('experiment_id'),
        'mode': workload.mode,
        'sequence_length': workload.sequence_length,
        'batch_size': workload.batch_size,
        'query_heads': workload.query_heads,
        'kv_heads': workload.kv_heads,
        'head_dim': workload.head_dim,
        'precision': workload.precision,
        'array_rows': workload.array_rows,
        'array_cols': workload.array_cols,
        'array_shape': workload.metadata.get('array_shape', f'{workload.array_rows}x{workload.array_cols}'),
        'decode_tokens': workload.decode_tokens,
        'decode_step': workload.decode_step,
        'softmax_variant': workload.softmax_variant,
        'exp_variant': workload.exp_variant,
        'reuse_kv_across_tokens': workload.reuse_kv_across_tokens,
        'total_cycles': payload['total_cycles'],
        'total_macs': payload['total_macs'],
        'total_stall_cycles': payload['total_stall_cycles'],
        'dram_reads': payload['dram_reads'],
        'dram_writes': payload['dram_writes'],
        'sram_reads': payload['sram_reads'],
        'sram_writes': payload['sram_writes'],
        'pe_utilization': payload['pe_utilization'],
        'weighted_pe_utilization': payload['weighted_pe_utilization'],
        'onchip_storage_bytes': payload['onchip_storage_bytes'],
        'kv_preload_bytes': payload['kv_preload_bytes'],
        'stage_count': len(payload['stages']),
        'stage_names': '|'.join(stage['name'] for stage in payload['stages']),
        'amortized_preload_bytes_per_token': metadata.get('amortized_preload_bytes_per_token', 0.0),
        'pipeline_depth': metadata.get('pipeline_depth', 0),
        'stream_groups': metadata.get('stream_groups', 0),
        'kv_resident_bytes': metadata.get('kv_resident_bytes', 0),
        'memory_stall_cycles': payload.get('memory_stall_cycles', 0),
        'kv_preload_bandwidth_cycles': payload.get('kv_preload_bandwidth_cycles', 0),
        'memory_model_applied': payload.get('memory_model_applied', False),
    }
    for stage_name, cycles in stage_cycles.items():
        row[f'stage_{stage_name}_cycles'] = cycles
    return row


def validate_against_reference(sim_result: Dict[str, Any], reference_result: Dict[str, Any], atol: float = 1e-5) -> ValidationSummary:
    sim_keys = set(sim_result.keys())
    ref_keys = set(reference_result.keys())
    missing = sorted(ref_keys - sim_keys)
    extra = sorted(sim_keys - ref_keys)

    passed = not missing
    message = 'reference shape matches simulator output' if passed else f'missing keys: {missing}'

    if passed:
        comparable_keys = ['mode', 'total_cycles', 'dram_reads', 'dram_writes', 'sram_reads', 'sram_writes']
        mismatches = {}
        for key in comparable_keys:
            if key in reference_result and sim_result.get(key) != reference_result.get(key):
                mismatches[key] = {
                    'expected': reference_result.get(key),
                    'actual': sim_result.get(key),
                }
        if mismatches:
            passed = False
            message = 'reference keys present but values differ'
        else:
            mismatches = {}
    else:
        mismatches = {}

    return ValidationSummary(
        passed=passed,
        message=message,
        metadata={
            'missing_keys': missing,
            'extra_keys': extra,
            'value_mismatches': mismatches,
            'atol': atol,
        },
    )
