from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .result_schema import MQASimulationResult


@dataclass(slots=True)
class ValidationSummary:
    passed: bool
    message: str
    meta: Dict[str, Any]


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
    }
    return payload


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
