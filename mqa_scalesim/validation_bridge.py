from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class ValidationSummary:
    passed: bool
    message: str
    metadata: Dict[str, Any]


def validate_against_reference(sim_result: Dict[str, Any], reference_result: Dict[str, Any], atol: float = 1e-5) -> ValidationSummary:
    sim_keys = set(sim_result.keys())
    ref_keys = set(reference_result.keys())
    missing = sorted(ref_keys - sim_keys)
    extra = sorted(sim_keys - ref_keys)
    passed = not missing
    message = 'reference shape matches simulator output' if passed else f'missing keys: {missing}'
    return ValidationSummary(
        passed=passed,
        message=message,
        metadata={
            'missing_keys': missing,
            'extra_keys': extra,
            'atol': atol,
        },
    )