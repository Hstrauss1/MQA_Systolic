from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PETiming:
    mac_cycles: int = 1
    compare_max_cycles: int = 1
    exp_cycles: int = 1
    renorm_cycles: int = 1
    value_accum_cycles: int = 1
    forward_cycles: int = 1
    final_norm_cycles: int = 1

    def validate(self) -> None:
        for field_name in (
            'mac_cycles',
            'compare_max_cycles',
            'exp_cycles',
            'renorm_cycles',
            'value_accum_cycles',
            'forward_cycles',
            'final_norm_cycles',
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f'{field_name} must be positive')
