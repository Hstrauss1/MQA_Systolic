from __future__ import annotations

from dataclasses import dataclass

from .pe_timing import PETiming


@dataclass(slots=True)
class SoftmaxOpCost:
    max_reduce_cycles: int
    exp_cycles: int
    renorm_cycles: int
    accumulate_cycles: int
    final_norm_cycles: int

    @property
    def total_cycles(self) -> int:
        return (
            self.max_reduce_cycles
            + self.exp_cycles
            + self.renorm_cycles
            + self.accumulate_cycles
            + self.final_norm_cycles
        )


def estimate_online_softmax_cost(tokens: int, timing: PETiming) -> SoftmaxOpCost:
    if tokens <= 0:
        raise ValueError('tokens must be positive')
    return SoftmaxOpCost(
        max_reduce_cycles=tokens * timing.compare_max_cycles,
        exp_cycles=tokens * timing.exp_cycles,
        renorm_cycles=tokens * timing.renorm_cycles,
        accumulate_cycles=tokens * timing.value_accum_cycles,
        final_norm_cycles=timing.final_norm_cycles,
    )
