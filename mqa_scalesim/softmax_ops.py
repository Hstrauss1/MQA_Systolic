from __future__ import annotations

from dataclasses import dataclass

from .pe_timing import PETiming


@dataclass(slots=True)
class OnlineSoftmaxCost:
    total_cycles: int
    max_cycles: int
    exp_cycles: int
    renorm_cycles: int
    value_accum_cycles: int


def estimate_online_softmax_cost(sequence_length: int, timing: PETiming) -> OnlineSoftmaxCost:
    max_cycles = sequence_length * timing.compare_max_cycles
    exp_cycles = sequence_length * timing.exp_cycles
    renorm_cycles = sequence_length * timing.renorm_cycles
    value_accum_cycles = sequence_length * timing.value_accum_cycles
    total_cycles = max_cycles + exp_cycles + renorm_cycles + value_accum_cycles
    return OnlineSoftmaxCost(
        total_cycles=total_cycles,
        max_cycles=max_cycles,
        exp_cycles=exp_cycles,
        renorm_cycles=renorm_cycles,
        value_accum_cycles=value_accum_cycles,
    )


def estimate_streaming_softmax_step_cost(timing: PETiming, value_vector_width: int) -> int:
    return (
        timing.compare_max_cycles
        + timing.exp_cycles
        + timing.renorm_cycles
        + timing.state_forward_cycles
        + (value_vector_width * timing.value_accum_cycles)
    )


def estimate_streaming_softmax_row_cost(sequence_length: int, timing: PETiming, value_vector_width: int) -> OnlineSoftmaxCost:
    max_cycles = sequence_length * timing.compare_max_cycles
    exp_cycles = sequence_length * timing.exp_cycles
    renorm_cycles = sequence_length * timing.renorm_cycles
    value_accum_cycles = sequence_length * value_vector_width * timing.value_accum_cycles
    total_cycles = max_cycles + exp_cycles + renorm_cycles + value_accum_cycles + (sequence_length * timing.state_forward_cycles)
    return OnlineSoftmaxCost(
        total_cycles=total_cycles,
        max_cycles=max_cycles,
        exp_cycles=exp_cycles,
        renorm_cycles=renorm_cycles,
        value_accum_cycles=value_accum_cycles,
    )


def estimate_final_normalization_cost(output_width: int, timing: PETiming) -> int:
    return output_width * timing.final_norm_cycles
