from __future__ import annotations

from .pe_timing import PETiming
from .result_schema import MQASimulationResult, MQAStageResult
from .softmax_ops import estimate_online_softmax_cost
from .workload import MQAWorkload


class BaselineMQADecodeSimulator:
    """Placeholder scheduler for conventional GEMM-style MQA decode."""

    def __init__(self, workload: MQAWorkload, timing: PETiming | None = None):
        self.workload = workload
        self.timing = timing or PETiming()
        self.workload.validate()
        self.timing.validate()

    def build_schedule(self) -> list[str]:
        return ['score_gemm', 'softmax_reduce', 'value_gemm', 'writeback']

    def simulate(self) -> MQASimulationResult:
        softmax_cost = estimate_online_softmax_cost(self.workload.sequence_length, self.timing)
        result = MQASimulationResult(mode=self.workload.mode)
        result.add_stage(MQAStageResult(name='score_gemm', cycles=self.workload.sequence_length * self.workload.head_dim * self.timing.mac_cycles))
        result.add_stage(MQAStageResult(name='softmax_reduce', cycles=softmax_cost.total_cycles))
        result.add_stage(MQAStageResult(name='value_gemm', cycles=self.workload.sequence_length * self.workload.head_dim * self.timing.value_accum_cycles))
        result.add_stage(MQAStageResult(name='writeback', cycles=self.timing.final_norm_cycles))
        result.metadata['schedule'] = self.build_schedule()
        return result
