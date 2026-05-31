from __future__ import annotations

from .pe_timing import PETiming
from .result_schema import MQASimulationResult, MQAStageResult
from .softmax_ops import estimate_online_softmax_cost
from .workload import MQAWorkload


class KVStationaryMQADecodeSimulator:
    """Placeholder scheduler for KV-stationary MQA decode."""

    def __init__(self, workload: MQAWorkload, timing: PETiming | None = None):
        self.workload = workload
        self.timing = timing or PETiming()
        self.workload.validate()
        self.timing.validate()

    def build_schedule(self) -> list[str]:
        return ['kv_preload', 'query_stream', 'online_softmax_accum', 'final_normalize', 'writeback']

    def simulate(self) -> MQASimulationResult:
        softmax_cost = estimate_online_softmax_cost(self.workload.sequence_length, self.timing)
        result = MQASimulationResult(mode=self.workload.mode)
        preload_cycles = self.workload.sequence_length * self.workload.head_dim
        stream_cycles = self.workload.sequence_length * (self.timing.mac_cycles + self.timing.forward_cycles)
        result.kv_preload_bytes = self.workload.sequence_length * self.workload.head_dim * 2
        result.add_stage(MQAStageResult(name='kv_preload', cycles=preload_cycles, dram_reads=result.kv_preload_bytes))
        result.add_stage(MQAStageResult(name='query_stream', cycles=stream_cycles))
        result.add_stage(MQAStageResult(name='online_softmax_accum', cycles=softmax_cost.total_cycles))
        result.add_stage(MQAStageResult(name='final_normalize', cycles=self.timing.final_norm_cycles))
        result.add_stage(MQAStageResult(name='writeback', cycles=1))
        result.metadata['schedule'] = self.build_schedule()
        return result
