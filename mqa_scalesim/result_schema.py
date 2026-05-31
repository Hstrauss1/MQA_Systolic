from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class MQAStageResult:
    name: str
    cycles: int = 0
    dram_reads: int = 0
    dram_writes: int = 0
    sram_reads: int = 0
    sram_writes: int = 0
    stall_cycles: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MQASimulationResult:
    mode: str
    total_cycles: int = 0
    pe_utilization: float = 0.0
    onchip_storage_bytes: int = 0
    kv_preload_bytes: int = 0
    dram_reads: int = 0
    dram_writes: int = 0
    sram_reads: int = 0
    sram_writes: int = 0
    stages: List[MQAStageResult] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def add_stage(self, stage: MQAStageResult) -> None:
        self.stages.append(stage)
        self.total_cycles += stage.cycles
        self.dram_reads += stage.dram_reads
        self.dram_writes += stage.dram_writes
        self.sram_reads += stage.sram_reads
        self.sram_writes += stage.sram_writes

    def to_dict(self) -> Dict[str, object]:
        return {
            'mode': self.mode,
            'total_cycles': self.total_cycles,
            'pe_utilization': self.pe_utilization,
            'onchip_storage_bytes': self.onchip_storage_bytes,
            'kv_preload_bytes': self.kv_preload_bytes,
            'dram_reads': self.dram_reads,
            'dram_writes': self.dram_writes,
            'sram_reads': self.sram_reads,
            'sram_writes': self.sram_writes,
            'stages': [
                {
                    'name': stage.name,
                    'cycles': stage.cycles,
                    'dram_reads': stage.dram_reads,
                    'dram_writes': stage.dram_writes,
                    'sram_reads': stage.sram_reads,
                    'sram_writes': stage.sram_writes,
                    'stall_cycles': stage.stall_cycles,
                    'metadata': dict(stage.metadata),
                }
                for stage in self.stages
            ],
            'metadata': dict(self.metadata),
        }