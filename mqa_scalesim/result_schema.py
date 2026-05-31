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
    macs: int = 0
    rows: int = 0
    cols: int = 0
    reduction_dim: int = 0
    row_tiles: int = 0
    col_tiles: int = 0
    reduction_tiles: int = 0
    spatial_pes: int = 0
    active_pes: int = 0
    occupancy: float = 0.0
    arithmetic_intensity: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            'name': self.name,
            'cycles': self.cycles,
            'dram_reads': self.dram_reads,
            'dram_writes': self.dram_writes,
            'sram_reads': self.sram_reads,
            'sram_writes': self.sram_writes,
            'stall_cycles': self.stall_cycles,
            'macs': self.macs,
            'rows': self.rows,
            'cols': self.cols,
            'reduction_dim': self.reduction_dim,
            'row_tiles': self.row_tiles,
            'col_tiles': self.col_tiles,
            'reduction_tiles': self.reduction_tiles,
            'spatial_pes': self.spatial_pes,
            'active_pes': self.active_pes,
            'occupancy': self.occupancy,
            'arithmetic_intensity': self.arithmetic_intensity,
            'metadata': dict(self.metadata),
        }


@dataclass(slots=True)
class MQASimulationResult:
    mode: str
    total_cycles: int = 0
    total_macs: int = 0
    total_stall_cycles: int = 0
    pe_utilization: float = 0.0
    weighted_pe_utilization: float = 0.0
    onchip_storage_bytes: int = 0
    kv_preload_bytes: int = 0
    dram_reads: int = 0
    dram_writes: int = 0
    sram_reads: int = 0
    sram_writes: int = 0
    stages: List[MQAStageResult] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    # Phase 6: populated by MQAMemoryBridge after the memory model runs
    memory_stall_cycles: int = 0
    kv_preload_bandwidth_cycles: int = 0
    memory_model_applied: bool = False

    def add_stage(self, stage: MQAStageResult) -> None:
        self.stages.append(stage)
        self.total_cycles += stage.cycles
        self.total_stall_cycles += stage.stall_cycles
        self.total_macs += stage.macs
        self.dram_reads += stage.dram_reads
        self.dram_writes += stage.dram_writes
        self.sram_reads += stage.sram_reads
        self.sram_writes += stage.sram_writes
        self._recompute_utilization()

    def _recompute_utilization(self) -> None:
        if not self.stages:
            self.pe_utilization = 0.0
            self.weighted_pe_utilization = 0.0
            return

        occupancies = [stage.occupancy for stage in self.stages if stage.cycles > 0]
        self.pe_utilization = sum(occupancies) / len(occupancies) if occupancies else 0.0

        weighted_num = sum(stage.occupancy * stage.cycles for stage in self.stages)
        self.weighted_pe_utilization = weighted_num / self.total_cycles if self.total_cycles > 0 else 0.0

    def finalize(self) -> None:
        self._recompute_utilization()

    def apply_memory_result(self, mem_result: object) -> None:
        """Merge MQAMemoryResult output into this simulation result (Phase 6)."""
        self.memory_stall_cycles = mem_result.stall_cycles
        self.kv_preload_bandwidth_cycles = mem_result.kv_preload_cycles
        self.total_stall_cycles += mem_result.stall_cycles
        self.total_cycles += mem_result.stall_cycles
        self.memory_model_applied = True
        self.metadata['memory_model'] = mem_result.to_dict()

    def to_dict(self) -> Dict[str, object]:
        return {
            'mode': self.mode,
            'total_cycles': self.total_cycles,
            'total_macs': self.total_macs,
            'total_stall_cycles': self.total_stall_cycles,
            'pe_utilization': self.pe_utilization,
            'weighted_pe_utilization': self.weighted_pe_utilization,
            'onchip_storage_bytes': self.onchip_storage_bytes,
            'kv_preload_bytes': self.kv_preload_bytes,
            'dram_reads': self.dram_reads,
            'dram_writes': self.dram_writes,
            'sram_reads': self.sram_reads,
            'sram_writes': self.sram_writes,
            'stages': [stage.to_dict() for stage in self.stages],
            'metadata': dict(self.metadata),
            'memory_stall_cycles': self.memory_stall_cycles,
            'kv_preload_bandwidth_cycles': self.kv_preload_bandwidth_cycles,
            'memory_model_applied': self.memory_model_applied,
        }
