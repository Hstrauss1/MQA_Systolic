from __future__ import annotations

import math

from .pe_timing import PETiming
from .result_schema import MQASimulationResult, MQAStageResult
from .softmax_ops import estimate_online_softmax_cost
from .workload import MQAWorkload


class BaselineMQADecodeSimulator:
    """Analytical baseline decode simulator for conventional GEMM-style MQA decode."""

    def __init__(self, workload: MQAWorkload, timing: PETiming | None = None):
        self.workload = workload
        self.timing = timing or PETiming()
        self.workload.validate()
        self.timing.validate()

    def build_schedule(self) -> list[str]:
        return ['score_gemm', 'softmax_reduce', 'value_gemm', 'writeback']

    @property
    def bytes_per_element(self) -> int:
        return {
            'int8': 1,
            'fp16': 2,
            'fp32': 4,
        }[self.workload.precision]

    def _ceil_div(self, x: int, y: int) -> int:
        return (x + y - 1) // y

    def _build_gemm_stage(self, name: str, rows: int, cols: int, reduction_dim: int, compute_cycles_per_step: int,
                          include_output_write: bool = True, output_resident: bool = False) -> MQAStageResult:
        array_rows = self.workload.array_rows
        array_cols = self.workload.array_cols
        bpe = self.bytes_per_element

        row_tiles = self._ceil_div(rows, array_rows)
        col_tiles = self._ceil_div(cols, array_cols)
        reduction_tiles = self._ceil_div(reduction_dim, array_rows)
        tile_instances = row_tiles * col_tiles

        macs = rows * cols * reduction_dim
        active_rows = min(rows, array_rows)
        active_cols = min(cols, array_cols)
        active_pes = active_rows * active_cols
        spatial_pes = array_rows * array_cols
        occupancy = active_pes / spatial_pes if spatial_pes > 0 else 0.0

        tile_compute_cycles = reduction_dim * compute_cycles_per_step
        tile_overhead = self.timing.gemm_tile_overhead() + reduction_tiles * self.timing.reduction_fan_in_cycles
        cycles = tile_instances * (tile_compute_cycles + tile_overhead)

        lhs_reads = rows * reduction_dim * bpe
        rhs_reads = reduction_dim * cols * bpe
        output_writes = rows * cols * bpe if include_output_write else 0
        output_reads = rows * cols * bpe if output_resident else 0

        total_dram_reads = lhs_reads + rhs_reads + output_reads
        total_dram_writes = output_writes
        total_sram_reads = lhs_reads + rhs_reads + output_reads
        total_sram_writes = output_writes
        total_bytes = total_dram_reads + total_dram_writes
        arithmetic_intensity = macs / total_bytes if total_bytes > 0 else 0.0

        return MQAStageResult(
            name=name,
            cycles=cycles,
            dram_reads=total_dram_reads,
            dram_writes=total_dram_writes,
            sram_reads=total_sram_reads,
            sram_writes=total_sram_writes,
            macs=macs,
            rows=rows,
            cols=cols,
            reduction_dim=reduction_dim,
            row_tiles=row_tiles,
            col_tiles=col_tiles,
            reduction_tiles=reduction_tiles,
            spatial_pes=spatial_pes,
            active_pes=active_pes,
            occupancy=occupancy,
            arithmetic_intensity=arithmetic_intensity,
            metadata={
                'stage_type': 'gemm',
                'tile_instances': tile_instances,
                'tile_compute_cycles': tile_compute_cycles,
                'tile_overhead_cycles': tile_overhead,
                'bytes_per_element': bpe,
            },
        )

    def _build_softmax_stage(self, rows: int, cols: int) -> MQAStageResult:
        bpe = self.bytes_per_element
        softmax_cost = estimate_online_softmax_cost(cols, self.timing)
        cycles_per_row = softmax_cost.total_cycles
        total_cycles = rows * cycles_per_row
        dram_reads = rows * cols * bpe
        dram_writes = rows * cols * bpe
        sram_reads = dram_reads
        sram_writes = dram_writes
        arithmetic_intensity = 0.0

        return MQAStageResult(
            name='softmax_reduce',
            cycles=total_cycles,
            dram_reads=dram_reads,
            dram_writes=dram_writes,
            sram_reads=sram_reads,
            sram_writes=sram_writes,
            macs=0,
            rows=rows,
            cols=cols,
            reduction_dim=cols,
            row_tiles=self._ceil_div(rows, self.workload.array_rows),
            col_tiles=1,
            reduction_tiles=self._ceil_div(cols, self.workload.array_cols),
            spatial_pes=self.workload.array_rows * self.workload.array_cols,
            active_pes=min(rows, self.workload.array_rows),
            occupancy=min(rows, self.workload.array_rows) / max(1, self.workload.array_rows * self.workload.array_cols),
            arithmetic_intensity=arithmetic_intensity,
            metadata={
                'stage_type': 'softmax',
                'cycles_per_row': cycles_per_row,
                'softmax_variant': self.workload.softmax_variant,
                'exp_variant': self.workload.exp_variant,
            },
        )

    def _build_writeback_stage(self, rows: int, cols: int) -> MQAStageResult:
        bpe = self.bytes_per_element
        row_tiles = self._ceil_div(rows, self.workload.array_rows)
        col_tiles = self._ceil_div(cols, self.workload.array_cols)
        tile_instances = row_tiles * col_tiles
        cycles = tile_instances * self.timing.writeback_overhead() + rows * self.timing.final_norm_cycles
        dram_writes = rows * cols * bpe
        sram_reads = rows * cols * bpe

        return MQAStageResult(
            name='writeback',
            cycles=cycles,
            dram_reads=0,
            dram_writes=dram_writes,
            sram_reads=sram_reads,
            sram_writes=0,
            macs=0,
            rows=rows,
            cols=cols,
            reduction_dim=1,
            row_tiles=row_tiles,
            col_tiles=col_tiles,
            reduction_tiles=1,
            spatial_pes=self.workload.array_rows * self.workload.array_cols,
            active_pes=min(rows, self.workload.array_rows) * min(cols, self.workload.array_cols),
            occupancy=(min(rows, self.workload.array_rows) * min(cols, self.workload.array_cols)) /
                      max(1, self.workload.array_rows * self.workload.array_cols),
            arithmetic_intensity=0.0,
            metadata={
                'stage_type': 'writeback',
                'tile_instances': tile_instances,
                'bytes_per_element': bpe,
            },
        )

    def simulate(self, run_memory_model: bool = True) -> MQASimulationResult:
        rows = self.workload.batch_size * self.workload.query_heads * self.workload.decode_tokens
        sequence_length = self.workload.sequence_length
        head_dim = self.workload.head_dim
        bpe = self.bytes_per_element

        score_stage = self._build_gemm_stage(
            name='score_gemm',
            rows=rows,
            cols=sequence_length,
            reduction_dim=head_dim,
            compute_cycles_per_step=self.timing.mac_cycles,
            include_output_write=True,
            output_resident=False,
        )

        softmax_stage = self._build_softmax_stage(rows=rows, cols=sequence_length)

        value_stage = self._build_gemm_stage(
            name='value_gemm',
            rows=rows,
            cols=head_dim,
            reduction_dim=sequence_length,
            compute_cycles_per_step=self.timing.value_accum_cycles,
            include_output_write=False,
            output_resident=True,
        )

        writeback_stage = self._build_writeback_stage(rows=rows, cols=head_dim)

        result = MQASimulationResult(mode=self.workload.mode)
        result.onchip_storage_bytes = (self.workload.ifmap_sram_kb + self.workload.filter_sram_kb + self.workload.ofmap_sram_kb) * 1024
        result.kv_preload_bytes = 0
        result.metadata['schedule'] = self.build_schedule()
        result.metadata['bytes_per_element'] = bpe
        result.metadata['workload_shape'] = {
            'rows': rows,
            'score_cols': sequence_length,
            'head_dim': head_dim,
        }

        for stage in (score_stage, softmax_stage, value_stage, writeback_stage):
            result.add_stage(stage)

        result.finalize()
        if run_memory_model:
            try:
                from .memory_bridge import MQAMemoryBridge
                bridge = MQAMemoryBridge(
                    workload=self.workload,
                    sim_result=result,
                    kv_preload_bytes=0,
                    verbose=False,
                )
                mem_result = bridge.run()
                result.apply_memory_result(mem_result)
            except ImportError:
                pass  # SCALE-Sim not installed; skip memory model silently
        return result
