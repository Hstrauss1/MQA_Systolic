from __future__ import annotations

from .pe_timing import PETiming
from .result_schema import MQASimulationResult, MQAStageResult
from .softmax_ops import estimate_online_softmax_cost
from .workload import MQAWorkload


class KVStationaryMQADecodeSimulator:
    """Structured placeholder scheduler for KV-stationary MQA decode using the Phase 3 result schema."""

    def __init__(self, workload: MQAWorkload, timing: PETiming | None = None):
        self.workload = workload
        self.timing = timing or PETiming()
        self.workload.validate()
        self.timing.validate()

    @property
    def bytes_per_element(self) -> int:
        return {
            'int8': 1,
            'fp16': 2,
            'fp32': 4,
        }[self.workload.precision]

    def build_schedule(self) -> list[str]:
        return ['kv_preload', 'query_stream', 'online_softmax_accum', 'final_normalize', 'writeback']

    def simulate(self) -> MQASimulationResult:
        rows = self.workload.batch_size * self.workload.query_heads * self.workload.decode_tokens
        seq_len = self.workload.sequence_length
        head_dim = self.workload.head_dim
        bpe = self.bytes_per_element

        softmax_cost = estimate_online_softmax_cost(seq_len, self.timing)
        result = MQASimulationResult(mode=self.workload.mode)
        result.onchip_storage_bytes = (self.workload.ifmap_sram_kb + self.workload.filter_sram_kb + self.workload.ofmap_sram_kb) * 1024
        result.kv_preload_bytes = seq_len * head_dim * 2 * bpe

        kv_preload = MQAStageResult(
            name='kv_preload',
            cycles=seq_len * head_dim + self.timing.dram_latency_cycles,
            dram_reads=result.kv_preload_bytes,
            dram_writes=0,
            sram_reads=0,
            sram_writes=result.kv_preload_bytes,
            macs=0,
            rows=seq_len,
            cols=head_dim,
            reduction_dim=1,
            row_tiles=1,
            col_tiles=1,
            reduction_tiles=1,
            spatial_pes=self.workload.array_rows * self.workload.array_cols,
            active_pes=min(seq_len, self.workload.array_rows),
            occupancy=min(seq_len, self.workload.array_rows) / max(1, self.workload.array_rows * self.workload.array_cols),
            arithmetic_intensity=0.0,
            metadata={'stage_type': 'preload', 'bytes_per_element': bpe},
        )

        query_stream_cycles = rows * seq_len * (self.timing.mac_cycles + self.timing.forward_cycles)
        query_stream = MQAStageResult(
            name='query_stream',
            cycles=query_stream_cycles,
            dram_reads=rows * head_dim * bpe,
            dram_writes=0,
            sram_reads=rows * head_dim * bpe + result.kv_preload_bytes,
            sram_writes=0,
            macs=rows * seq_len * head_dim,
            rows=rows,
            cols=seq_len,
            reduction_dim=head_dim,
            row_tiles=max(1, (rows + self.workload.array_rows - 1) // self.workload.array_rows),
            col_tiles=max(1, (seq_len + self.workload.array_cols - 1) // self.workload.array_cols),
            reduction_tiles=max(1, (head_dim + self.workload.array_rows - 1) // self.workload.array_rows),
            spatial_pes=self.workload.array_rows * self.workload.array_cols,
            active_pes=min(rows, self.workload.array_rows) * min(seq_len, self.workload.array_cols),
            occupancy=(min(rows, self.workload.array_rows) * min(seq_len, self.workload.array_cols)) /
                      max(1, self.workload.array_rows * self.workload.array_cols),
            arithmetic_intensity=(rows * seq_len * head_dim) / max(1, rows * head_dim * bpe + result.kv_preload_bytes),
            metadata={'stage_type': 'stream'},
        )

        softmax_stage = MQAStageResult(
            name='online_softmax_accum',
            cycles=rows * softmax_cost.total_cycles,
            dram_reads=0,
            dram_writes=0,
            sram_reads=rows * seq_len * bpe,
            sram_writes=rows * head_dim * bpe,
            macs=0,
            rows=rows,
            cols=seq_len,
            reduction_dim=seq_len,
            row_tiles=max(1, (rows + self.workload.array_rows - 1) // self.workload.array_rows),
            col_tiles=1,
            reduction_tiles=max(1, (seq_len + self.workload.array_cols - 1) // self.workload.array_cols),
            spatial_pes=self.workload.array_rows * self.workload.array_cols,
            active_pes=min(rows, self.workload.array_rows),
            occupancy=min(rows, self.workload.array_rows) / max(1, self.workload.array_rows * self.workload.array_cols),
            arithmetic_intensity=0.0,
            metadata={
                'stage_type': 'softmax',
                'softmax_variant': self.workload.softmax_variant,
                'exp_variant': self.workload.exp_variant,
            },
        )

        final_normalize = MQAStageResult(
            name='final_normalize',
            cycles=rows * head_dim * self.timing.final_norm_cycles,
            dram_reads=0,
            dram_writes=0,
            sram_reads=rows * head_dim * bpe,
            sram_writes=rows * head_dim * bpe,
            macs=0,
            rows=rows,
            cols=head_dim,
            reduction_dim=1,
            row_tiles=max(1, (rows + self.workload.array_rows - 1) // self.workload.array_rows),
            col_tiles=max(1, (head_dim + self.workload.array_cols - 1) // self.workload.array_cols),
            reduction_tiles=1,
            spatial_pes=self.workload.array_rows * self.workload.array_cols,
            active_pes=min(rows, self.workload.array_rows) * min(head_dim, self.workload.array_cols),
            occupancy=(min(rows, self.workload.array_rows) * min(head_dim, self.workload.array_cols)) /
                      max(1, self.workload.array_rows * self.workload.array_cols),
            arithmetic_intensity=0.0,
            metadata={'stage_type': 'normalize'},
        )

        writeback = MQAStageResult(
            name='writeback',
            cycles=rows * self.timing.writeback_overhead(),
            dram_reads=0,
            dram_writes=rows * head_dim * bpe,
            sram_reads=rows * head_dim * bpe,
            sram_writes=0,
            macs=0,
            rows=rows,
            cols=head_dim,
            reduction_dim=1,
            row_tiles=max(1, (rows + self.workload.array_rows - 1) // self.workload.array_rows),
            col_tiles=max(1, (head_dim + self.workload.array_cols - 1) // self.workload.array_cols),
            reduction_tiles=1,
            spatial_pes=self.workload.array_rows * self.workload.array_cols,
            active_pes=min(rows, self.workload.array_rows),
            occupancy=min(rows, self.workload.array_rows) / max(1, self.workload.array_rows * self.workload.array_cols),
            arithmetic_intensity=0.0,
            metadata={'stage_type': 'writeback'},
        )

        result.metadata['schedule'] = self.build_schedule()
        result.metadata['bytes_per_element'] = bpe

        for stage in (kv_preload, query_stream, softmax_stage, final_normalize, writeback):
            result.add_stage(stage)

        result.finalize()
        return result
