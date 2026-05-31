from __future__ import annotations

from dataclasses import dataclass

from .pe_timing import PETiming
from .result_schema import MQASimulationResult, MQAStageResult
from .softmax_ops import estimate_final_normalization_cost, estimate_streaming_softmax_row_cost
from .workload import MQAWorkload


@dataclass(slots=True)
class KVStreamingDerived:
    rows: int
    sequence_length: int
    head_dim: int
    kv_heads: int
    kv_group_count: int
    heads_per_kv_group: int
    bytes_per_element: int
    softmax_state_bytes_per_row: int
    resident_kv_bytes: int
    kv_preload_bytes: int
    query_bytes: int
    output_bytes: int
    array_rows: int
    array_cols: int
    array_area: int
    pipeline_depth: int
    kv_columns_per_group: int
    active_rows: int
    active_cols: int
    stream_groups: int
    preload_amortization_tokens: int


class KVStationaryMQADecodeSimulator:
    """Streaming analytical model for KV-stationary MQA decode."""

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

    @property
    def softmax_state_bytes(self) -> int:
        precision = self.workload.effective_softmax_state_precision
        return {
            'int8': 1,
            'fp16': 2,
            'fp32': 4,
        }[precision]

    def build_schedule(self) -> list[str]:
        return ['kv_preload', 'query_stream', 'online_softmax_accum', 'final_normalize', 'writeback']

    def _ceil_div(self, x: int, y: int) -> int:
        return (x + y - 1) // y

    def _derive_problem_shape(self) -> KVStreamingDerived:
        rows = self.workload.batch_size * self.workload.query_heads * self.workload.decode_tokens
        sequence_length = self.workload.sequence_length
        head_dim = self.workload.head_dim
        kv_heads = self.workload.kv_heads
        kv_group_count = kv_heads
        heads_per_kv_group = self.workload.heads_per_kv_group
        bpe = self.bytes_per_element
        state_bpe = self.softmax_state_bytes
        kv_columns_per_group = self.workload.kv_block_size or sequence_length
        kv_columns_per_group = min(kv_columns_per_group, sequence_length)
        array_rows = self.workload.array_rows
        array_cols = self.workload.array_cols
        pipeline_depth = self.workload.pipeline_depth_override or (array_cols + max(1, head_dim // max(1, array_rows)))
        active_rows = min(rows, array_rows)
        active_cols = min(kv_columns_per_group, array_cols)
        stream_groups = self._ceil_div(rows, max(1, self.workload.stream_group_rows))
        resident_kv_bytes = sequence_length * head_dim * kv_heads * 2 * bpe
        preload_amortization_tokens = self.workload.decode_tokens if self.workload.reuse_kv_across_tokens else 1
        kv_preload_bytes = resident_kv_bytes if self.workload.reuse_kv_across_tokens else resident_kv_bytes * self.workload.decode_tokens
        query_bytes = rows * head_dim * bpe
        output_bytes = rows * head_dim * bpe
        softmax_state_bytes_per_row = (2 * state_bpe) + (head_dim * state_bpe)

        return KVStreamingDerived(
            rows=rows,
            sequence_length=sequence_length,
            head_dim=head_dim,
            kv_heads=kv_heads,
            kv_group_count=kv_group_count,
            heads_per_kv_group=heads_per_kv_group,
            bytes_per_element=bpe,
            softmax_state_bytes_per_row=softmax_state_bytes_per_row,
            resident_kv_bytes=resident_kv_bytes,
            kv_preload_bytes=kv_preload_bytes,
            query_bytes=query_bytes,
            output_bytes=output_bytes,
            array_rows=array_rows,
            array_cols=array_cols,
            array_area=array_rows * array_cols,
            pipeline_depth=pipeline_depth,
            kv_columns_per_group=kv_columns_per_group,
            active_rows=active_rows,
            active_cols=active_cols,
            stream_groups=stream_groups,
            preload_amortization_tokens=preload_amortization_tokens,
        )

    def _simulate_kv_preload(self, d: KVStreamingDerived) -> MQAStageResult:
        fill_cycles = self._ceil_div(d.sequence_length * d.head_dim * d.kv_heads * 2, max(1, d.array_cols))
        cycles = self.timing.kv_preload_setup_cycles + self.timing.dram_latency_cycles + fill_cycles
        return MQAStageResult(
            name='kv_preload',
            cycles=cycles,
            dram_reads=d.kv_preload_bytes,
            dram_writes=0,
            sram_reads=0,
            sram_writes=d.kv_preload_bytes,
            macs=0,
            rows=d.sequence_length * d.kv_heads,
            cols=d.head_dim,
            reduction_dim=1,
            row_tiles=self._ceil_div(d.sequence_length * d.kv_heads, d.array_rows),
            col_tiles=self._ceil_div(d.head_dim, d.array_cols),
            reduction_tiles=1,
            spatial_pes=d.array_area,
            active_pes=min(d.sequence_length * d.kv_heads, d.array_rows) * min(d.head_dim, d.array_cols),
            occupancy=(min(d.sequence_length * d.kv_heads, d.array_rows) * min(d.head_dim, d.array_cols)) / max(1, d.array_area),
            arithmetic_intensity=0.0,
            metadata={
                'stage_type': 'preload',
                'kv_resident_bytes': d.resident_kv_bytes,
                'preload_amortization_tokens': d.preload_amortization_tokens,
            },
        )

    def _simulate_query_wavefront(self, d: KVStreamingDerived) -> MQAStageResult:
        wavefront_steps = d.sequence_length * d.stream_groups * d.kv_group_count
        startup_cycles = d.stream_groups * self.timing.stream_startup_cycles
        head_switch_cycles = max(0, d.kv_group_count - 1) * self.timing.head_switch_cycles * d.stream_groups
        step_cycles = wavefront_steps * self.timing.stream_step_cost()
        drain_cycles = d.stream_groups * self.timing.drain_cost(d.pipeline_depth)
        cycles = startup_cycles + step_cycles + head_switch_cycles + drain_cycles
        query_reads = d.query_bytes
        kv_reads = d.resident_kv_bytes
        macs = d.rows * d.sequence_length * d.head_dim
        state_forward_bytes = d.rows * d.sequence_length * d.softmax_state_bytes_per_row
        return MQAStageResult(
            name='query_stream',
            cycles=cycles,
            dram_reads=query_reads + (0 if self.workload.reuse_kv_across_tokens else d.resident_kv_bytes),
            dram_writes=0,
            sram_reads=query_reads + kv_reads,
            sram_writes=state_forward_bytes,
            macs=macs,
            rows=d.rows,
            cols=d.sequence_length,
            reduction_dim=d.head_dim,
            row_tiles=self._ceil_div(d.rows, d.array_rows),
            col_tiles=self._ceil_div(d.sequence_length, d.array_cols),
            reduction_tiles=self._ceil_div(d.head_dim, d.array_rows),
            spatial_pes=d.array_area,
            active_pes=d.active_rows * d.active_cols,
            occupancy=(d.active_rows * d.active_cols) / max(1, d.array_area),
            arithmetic_intensity=macs / max(1, query_reads + kv_reads),
            stall_cycles=head_switch_cycles,
            metadata={
                'stage_type': 'stream',
                'pipeline_startup_cycles': startup_cycles,
                'pipeline_drain_cycles': drain_cycles,
                'state_forward_bytes': state_forward_bytes,
                'wavefront_steps': wavefront_steps,
            },
        )

    def _simulate_online_softmax_stream(self, d: KVStreamingDerived) -> MQAStageResult:
        row_cost = estimate_streaming_softmax_row_cost(d.sequence_length, self.timing, d.head_dim)
        cycles = d.rows * row_cost.total_cycles
        state_reads = d.rows * d.sequence_length * d.softmax_state_bytes_per_row
        partial_output_writes = d.rows * d.head_dim * d.bytes_per_element
        return MQAStageResult(
            name='online_softmax_accum',
            cycles=cycles,
            dram_reads=0,
            dram_writes=0,
            sram_reads=state_reads,
            sram_writes=partial_output_writes,
            macs=d.rows * d.sequence_length * d.head_dim,
            rows=d.rows,
            cols=d.sequence_length,
            reduction_dim=d.head_dim,
            row_tiles=self._ceil_div(d.rows, d.array_rows),
            col_tiles=self._ceil_div(d.sequence_length, d.array_cols),
            reduction_tiles=self._ceil_div(d.head_dim, d.array_cols),
            spatial_pes=d.array_area,
            active_pes=d.active_rows * d.active_cols,
            occupancy=(d.active_rows * d.active_cols) / max(1, d.array_area),
            arithmetic_intensity=(d.rows * d.sequence_length * d.head_dim) / max(1, state_reads + partial_output_writes),
            metadata={
                'stage_type': 'softmax_stream',
                'cycles_per_row': row_cost.total_cycles,
                'softmax_variant': self.workload.softmax_variant,
                'exp_variant': self.workload.exp_variant,
            },
        )

    def _simulate_final_normalize(self, d: KVStreamingDerived) -> MQAStageResult:
        normalize_cycles = d.rows * estimate_final_normalization_cost(d.head_dim, self.timing)
        tail_cycles = d.stream_groups * self.timing.drain_cost(d.pipeline_depth)
        cycles = normalize_cycles + tail_cycles
        sram_reads = d.output_bytes
        sram_writes = d.output_bytes
        return MQAStageResult(
            name='final_normalize',
            cycles=cycles,
            dram_reads=0,
            dram_writes=0,
            sram_reads=sram_reads,
            sram_writes=sram_writes,
            macs=0,
            rows=d.rows,
            cols=d.head_dim,
            reduction_dim=1,
            row_tiles=self._ceil_div(d.rows, d.array_rows),
            col_tiles=self._ceil_div(d.head_dim, d.array_cols),
            reduction_tiles=1,
            spatial_pes=d.array_area,
            active_pes=min(d.rows, d.array_rows) * min(d.head_dim, d.array_cols),
            occupancy=(min(d.rows, d.array_rows) * min(d.head_dim, d.array_cols)) / max(1, d.array_area),
            arithmetic_intensity=0.0,
            metadata={
                'stage_type': 'normalize',
                'pipeline_drain_cycles': tail_cycles,
            },
        )

    def _simulate_writeback(self, d: KVStreamingDerived) -> MQAStageResult:
        tile_instances = self._ceil_div(d.rows, d.array_rows) * self._ceil_div(d.head_dim, d.array_cols)
        cycles = tile_instances * self.timing.writeback_overhead() + d.stream_groups * self.timing.head_switch_cycles
        return MQAStageResult(
            name='writeback',
            cycles=cycles,
            dram_reads=0,
            dram_writes=d.output_bytes,
            sram_reads=d.output_bytes,
            sram_writes=0,
            macs=0,
            rows=d.rows,
            cols=d.head_dim,
            reduction_dim=1,
            row_tiles=self._ceil_div(d.rows, d.array_rows),
            col_tiles=self._ceil_div(d.head_dim, d.array_cols),
            reduction_tiles=1,
            spatial_pes=d.array_area,
            active_pes=min(d.rows, d.array_rows),
            occupancy=min(d.rows, d.array_rows) / max(1, d.array_area),
            arithmetic_intensity=0.0,
            metadata={
                'stage_type': 'writeback',
                'amortized_preload_bytes_per_token': d.kv_preload_bytes / max(1, d.preload_amortization_tokens),
            },
        )

    def _finalize_result(self, d: KVStreamingDerived, stages: list[MQAStageResult]) -> MQASimulationResult:
        result = MQASimulationResult(mode=self.workload.mode)
        result.onchip_storage_bytes = (self.workload.ifmap_sram_kb + self.workload.filter_sram_kb + self.workload.ofmap_sram_kb) * 1024
        result.kv_preload_bytes = d.kv_preload_bytes
        result.metadata['schedule'] = self.build_schedule()
        result.metadata['kv_resident_bytes'] = d.resident_kv_bytes
        result.metadata['pipeline_depth'] = d.pipeline_depth
        result.metadata['stream_groups'] = d.stream_groups
        result.metadata['preload_amortization_tokens'] = d.preload_amortization_tokens
        result.metadata['amortized_preload_bytes_per_token'] = d.kv_preload_bytes / max(1, d.preload_amortization_tokens)
        for stage in stages:
            result.add_stage(stage)
        result.finalize()
        return result

    def simulate(self) -> MQASimulationResult:
        d = self._derive_problem_shape()
        stages = [
            self._simulate_kv_preload(d),
            self._simulate_query_wavefront(d),
            self._simulate_online_softmax_stream(d),
            self._simulate_final_normalize(d),
            self._simulate_writeback(d),
        ]
        return self._finalize_result(d, stages)
