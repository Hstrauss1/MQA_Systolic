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
    tile_launch_overhead: int = 1
    tile_drain_overhead: int = 1
    reduction_fan_in_cycles: int = 1
    sram_read_cycles_per_tile: int = 1
    sram_write_cycles_per_tile: int = 1
    dram_latency_cycles: int = 1
    stream_startup_cycles: int = 2
    wavefront_step_cycles: int = 1
    state_forward_cycles: int = 1
    partial_output_forward_cycles: int = 1
    pipeline_drain_cycles: int = 2
    head_switch_cycles: int = 1
    kv_preload_setup_cycles: int = 2

    def validate(self) -> None:
        for field_name in (
            'mac_cycles',
            'compare_max_cycles',
            'exp_cycles',
            'renorm_cycles',
            'value_accum_cycles',
            'forward_cycles',
            'final_norm_cycles',
            'tile_launch_overhead',
            'tile_drain_overhead',
            'reduction_fan_in_cycles',
            'sram_read_cycles_per_tile',
            'sram_write_cycles_per_tile',
            'dram_latency_cycles',
            'stream_startup_cycles',
            'wavefront_step_cycles',
            'state_forward_cycles',
            'partial_output_forward_cycles',
            'pipeline_drain_cycles',
            'head_switch_cycles',
            'kv_preload_setup_cycles',
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f'{field_name} must be positive')

    def gemm_tile_overhead(self) -> int:
        return self.tile_launch_overhead + self.tile_drain_overhead + self.sram_read_cycles_per_tile

    def writeback_overhead(self) -> int:
        return self.tile_launch_overhead + self.sram_write_cycles_per_tile

    def stream_step_cost(self) -> int:
        return self.mac_cycles + self.wavefront_step_cycles + self.state_forward_cycles + self.partial_output_forward_cycles

    def drain_cost(self, depth: int) -> int:
        return self.pipeline_drain_cycles + max(0, depth - 1) * self.wavefront_step_cycles

    @staticmethod
    def bandwidth_bytes_to_words(bandwidth_bytes_per_cycle: float, word_size: int) -> int:
        """Convert a DRAM bandwidth in bytes/cycle to words/cycle (Phase 6).

        Used by MQAMemoryBridge to translate workload.dram_bandwidth into the
        ifmap_backing_buf_bw word-count form expected by
        double_buffered_scratchpad.set_params().

        Parameters
        ----------
        bandwidth_bytes_per_cycle : float
            Raw DRAM bandwidth in bytes per cycle.
        word_size : int
            Size of one word in bytes (e.g. 2 for fp16, 4 for fp32).

        Returns
        -------
        int
            Words per cycle, floored, with a minimum of 1.
        """
        if word_size <= 0:
            return 1
        return max(1, int(bandwidth_bytes_per_cycle / word_size))
