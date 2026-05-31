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
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f'{field_name} must be positive')

    def gemm_tile_overhead(self) -> int:
        return self.tile_launch_overhead + self.tile_drain_overhead + self.sram_read_cycles_per_tile

    def writeback_overhead(self) -> int:
        return self.tile_launch_overhead + self.sram_write_cycles_per_tile
