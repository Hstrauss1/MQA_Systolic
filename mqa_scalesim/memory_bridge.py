"""memory_bridge.py — Phase 6

Translation layer between MQASimulationResult stage data and the SCALE-Sim
double_buffered_scratchpad memory subsystem.  Importing this module does *not*
require SCALE-Sim to be installed; the import happens lazily inside run() so
that the rest of mqa_scalesim can be used standalone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .result_schema import MQASimulationResult, MQAStageResult
from .workload import MQAWorkload


# ---------------------------------------------------------------------------
# Result container returned by MQAMemoryBridge.run()
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MQAMemoryResult:
    """Memory-model output produced by the bridge for one simulation run."""
    stall_cycles: int = 0
    dram_read_cycles: int = 0
    dram_write_cycles: int = 0
    sram_hit_cycles: int = 0
    kv_preload_cycles: int = 0
    corrected_total_cycles: int = 0
    per_stage: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'stall_cycles': self.stall_cycles,
            'dram_read_cycles': self.dram_read_cycles,
            'dram_write_cycles': self.dram_write_cycles,
            'sram_hit_cycles': self.sram_hit_cycles,
            'kv_preload_cycles': self.kv_preload_cycles,
            'corrected_total_cycles': self.corrected_total_cycles,
            'per_stage': self.per_stage,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_to_words(total_bytes: int, word_size: int) -> int:
    """Convert a byte count to a word count, rounding up."""
    return max(1, math.ceil(total_bytes / max(1, word_size)))


def _bandwidth_bytes_to_words(bandwidth_bytes_per_cycle: float, word_size: int) -> int:
    """Convert a bandwidth in bytes/cycle to words/cycle (floor, minimum 1)."""
    return max(1, int(bandwidth_bytes_per_cycle / max(1, word_size)))


def _make_demand_matrix(total_words: int, bandwidth_words: int) -> np.ndarray:
    """Build a 2-D demand matrix (num_cycles × bandwidth_words) filled with
    sequential addresses, compatible with read_buffer.service_reads().

    The last row may be partially filled; unused slots are set to -1
    (SCALE-Sim's convention for 'no request').
    """
    if total_words <= 0:
        return np.full((1, bandwidth_words), -1, dtype=int)
    num_cycles = math.ceil(total_words / bandwidth_words)
    mat = np.full((num_cycles, bandwidth_words), -1, dtype=int)
    addr = 0
    for row in range(num_cycles):
        for col in range(bandwidth_words):
            if addr < total_words:
                mat[row, col] = addr
                addr += 1
    return mat


def _null_demand_matrix(num_cycles: int, bandwidth_words: int) -> np.ndarray:
    """Return an all-(-1) demand matrix for a buffer that has no activity."""
    return np.full((max(1, num_cycles), bandwidth_words), -1, dtype=int)


# ---------------------------------------------------------------------------
# Main bridge class
# ---------------------------------------------------------------------------

class MQAMemoryBridge:
    """Drive the SCALE-Sim scratchpad memory model from MQA stage results.

    Parameters
    ----------
    workload:
        The MQAWorkload that produced *sim_result*.
    sim_result:
        The MQASimulationResult returned by BaselineMQADecodeSimulator or
        KVStationaryMQADecodeSimulator.  The bridge reads stage traffic but
        does **not** mutate this object; call result.apply_memory_result()
        afterwards if you want the fields merged.
    kv_preload_bytes:
        For KV-stationary mode: the total number of bytes transferred from
        DRAM to SRAM in the one-time KV preload stage.  Leave 0 for baseline.
    verbose:
        Passed through to the scratchpad so the tqdm progress bar is shown.
    """

    def __init__(
        self,
        workload: MQAWorkload,
        sim_result: MQASimulationResult,
        kv_preload_bytes: int = 0,
        verbose: bool = False,
    ) -> None:
        self.workload = workload
        self.sim_result = sim_result
        self.kv_preload_bytes = kv_preload_bytes
        self.verbose = verbose

        # Derived constants -------------------------------------------------
        self.word_size: int = workload.precision_bytes()
        dram_bw_bytes = workload.dram_bandwidth if getattr(workload, 'dram_bandwidth', None) is not None else 8.0
        self.dram_bw_words: int = _bandwidth_bytes_to_words(
            dram_bw_bytes,
            self.word_size,
        )
        self.sram_total_bytes: int = (
            (workload.ifmap_sram_kb + workload.filter_sram_kb + workload.ofmap_sram_kb) * 1024
        )
        self.ifmap_sram_bytes: int = workload.ifmap_sram_kb * 1024
        self.filter_sram_bytes: int = workload.filter_sram_kb * 1024
        self.ofmap_sram_bytes: int = workload.ofmap_sram_kb * 1024

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> MQAMemoryResult:
        """Execute the memory simulation and return an MQAMemoryResult.

        Imports SCALE-Sim lazily so that the module can be imported without
        a full SCALE-Sim installation.
        """
        # Lazy SCALE-Sim import -------------------------------------------
        try:
            from scalesim.memory.double_buffered_scratchpad_mem import (
                double_buffered_scratchpad,
            )
            from scalesim.memory.read_buffer import read_buffer
            from scalesim.memory.write_buffer import write_buffer
            from scalesim.memory.read_port import read_port
            from scalesim.memory.write_port import write_port
        except ImportError as exc:
            raise ImportError(
                'SCALE-Sim must be installed (pip install scalesim) to use '
                'MQAMemoryBridge.  Original error: ' + str(exc)
            ) from exc

        mem_result = MQAMemoryResult()

        # ------------------------------------------------------------------
        # Step 1: one-time KV preload (KV-stationary mode only)
        # ------------------------------------------------------------------
        kv_preload_cycles = 0
        if self.kv_preload_bytes > 0:
            preload_words = _bytes_to_words(self.kv_preload_bytes, self.word_size)
            kv_preload_cycles = math.ceil(preload_words / self.dram_bw_words)
            mem_result.kv_preload_cycles = kv_preload_cycles
            mem_result.per_stage.append({
                'stage': 'kv_preload',
                'preload_words': preload_words,
                'preload_cycles': kv_preload_cycles,
            })

        # ------------------------------------------------------------------
        # Step 2: drive each compute stage through the scratchpad
        # ------------------------------------------------------------------
        total_stall_cycles = 0
        total_dram_read_cycles = 0
        total_dram_write_cycles = 0

        for stage in self.sim_result.stages:
            stage_mem = self._run_stage(stage, double_buffered_scratchpad, read_port, write_port)
            total_stall_cycles += stage_mem['stall_cycles']
            total_dram_read_cycles += stage_mem['dram_read_cycles']
            total_dram_write_cycles += stage_mem['dram_write_cycles']
            mem_result.per_stage.append(stage_mem)

        # ------------------------------------------------------------------
        # Aggregate
        # ------------------------------------------------------------------
        mem_result.stall_cycles = total_stall_cycles
        mem_result.dram_read_cycles = total_dram_read_cycles
        mem_result.dram_write_cycles = total_dram_write_cycles
        mem_result.sram_hit_cycles = max(
            0,
            self.sim_result.total_cycles - total_stall_cycles - kv_preload_cycles,
        )
        mem_result.corrected_total_cycles = (
            self.sim_result.total_cycles + total_stall_cycles + kv_preload_cycles
        )
        return mem_result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        stage: MQAStageResult,
        scratchpad_cls,
        rdport_cls,
        wrport_cls,
    ) -> dict:
        """Run one MQAStageResult through a fresh scratchpad instance.

        The three demand matrices are constructed from the stage's aggregate
        SRAM and DRAM traffic numbers and the workload's DRAM bandwidth.
        """
        # --- build demand matrices ----------------------------------------
        ifmap_words = _bytes_to_words(stage.sram_reads, self.word_size)
        filter_words = _bytes_to_words(
            stage.dram_reads - min(stage.dram_reads, stage.sram_reads), self.word_size
        ) if stage.dram_reads > stage.sram_reads else 1
        ofmap_words = _bytes_to_words(
            max(stage.sram_writes, stage.dram_writes), self.word_size
        )

        bw = self.dram_bw_words
        ifmap_demand = _make_demand_matrix(ifmap_words, bw)
        filter_demand = _null_demand_matrix(ifmap_demand.shape[0], bw)
        ofmap_demand  = _make_demand_matrix(ofmap_words, bw)

        # Pad to equal height (scratchpad requires matched row counts)
        max_rows = max(ifmap_demand.shape[0], ofmap_demand.shape[0])
        if ifmap_demand.shape[0] < max_rows:
            pad = np.full((max_rows - ifmap_demand.shape[0], bw), -1, dtype=int)
            ifmap_demand = np.vstack([ifmap_demand, pad])
        if filter_demand.shape[0] < max_rows:
            pad = np.full((max_rows - filter_demand.shape[0], bw), -1, dtype=int)
            filter_demand = np.vstack([filter_demand, pad])
        if ofmap_demand.shape[0] < max_rows:
            pad = np.full((max_rows - ofmap_demand.shape[0], bw), -1, dtype=int)
            ofmap_demand = np.vstack([ofmap_demand, pad])

        # --- build and configure scratchpad --------------------------------
        sp = scratchpad_cls()
        sp.set_params(
            layer_id=0,
            verbose=self.verbose,
            estimate_bandwidth_mode=False,
            word_size=self.word_size,
            ifmap_buf_size_bytes=max(self.ifmap_sram_bytes, self.word_size * bw),
            filter_buf_size_bytes=max(self.filter_sram_bytes, self.word_size * bw),
            ofmap_buf_size_bytes=max(self.ofmap_sram_bytes, self.word_size * bw),
            rd_buf_active_frac=0.5,
            wr_buf_active_frac=0.5,
            ifmap_backing_buf_bw=bw,
            filter_backing_buf_bw=bw,
            ofmap_backing_buf_bw=bw,
            mqa_mode=False,  # non-MQA path; preload handled separately above
        )
        sp.set_read_buf_prefetch_matrices(
            ifmap_prefetch_mat=ifmap_demand,
            filter_prefetch_mat=filter_demand,
        )
        sp.service_memory_requests(ifmap_demand, filter_demand, ofmap_demand)

        stall_cycles = sp.get_stall_cycles()
        total_cycles = sp.get_total_compute_cycles()

        # DRAM cycle estimates from bandwidth and byte counts
        dram_read_cycles = math.ceil(
            _bytes_to_words(stage.dram_reads, self.word_size) / self.dram_bw_words
        ) if stage.dram_reads > 0 else 0
        dram_write_cycles = math.ceil(
            _bytes_to_words(stage.dram_writes, self.word_size) / self.dram_bw_words
        ) if stage.dram_writes > 0 else 0

        return {
            'stage': stage.name,
            'stall_cycles': stall_cycles,
            'scratchpad_total_cycles': total_cycles,
            'dram_read_cycles': dram_read_cycles,
            'dram_write_cycles': dram_write_cycles,
            'ifmap_demand_rows': ifmap_demand.shape[0],
            'ofmap_demand_rows': ofmap_demand.shape[0],
        }
