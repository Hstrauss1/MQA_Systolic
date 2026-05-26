"""Analytical 2D KV-stationary systolic-style attention model for MQA decode.

This is a custom architectural model and does not claim cycle accuracy against
SCALE-Sim. The intent is to model a true 2D array where:

- rows = parallel query lanes
- cols = KV-stationary token lanes

Each processing element PE[row][col] consumes a query-state packet moving
left-to-right while K/V data remains stationary within the column.

Each column stage is a two-stage pipeline:

  Stage 1 — Lower MAC array (Q·K dot product):
    lower_mac_count independent serial MACs, shared across all row lanes.
    Each MAC handles one Q vector's full d-cycle dot product, then is
    immediately available for the next Q vector from a later row.
    With lower_mac_count MACs and N simultaneous Q vectors in flight,
    effective_macs = min(lower_mac_count, N) MACs fire in parallel and
    the MAC initiation interval (row_stagger) = ceil(d / effective_macs).

  Stage 2 — Upper Running Attention PE (online softmax update):
    Receives score_i from Stage 1 and the running state (M, L, O) from the
    left neighbour. Performs:
        Mout = max(score, Min)
        exp_old = exp(Min - Mout)          -- exp lookup, exp_latency_cycles
        exp_new = exp(score - Mout)        -- exp lookup, parallel with above
        Lout = Lin * exp_old + exp_new
        Oout = Oin * exp_old + exp_new * V -- d-wide vector ops, pe_mac_width MACs
    Takes `exp_latency_cycles + ceil(3*d / pe_mac_width)` cycles.

The two stages are sequential within a column: Stage 1 produces the score,
then Stage 2 consumes it.  A Q packet must wait for BOTH stages before
moving to the next column — giving two distinct timing parameters:

    column_dwell = d + exp_latency_cycles + ceil(3 * d / pe_mac_width)
        How long each Q packet occupies a column (independent of MAC count).

    row_stagger  = ceil(d / effective_macs)
        Initiation interval: how soon the next Q can start at the same column.
        Equal to column_dwell only when effective_macs == 1 (single MAC).

Pipeline fill/steady/flush phases are counted in pipeline steps; the
column-traversal uses column_dwell cycles per step and the vertical-K
propagation uses row_stagger cycles per step.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Sequence, Tuple


def _queries_per_row(num_queries: int, num_rows: int) -> List[int]:
    """Return the number of query packets assigned to each row lane."""
    if num_queries <= 0 or num_rows <= 0:
        return []

    active_rows = min(num_queries, num_rows)
    counts = [0 for _ in range(active_rows)]
    for query_id in range(num_queries):
        counts[query_id % active_rows] += 1
    return counts


def _compute_cycles_per_stage(
    d: int,
    exp_latency_cycles: int,
    pe_mac_width: int,
    lower_mac_count: int = 1,
    max_packets_per_row: int = 1,
) -> Tuple[int, int, int, int]:
    """Return (lower_mac_latency, upper_pe, column_dwell, packet_stagger).

    Each row PE has lower_mac_count time-interleaved serial MAC units.  The K
    stream enters MAC 0 on cycle 0 alongside Q0; MAC 1 taps the stream
    ceil(d/lower_mac_count) cycles later with Q1; and so on.  K is stored once
    as a d-element shift register with lower_mac_count tap points — no duplication.

    lower_mac_latency  = d
        Each MAC is a fully serial accumulator: d multiply-accumulates over d
        cycles.  Fixed regardless of lower_mac_count.

    upper_pe  = exp_latency_cycles + ceil(3*d / pe_mac_width)
        Online-softmax update per score: exp lookups + Oout = Oin*a + exp_new*V.

    column_dwell  = d + upper_pe   (always, independent of lower_mac_count)
        Q must wait for its own MAC (d cycles) then the upper PE (upper_pe cycles)
        before its updated (M,L,O) state is ready to carry to the next column.

    packet_stagger  = ceil(d / effective_macs)
        How soon the next Q packet can start at the same column.
        effective_macs = min(lower_mac_count, max_packets_per_row).
        With 16 MACs and ≥16 Q packets flowing through the row: stagger = 8.
        With 1 packet (single decode): effective_macs=1, stagger=128.
        Upper PE utilisation ≈ upper_pe / packet_stagger.
        With 16 MACs + prefill: 7/8 = 87.5%.  With 1 MAC: 7/135 = 5%.
    """
    effective_macs    = min(lower_mac_count, max(1, max_packets_per_row))
    lower_mac_latency = d
    upper_pe          = exp_latency_cycles + math.ceil(3 * d / pe_mac_width)
    column_dwell      = lower_mac_latency + upper_pe
    packet_stagger    = math.ceil(d / effective_macs)
    return lower_mac_latency, upper_pe, column_dwell, packet_stagger


def _pipeline_phase_steps(packet_count: int, active_cols: int) -> Tuple[int, int, int]:
    """Return fill, steady-state, and flush counts in pipeline *steps*.

    One step = one column-stage traversal = cycles_per_stage hardware cycles.
    Multiply the returned values by cycles_per_stage to get actual cycles.

    Naming note
    -----------
    The third value is called "flush" (not "drain"): it is the time the last
    packet spends traversing columns after the pipeline is no longer accepting
    new packets.  Every flush step has a packet doing useful work in the array
    — there is no idle computation, only idle *slots*.  For a single packet
    (decode), fill=0, steady=1, flush=active_cols-1: the packet is processing
    K/V tokens at every step.  Flush is only a true overhead when amortised
    over few packets; for large packet counts (prefill, batching) it becomes
    negligible.
    """
    if packet_count <= 0 or active_cols <= 0:
        return 0, 0, 0

    if packet_count >= active_cols:
        fill_steps   = active_cols - 1
        steady_steps = packet_count - active_cols + 1
        flush_steps  = active_cols - 1
        return fill_steps, steady_steps, flush_steps

    fill_steps   = packet_count - 1
    steady_steps = 1
    flush_steps  = active_cols - 1
    return fill_steps, steady_steps, flush_steps


def _format_cycle_trace(
    step_index: int,
    cycle_index: int,
    cycles_per_stage: int,
    active_packets: Sequence[Tuple[int, int, int]],
    num_rows: int,
    num_cols: int,
) -> str:
    """Create a compact ASCII snapshot of query packet locations."""
    lines = [f"Step {step_index}  (cycle {cycle_index}, {cycles_per_stage} cycles/stage):"]
    if active_packets:
        for query_id, row, col in active_packets:
            lines.append(f"  Q{query_id} -> PE({row},{col})")
    else:
        lines.append("  (idle)")

    occupancy = [[".." for _ in range(num_cols)] for _ in range(num_rows)]
    for query_id, row, col in active_packets:
        occupancy[row][col] = f"Q{query_id}"

    lines.append("  Occupancy:")
    for row_idx, row_cells in enumerate(occupancy):
        lines.append(f"  row{row_idx}: " + " ".join(row_cells))
    return "\n".join(lines)


def simulate_2d_kv_stationary_array(
    H: int,
    T: int,
    d: int,
    array_rows: int,
    array_cols: int,
    bytes_per_element: int,
    memory_bandwidth_bytes_per_cycle: int,
    exp_latency_cycles: int = 4,
    pe_mac_width: int = 1,
    lower_mac_count: int = 1,
    batch_size: int = 1,
    head_parallelism: int = 1,
    merge_extensions: int = 0,
    query_tokens: int = 1,
    debug_cycles: int = 3,
) -> Dict[str, object]:
    """Simulate a 2D systolic-style KV-stationary attention array.

    The 2D mapping is explicit:

    - rows = parallel query lanes
    - cols = KV-stationary token lanes

    Queries are assigned to horizontal row lanes and move left-to-right across
    the array. K/V storage is shared vertically within a column tile and is
    loaded once per column tile per sequence, not once per head.

    Parameters
    ----------
    H : int
        Number of query heads.
    T : int
        Sequence length (KV cache tokens).
    d : int
        Head dimension.
    array_rows : int
        Number of parallel query row lanes in the array.
    array_cols : int
        Number of token columns per tile.
    bytes_per_element : int
        Bytes per data element (e.g. 2 for FP16).
    memory_bandwidth_bytes_per_cycle : int
        DRAM bandwidth in bytes per hardware clock cycle.
    exp_latency_cycles : int
        Hardware cycles for one exp operation in the upper Running Attention PE.
        Typical values: 4 (LUT), 8-16 (CORDIC). Default 4.
    pe_mac_width : int
        Number of parallel MACs in the upper PE's vector accumulation unit.
        1 = fully serial; d = fully parallel (one cycle for d-wide ops).
        Default 1.
    lower_mac_count : int
        Number of MACs in the lower dot-product unit per PE column.
        Each MAC services a different Q token offset by ceil(d / lower_mac_count)
        cycles, creating a time-interleaved micro-pipeline within the column.
        With lower_mac_count=16 and d=128: offset = 8 cycles, and the column
        accepts a new Q token every max(8, upper_pe) cycles instead of every
        d + upper_pe = 135 cycles.  Requires lower_mac_count <= d.
        Default 1 (original sequential behaviour).
    batch_size : int
        Number of decode sequences processed simultaneously. Each sequence has
        its own KV cache; K/V sharing across heads is per-sequence (MQA).
        Increasing batch_size fills the pipeline and improves PE utilization.
        The pipeline saturates when H * batch_size * head_parallelism >= array_rows * array_cols.
        Default 1.
    head_parallelism : int
        Number of row lanes assigned to each (head, sequence) pair. Each lane
        processes a contiguous T/head_parallelism token range. After all lanes
        finish, their (M, L, O) running states are merged using the log-sum-exp
        merge formula — identical math to FlashAttention's split-K reduction.
        Requires H * batch_size * head_parallelism <= array_rows.
        Benefits: fills the Y-axis of the array, halves per-lane token count
        (reducing drain time per head), and halves the batch size needed for
        pipeline saturation.
        Default 1.
    merge_extensions : int
        Number of inline merge-tree levels added above the KV-stationary array.
        Each level doubles the row count and halves the column count, inserting
        2D merge PEs at the level boundary.  A merge PE receives (M, L, O) state
        from BOTH the left neighbour (regular pipeline) and the top neighbour
        (sister sub-array), applies the online-softmax merge formula inline, and
        forwards the combined state — zero post-pipeline cost.

        Merge PE cost per level:
            upper: 2*exp_latency + ceil(6*d / pe_mac_width)  (two rescalings + add)
            stage: max(lower_mac_throughput, merge_upper_pe)

        A small FIFO sync-buffer is required at each merge junction to absorb
        the skew between the regular PE (cycles_per_stage) and the merge PE
        (merge_stage_cycles).  The total sync buffer grows as
            H * batch * sum(2^(k-1) * skew * (d+2) * bpe  for k in 1..n).

        Optimal setting balances drain overhead and vertical-fill overhead:
            drain = array_cols - 1 = T/2^n - 1
            vfill = H * 2^n - 1
        which are equal when 2^n ≈ sqrt(T / H).  For H=64, T=8192 this gives
        n≈3 (both ≈511 steps, TBOT≈12 280 cycles vs 66 040 without extensions).
        Default 0 (no merge extensions).
    query_tokens : int
        Number of Q tokens per head per sequence flowing through the array.
        1  = decode  (one new token attends over T KV cache entries).
        T  = prefill (all T tokens of the prompt attend over themselves).
        Any value in between is a chunked-prefill or speculative decode step.
        With query_tokens = Q_T, each row lane carries Q_T packets instead of
        1, filling the pipeline and reducing drain% from ~100% toward ~2*array_cols/Q_T.
        DRAM scales accordingly: Q reads = H * batch_size * query_tokens * d.
        Default 1 (decode).
    debug_cycles : int
        Number of first-tile pipeline steps to include in the ASCII trace.
    """
    if min(H, T, d, array_rows, array_cols, bytes_per_element) <= 0:
        raise ValueError("All dimensions and bytes_per_element must be positive.")
    if memory_bandwidth_bytes_per_cycle <= 0:
        raise ValueError("memory_bandwidth_bytes_per_cycle must be positive.")
    if exp_latency_cycles <= 0:
        raise ValueError("exp_latency_cycles must be positive.")
    if pe_mac_width <= 0:
        raise ValueError("pe_mac_width must be positive.")
    if lower_mac_count <= 0 or lower_mac_count > d:
        raise ValueError(f"lower_mac_count must be between 1 and d ({d}).")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if head_parallelism <= 0:
        raise ValueError("head_parallelism must be positive.")
    if merge_extensions < 0:
        raise ValueError("merge_extensions must be >= 0.")
    if query_tokens <= 0:
        raise ValueError("query_tokens must be positive.")
    # Effective head parallelism: head_parallelism post-pipeline splits plus
    # 2^merge_extensions inline merge-tree splits.  All of these need dedicated rows.
    # batch_size beyond that simply time-multiplexes and is handled separately.
    merge_rows_per_head = 2 ** merge_extensions
    effective_hp = head_parallelism * merge_rows_per_head
    if H * effective_hp > array_rows:
        raise ValueError(
            f"H * effective_head_parallelism ({H * effective_hp}) exceeds array_rows "
            f"({array_rows}). Each head needs effective_hp={effective_hp} dedicated row "
            f"lanes (head_parallelism={head_parallelism} × 2^merge_extensions={merge_rows_per_head}). "
            f"Set array_rows >= {H * effective_hp} or reduce head_parallelism/merge_extensions."
        )

    # --- Array layout (compute first — needed for packets_in_flight) ---------
    # Each (head, sequence) pair is split across effective_hp row lanes.
    #   merge_extensions levels: inline merge PEs, zero post-pipeline overhead.
    #   head_parallelism splits:  post-pipeline log-sum-exp merge.
    # Each lane processes a T/effective_hp token range independently.
    tokens_per_lane = math.ceil(T / effective_hp)

    # Post-pipeline merge cost: only the head_parallelism split contributes.
    # merge_extensions splits are resolved inline by the merge PEs.
    merge_cycles_per_result = (head_parallelism - 1) * (
        exp_latency_cycles + math.ceil(2 * d / pe_mac_width)
    )
    total_merge_cycles = H * batch_size * merge_cycles_per_result

    # Dedicated-row mapping: each (head, sequence, lane) triple owns a fixed row.
    # Within that row, all query_tokens Q vectors of that head stream through the
    # columns one after another — offset by 1 column per step (systolic skew).
    # If more dedicated lanes are needed than physical rows exist, lanes are
    # time-multiplexed: one physical row handles ceil(lanes / array_rows) lanes
    # sequentially, each contributing query_tokens packets.
    dedicated_lanes = H * batch_size * effective_hp
    active_query_rows = min(dedicated_lanes, array_rows)
    lanes_per_row = math.ceil(dedicated_lanes / array_rows)   # 1 when lanes <= rows
    max_packets_per_row = query_tokens * lanes_per_row
    token_tiles = math.ceil(tokens_per_lane / array_cols)

    # Saturation check: pipeline is fully utilised when max_packets_per_row >= array_cols
    saturated = max_packets_per_row >= array_cols

    # --- Hardware stage latencies (computed after layout — depends on packets_in_flight) --
    # Each (row, col) PE has lower_mac_count independent serial MACs.  Each MAC
    # handles one Q packet's full d-cycle dot product; all lower_mac_count MACs
    # belong to the same row and are time-interleaved across packets IN THAT ROW.
    #
    # Each MAC needs its own copy of K[t] (reading different elements simultaneously),
    # so on-chip K storage per PE = lower_mac_count × d × bytes_per_element.
    #
    # column_dwell   : Q packet's dwell time per column = d + upper_pe  (always)
    #                  governs both Q's horizontal progress and inter-row K propagation.
    # packet_stagger : initiation interval WITHIN a row = ceil(d / effective_macs)
    #                  only affects rows with max_packets_per_row > 1 (prefill / batching).
    # cycles_per_stage = column_dwell  (alias kept for backward compatibility)
    lower_mac_cycles, upper_pe_cycles, column_dwell, packet_stagger = _compute_cycles_per_stage(
        d, exp_latency_cycles, pe_mac_width, lower_mac_count,
        max_packets_per_row=max_packets_per_row,
    )
    cycles_per_stage = column_dwell   # Q's dwell time at each column (also inter-row rate)

    # On-chip K buffer: each PE stores 1 copy of K[t] (d elements).
    # Access bandwidth = lower_mac_count elements per cycle (lower_mac_count-ported or
    # lower_mac_count-banked register).  Storage is NOT multiplied by lower_mac_count.
    k_buffer_bytes_per_pe = d * bytes_per_element
    total_k_buffer_bytes  = array_rows * array_cols * k_buffer_bytes_per_pe
    k_read_ports_per_pe   = lower_mac_count   # simultaneous K element reads per cycle

    # --- Merge-extension PE timing -------------------------------------------
    # Merge PE: receives (M,L,O) from left neighbour AND top merge-tree neighbour.
    # Operations per stage:
    #   compute m_in = max(m_left, m_top, score)           -- 2 compares
    #   exp(m_top  - m_in), exp(m_left - m_in)             -- 2 exp lookups (parallel)
    #   O_in = O_top*a_top + O_left*a_left + exp_new*V     -- 6*d multiply/add ops
    #   L_in = L_top*a_top + L_left*a_left + exp_new       -- scalar (hidden)
    # Merge PE: receives (M,L,O) from left and top neighbours; no Q·K dot product.
    # Its latency is purely the upper-PE-style state merge (two rescalings + vector add).
    merge_upper_pe_cycles = 2 * exp_latency_cycles + math.ceil(6 * d / pe_mac_width)
    merge_stage_cycles = merge_upper_pe_cycles if merge_extensions > 0 else 0
    # Skew: extra cycles a merge PE takes vs a regular PE (column_dwell).
    skew_per_level = max(0, merge_stage_cycles - column_dwell) if merge_extensions > 0 else 0

    # Sync buffer total: at tree level k (1..n) there are 2^(k-1) junctions per
    # (head × sequence).  Each junction buffers skew_per_level state entries of
    # (d + 2) elements: output vector O (d) + max-so-far M (1) + denom L (1).
    state_bytes_per_entry = (d + 2) * bytes_per_element
    sync_buffer_bytes = (
        H * batch_size * state_bytes_per_entry * skew_per_level
        * sum(2 ** (k - 1) for k in range(1, merge_extensions + 1))
    ) if merge_extensions > 0 else 0

    # --- DRAM traffic (independent of cycle model) ---------------------------
    # Each sequence in the batch has its own KV cache (sequences are independent).
    # Within each sequence, K/V is shared across all H query heads (MQA property).
    # Q reads and output writes scale with query_tokens: prefill reads all Q_T vectors.
    # K/V is still loaded once per sequence regardless of query_tokens.
    query_reads_bytes = H * batch_size * query_tokens * d * bytes_per_element
    output_writes_bytes = H * batch_size * query_tokens * d * bytes_per_element
    kv_load_bytes = batch_size * 2 * T * d * bytes_per_element
    total_dram_bytes = query_reads_bytes + kv_load_bytes + output_writes_bytes

    # --- MAC counts ----------------------------------------------------------
    # Lower MAC: Q·K dot product for every (query_token, head, kv_token, sequence).
    dot_product_macs = H * batch_size * query_tokens * T * d
    # Upper PE: Oout = Oin*exp_old + exp_new*V needs 3*d multiply/add ops per (q, kv).
    value_macs = H * batch_size * query_tokens * T * 3 * d
    total_macs = dot_product_macs + value_macs

    # --- Memory bandwidth cycles --------------------------------------------
    query_io_cycles = math.ceil(query_reads_bytes / memory_bandwidth_bytes_per_cycle)
    output_io_cycles = math.ceil(output_writes_bytes / memory_bandwidth_bytes_per_cycle)

    # --- Compute cycles (pipeline steps → hardware cycles) ------------------
    #
    # Latency vs throughput
    # ─────────────────────
    # THROUGHPUT: once the pipeline is continuously fed, one result exits every
    # cycles_per_stage hardware cycles — regardless of T or array dimensions.
    # This is the correct TBOT when the pipeline is kept full (batching).
    #
    # LATENCY: time from the first Q entering col 0 to the last O exiting the
    # last column.  For a tile with active_cols columns and active_query_rows rows:
    #   pipeline_depth = active_cols + active_query_rows - 1  steps
    #   latency        = pipeline_depth * cycles_per_stage    cycles
    # For a single-token decode pass (query_tokens=1, batch=1), estimated_cycles
    # equals this latency because there is only one token to wait for.
    #
    # The "flush" steps (formerly called "drain") are NOT idle time — the last
    # packet is traversing the remaining columns doing useful dot-product work.
    # Flush is only a scheduling overhead when amortised over few packets.
    #
    # Two-rate pipeline model
    # ──────────────────────
    # column_dwell (= cycles_per_stage) governs both:
    #   1. Q's left-to-right progress  (Q moves to next column after column_dwell cycles)
    #   2. Inter-row stagger           (K propagates one row every column_dwell cycles)
    #
    # packet_stagger governs only WITHIN-ROW Q scheduling:
    #   Each row's PE accepts a new Q packet every packet_stagger cycles.
    #   With lower_mac_count=16 MACs: packet_stagger = 8.
    #   With lower_mac_count=1  MAC:  packet_stagger = column_dwell (no benefit).
    #   Has ZERO effect on single-decode (max_packets_per_row=1).
    #
    # LATENCY (single packet per row):
    #   pipeline_depth_steps × column_dwell   (unchanged from 1-MAC model)
    #
    # TOTAL TILE CYCLES (P packets per row, R rows, C active cols):
    #   (R + C - 1) × column_dwell + (P - 1) × packet_stagger
    #   = pipeline_latency + (max_packets_per_row - 1) × packet_stagger
    #
    # THROUGHPUT (continuously fed pipeline, many packets):
    #   Bottleneck is the slower of: packet_stagger (within-row) or
    #   column_dwell / max_packets_per_row (row-feeding rate).
    #   throughput = max(packet_stagger, ceil(column_dwell / max_packets_per_row))
    #
    throughput_cycles_per_token = max(
        packet_stagger,
        math.ceil(column_dwell / max_packets_per_row),
    )

    total_fill_steps  = 0
    total_steady_steps = 0
    total_flush_steps  = 0
    total_compute_cycles = 0
    total_active_pe_cycles = 0
    total_memory_cycles = 0
    debug_trace: List[str] = []

    # Vertical-K propagation: K[t] ripples down through rows with a 1-step
    # delay per row, extending every tile by (active_query_rows - 1) steps.
    vertical_fill_steps_per_tile = active_query_rows - 1

    # Pipeline depth (per tile): steps from first Q entering col 0 to last Q
    # exiting the last column, including the vertical-K skew.
    # With 1 packet per row: latency = pipeline_depth_steps × column_dwell.
    # packet_stagger adds (max_packets_per_row - 1) × packet_stagger extra cycles
    # for rows that carry multiple Q packets — but this does NOT affect latency
    # for single-decode (max_packets_per_row = 1).
    first_tile_active_cols = min(array_cols, T)
    pipeline_depth_steps = first_tile_active_cols + active_query_rows - 1
    pipeline_latency_cycles = pipeline_depth_steps * column_dwell

    global_step_offset = 0
    for tile_idx in range(token_tiles):
        active_cols = min(array_cols, T - tile_idx * array_cols)
        fill_steps, steady_steps, flush_steps = _pipeline_phase_steps(
            max_packets_per_row,
            active_cols,
        )
        # Two-rate tile cost:
        #   (active_query_rows + active_cols - 1) × column_dwell   [pipeline depth × dwell]
        #     = inter-row K propagation + Q's column traversal, both at column_dwell rate.
        #   + (max_packets_per_row - 1) × packet_stagger            [within-row scheduling]
        #     = extra time for rows with multiple Q packets (zero for single decode).
        tile_total_steps  = fill_steps + steady_steps + flush_steps + vertical_fill_steps_per_tile
        tile_compute_cycles = (
            (active_query_rows + active_cols - 1) * column_dwell
            + (max_packets_per_row - 1) * packet_stagger
        )

        tile_kv_load_bytes = 2 * active_cols * d * bytes_per_element
        tile_memory_cycles = math.ceil(
            tile_kv_load_bytes / memory_bandwidth_bytes_per_cycle
        )

        total_fill_steps   += fill_steps
        total_steady_steps += steady_steps
        total_flush_steps  += flush_steps
        total_compute_cycles += tile_compute_cycles
        total_memory_cycles  += tile_memory_cycles
        # Active PE cycles: each dedicated lane visits each active column once.
        # Each visit costs column_dwell cycles (the Q's dwell time at the column).
        total_active_pe_cycles += dedicated_lanes * query_tokens * active_cols * column_dwell

        if tile_idx == 0:
            trace_steps = min(tile_total_steps, max(0, debug_cycles))
            for local_step in range(trace_steps):
                active_packets: List[Tuple[int, int, int]] = []
                for query_id in range(H):
                    row = query_id % active_query_rows
                    packet_index = query_id // active_query_rows
                    col = local_step - packet_index
                    if 0 <= col < active_cols:
                        active_packets.append((query_id, row, col))

                debug_trace.append(
                    _format_cycle_trace(
                        step_index=global_step_offset + local_step,
                        cycle_index=(global_step_offset + local_step) * cycles_per_stage,
                        cycles_per_stage=cycles_per_stage,
                        active_packets=active_packets,
                        num_rows=active_query_rows,
                        num_cols=active_cols,
                    )
                )

        global_step_offset += tile_total_steps

    memory_service_cycles = query_io_cycles + total_memory_cycles + output_io_cycles
    # Merge runs after the pipeline drains — sequential with compute.
    total_cycles = max(total_compute_cycles, memory_service_cycles) + total_merge_cycles
    pe_capacity_cycles = total_compute_cycles * array_rows * array_cols
    pe_utilization = (
        total_active_pe_cycles / pe_capacity_cycles if pe_capacity_cycles else 0.0
    )
    arithmetic_intensity = total_macs / total_dram_bytes
    # Throughput: total Q token-head pairs processed per hardware cycle.
    query_throughput = H * batch_size * query_tokens / total_cycles

    return {
        "H": H,
        "T": T,
        "d": d,
        "query_tokens": query_tokens,
        "batch_size": batch_size,
        "head_parallelism": head_parallelism,
        "merge_extensions": merge_extensions,
        "merge_rows_per_head": merge_rows_per_head,
        "effective_head_parallelism": effective_hp,
        "tokens_per_lane": tokens_per_lane,
        "merge_upper_pe_cycles": merge_upper_pe_cycles,
        "merge_stage_cycles": merge_stage_cycles,
        "skew_per_level": skew_per_level,
        "sync_buffer_bytes": sync_buffer_bytes,
        "merge_cycles_per_result": merge_cycles_per_result,
        "total_merge_cycles": total_merge_cycles,
        "dedicated_lanes": dedicated_lanes,
        "lanes_per_row": lanes_per_row,
        "max_packets_per_row": max_packets_per_row,
        "array_rows": array_rows,
        "array_cols": array_cols,
        "active_query_rows": active_query_rows,
        "token_tiles": token_tiles,
        "saturated": saturated,
        "bytes_per_element": bytes_per_element,
        "memory_bandwidth_bytes_per_cycle": memory_bandwidth_bytes_per_cycle,
        # --- Latency vs throughput -------------------------------------------
        # throughput_cycles_per_token: sustained rate when the pipeline is
        #   continuously fed (e.g. large batch, prefill).  = row_stagger.
        # pipeline_latency_cycles: time from the first Q entering col 0 to the
        #   last O exiting the last column.
        #   = (active_query_rows-1)*row_stagger + first_tile_active_cols*column_dwell.
        # For decode batch=1: estimated_cycles ≈ pipeline_latency_cycles.
        # For large batches / prefill: throughput_cycles_per_token = row_stagger.
        "throughput_cycles_per_token": throughput_cycles_per_token,
        "pipeline_depth_steps": pipeline_depth_steps,
        "pipeline_latency_cycles": pipeline_latency_cycles,
        # 2D systolic vertical-K propagation (inter-row rate = column_dwell)
        "vertical_fill_steps_per_tile": vertical_fill_steps_per_tile,
        "vertical_fill_cycles_total": vertical_fill_steps_per_tile * token_tiles * column_dwell,
        # Stage latency breakdown — two-rate model
        #   column_dwell    : Q's dwell time per column = d + upper_pe  (governs latency)
        #   packet_stagger  : within-row initiation interval = ceil(d / effective_macs)
        #                     (only matters when max_packets_per_row > 1)
        #   cycles_per_stage = column_dwell  (backward-compat alias)
        "exp_latency_cycles": exp_latency_cycles,
        "pe_mac_width": pe_mac_width,
        "lower_mac_count": lower_mac_count,
        "lower_mac_throughput_cycles": lower_mac_cycles,
        "upper_pe_cycles_per_stage": upper_pe_cycles,
        "column_dwell": column_dwell,
        "packet_stagger": packet_stagger,
        "cycles_per_stage": cycles_per_stage,   # = column_dwell
        # On-chip K buffer cost: each PE needs lower_mac_count copies of K[t]
        # so all MACs can read simultaneously (16 K streams per column PE).
        # DRAM traffic is unchanged; this is local register file area.
        "k_buffer_bytes_per_pe": k_buffer_bytes_per_pe,   # 1× K[t] per PE (not ×lower_mac_count)
        "k_read_ports_per_pe": k_read_ports_per_pe,        # simultaneous K element reads needed
        "total_k_buffer_bytes": total_k_buffer_bytes,
        # MAC counts
        "dot_product_macs": dot_product_macs,
        "value_macs": value_macs,
        "total_macs": total_macs,
        # DRAM traffic
        "query_reads_bytes": query_reads_bytes,
        "kv_load_bytes": kv_load_bytes,
        "output_writes_bytes": output_writes_bytes,
        "total_dram_bytes": total_dram_bytes,
        # Pipeline step counts (× cycles_per_stage = hardware cycles)
        "pipeline_fill_steps": total_fill_steps,
        "steady_state_steps": total_steady_steps,
        "flush_steps": total_flush_steps,
        # Hardware cycle counts (horizontal phases use column_dwell per step)
        "pipeline_fill_cycles": total_fill_steps * column_dwell,
        "steady_state_cycles": total_steady_steps * column_dwell,
        "flush_cycles": total_flush_steps * column_dwell,
        "compute_cycles": total_compute_cycles,
        "memory_service_cycles": memory_service_cycles,
        "total_cycles": total_cycles,
        "estimated_cycles": total_cycles,
        "pe_active_cycles": total_active_pe_cycles,
        "pe_utilization": pe_utilization,
        "arithmetic_intensity": arithmetic_intensity,
        "query_throughput_q_per_cycle": query_throughput,
        "debug_trace": debug_trace,
    }


def kv_stationary_metrics(
    H: int,
    T: int,
    d: int,
    array_rows: int,
    array_cols: int,
    bytes_per_element: int,
    memory_bandwidth_bytes_per_cycle: int,
    exp_latency_cycles: int = 4,
    pe_mac_width: int = 1,
    lower_mac_count: int = 1,
    batch_size: int = 1,
    head_parallelism: int = 1,
    merge_extensions: int = 0,
    query_tokens: int = 1,
) -> Dict[str, object]:
    """Wrapper for the 2D KV-stationary array model."""
    return simulate_2d_kv_stationary_array(
        H=H,
        T=T,
        d=d,
        array_rows=array_rows,
        array_cols=array_cols,
        bytes_per_element=bytes_per_element,
        memory_bandwidth_bytes_per_cycle=memory_bandwidth_bytes_per_cycle,
        exp_latency_cycles=exp_latency_cycles,
        pe_mac_width=pe_mac_width,
        lower_mac_count=lower_mac_count,
        batch_size=batch_size,
        head_parallelism=head_parallelism,
        merge_extensions=merge_extensions,
        query_tokens=query_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate analytical 2D KV-stationary MQA performance."
    )
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--array-rows", type=int, default=64)
    parser.add_argument("--array-cols", type=int, default=64)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--memory-bandwidth-bytes-per-cycle", type=int, default=512)
    parser.add_argument(
        "--exp-latency-cycles",
        type=int,
        default=4,
        help="Hardware cycles for one exp operation in the upper Running Attention PE "
             "(e.g. 4 for LUT-based, 8-16 for CORDIC). Default: 4.",
    )
    parser.add_argument(
        "--pe-mac-width",
        type=int,
        default=1,
        help="Parallel MACs in the upper PE vector accumulation unit. "
             "1 = fully serial; d = fully parallel. Default: 1.",
    )
    parser.add_argument(
        "--lower-mac-count",
        type=int,
        default=1,
        help="Number of interleaved MACs in the lower dot-product unit per column. "
             "Each services a different Q token offset by d/lower_mac_count cycles. "
             "16 reduces cycles_per_stage from 135 to 8 (with d=128, pe_mac_width=128). "
             "Default: 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of decode sequences processed simultaneously. "
             "Increases pipeline utilisation. Saturates at H*B*R >= array_rows*array_cols.",
    )
    parser.add_argument(
        "--head-parallelism",
        type=int,
        default=1,
        help="Row lanes per (head, sequence). Splits the token dimension across rows, "
             "filling the Y-axis and halving per-lane token count. "
             "Requires H * batch_size * head_parallelism <= array_rows.",
    )
    parser.add_argument(
        "--merge-extensions",
        type=int,
        default=0,
        help="Number of inline merge-tree levels. Each level doubles array rows and "
             "halves array cols, inserting merge PEs that combine (M,L,O) state inline "
             "with zero post-pipeline overhead. n=3 is optimal for H=64, T=8192 "
             "(drain = vfill ≈ 511 steps). Default: 0.",
    )
    parser.add_argument(
        "--query-tokens",
        type=int,
        default=1,
        help="Number of Q tokens per head per sequence. "
             "1 = decode (one new token); T = prefill (full sequence). "
             "More query tokens fill the pipeline and reduce drain overhead. Default: 1.",
    )
    parser.add_argument(
        "--debug-cycles",
        type=int,
        default=3,
        help="Number of first-tile pipeline steps to print in the ASCII trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = simulate_2d_kv_stationary_array(
        H=args.heads,
        T=args.seq_len,
        d=args.head_dim,
        array_rows=args.array_rows,
        array_cols=args.array_cols,
        bytes_per_element=args.bytes_per_element,
        memory_bandwidth_bytes_per_cycle=args.memory_bandwidth_bytes_per_cycle,
        exp_latency_cycles=args.exp_latency_cycles,
        pe_mac_width=args.pe_mac_width,
        lower_mac_count=args.lower_mac_count,
        batch_size=args.batch_size,
        head_parallelism=args.head_parallelism,
        merge_extensions=args.merge_extensions,
        query_tokens=args.query_tokens,
        debug_cycles=args.debug_cycles,
    )
    print(json.dumps({k: v for k, v in results.items() if k != "debug_trace"}, indent=2))
    if results["debug_trace"]:
        print("\nASCII pipeline trace:")
        for snapshot in results["debug_trace"]:
            print(snapshot)
            print()


if __name__ == "__main__":
    main()
