"""Analytical 2D KV-stationary systolic-style attention model for MQA decode.

This is a custom architectural model and does not claim cycle accuracy against
SCALE-Sim. The intent is to model a true 2D array where:

- rows = parallel query lanes
- cols = KV-stationary token lanes

Each processing element PE[row][col] consumes a query-state packet moving
left-to-right while K/V data remains stationary within the column.
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


def _pipeline_phase_cycles(packet_count: int, active_cols: int) -> Tuple[int, int, int]:
    """Split a horizontal packet pipeline into fill, steady, and drain cycles."""
    if packet_count <= 0 or active_cols <= 0:
        return 0, 0, 0

    if packet_count >= active_cols:
        fill_cycles = active_cols - 1
        steady_state_cycles = packet_count - active_cols + 1
        drain_cycles = active_cols - 1
        return fill_cycles, steady_state_cycles, drain_cycles

    fill_cycles = packet_count - 1
    steady_state_cycles = 1
    drain_cycles = active_cols - 1
    return fill_cycles, steady_state_cycles, drain_cycles


def _format_cycle_trace(
    cycle_index: int,
    active_packets: Sequence[Tuple[int, int, int]],
    num_rows: int,
    num_cols: int,
) -> str:
    """Create a compact ASCII snapshot of query packet locations."""
    lines = [f"Cycle {cycle_index}:"]
    if active_packets:
        for query_id, row, col in active_packets:
            lines.append(f"Q{query_id} -> PE({row},{col})")
    else:
        lines.append("(idle)")

    occupancy = [[".." for _ in range(num_cols)] for _ in range(num_rows)]
    for query_id, row, col in active_packets:
        occupancy[row][col] = f"Q{query_id}"

    lines.append("Occupancy:")
    for row_idx, row_cells in enumerate(occupancy):
        lines.append(f"row{row_idx}: " + " ".join(row_cells))
    return "\n".join(lines)


def simulate_2d_kv_stationary_array(
    H: int,
    T: int,
    d: int,
    array_rows: int,
    array_cols: int,
    bytes_per_element: int,
    memory_bandwidth_bytes_per_cycle: int,
    debug_cycles: int = 3,
) -> Dict[str, object]:
    """Simulate a 2D systolic-style KV-stationary attention array.

    The 2D mapping is explicit:

    - rows = parallel query lanes
    - cols = KV-stationary token lanes

    Queries are assigned to horizontal row lanes and move left-to-right across
    the array. K/V storage is shared vertically within a column tile and is
    loaded once per column tile, not once per row.
    """
    if min(H, T, d, array_rows, array_cols, bytes_per_element) <= 0:
        raise ValueError("All dimensions and bytes_per_element must be positive.")
    if memory_bandwidth_bytes_per_cycle <= 0:
        raise ValueError("memory_bandwidth_bytes_per_cycle must be positive.")

    active_query_rows = min(H, array_rows)
    token_tiles = math.ceil(T / array_cols)
    query_packets_per_row = _queries_per_row(H, array_rows)
    max_packets_per_row = max(query_packets_per_row)

    total_fill_cycles = 0
    total_steady_state_cycles = 0
    total_drain_cycles = 0
    total_compute_cycles = 0
    total_active_pe_cycles = 0
    total_memory_cycles = 0
    debug_trace: List[str] = []

    query_reads_bytes = H * d * bytes_per_element
    output_writes_bytes = H * d * bytes_per_element
    kv_load_bytes = 2 * T * d * bytes_per_element
    total_dram_bytes = query_reads_bytes + kv_load_bytes + output_writes_bytes

    # Each PE visit corresponds to one query/token interaction. The d-wide dot
    # product and d-wide value accumulation are tracked as MAC counts.
    dot_product_macs = H * T * d
    value_macs = H * T * d
    total_macs = dot_product_macs + value_macs

    query_io_cycles = math.ceil(query_reads_bytes / memory_bandwidth_bytes_per_cycle)
    output_io_cycles = math.ceil(
        output_writes_bytes / memory_bandwidth_bytes_per_cycle
    )

    global_cycle_offset = 0
    for tile_idx in range(token_tiles):
        active_cols = min(array_cols, T - tile_idx * array_cols)
        fill_cycles, steady_state_cycles, drain_cycles = _pipeline_phase_cycles(
            max_packets_per_row,
            active_cols,
        )
        tile_compute_cycles = fill_cycles + steady_state_cycles + drain_cycles
        tile_kv_load_bytes = 2 * active_cols * d * bytes_per_element
        tile_memory_cycles = math.ceil(
            tile_kv_load_bytes / memory_bandwidth_bytes_per_cycle
        )

        total_fill_cycles += fill_cycles
        total_steady_state_cycles += steady_state_cycles
        total_drain_cycles += drain_cycles
        total_compute_cycles += tile_compute_cycles
        total_memory_cycles += tile_memory_cycles
        total_active_pe_cycles += H * active_cols

        if tile_idx == 0:
            trace_cycles = min(tile_compute_cycles, max(0, debug_cycles))
            for local_cycle in range(trace_cycles):
                active_packets: List[Tuple[int, int, int]] = []
                for query_id in range(H):
                    row = query_id % active_query_rows
                    packet_index = query_id // active_query_rows
                    col = local_cycle - packet_index
                    if 0 <= col < active_cols:
                        active_packets.append((query_id, row, col))

                debug_trace.append(
                    _format_cycle_trace(
                        cycle_index=global_cycle_offset + local_cycle,
                        active_packets=active_packets,
                        num_rows=active_query_rows,
                        num_cols=active_cols,
                    )
                )

        global_cycle_offset += tile_compute_cycles

    memory_service_cycles = query_io_cycles + total_memory_cycles + output_io_cycles
    total_cycles = total_compute_cycles + memory_service_cycles
    pe_capacity_cycles = total_compute_cycles * array_rows * array_cols
    pe_utilization = (
        total_active_pe_cycles / pe_capacity_cycles if pe_capacity_cycles else 0.0
    )
    arithmetic_intensity = total_macs / total_dram_bytes
    query_throughput = H / total_cycles

    return {
        "H": H,
        "T": T,
        "d": d,
        "array_rows": array_rows,
        "array_cols": array_cols,
        "active_query_rows": active_query_rows,
        "token_tiles": token_tiles,
        "query_packets_per_row": query_packets_per_row,
        "bytes_per_element": bytes_per_element,
        "memory_bandwidth_bytes_per_cycle": memory_bandwidth_bytes_per_cycle,
        "dot_product_macs": dot_product_macs,
        "value_macs": value_macs,
        "total_macs": total_macs,
        "query_reads_bytes": query_reads_bytes,
        "kv_load_bytes": kv_load_bytes,
        "output_writes_bytes": output_writes_bytes,
        "total_dram_bytes": total_dram_bytes,
        "pipeline_fill_cycles": total_fill_cycles,
        "steady_state_cycles": total_steady_state_cycles,
        "drain_cycles": total_drain_cycles,
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
) -> Dict[str, object]:
    """Backward-compatible wrapper for the 2D KV-stationary array model."""
    return simulate_2d_kv_stationary_array(
        H=H,
        T=T,
        d=d,
        array_rows=array_rows,
        array_cols=array_cols,
        bytes_per_element=bytes_per_element,
        memory_bandwidth_bytes_per_cycle=memory_bandwidth_bytes_per_cycle,
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
        "--debug-cycles",
        type=int,
        default=3,
        help="Number of first-tile cycles to print in the ASCII pipeline trace.",
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
        debug_cycles=args.debug_cycles,
    )
    print(json.dumps(results, indent=2))
    if results["debug_trace"]:
        print("\nASCII pipeline trace:")
        for snapshot in results["debug_trace"]:
            print(snapshot)
            print()


if __name__ == "__main__":
    main()
