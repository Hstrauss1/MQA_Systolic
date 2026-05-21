"""Compare baseline GEMM-style MQA against an analytical KV-stationary model.

The baseline corresponds to the standard two-GEMM decomposition that can be
modeled in SCALE-Sim. The KV-stationary path here is an analytical extension
for comparison only and is not cycle-accurate.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from baseline_mqa_model import baseline_mqa_metrics
from kv_stationary_model import kv_stationary_metrics


DEFAULT_SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096, 8192]
DEFAULT_H = 32
DEFAULT_D = 128
DEFAULT_ARRAY_ROWS = 64
DEFAULT_ARRAY_COLS = 64
DEFAULT_BYTES_PER_ELEMENT = 2
DEFAULT_MEMORY_BANDWIDTH_BYTES_PER_CYCLE = 512
RESULTS_CSV = "results.csv"


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def build_result_row(T: int) -> Dict[str, float]:
    baseline = baseline_mqa_metrics(
        H=DEFAULT_H,
        T=T,
        d=DEFAULT_D,
        array_rows=DEFAULT_ARRAY_ROWS,
        array_cols=DEFAULT_ARRAY_COLS,
        bytes_per_element=DEFAULT_BYTES_PER_ELEMENT,
        memory_bandwidth_bytes_per_cycle=DEFAULT_MEMORY_BANDWIDTH_BYTES_PER_CYCLE,
    )
    kv_stationary = kv_stationary_metrics(
        H=DEFAULT_H,
        T=T,
        d=DEFAULT_D,
        array_rows=DEFAULT_ARRAY_ROWS,
        array_cols=DEFAULT_ARRAY_COLS,
        bytes_per_element=DEFAULT_BYTES_PER_ELEMENT,
        memory_bandwidth_bytes_per_cycle=DEFAULT_MEMORY_BANDWIDTH_BYTES_PER_CYCLE,
    )

    baseline_cycles = baseline["estimated_cycles"]
    kv_cycles = kv_stationary["estimated_cycles"]
    baseline_dram_mb = bytes_to_mb(baseline["total_dram_bytes"])
    kv_dram_mb = bytes_to_mb(kv_stationary["total_dram_bytes"])

    return {
        "T": T,
        "baseline_cycles": baseline_cycles,
        "kv_stationary_cycles": kv_cycles,
        "kv_stationary_fill_cycles": kv_stationary["pipeline_fill_cycles"],
        "kv_stationary_steady_cycles": kv_stationary["steady_state_cycles"],
        "kv_stationary_drain_cycles": kv_stationary["drain_cycles"],
        "baseline_dram_MB": baseline_dram_mb,
        "kv_stationary_dram_MB": kv_dram_mb,
        "memory_reduction_ratio": baseline_dram_mb / kv_dram_mb,
        "speedup_estimate": baseline_cycles / kv_cycles,
        "baseline_AI": baseline["arithmetic_intensity"],
        "kv_stationary_AI": kv_stationary["arithmetic_intensity"],
        "kv_stationary_utilization": kv_stationary["pe_utilization"],
        "kv_stationary_q_per_cycle": kv_stationary["query_throughput_q_per_cycle"],
    }


def format_table(rows: List[Dict[str, float]]) -> str:
    headers = [
        "T",
        "baseline_cycles",
        "kv_stationary_cycles",
        "kv_stationary_fill_cycles",
        "kv_stationary_steady_cycles",
        "kv_stationary_drain_cycles",
        "baseline_dram_MB",
        "kv_stationary_dram_MB",
        "memory_reduction_ratio",
        "speedup_estimate",
        "baseline_AI",
        "kv_stationary_AI",
        "kv_stationary_utilization",
        "kv_stationary_q_per_cycle",
    ]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            [
                f"{int(row['T'])}",
                f"{int(row['baseline_cycles'])}",
                f"{int(row['kv_stationary_cycles'])}",
                f"{int(row['kv_stationary_fill_cycles'])}",
                f"{int(row['kv_stationary_steady_cycles'])}",
                f"{int(row['kv_stationary_drain_cycles'])}",
                f"{row['baseline_dram_MB']:.4f}",
                f"{row['kv_stationary_dram_MB']:.4f}",
                f"{row['memory_reduction_ratio']:.4f}",
                f"{row['speedup_estimate']:.4f}",
                f"{row['baseline_AI']:.4f}",
                f"{row['kv_stationary_AI']:.4f}",
                f"{row['kv_stationary_utilization']:.4f}",
                f"{row['kv_stationary_q_per_cycle']:.6f}",
            ]
        )

    widths = [
        max(len(header), max(len(row[idx]) for row in formatted_rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "  ".join(
        header.ljust(widths[idx]) for idx, header in enumerate(headers)
    )
    separator_line = "  ".join("-" * width for width in widths)
    body_lines = [
        "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        for row in formatted_rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def write_results_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    fieldnames = [
        "T",
        "baseline_cycles",
        "kv_stationary_cycles",
        "kv_stationary_fill_cycles",
        "kv_stationary_steady_cycles",
        "kv_stationary_drain_cycles",
        "baseline_dram_MB",
        "kv_stationary_dram_MB",
        "memory_reduction_ratio",
        "speedup_estimate",
        "baseline_AI",
        "kv_stationary_AI",
        "kv_stationary_utilization",
        "kv_stationary_q_per_cycle",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = [build_result_row(T) for T in DEFAULT_SEQUENCE_LENGTHS]
    print(format_table(rows))
    output_path = Path(RESULTS_CSV)
    write_results_csv(rows, output_path)
    print(f"\nSaved comparison results to {output_path}")


if __name__ == "__main__":
    main()
