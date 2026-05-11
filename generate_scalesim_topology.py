"""Generate a SCALE-Sim GEMM topology for baseline MQA.

This file writes the baseline two-GEMM mapping used with SCALE-Sim's `-i gemm`
mode. It does not attempt to encode the custom KV-stationary architecture.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_OUTPUT = "mqa_baseline_gemm.csv"


def write_topology_csv(H: int, T: int, d: int, output_path: Path) -> None:
    """Write the baseline MQA GEMM topology in M, N, K format."""
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # SCALE-Sim's GEMM parser drops the final CSV field on each row, and the
        # repo's existing MNK topologies therefore include a trailing empty
        # column. We match that format here for direct compatibility.
        writer.writerow(["Layer", "M", "N", "K", ""])
        writer.writerow(["QK_scores", H, T, d, ""])
        writer.writerow(["AV_output", H, d, T, ""])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a SCALE-Sim GEMM topology CSV for baseline MQA."
    )
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    write_topology_csv(args.heads, args.seq_len, args.head_dim, output_path)
    print(f"Wrote SCALE-Sim GEMM topology to {output_path}")


if __name__ == "__main__":
    main()
