"""Plot MQA comparison results from results.csv using matplotlib only."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_CSV = Path("results.csv")
OUTPUT_PNG = Path("results.png")


def load_results(path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {
        "T": [],
        "baseline_cycles": [],
        "kv_stationary_cycles": [],
        "baseline_dram_MB": [],
        "kv_stationary_dram_MB": [],
        "baseline_AI": [],
        "kv_stationary_AI": [],
    }

    with path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))

    return data


def plot_results(data: Dict[str, List[float]], output_path: Path) -> None:
    seq = data["T"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(seq, data["baseline_cycles"], marker="o", label="Baseline GEMM")
    axes[0].plot(seq, data["kv_stationary_cycles"], marker="s", label="KV-stationary")
    axes[0].set_title("Cycles vs Sequence Length")
    axes[0].set_xlabel("Sequence Length (T)")
    axes[0].set_ylabel("Estimated Cycles")
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(seq, data["baseline_dram_MB"], marker="o", label="Baseline GEMM")
    axes[1].plot(seq, data["kv_stationary_dram_MB"], marker="s", label="KV-stationary")
    axes[1].set_title("DRAM Traffic vs Sequence Length")
    axes[1].set_xlabel("Sequence Length (T)")
    axes[1].set_ylabel("DRAM Traffic (MB)")
    axes[1].set_xscale("log", base=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(seq, data["baseline_AI"], marker="o", label="Baseline GEMM")
    axes[2].plot(seq, data["kv_stationary_AI"], marker="s", label="KV-stationary")
    axes[2].set_title("Arithmetic Intensity vs Sequence Length")
    axes[2].set_xlabel("Sequence Length (T)")
    axes[2].set_ylabel("Arithmetic Intensity (MACs/byte)")
    axes[2].set_xscale("log", base=2)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main() -> None:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            "results.csv not found. Run `python compare.py` before plotting."
        )

    data = load_results(RESULTS_CSV)
    plot_results(data, OUTPUT_PNG)


if __name__ == "__main__":
    main()
