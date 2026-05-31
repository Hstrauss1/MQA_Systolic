#!/usr/bin/env python3
"""Plot Phase 7 integrated MQA sweep and comparison outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path('outputs')
DEFAULT_SWEEP_INPUT_NAME = 'sweep_results.csv'
DEFAULT_COMPARE_INPUT_NAME = 'comparison.csv'
DEFAULT_PLOTS_DIR_NAME = 'plots'


def find_latest_run_dir(output_root: Path) -> Path:
    if not output_root.exists():
        raise FileNotFoundError(f'Output root does not exist: {output_root}')
    candidates = sorted(path for path in output_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f'No run directories found under {output_root}')
    return candidates[-1]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_total_cycles_vs_sequence_length(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, group in df.groupby('mode'):
        grouped = group.groupby('sequence_length', as_index=False)['total_cycles'].mean()
        ax.plot(grouped['sequence_length'], grouped['total_cycles'], marker='o', label=mode)
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Total cycles')
    ax.set_title('Phase 7 total cycles vs sequence length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'total_cycles_vs_sequence_length.png', dpi=200)
    plt.close(fig)


def plot_total_dram_vs_sequence_length(df: pd.DataFrame, output_dir: Path) -> None:
    working = df.copy()
    working['total_dram'] = working['dram_reads'] + working['dram_writes']
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, group in working.groupby('mode'):
        grouped = group.groupby('sequence_length', as_index=False)['total_dram'].mean()
        ax.plot(grouped['sequence_length'], grouped['total_dram'], marker='o', label=mode)
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Total DRAM traffic')
    ax.set_title('Phase 7 total DRAM traffic vs sequence length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'total_dram_vs_sequence_length.png', dpi=200)
    plt.close(fig)


def plot_weighted_util_vs_array_shape(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = df.groupby(['array_shape', 'mode'], as_index=False)['weighted_pe_utilization'].mean()
    pivot = grouped.pivot(index='array_shape', columns='mode', values='weighted_pe_utilization').fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind='bar', ax=ax)
    ax.set_xlabel('Array shape')
    ax.set_ylabel('Weighted PE utilization')
    ax.set_title('Phase 7 weighted PE utilization vs array shape')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'weighted_pe_util_vs_array_shape.png', dpi=200)
    plt.close(fig)


def plot_speedup_vs_sequence_length(compare_df: pd.DataFrame, output_dir: Path) -> None:
    grouped = compare_df.groupby('sequence_length', as_index=False)['baseline_over_kv_cycle_ratio'].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped['sequence_length'], grouped['baseline_over_kv_cycle_ratio'], marker='o', color='tab:green')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Baseline / KV cycle ratio')
    ax.set_title('Phase 7 baseline vs KV cycle ratio')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'baseline_vs_kv_cycle_ratio.png', dpi=200)
    plt.close(fig)


def plot_memory_overhead_breakdown(compare_df: pd.DataFrame, output_dir: Path) -> None:
    grouped = compare_df.groupby('sequence_length', as_index=False)[[
        'baseline_memory_stall_cycles',
        'kv_memory_stall_cycles',
        'kv_preload_bandwidth_cycles',
    ]].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grouped['sequence_length'], grouped['baseline_memory_stall_cycles'], marker='o', label='baseline stall')
    ax.plot(grouped['sequence_length'], grouped['kv_memory_stall_cycles'], marker='o', label='kv stall')
    ax.plot(grouped['sequence_length'], grouped['kv_preload_bandwidth_cycles'], marker='o', label='kv preload bw')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Cycles')
    ax.set_title('Phase 7 memory overhead breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'memory_overhead_breakdown.png', dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot Phase 7 MQA sweep and comparison results.')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--sweep-input', type=Path, default=None)
    parser.add_argument('--compare-input', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(args.output_root)
    sweep_input = args.sweep_input or (run_dir / DEFAULT_SWEEP_INPUT_NAME)
    compare_input = args.compare_input or (run_dir / DEFAULT_COMPARE_INPUT_NAME)
    output_dir = args.output_dir or (run_dir / DEFAULT_PLOTS_DIR_NAME)

    sweep_df = pd.read_csv(sweep_input)
    compare_df = pd.read_csv(compare_input)
    ensure_output_dir(output_dir)

    plot_total_cycles_vs_sequence_length(sweep_df, output_dir)
    plot_total_dram_vs_sequence_length(sweep_df, output_dir)
    plot_weighted_util_vs_array_shape(sweep_df, output_dir)
    plot_speedup_vs_sequence_length(compare_df, output_dir)
    plot_memory_overhead_breakdown(compare_df, output_dir)

    print(f'Phase 7 plots written to {output_dir}')
    print(f'RUN_DIR: {run_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
