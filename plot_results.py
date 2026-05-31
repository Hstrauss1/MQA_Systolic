#!/usr/bin/env python3
"""Plot Phase 5 integrated MQA sweep outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path('phase5_outputs/phase5_sweep_results.csv')
DEFAULT_OUTPUT_DIR = Path('phase5_outputs/plots')


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_total_cycles_vs_sequence_length(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, group in df.groupby('mode'):
        grouped = group.groupby('sequence_length', as_index=False)['total_cycles'].mean()
        ax.plot(grouped['sequence_length'], grouped['total_cycles'], marker='o', label=mode)
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Total cycles')
    ax.set_title('Total cycles vs sequence length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'total_cycles_vs_sequence_length.png', dpi=200)
    plt.close(fig)


def plot_amortized_preload_vs_decode_tokens(df: pd.DataFrame, output_dir: Path) -> None:
    kv = df[df['mode'] == 'kv_stationary_mqa_decode'].copy()
    if kv.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = kv.groupby('decode_tokens', as_index=False)['amortized_preload_bytes_per_token'].mean()
    ax.plot(grouped['decode_tokens'], grouped['amortized_preload_bytes_per_token'], marker='o', color='tab:orange')
    ax.set_xlabel('Decode tokens')
    ax.set_ylabel('Amortized preload bytes / token')
    ax.set_title('KV preload amortization vs decode tokens')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'amortized_preload_vs_decode_tokens.png', dpi=200)
    plt.close(fig)


def plot_dram_reads_vs_sequence_length(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, group in df.groupby('mode'):
        grouped = group.groupby('sequence_length', as_index=False)['dram_reads'].mean()
        ax.plot(grouped['sequence_length'], grouped['dram_reads'], marker='o', label=mode)
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('DRAM reads')
    ax.set_title('DRAM reads vs sequence length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'dram_reads_vs_sequence_length.png', dpi=200)
    plt.close(fig)


def plot_weighted_util_vs_array_shape(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = df.groupby(['array_shape', 'mode'], as_index=False)['weighted_pe_utilization'].mean()
    pivot = grouped.pivot(index='array_shape', columns='mode', values='weighted_pe_utilization').fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind='bar', ax=ax)
    ax.set_xlabel('Array shape')
    ax.set_ylabel('Weighted PE utilization')
    ax.set_title('Weighted PE utilization vs array shape')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'weighted_pe_util_vs_array_shape.png', dpi=200)
    plt.close(fig)


def plot_speedup_vs_sequence_length(df: pd.DataFrame, output_dir: Path) -> None:
    baseline = df[df['mode'] == 'baseline_mqa_decode'].copy()
    kv = df[df['mode'] == 'kv_stationary_mqa_decode'].copy()
    join_keys = ['experiment_id', 'sequence_length', 'decode_tokens', 'array_shape']
    merged = baseline.merge(kv, on=join_keys, suffixes=('_baseline', '_kv'))
    if merged.empty:
        return
    merged['baseline_over_kv_speedup'] = merged['total_cycles_baseline'] / merged['total_cycles_kv']
    grouped = merged.groupby('sequence_length', as_index=False)['baseline_over_kv_speedup'].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped['sequence_length'], grouped['baseline_over_kv_speedup'], marker='o', color='tab:green')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Baseline / KV total-cycle ratio')
    ax.set_title('Baseline vs KV speedup across sequence length')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'baseline_vs_kv_speedup_vs_sequence_length.png', dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot integrated Phase 5 MQA results.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    ensure_output_dir(args.output_dir)

    plot_total_cycles_vs_sequence_length(df, args.output_dir)
    plot_amortized_preload_vs_decode_tokens(df, args.output_dir)
    plot_dram_reads_vs_sequence_length(df, args.output_dir)
    plot_weighted_util_vs_array_shape(df, args.output_dir)
    plot_speedup_vs_sequence_length(df, args.output_dir)

    print(f'Plots written to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
