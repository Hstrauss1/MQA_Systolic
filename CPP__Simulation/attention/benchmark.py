from __future__ import annotations

import csv
from dataclasses import replace
import sys
from typing import Dict, List, Optional

import torch

from attention.baseline import baseline_mqa_attention
from attention.block_streaming import block_streaming_mqa_attention
from attention.integer_streaming import ExpLUT, integer_streaming_mqa_attention
from attention.streaming import streaming_mqa_attention
from attention.utils import (
    MQAConfig,
    attention_error_metrics,
    benchmark_forward,
    format_metrics,
    generate_random_inputs,
)


def validate_attention_implementations(
    config: MQAConfig,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Compare streaming implementations against the baseline reference.
    """
    q, k, v = generate_random_inputs(config, seed=seed)
    baseline = baseline_mqa_attention(q, k, v, causal=True)
    streaming = streaming_mqa_attention(q, k, v, causal=True)
    block_streaming = block_streaming_mqa_attention(q, k, v, block_size=64, causal=True)
    lut_streaming = integer_streaming_mqa_attention(
        q,
        k,
        v,
        causal=True,
        exp_approx=ExpLUT(device=str(q.device), dtype=q.dtype),
    )

    return {
        "streaming_vs_baseline": attention_error_metrics(baseline, streaming),
        "block_streaming_vs_baseline": attention_error_metrics(baseline, block_streaming),
        "lut_streaming_vs_baseline": attention_error_metrics(baseline, lut_streaming),
    }


def run_benchmarks(
    base_config: MQAConfig,
    seq_lens: List[int] | None = None,
    block_sizes: Optional[List[int]] = None,
    warmup: int = 5,
    iters: int = 25,
) -> List[Dict[str, float]]:
    """
    Benchmark baseline, exact streaming, and LUT streaming attention.
    """
    seq_lens = seq_lens or [128, 256, 512, 1024]
    block_sizes = block_sizes or [16, 32, 64, 128]
    results: List[Dict[str, float]] = []
    device = torch.device(base_config.device)

    for seq_len in seq_lens:
        config = replace(base_config, seq_len=seq_len)
        q, k, v = generate_random_inputs(config, seed=seq_len)

        baseline_out = baseline_mqa_attention(q, k, v, causal=True)

        variants: List[tuple[str, Optional[int], object]] = [
            ("baseline", None, lambda: baseline_mqa_attention(q, k, v, causal=True)),
            ("token_streaming", None, lambda: streaming_mqa_attention(q, k, v, causal=True)),
            ("integer_streaming", None, lambda: integer_streaming_mqa_attention(
                q,
                k,
                v,
                causal=True,
                exp_approx=ExpLUT(device=str(device), dtype=q.dtype),
            )),
        ]

        for block_size in block_sizes:
            variants.append(
                (
                    "block_streaming",
                    block_size,
                    lambda block_size=block_size: block_streaming_mqa_attention(
                        q,
                        k,
                        v,
                        block_size=block_size,
                        causal=True,
                    ),
                )
            )

        for name, block_size, fn in variants:
            metrics = benchmark_forward(fn=fn, seq_len=seq_len, device=device, warmup=warmup, iters=iters)
            output = fn()
            error = attention_error_metrics(baseline_out, output)
            results.append(
                {
                    "mode": name,
                    "seq_len": seq_len,
                    "block_size": block_size,
                    **metrics,
                    **error,
                }
            )

    return results


def print_validation_report(config: MQAConfig, stream: Optional[object] = None) -> None:
    validation = validate_attention_implementations(config)
    stream = stream or sys.stderr
    print("Validation report", file=stream)
    for name, metrics in validation.items():
        print(f"  {name}: {format_metrics(metrics)}", file=stream)


def write_benchmark_csv(base_config: MQAConfig, stream: Optional[object] = None) -> None:
    """
    Emit benchmark rows as CSV so external tooling can ingest the results.
    """
    stream = stream or sys.stdout
    rows = run_benchmarks(base_config)
    fieldnames = [
        "mode",
        "seq_len",
        "block_size",
        "latency_ms",
        "tokens_per_sec",
        "peak_memory_mb",
        "mae",
        "rmse",
        "max_abs_error",
        "cosine_similarity",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main() -> None:
    config = MQAConfig(
        batch_size=1,
        seq_len=128,
        d_model=512,
        num_heads=8,
        d_k=64,
        num_kv_heads=1,
        dtype=torch.float32,
        device="cpu",
    )
    print_validation_report(config, stream=sys.stderr)
    write_benchmark_csv(config)


if __name__ == "__main__":
    main()
