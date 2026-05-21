from __future__ import annotations

import argparse
import time

import torch

from attention.baseline import baseline_mqa_attention
from attention.integer_streaming import ExpLUT, integer_streaming_mqa_attention
from attention.streaming import streaming_mqa_attention
from attention.utils import MQAConfig, attention_error_metrics, generate_random_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MQA attention forward pass.")
    parser.add_argument("--mode", choices=["baseline", "streaming", "integer"], required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    if args.device == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 is only supported on CUDA in this prototype runner.")
    if args.d_model != args.num_heads * args.d_k:
        raise ValueError("This prototype assumes d_model == num_heads * d_k.")

    config = MQAConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_k=args.d_k,
        num_kv_heads=1,
        dtype=dtype,
        device=args.device,
    )
    q, k, v = generate_random_inputs(config, seed=0)

    if args.mode == "baseline":
        fn = lambda: baseline_mqa_attention(q, k, v, causal=args.causal)
    elif args.mode == "streaming":
        fn = lambda: streaming_mqa_attention(q, k, v, causal=args.causal)
    else:
        fn = lambda: integer_streaming_mqa_attention(
            q,
            k,
            v,
            causal=args.causal,
            exp_approx=ExpLUT(device=args.device, dtype=dtype),
        )

    maybe_synchronize(torch.device(args.device))
    start = time.perf_counter()
    output = fn()
    maybe_synchronize(torch.device(args.device))
    latency_s = time.perf_counter() - start

    print(f"mode={args.mode}")
    print(f"output_shape={tuple(output.shape)}")
    print(f"latency_ms={latency_s * 1000.0:.6f}")
    print(f"tokens_per_sec={args.seq_len / latency_s:.6f}")

    if args.mode != "baseline":
        baseline = baseline_mqa_attention(q, k, v, causal=args.causal)
        error = attention_error_metrics(baseline, output)
        print("error_vs_baseline")
        for key, value in error.items():
            print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
