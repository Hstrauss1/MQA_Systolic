from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class MQAConfig:
    """Configuration for a simple Multi-Query Attention experiment."""

    batch_size: int = 1
    seq_len: int = 128
    d_model: int = 512
    num_heads: int = 8
    d_k: int = 64
    num_kv_heads: int = 1
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.d_k

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.d_k

    def validate(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped MQA sharing.")
        if self.d_model != self.q_dim:
            raise ValueError("This prototype assumes d_model == num_heads * d_k.")


def make_causal_mask(
    query_len: int,
    key_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Return a causal mask with shape [query_len, key_len].

    Valid positions contain True. Invalid future positions contain False.
    """
    q_pos = torch.arange(query_len, device=device).unsqueeze(1)
    k_pos = torch.arange(key_len, device=device).unsqueeze(0)
    return k_pos <= q_pos


def generate_random_inputs(config: MQAConfig, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate Q/K/V tensors for MQA experiments.

    Q shape: [batch, seq_len, num_heads, d_k]
    K shape: [batch, seq_len, num_kv_heads, d_k]
    V shape: [batch, seq_len, num_kv_heads, d_k]
    """
    config.validate()
    generator = torch.Generator(device=config.device)
    generator.manual_seed(seed)

    q = torch.randn(
        config.batch_size,
        config.seq_len,
        config.num_heads,
        config.d_k,
        dtype=config.dtype,
        device=config.device,
        generator=generator,
    )
    k = torch.randn(
        config.batch_size,
        config.seq_len,
        config.num_kv_heads,
        config.d_k,
        dtype=config.dtype,
        device=config.device,
        generator=generator,
    )
    v = torch.randn(
        config.batch_size,
        config.seq_len,
        config.num_kv_heads,
        config.d_k,
        dtype=config.dtype,
        device=config.device,
        generator=generator,
    )
    return q, k, v


def generate_random_inputs_with_lengths(
    config: MQAConfig,
    q_len: Optional[int] = None,
    kv_len: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate Q/K/V tensors while allowing q_len and kv_len to differ.

    Q shape: [batch, q_len, num_heads, d_k]
    K shape: [batch, kv_len, num_kv_heads, d_k]
    V shape: [batch, kv_len, num_kv_heads, d_k]
    """
    config.validate()
    q_len = q_len if q_len is not None else config.seq_len
    kv_len = kv_len if kv_len is not None else config.seq_len

    generator = torch.Generator(device=config.device)
    generator.manual_seed(seed)

    q = torch.randn(
        config.batch_size,
        q_len,
        config.num_heads,
        config.d_k,
        dtype=config.dtype,
        device=config.device,
        generator=generator,
    )
    k = torch.randn(
        config.batch_size,
        kv_len,
        config.num_kv_heads,
        config.d_k,
        dtype=config.dtype,
        device=config.device,
        generator=generator,
    )
    v = torch.randn(
        config.batch_size,
        kv_len,
        config.num_kv_heads,
        config.d_k,
        dtype=config.dtype,
        device=config.device,
        generator=generator,
    )
    return q, k, v


def repeat_kv_for_mqa(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Repeat shared KV heads so each query head has an aligned KV head.

    Input shape:  [batch, seq_len, num_kv_heads, d_k]
    Output shape: [batch, seq_len, num_heads, d_k]
    """
    batch, seq_len, num_kv_heads, d_k = x.shape
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads.")
    group_size = num_heads // num_kv_heads
    return x.repeat_interleave(group_size, dim=2).reshape(batch, seq_len, num_heads, d_k)


def safe_exp_difference(
    x: torch.Tensor,
    y: torch.Tensor,
    exp_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Compute exp_fn(x - y) while mapping non-finite deltas to zero.

    This is important for masked attention states where both x and y may be
    -inf, which would otherwise produce nan in the subtraction.
    """
    delta = x - y
    out = torch.zeros_like(delta)
    valid = torch.isfinite(delta)
    if valid.any():
        out[valid] = exp_fn(delta[valid])
    return out


def attention_error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    """Compute scalar error metrics between two tensors of identical shape."""
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")

    diff = (reference - candidate).detach().float()
    ref = reference.detach().float().reshape(-1)
    cand = candidate.detach().float().reshape(-1)

    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff * diff).mean()).item()
    max_abs_error = diff.abs().max().item()
    cosine_similarity = torch.nn.functional.cosine_similarity(ref, cand, dim=0).item()

    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "cosine_similarity": cosine_similarity,
    }


def maybe_synchronize(device: torch.device) -> None:
    """Synchronize when the selected device is CUDA so timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float:
    """
    Return peak allocated memory in MB.

    CUDA is measured with PyTorch's built-in memory stats.
    CPU peak memory is not tracked here and returns NaN for clarity.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    return float("nan")


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def benchmark_forward(
    fn: Callable[[], torch.Tensor],
    seq_len: int,
    device: torch.device,
    warmup: int = 5,
    iters: int = 25,
) -> Dict[str, float]:
    """Benchmark a forward function and report latency and throughput."""
    for _ in range(warmup):
        _ = fn()
    maybe_synchronize(device)

    reset_peak_memory(device)
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    maybe_synchronize(device)
    elapsed = time.perf_counter() - start

    avg_latency_s = elapsed / iters
    tokens_per_sec = seq_len / avg_latency_s
    return {
        "latency_ms": avg_latency_s * 1_000.0,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": peak_memory_mb(device),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    pieces = []
    for key, value in metrics.items():
        if math.isnan(value):
            pieces.append(f"{key}=nan")
        else:
            pieces.append(f"{key}={value:.6f}")
    return ", ".join(pieces)
