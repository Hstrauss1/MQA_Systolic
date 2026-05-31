"""Numerical correctness check for KV-stationary MQA.

This script verifies that the streaming online-softmax formulation used by the
KV-stationary analytical model is numerically equivalent to standard NumPy
attention. It does not validate SCALE-Sim output.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def reference_mqa(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference MQA decode using standard dense attention."""
    scores = Q @ K.T
    attn = softmax(scores, axis=-1)
    out = attn @ V
    return out


def kv_stationary_mqa(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Streaming online-softmax MQA.

    Each query head is processed token-by-token, updating the running max,
    normalization term, and output accumulator. This mirrors the intended
    numerical behavior of a KV-stationary streaming implementation.
    """
    num_heads, head_dim = Q.shape
    seq_len = K.shape[0]

    output = np.zeros((num_heads, head_dim), dtype=np.float64)

    for h in range(num_heads):
        m = -np.inf
        l = 0.0
        o = np.zeros(head_dim, dtype=np.float64)

        for i in range(seq_len):
            s = float(np.dot(Q[h], K[i]))
            m_new = max(m, s)
            scale_old = 0.0 if np.isneginf(m) else float(np.exp(m - m_new))
            scale_new = float(np.exp(s - m_new))

            o = o * scale_old + scale_new * V[i]
            l = l * scale_old + scale_new
            m = m_new

        output[h] = o / l

    return output.astype(Q.dtype, copy=False)


def generate_random_inputs(
    num_heads: int, seq_len: int, head_dim: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate reproducible random test inputs."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((num_heads, head_dim), dtype=np.float32)
    K = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    V = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    return Q, K, V


def run_correctness_test(
    num_heads: int, seq_len: int, head_dim: int, seed: int
) -> float:
    """Run the numerical equivalence test and return max absolute error."""
    Q, K, V = generate_random_inputs(num_heads, seq_len, head_dim, seed)
    reference = reference_mqa(Q, K, V)
    kv_stationary = kv_stationary_mqa(Q, K, V)

    max_abs_error = float(np.max(np.abs(reference - kv_stationary)))
    passed = np.allclose(reference, kv_stationary, rtol=1e-4, atol=1e-4)

    print(f"H={num_heads}, T={seq_len}, d={head_dim}, seed={seed}")
    print(f"Max absolute error: {max_abs_error:.8e}")
    print(f"Allclose(rtol=1e-4, atol=1e-4): {passed}")

    if not passed:
        raise AssertionError(
            "KV-stationary online-softmax result does not match reference MQA."
        )

    return max_abs_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check numerical correctness of KV-stationary MQA."
    )
    parser.add_argument("--heads", type=int, default=32, help="Number of query heads.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length.")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_correctness_test(args.heads, args.seq_len, args.head_dim, args.seed)


if __name__ == "__main__":
    main()
