from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from attention.streaming import streaming_mqa_attention


@dataclass
class ExpLUT:
    """
    Simple lookup-table approximation for exp(x) over a bounded interval.

    This is intentionally minimal and modular so later experiments can swap in:
    - tighter quantization schemes
    - piecewise-linear approximations
    - fixed-point kernels
    """

    xmin: float = -16.0
    xmax: float = 0.0
    num_entries: int = 4096
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        grid = torch.linspace(self.xmin, self.xmax, self.num_entries, device=self.device, dtype=self.dtype)
        self.grid = grid
        self.values = torch.exp(grid)
        self.step = (self.xmax - self.xmin) / (self.num_entries - 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        clipped = x.clamp(self.xmin, self.xmax)
        index = torch.round((clipped - self.xmin) / self.step).long()
        return self.values[index]


def integer_streaming_mqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    exp_approx: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Streaming MQA path that swaps torch.exp for an approximation interface.

    The surrounding online-softmax logic is unchanged, which keeps correctness
    comparisons straightforward while enabling future quantized experiments.
    """
    if exp_approx is None:
        exp_approx = ExpLUT(device=str(q.device), dtype=q.dtype)
    return streaming_mqa_attention(q=q, k=k, v=v, causal=causal, exp_fn=exp_approx)
