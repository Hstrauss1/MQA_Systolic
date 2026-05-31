from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

MQAMode = Literal['baseline_mqa_decode', 'kv_stationary_mqa_decode']
Precision = Literal['int8', 'fp16', 'fp32']
BandwidthMode = Literal['calc', 'user']


@dataclass(slots=True)
class MQAWorkload:
    """Canonical workload definition for decode-time MQA experiments."""

    mode: MQAMode
    sequence_length: int
    batch_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    precision: Precision = 'int8'
    array_rows: int = 16
    array_cols: int = 16
    ifmap_sram_kb: int = 64
    filter_sram_kb: int = 64
    ofmap_sram_kb: int = 64
    bandwidth_mode: BandwidthMode = 'calc'
    dram_bandwidth: Optional[float] = None
    decode_tokens: int = 1
    decode_step: Optional[int] = None
    softmax_variant: str = 'online'
    exp_variant: str = 'lookup'
    kv_block_size: Optional[int] = None
    stream_group_rows: int = 1
    pipeline_depth_override: Optional[int] = None
    reuse_kv_across_tokens: bool = True
    softmax_state_precision: Optional[Precision] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in ('baseline_mqa_decode', 'kv_stationary_mqa_decode'):
            raise ValueError(f'Unsupported MQA mode: {self.mode}')
        for name in ('sequence_length', 'batch_size', 'query_heads', 'kv_heads', 'head_dim', 'array_rows', 'array_cols'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.decode_tokens <= 0:
            raise ValueError('decode_tokens must be positive')
        if self.bandwidth_mode == 'user' and self.dram_bandwidth is None:
            raise ValueError('dram_bandwidth must be set when bandwidth_mode="user"')
        if self.bandwidth_mode == 'calc' and self.dram_bandwidth is not None and self.dram_bandwidth <= 0:
            raise ValueError('dram_bandwidth must be positive when provided')
        if self.query_heads < self.kv_heads:
            raise ValueError('query_heads must be >= kv_heads for MQA/GQA style decode')
        if self.query_heads % self.kv_heads != 0:
            raise ValueError('query_heads must be divisible by kv_heads')
        if self.kv_block_size is not None and self.kv_block_size <= 0:
            raise ValueError('kv_block_size must be positive when provided')
        if self.stream_group_rows <= 0:
            raise ValueError('stream_group_rows must be positive')
        if self.pipeline_depth_override is not None and self.pipeline_depth_override <= 0:
            raise ValueError('pipeline_depth_override must be positive when provided')

    @property
    def heads_per_kv_group(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def effective_softmax_state_precision(self) -> Precision:
        return self.softmax_state_precision or self.precision

    def precision_bytes(self) -> int:
        return {
            'int8': 1,
            'fp16': 2,
            'fp32': 4,
        }[self.precision]

    def softmax_state_precision_bytes(self) -> int:
        return {
            'int8': 1,
            'fp16': 2,
            'fp32': 4,
        }[self.effective_softmax_state_precision]
