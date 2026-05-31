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
            raise ValueError('dram_bandwidth must be provided when bandwidth_mode="user"')

    @property
    def kv_sharing_ratio(self) -> float:
        return self.query_heads / self.kv_heads

    @property
    def array_shape(self) -> tuple[int, int]:
        return self.array_rows, self.array_cols

    def to_dict(self) -> Dict[str, object]:
        return {
            'mode': self.mode,
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'query_heads': self.query_heads,
            'kv_heads': self.kv_heads,
            'head_dim': self.head_dim,
            'precision': self.precision,
            'array_rows': self.array_rows,
            'array_cols': self.array_cols,
            'ifmap_sram_kb': self.ifmap_sram_kb,
            'filter_sram_kb': self.filter_sram_kb,
            'ofmap_sram_kb': self.ofmap_sram_kb,
            'bandwidth_mode': self.bandwidth_mode,
            'dram_bandwidth': self.dram_bandwidth,
            'decode_tokens': self.decode_tokens,
            'decode_step': self.decode_step,
            'softmax_variant': self.softmax_variant,
            'exp_variant': self.exp_variant,
            'metadata': dict(self.metadata),
        }