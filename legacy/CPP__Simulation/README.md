# Streaming MQA Prototype

This repository contains a small PyTorch research prototype for experimenting with Multi-Query Attention (MQA) inference kernels that avoid materializing the full attention matrix.

## Architecture

The code is organized under [attention/](/Users/hudsons/Code/anything/testFiles/attention):

- [attention/utils.py](/Users/hudsons/Code/anything/testFiles/attention/utils.py): shared config, random input generation, benchmarking helpers, and numerical error metrics
- [attention/baseline.py](/Users/hudsons/Code/anything/testFiles/attention/baseline.py): correctness reference using materialized `QK^T`
- [attention/streaming.py](/Users/hudsons/Code/anything/testFiles/attention/streaming.py): token-by-token online-softmax streaming MQA
- [attention/block_streaming.py](/Users/hudsons/Code/anything/testFiles/attention/block_streaming.py): block-streaming MQA that carries online-softmax state across KV blocks
- [attention/integer_streaming.py](/Users/hudsons/Code/anything/testFiles/attention/integer_streaming.py): modular LUT-backed `exp()` approximation path
- [attention/benchmark.py](/Users/hudsons/Code/anything/testFiles/attention/benchmark.py): validation and CSV benchmark sweep
- [attention/tests.py](/Users/hudsons/Code/anything/testFiles/attention/tests.py): pytest-compatible correctness tests

## Why Streaming Avoids Materialized Attention

The baseline implementation forms a score tensor with shape `[B, H, Q, K]`. That is simple and correct, but memory grows quadratically with sequence length because every query-key score is stored at once.

The streaming implementations avoid this by keeping only running softmax state per query/head:

- `m`: running maximum score
- `l`: running softmax denominator in shifted form
- `o`: running weighted value accumulator

For each new key or KV block, they update:

```text
m_out = max(m_in, s)
l_out = l_in * exp(m_in - m_out) + sum(exp(s - m_out))
o_out = o_in * exp(m_in - m_out) + sum(V * exp(s - m_out))
```

Because only the current token or current KV block is resident, the implementation never allocates the full `[B, H, Q, K]` attention matrix in the streaming kernels.

## Mapping To KV-Stationary Systolic Attention

Conceptually, the streaming kernels are closer to an accelerator-friendly dataflow than the baseline:

- KV-stationary means keys and values are streamed in and reused while query-side accumulators stay live.
- The token-streaming kernel is the simplest form: one KV position arrives, contributes to all active query/head accumulators, then is retired.
- The block-streaming kernel is a more realistic software stepping stone for Triton or CUDA because it amortizes launch and memory overhead over a KV tile while still avoiding full attention materialization.
- The online-softmax state `(m, l, o)` is exactly the compact state that a systolic or tiled implementation would need to preserve across tiles.

This makes the prototype useful both for PyTorch benchmarking and for reasoning about future mappings into custom kernels or accelerator simulators such as SCALE-Sim.

## Running One Forward Pass

Use [run.py](/Users/hudsons/Code/anything/testFiles/run.py):

```bash
python3 run.py --mode baseline --seq-len 128 --device cpu --dtype fp32 --causal
python3 run.py --mode streaming --seq-len 128 --device cpu --dtype fp32 --causal
python3 run.py --mode integer --seq-len 128 --device cpu --dtype fp32 --causal
```

The runner prints:

- output shape
- forward latency
- tokens/sec
- error versus the baseline reference for non-baseline modes

## Running Tests

```bash
pytest attention/tests.py
```

The tests cover:

- baseline output shape
- streaming numerical closeness to the baseline
- causal masking correctness
- integer/LUT execution
- `seq_len = 1`
- `q_len != kv_len`

## Running Benchmarks

```bash
python3 -m attention.benchmark > benchmark.csv
```

The CSV includes:

- `mode`
- `seq_len`
- `block_size`
- `latency_ms`
- `tokens_per_sec`
- `peak_memory_mb`
- `mae`
- `rmse`
- `max_abs_error`
- `cosine_similarity`

The benchmark sweep compares:

- baseline
- token streaming
- block streaming with `block_size` in `16, 32, 64, 128`
- integer streaming

## Notes

- The baseline path remains the correctness reference.
- The token-streaming and block-streaming implementations do not silently fall back to materialized full attention.
- Correctness and clear tensor-shape documentation were prioritized ahead of low-level optimization.
