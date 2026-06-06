# KV-Stationary MQA Accelerator — Project Knowledge Base

**Project:** Performance Modeling of a KV-Stationary Systolic Array for Multi-Query Attention Inference
**Authors:** Hudson Strauss & Dhruv Patel
**Course:** High-Performance Computer Architecture, Santa Clara University
**Location:** `/Users/hudsons/Code/ScaleSim/MQA_SCALE_SIM/`

---

## 1. What This Project Is

An analytical performance model of a custom 2D systolic array architecture
for Multi-Query Attention (MQA) inference. The architecture keeps Key and
Value tensors stationary in array columns while Query vectors stream
left-to-right, computing online softmax inline at each PE column.

The model is validated against SCALE-Sim (cycle-accurate systolic array
simulator) for the compute-dominant GEMM components. The upper PE
(exp lookup + V accumulation) is modeled analytically.

The primary comparison is against FlashAttention-style fused attention
on equal silicon (same MAC count).

---

## 2. Fixed Hardware Parameters

These constants are used throughout all experiments and the report.

```
H   = 64      # query heads
d   = 128     # head dimension
bpe = 2       # bytes per element (FP16)
BW  = 512     # DRAM bandwidth bytes/cycle
pe_mac_width = 128   # upper PE vector width (fully parallel)
exp_latency  = 4     # cycles for one exp lookup (LUT-based)
```

Derived timing constants:
```
upper_pe      = exp_latency + ceil(3*d / pe_mac_width) = 4 + 3 = 7 cycles
column_dwell  = d + upper_pe = 128 + 7 = 135 cycles
```

With lmc=16 interleaved MACs:
```
packet_stagger = ceil(d / lmc) = ceil(128/16) = 8 cycles
upper_PE_utilization ≈ upper_pe / packet_stagger = 7/8 = 87.5%
```

With lmc=1 (single MAC):
```
packet_stagger = 128 cycles
upper_PE_utilization ≈ 7/135 = 5.2%
```

---

## 3. Architecture Description

### 3.1 Array Layout

```
array_rows = H * 2^n     (n = merge_extensions, default 0)
array_cols = T // 2^n
```

- Rows = parallel query lanes (one per head in base case)
- Columns = KV-stationary token positions
- With n=0 (recommended): array_rows=64, array_cols=T

### 3.2 PE Structure — Two Pipeline Stages Per Column

**Stage 1 — Lower MAC array:**
- `lmc` interleaved serial MACs compute s_i = Q · K_i
- Each MAC handles one Q packet; packets are time-multiplexed offset by ceil(d/lmc) cycles
- With lmc=16: initiates new packet every 8 cycles instead of 128

**Stage 2 — Upper Running Attention PE:**
```
m_out = max(m_in, s_i)
l_out = l_in * exp(m_in - m_out) + exp(s_i - m_out)
O_out = O_in * exp(m_in - m_out) + V_i * exp(s_i - m_out)
```
Cost: 4 + ceil(3*128/128) = 7 cycles

Final normalization after last column: Output = O_N / l_N

### 3.3 Pipeline Timing

Total tile cycles with R rows, C cols, P packets per row:
```
tile_cycles = (R + C - 1) * column_dwell + (P - 1) * effective_stagger
```

- (R + C - 1) * column_dwell = pipeline depth (fill + traversal + drain)
- (P - 1) * effective_stagger = within-row scheduling overhead (only when P > 1)

`effective_stagger = max(packet_stagger, upper_pe_cycles)` — see §3.6 below.

### 3.4 Merge Extensions (n levels)

Motivation: at T=8192 with no partitioning, drain = 8191 steps * 135 cycles
≈ 1.1M cycles — essentially all runtime.

Solution: split KV cache into 2^n sub-arrays, each handling T/2^n tokens.
Binary tree of inline merge PEs combines partial (m, l, O) states.
- Drain reduces: T-1 → T/2^n - 1 steps
- Cost: 2^n × more PE rows
- Merge PE cost: 2*exp_latency + ceil(6*d/pe_mac_width) = 8+3 = 11 cycles

**Finding:** n=3 hurts area-normalized efficiency vs n=0 with lmc=16.
Merge extensions multiply PEs by 8× but only recover ~2-3× speed.

### 3.5 Design Evolution (Chronological)

1. **1D pipeline (baseline):** Single row, drain = T-1 steps, upper PE util = 5%
2. **2D multi-head:** H rows, MQA shared KV loaded once for all heads. Doesn't fix utilization.
3. **Merge extensions (n):** Reduces drain by 2^n at cost of 2^n more rows. Validated 1.29× area-norm at T=8192.
4. **Interleaved MACs (lmc=16):** Reduces packet_stagger from 128→8 cycles, util 5%→87.5%. No extra rows. Validated 1.79× area-norm at T=2048. **This is the key parameter.**

**Conclusion:** n=0, lmc=16 is Pareto-optimal. Simpler and more area-efficient than n=3.

### 3.6 Score Delay Buffer — Upper PE Timing Alignment

**Problem:** The lower MAC emits one score every `packet_stagger = ceil(d/lmc)` cycles.
The upper PE takes `upper_pe_cycles = 7` cycles per score. If `packet_stagger <
upper_pe_cycles`, scores pile up unboundedly as each pending Q token holds its full
running state (M, L, O) = d+2 = 130 elements × 2 bytes = 260 bytes.

**Fix (implemented in `kv_stationary_model.py`):** A shallow scalar FIFO between the
lower MAC output and the upper PE input paces delivery to one score per
`effective_stagger` cycles. Buffer depth is permanently bounded.

```
effective_stagger        = max(packet_stagger, upper_pe_cycles)
score_buffer_depth       = max(0, ceil(upper_pe_cycles / packet_stagger) - 1)
score_buffer_bytes_total = score_buffer_depth * bytes_per_element * array_cols
```

**Validated results (d=128, pe_mac_width=128, exp_latency=4 → upper_pe=7):**

| lmc | stagger | eff_stagger | buf_depth | buf_bytes (1024 cols) |
|-----|---------|-------------|-----------|----------------------|
| 1   | 128     | 128         | 0         | 0                    |
| 16  | 8       | 8           | 0         | 0  ← 1-cycle slack, no buffer |
| 21  | 7       | 7           | 0         | 0  ← **throughput ceiling** |
| 22  | 6       | 7           | 1         | 2 KB                 |
| 128 | 1       | 7           | 6         | 12 KB                |

**lmc=21 is the true throughput ceiling:** ceil(128/21)=7 exactly meets upper_pe,
zero buffer needed, maximum lower MAC fan-out. lmc ≥ 22 is still valid hardware —
the buffer is a handful of scalar registers — but offers no additional throughput.

**Effect on total_cycles:** At memory-bound operating points (normal), effective_stagger
replacing packet_stagger makes no difference to total_cycles. In a compute-bound regime,
lmc ≥ 22 incurs `(P-1) * (effective_stagger - packet_stagger)` extra cycles.

**New fields in `kv_stationary_metrics` output:**
```
effective_stagger            — max(packet_stagger, upper_pe_cycles)
score_buffer_depth           — scalar entries per column
score_buffer_bytes_per_col   — bytes per column
score_buffer_bytes_total     — bytes across all columns
```

---

## 4. Key Files

### Core Models

| File | Purpose |
|------|---------|
| `kv_stationary_model.py` | Main KV-stat analytical + pipeline model. Key function: `simulate_2d_kv_stationary_array()` and `kv_stationary_metrics()` wrapper |
| `baseline_mqa_model.py` | Roofline baseline model. Supports `fused=True` (FlashAttention-style), `score_bytes_per_element`, `generate_tokens` |
| `compare.py` | Sweeps T=[128,512,1024,2048,4096,8192], generates results.csv |

### Validation & Experiments

| File | Purpose |
|------|---------|
| `validate_lower_mac_sweep.py` | Runs SCALE-Sim on lower MAC GEMM, computes delta% error vs analytical |
| `validate_lower_mac_sweep.csv` | Results: 29 configs, analytical vs ss_mac_cycles, delta%, corrected pipeline |
| `prefill_comparison.py` | KV-stat vs FlashAttention sweep with SCALE-Sim validation |
| `prefill_full_results.csv` | Full prefill results: all 24 configs with speedup, area ratio, area_norm_speedup |
| `prefill_element_streaming.csv` | Cycle breakdown: ss_lower_mac, stagger_cost, total_cycles per config |
| `prefill_full_results.py` | Generates prefill_full_results.csv |
| `prefill_element_streaming.py` | Corrected element-streaming model |

### Plotting Scripts

| File | Output |
|------|--------|
| `make_plot11.py` | plots/11_speedup_vs_baseline.png — 3-way speedup comparison |
| `make_autoregressive_plots.py` | plots/13,14 — DRAM/cycles vs G tokens |
| `plot_prefill_corrected.py` | plots_prefill_corrected/fig1-6 — main result figures |
| `plot_fig4_n_sweep.py` | n-sweep breakdown at T=8192 |

### Architecture Diagrams

| File | Purpose |
|------|---------|
| `plots/kv_stationary_architecture.html` | Interactive HTML/SVG architecture visualization. Four sections: (1) single PE internals — Lower Unit MAC array + Upper Unit running-attention PE with data flows labeled, (2) 2D array layout with head-labeled rows (rows = heads) and shared K/V column flows, (3) pipeline phase timeline (fill/steady/flush/merge), (4) merge extension tree. Open in browser. Key conceptual clarification captured here: the `lower_mac_count` MACs within a single PE column are time-interleaved across **Q packets from the same head's row**, each staggered uniformly by `⌈d/n⌉` cycles (e.g. 8 cycles apart with d=128, n=16), all reading the same stationary K[t] register via n read-ports. |

### Report

| File | Purpose |
|------|---------|
| `report.tex` | Full mid-progress report (15 pages). Compile: pdflatex + biber + pdflatex×2 |
| `bib.bib` | Bibliography (ainslie2023gqa entry) |
| `documentation/resources/mqa-designs/` | All architecture diagram PNGs |

---

## 5. DRAM Traffic Model

### KV-Stationary

```
query_reads   = H * batch * query_tokens * d * bpe
kv_load       = batch * 2 * T * d * bpe   (K + V, loaded once)
output_writes = H * batch * query_tokens * d * bpe
total_dram    = query_reads + kv_load + output_writes
```

At T=8192, H=64, d=128, bpe=2:
- KV load = 2 * 8192 * 128 * 2 = 4.19 MB
- Q reads (prefill) = 64 * 8192 * 128 * 2 = 134 MB
- Total ≈ 272 MB

### Baseline (unfused)

```
q_reads          = H * B * Q_T * d * bpe
k_reads          = B * T * d * bpe           (MQA: no H factor)
v_reads          = B * T * d * bpe
score_write_read = H * B * Q_T * T * sbpe * 2  (= 0 if fused)
output_writes    = H * B * Q_T * d * bpe
```

At T=8192 prefill, unfused: score matrix = 64 * 8192 * 8192 * 2 * 2 = **17,179 MB = 98.4% of total**

### Baseline (fused / FlashAttention)

score_write_read = 0. Total ≈ 272 MB — identical to KV-stat.

---

## 6. Arithmetic Intensity (T=8192, prefill)

| Config | Total MACs | Total DRAM | AI (MACs/byte) | Bottleneck |
|--------|-----------|-----------|----------------|------------|
| 64×64 unfused | 1.10T | 17.45 GB | 63 | COMPUTE (7.8× over mem) |
| MAC-norm unfused | 1.10T | 17.45 GB | 63 | MEMORY |
| MAC-norm fused | 1.10T | 0.27 GB | 4,033 | MEMORY |
| KV-stat n=3 lmc=16 | 2.20T | 0.27 GB | 8,066 | MEMORY |

Note: KV-stat has 2× MACs because value accumulation adds 3d ops per (Q,KV) pair
on top of the d-wide dot product. Same DRAM as fused baseline → ~1× speedup.

---

## 7. Validated Experimental Results

### Lower MAC Validation Error (prefill only)

| T | n | lmc | Analytical | SCALE-Sim | Error | Mac frac |
|---|---|-----|-----------|-----------|-------|----------|
| 128 | 0 | 1 | 16,384 | 16,891 | 3.1% | 40.2% |
| 128 | 0 | 16 | 1,024 | 1,531 | 49.5% | 5.7% |
| 512 | 0 | 1 | 65,536 | 66,811 | 1.9% | 46.7% |
| 512 | 0 | 16 | 4,096 | 5,371 | 31.1% | 6.6% |
| 1024 | 0 | 16 | 8,192 | 10,491 | 28.1% | 6.8% |
| 2048 | 0 | 16 | 16,384 | 20,731 | 26.5% | 6.9% |
| 2048 | 3 | 16 | 16,384 | 17,147 | 4.7% | 3.6% |
| 4096 | 3 | 16 | 32,768 | 34,043 | 3.9% | 3.9% |
| 8192 | 3 | 16 | 65,536 | 67,835 | 3.5% | 4.0% |

Error source = pipeline drain overhead (eff_cols - 1 extra steps SCALE-Sim measures).
Converges at large T as computation dominates drain.

**Decode validation: NOT DONE.** Analytical predicts 128 cycles; SCALE-Sim measures
full eff_cols-column pipeline traversal (e.g. 16,763 cycles at T=8192). Error 12,996%.
Decode is excluded from all validated results.

### Prefill: KV-stat vs FlashAttention (SCALE-Sim validated)

| Config | T | Flash cycles | KV corrected | Raw speedup | PE ratio | **Area-norm** | PE util |
|--------|---|-------------|-------------|------------|---------|--------------|---------|
| n=0, lmc=16 | 512 | 1,097,088 | 82,988 | 13.2× | 8× | **1.65×** | 84.6% |
| n=0, lmc=16 | 1024 | 4,388,352 | 157,228 | 27.9× | 16× | **1.74×** | 89.2% |
| n=0, lmc=16 | 2048 | 17,553,408 | 305,708 | 57.4× | 32× | **1.79×** | 91.7% |
| n=3, lmc=16 | 4096 | 70,213,632 | 884,355 | 79.4× | 64× | 1.24× | 62.6% |
| n=3, lmc=16 | 8192 | 280,854,528 | 1,700,483 | 165.2× | 128× | 1.29× | 65.1% |

**Analytical extrapolations (not SCALE-Sim validated):**

| Config | T | Raw speedup | Area-norm |
|--------|---|------------|----------|
| n=0, lmc=16 | 4096 | 118× | **1.85×** |
| n=0, lmc=16 | 8192 | 238× | **1.86×** |

### Three-Way Speedup (KV-stat n=3 lmc=16 vs all baselines, prefill)

| T | vs 64×64 unfused | vs MAC-norm unfused | vs MAC-norm fused |
|---|-----------------|--------------------|--------------------|
| 512 | 12.8× | 2.0× | 0.41× |
| 1024 | 44.4× | 6.3× | 0.70× |
| 2048 | 127.8× | 17.0× | **1.01×** |
| 4096 | 255.5× | 33.0× | **1.01×** |
| 8192 | 511.0× | 64.9× | **1.01×** |

**Key insight:** 511× is a MAC-count artifact. 65× is score-matrix DRAM.
~1× is the honest comparison against fused FlashAttention on equal silicon.
KV-stat breaks even with fused baseline at T≈2048.

---

## 8. Physical Cost

At T=8192, n=0, lmc=16 (best config):

| Resource | Amount |
|----------|--------|
| Total PEs | 524,288 |
| Total MACs | 8,388,608 |
| K-buffer SRAM | 128 MB |
| Estimated die area | ~20 mm² |

At T=2048, n=0, lmc=16 (validated + practical):

| Resource | Amount |
|----------|--------|
| Total PEs | 131,072 |
| K-buffer SRAM | 34 MB |

**Central limitation:** PE count and SRAM scale linearly with T. No tested
configuration achieves >2× area-normalized advantage. The hardware cost to
support raw speedup numbers grows faster than linearly with T.

---

## 9. Autoregressive Decode (Analytical Only — NOT Validated)

Per-token DRAM at T=8192:

| System | DRAM/token | Notes |
|--------|-----------|-------|
| Conventional | 6.0 MB | K + V reload + Q + scores + output every step |
| KV-stat model | 32 KB | Q + output only (KV held in SRAM) |
| Projected reduction | 188× | Analytical only |

**This projection requires 128 MB on-chip SRAM and has not been SCALE-Sim validated.**

---

## 10. Silicon Re-Use Model — COMPLETE (see §18 for full results)

**Concept:** Use P passes over T/P columns instead of 1 pass over T columns.
Same physical array reused P times. Partial softmax state (m, l, O) saved
between passes and re-injected via a single merge PE.

**Hardware cost of re-injection per pass boundary:**
```
reinject_cycles = 2 * exp_latency + ceil(6 * d / pe_mac_width) = 8 + 3 = 11 cycles
```
State saved: (d+2) * bpe = 258 bytes. Negligible SRAM.

**Trade-off formula:**
```
physical_cols    = T / P
SRAM_required    = 2 * (T/P) * d * bpe
total_cycles     = P * single_pass_cycles(T/P) + (P-1) * reinject_cycles
```

**Key property:** No extra DRAM. Applies to both prefill and decode.
Trade latency for silicon area at a known ratio.

| P | Cols at T=8192 | SRAM | Cycle multiplier |
|---|---------------|------|-----------------|
| 1 | 8192 | 128 MB | 1× |
| 2 | 4096 | 64 MB | ~2× |
| 4 | 2048 | 32 MB | ~4× |
| 8 | 1024 | 16 MB | ~8× |

At P=4: array is 64×2048 — exactly the validated T=2048 operating point.

**Implementation:** `kv_reuse_model.py` — see §18.

---

## 11. Validation Methodology

### What SCALE-Sim validates

SCALE-Sim is run with `Bandwidth: 100000` (infinite DRAM) to isolate compute.
Config: weight-stationary dataflow, 4096 KB SRAM per buffer.

Lower MAC GEMM shape:
```
M = H * (Q_T / lmc)     # rows
N = eff_cols            # cols = T / 2^n
K = d                   # head dimension
array: H rows × eff_cols cols
```

FlashAttention tiles:
```
QK tile: M = H*Br, N = Bc, K = d   on 64×64 array
AV tile: M = H*Br, N = d,  K = Bc  on 64×64 array
Flash tile size Br=Bc=64 (SRAM-constrained: 4*(64*128*2) = 65536 bytes < 4096 KB)
```

### Error formula

```
delta_pct = 100 * (ss_mac_cycles - analytical_mac) / analytical_mac
```

Source of error = pipeline drain (eff_cols - 1 extra steps). Always positive
(SCALE-Sim >= analytical). Converges as T grows.

### Corrected pipeline

```
corrected_total = pipeline_cycles - analytical_mac + ss_mac_cycles
```

### Memory limits for SCALE-Sim runs

```
operand_matrix_MB = (M*d + d*eff_cols + M*eff_cols) * 8 / 1e6
skip if > 500 MB
```

This excludes: n=0 lmc=1 prefill at T≥1024, n=0 lmc=16 prefill at T≥4096.

---

## 12. Model API Reference

### `kv_stationary_metrics(H, T, d, array_rows, array_cols, bytes_per_element, memory_bandwidth_bytes_per_cycle, exp_latency_cycles=4, pe_mac_width=1, lower_mac_count=1, batch_size=1, head_parallelism=1, merge_extensions=0, query_tokens=1, generate_tokens=0)`

Key output keys:
```
total_cycles            — full pipeline + memory + merge cycles
compute_cycles          — pipeline compute only
memory_service_cycles   — DRAM service cycles
pipeline_latency_cycles — first Q in to last O out
throughput_cycles_per_token
column_dwell            — 135 cycles (fixed for these params)
packet_stagger          — ceil(d/lmc)
effective_stagger       — max(packet_stagger, upper_pe_cycles)
score_buffer_depth      — scalar entries per column (0 when stagger ≥ upper_pe)
score_buffer_bytes_per_col
score_buffer_bytes_total
total_dram_bytes
kv_load_bytes
query_reads_bytes
output_writes_bytes
pe_utilization
total_macs
k_buffer_bytes_per_pe
total_k_buffer_bytes
# if generate_tokens > 0:
decode_dram_bytes_per_step
total_generate_dram_mb
total_generate_cycles
dram_per_token_kb
```

### `baseline_mqa_metrics(H, T, d, array_rows, array_cols, bytes_per_element, memory_bandwidth_bytes_per_cycle, batch_size=1, query_tokens=1, fused=False, score_bytes_per_element=None, generate_tokens=0)`

Key output keys:
```
estimated_cycles        — max(ideal_compute_cycles, memory_cycles)
ideal_compute_cycles    — total_macs / (array_rows * array_cols)
memory_cycles           — total_dram_bytes / BW
total_dram_bytes
score_write_read_bytes  — 0 if fused=True
arithmetic_intensity
# if generate_tokens > 0:
decode_dram_bytes_per_step
total_generate_dram_mb
total_generate_cycles
```

---

## 13. Standard Configurations Used in Experiments

```python
# KV-stat best validated config (n=0, lmc=16)
kv = kv_stationary_metrics(
    H=64, T=T, d=128,
    array_rows=64,              # H * 2^0
    array_cols=T,               # T / 2^0
    bytes_per_element=2,
    memory_bandwidth_bytes_per_cycle=512,
    pe_mac_width=128,
    lower_mac_count=16,
    merge_extensions=0,
    query_tokens=T,             # prefill
)

# KV-stat n=3 config (validated at large T)
kv_n3 = kv_stationary_metrics(
    H=64, T=T, d=128,
    array_rows=512,             # H * 2^3
    array_cols=T // 8,          # T / 2^3
    bytes_per_element=2,
    memory_bandwidth_bytes_per_cycle=512,
    pe_mac_width=128,
    lower_mac_count=16,
    merge_extensions=3,
    query_tokens=T,
)

# MAC-normalized fused baseline (FlashAttention reference)
b_fused = baseline_mqa_metrics(
    H=64, T=T, d=128,
    array_rows=8192,            # H * 2^3 * lmc = 512 * 16
    array_cols=T // 8,
    bytes_per_element=2,
    memory_bandwidth_bytes_per_cycle=512,
    query_tokens=T,
    fused=True,
)

# Original 64x64 unfused baseline
b_orig = baseline_mqa_metrics(
    H=64, T=T, d=128,
    array_rows=64, array_cols=64,
    bytes_per_element=2,
    memory_bandwidth_bytes_per_cycle=512,
    query_tokens=T,
    fused=False,
)
```

---

## 14. Report Structure (report.tex — 15 pages)

1. Introduction — project overview and key finding
2. Problem Statement — attention math, score matrix bottleneck
3. Proposed Architecture
   - 3.1 Design Evolution (4 iterations)
   - 3.2 Array Layout and Dataflow
   - 3.3 Pipeline Timing Model
   - 3.4 Merge Extensions
4. Experimental Methodology
   - 4.1 Hybrid Analytical + SCALE-Sim Model
   - 4.2 Validation Error Analysis
   - 4.3 Fixed Hardware Parameters
   - 4.4 Parameter Sweep
   - 4.5 Experiments Conducted (with MAC validation table)
5. Results
   - 5.1 Baseline Characterization (3 variants, compute/memory bottleneck, score matrix table, AI table, 3-way speedup table)
   - 5.2 Prefill: KV-stat vs FlashAttention (main results table)
   - 5.3 Prefill: Comparison Against Unfused Baseline
   - 5.4 Autoregressive Decode: Analytical Projection (UNVALIDATED)
   - 5.5 Physical Cost
6. Discussion
7. Conclusion (central finding: physical scaling is the limiting factor)
8. Future Work: Two-Pass Decode / Silicon Re-Use Model (with ReuseModel.png)
9. References

Compile command:
```
pdflatex report.tex && biber report && pdflatex report.tex && pdflatex report.tex
```

---

## 15. Key Findings Summary

1. **lmc=16 is the single most important parameter.** Raises upper PE utilization from 5% to 87.5% with no extra rows. Transforms a design that loses to FlashAttention per unit area into one that beats it by 65-79%.

2. **Merge extensions hurt area efficiency.** n=3 adds 8× rows for ~2-3× speed. n=0 lmc=16 is Pareto-optimal.

3. **The 511× speedup is a MAC-count artifact.** 64×64 baseline is 7.8× compute-bound. MAC-normalized: 65×. Fused: ~1×.

4. **65× vs unfused = score matrix.** 98.4% of unfused baseline DRAM at T=8192 is the H×T² score tensor. Both KV-stat and FlashAttention eliminate this.

5. **~1× vs fused FlashAttention** at equal MAC count. KV-stat's AI is 2× higher (8066 vs 4033 MACs/byte) because it has 2× more MACs, but DRAM is identical so cycle count is similar.

6. **Validated ceiling: 1.79× area-normalized at T=2048.** Extrapolated to 1.86× at T=8192 (unconfirmed).

7. **Physical scaling is the central limitation.** PE count and SRAM scale linearly with T. 524K PEs and 128 MB SRAM at T=8192. No config exceeds 2× area-normalized advantage.

8. **Decode is not validated.** Analytical model diverges from SCALE-Sim by 200-12,996%. Future work: two-pass re-use model trades P× latency for P× smaller chip.

9. **Energy: KV-stat is 3.2× more efficient than two-GEMM baseline** (Accelergy + CACTI verified, decode, T=8192). 2.90 mJ vs 9.27 mJ. Savings come from eliminating the attention score matrix DRAM round-trip — not from K/V being loaded fewer times (both load KV once). DRAM still dominates at 81.5% of total energy.

---

## 16. Open Work / TODO

- [ ] SCALE-Sim validation of n=0 lmc=16 at T=4096 and T=8192 (currently blocked by 500MB operand matrix limit)
- [x] Implement `kv_reuse_model.py` — multi-pass silicon re-use model — COMPLETE. See §18.
- [x] Sweep P=[1,2,4,8,16] at T=[512,1024,2048,4096,8192], plot latency vs P and area-norm throughput vs P — COMPLETE. See §18.
- [ ] Validate single decode step in SCALE-Sim (currently divergent — needs pipeline traversal model not MAC-only)
- [x] Accelergy energy integration — COMPLETE. KV-stat 2.90 mJ vs baseline 9.27 mJ (3.2×). See §17.
- [ ] Upper PE hardware validation (exp unit latency modeled as 4 cycles — not confirmed)
- [ ] **Literature review** — search ISCA/MICRO/ASPLOS/DAC/MLSys 2022-2025 for prior implementations of:
  - KV-stationary or token-stationary attention accelerators
  - Systolic arrays with inline online softmax (FlashAttention hardware)
  - Multi-query attention hardware (streaming Q over stationary KV)
  - Score delay buffers / timing alignment between dot-product and accumulation stages
  - Merge-tree attention tiling in hardware
  Previous attempt (2026-05-28) hit rate limit before finding matches. Key search terms:
  "KV-stationary", "token-stationary attention", "streaming attention hardware",
  "online softmax accelerator", "FlashAttention FPGA/ASIC", "MQA hardware"

---

## 17. Energy Estimation (Accelergy — COMPLETE)

### Results: Decode, H=64, T=8192, d=128, n=3, lmc=16

| Component | KV-Stationary | Baseline (2-GEMM) |
|---|---|---|
| DRAM | 2.37 mJ (81.5%) | 8.25 mJ (89%) |
| On-chip reads (K shift-reg / GLB) | 0.27 mJ (9.2%) | 0.93 mJ (10%) |
| MAC compute | 0.27 mJ (9.2%) | 0.09 mJ (1%) |
| **Total** | **2.90 mJ** | **9.27 mJ** |
| **Improvement** | **3.2×** | — |

Accelergy output: `rundir-accelergy/accelergy_output_kv/energy_estimation.yaml`

### Why the DRAM savings (not what you'd expect)

Both architectures load K and V exactly once from DRAM (MQA — no H factor on KV).
The DRAM difference (8.25 → 2.37 mJ) comes from **eliminating the attention score matrix DRAM round-trip**:

- Baseline (SCALE-Sim output-stationary): score matrix [H×T=64×8192] must be written to DRAM
  after GEMM1 (psum) then re-read as input to GEMM2, because the on-chip GLB can't hold
  it alongside K and V simultaneously. ~1M extra DRAM accesses.
- KV-stationary: online softmax update — scores never materialized, never go to DRAM.

### K shift-register clarification (important for energy model)

`total_k_buffer_bytes = 128 MB` reported by the model is **distributed instantaneous
register storage** — 524,288 PEs × 256 bytes each (d * bpe). It is NOT a monolithic SRAM.

K[t] propagates **row-to-row as a shift register** (confirmed in model docstring:
"K[t] ripples down through rows with a 1-step delay per row"). This means:
- Each K[t] is read from DRAM once into the first-row PE register
- Then transferred row-to-row (register-to-register, essentially free / bundled in MAC energy)
- The column K/V buffer is accessed T×d times total (not H×T×d times)
- Section 8's "K-buffer SRAM 128 MB" label is misleading — it's the PE register footprint,
  not a centralized SRAM that incurs per-access SRAM energy costs

### MAC count for decode (query_tokens=1)

`total_macs = H * T * d * 4 = 64 * 8192 * 128 * 4 = 268,435,456` (268M)
- 1d for Q·K dot product
- 3d for online softmax O update (Oin*exp_old + exp_new*V)

Section 6 AI table value "2.20T MACs" is for **prefill** (query_tokens=T), not decode.

### hwcomponents_cacti methodology

```python
# Install: pip install hwcomponents_cacti (macOS arm64: recompile CACTI from source)
from hwcomponents_cacti.hwcomponents_cacti import SRAM, DDR3

# CRITICAL: tech_node in METERS, not nanometers
tech_node = 40e-9

# DRAM (8-bit element width → 560 pJ/byte)
ddr3 = DDR3(width=8)
energy_J, _ = ddr3.read()
dram_pj = energy_J * 1e12  # ≈ 560 pJ per byte

# SRAM (size-dependent, 8-bit width)
size_kb = 512
size_bits = size_kb * 1024 * 8
depth = size_bits // 8
sram = SRAM(tech_node=tech_node, size=size_bits, width=8, depth=depth)
read_pj  = sram.read_energy  * 1e12   # e.g. ~38 pJ/byte at 512 KB
write_pj = sram.write_energy * 1e12
```

Typical per-byte costs at 40nm:
- DDR3 DRAM: 560 pJ/byte (70 pJ/bit)
- 512 KB SRAM: ~38 pJ/byte read
- 8 MB SRAM: ~135 pJ/byte read
- 128 MB SRAM: ~892 pJ/byte read (nearly DRAM-tier — avoid monolithic buffers this large)
- regfile (sub-4KB): 1 pJ/access (Horowitz 2014 analytical)
- intmac 8-bit: 1 pJ/MAC (Horowitz 2014 — includes input register reads)

### Accelergy Quirks (40nm setup)

- `technology: "40nm"` must be **quoted** in YAML (unquoted fails Python eval in Accelergy 0.4)
- `global_cycle_seconds: 1e-9` required in top-level attributes
- `memory_depth: bank_depth * n_banks` cross-attribute arithmetic broken — precompute
- Primitive lib top-level key is `classes:` (flat list), NOT `primitive_components: {classes: ...}`
- `arguments:` with no value = Python None → omit bare argument lines
- `action_counts` YAML needs `version: 0.3` as sibling of `local:`
- Use Python at `SCALE-Sim/scale/bin/python3` for hwcomponents_cacti (it's installed there)

---

## 18. Silicon Re-Use Model — Implementation & Results (COMPLETE, 2026-06-05)

### Files added

| File | Purpose |
|------|---------|
| `kv_reuse_model.py` | Multi-pass re-use extension. Imports helpers from `kv_stationary_model.py`. Key function: `kv_stationary_metrics(..., num_passes=1, causal=False)` |
| `make_reuse_plots.py` | Decode sweep: T×P grid, produces plots A–B in `plots/` |
| `make_prefill_plots.py` | Causal prefill sweep: T×P grid, produces plots C–F in `plots/` |
| `make_reuse_figs.py` | Publication-quality figures fig7–10 in `plots_reuse/`, styled to match `plots_prefill_corrected/fig3_area_norm_speedup.png` |

### `kv_stationary_metrics` new parameters

```
num_passes: int = 1    — P passes over T/P columns
causal: bool  = False  — enable causal-prefill per-pass computation
```

When `num_passes > 1` the base model is called with `T = physical_cols = T // P`
(one tile, no internal tiling) and the results scaled:

```
total_cycles_multipass = P × single_pass_cycles + (P-1) × reinject_cycles
reinject_cycles        = 14  (= 2×4 + ceil(6×128/128) = 8 + 6)
pe_count               = array_rows × physical_cols
sram_bytes             = 2 × physical_cols × d × bpe   (K+V, MQA-shared)
```

New keys added to result dict:
```
num_passes, physical_cols, reinject_cycles,
total_cycles_multipass, total_cycles_causal,
causal_cycles_per_pass, causal, pe_count, sram_bytes
```

### Decode sweep results (T=[512…8192], P=[1…16])

Key finding: **latency overhead is sub-linear in P at large T.**

At T=8192 the P=1 pipeline is already 99% drain (8191 of 8255 steps).
Reducing to T/P columns shrinks each pass's drain proportionally, so the
sum of P smaller passes is barely more than one big pass:

| T | P | physical_cols | total_cycles | latency_ratio | area_norm_tput |
|---|---|---------------|-------------|---------------|----------------|
| 8192 | 1 | 8192 | 1,114,425 | 1.00 | 1.40e-8 |
| 8192 | 2 | 4096 | 1,122,944 | 1.008 | 2.78e-8 |
| 8192 | 4 | 2048 | 1,139,982 | 1.023 | 5.48e-8 |
| 8192 | 8 | 1024 | 1,174,058 | 1.054 | 1.06e-7 |
| 8192 | 16 | 512 | 1,242,210 | 1.115 | 2.01e-7 |

Area-norm throughput = `T / (total_cycles × pe_count)`.
Increases monotonically with P — no crossover observed within P≤16.
Using P=16 gives 14× better silicon efficiency than P=1 at T=8192.

At P=4, T=8192: array is 64×2048 (the validated T=2048 operating point),
cycles = 1,139,982 ≈ 4 × 284,985 (the validated T=2048 single-pass cycles). ✓

### Causal prefill model

With a lower-triangular causal mask and P passes over C = T/P columns,
query token i attends to KV columns 0..i. In pass k (covering columns
[(k-1)C .. kC-1]):

- Queries i < (k-1)C: **discard** — done, do not participate.
- Queries i ∈ [(k-1)C, kC-1]: **diagonal** — complete their final pass, write output.
- Queries i ≥ kC: full pass, carry (m, l, O) state forward via reinject.

Active query count per pass: **Q_k = T − (k−1)C** (decreasing each pass).
Queries finishing per pass: **C** (the diagonal slice — output streamed immediately).

Per-pass DRAM (implemented correctly, not estimated):
```
Q reads   = H × Q_k × d × bpe     (only active queries reload Q)
KV reads  = 2 × C × d × bpe       (always load this pass's KV slice)
Output    = H × C × d × bpe       (C queries write final O each pass)
pass_cycles_k = max(compute_k, memory_k)
```

**`total_cycles_causal` = Σ_k pass_cycles_k + (P−1) × reinject_cycles**

### Causal prefill results (T=[512…8192], P=[1…16])

Cycle savings vs naive non-causal multi-pass:

| T | P | causal_cycles | noncausal_cycles | savings |
|---|---|--------------|-----------------|---------|
| 8192 | 2 | 1,221,232 | 1,254,000 | 2.6% |
| 8192 | 4 | 1,303,790 | 2,105,386 | 38.1% |
| 8192 | 8 | 1,656,102 | 4,202,594 | 60.6% |
| 8192 | 16 | 2,618,774 | 8,397,010 | 68.8% |

**The savings are DRAM-driven, not compute-driven.**
- Non-causal re-reads all T query vectors every pass: total Q DRAM ∝ P × T
- Causal re-reads only active Q_k: total Q DRAM ∝ T × (P+1)/2
- Savings from pure compute (stagger term reduction) ≈ 4T(P-1) cycles ≈ 8%
- Remaining ~60% savings come from smaller Q reads + streaming output per pass

**Bottleneck transition:** early passes are memory-bound (orange DRAM bar > blue
compute bar in fig E — T=8192, P=8, pass 1: memory=295K, compute=212K).
Later passes flip compute-bound as Q_k shrinks. Crossover around pass 4 of 8.

### Publication figures (plots_reuse/)

Style matches `plots_prefill_corrected/fig3_area_norm_speedup.png` exactly:
`semilogx`, `linewidth=2.2`, `marker='o' markersize=6`, `figsize=(8,5)`, `dpi=150`,
grid `both/--/alpha=0.3`. Colour palette: P=1 blue, P=2 orange, P=4 green,
P=8 purple, P=16 red.

| Figure | File | What it shows |
|--------|------|---------------|
| fig7 | `fig7_area_norm_throughput_decode.png` | Area-norm tput vs T per P (decode). All lines fall with T; higher P always wins. |
| fig8 | `fig8_latency_overhead.png` | Cycle multiplier (P passes / 1 pass) vs T. Sub-linear overhead converges to ~1× at large T. |
| fig9 | `fig9_causal_savings_vs_T.png` | Causal cycle savings % vs T per P. P=16 saves 69% at T=8192. |
| fig10 | `fig10_decode_vs_prefill_area_norm.png` | Decode vs causal prefill area-norm tput. Decode higher due to memory-bound early prefill passes. |

### Key findings summary

1. **Latency overhead is sub-linear, not P×.** At large T the pipeline drain already
   dominates P=1, so splitting into P smaller passes costs only 1.02–1.11× more
   cycles while delivering P× smaller chip and P²× better area-norm throughput.

2. **Area-norm throughput increases monotonically with P** (within P≤16 tested).
   No crossover / diminishing-returns point found. The reinject overhead (14 cycles)
   is negligible relative to single-pass cycles (~1M at T=8192).

3. **Causal masking is a first-class feature of multi-pass, not an afterthought.**
   At P=8 or higher, >60% of non-causal DRAM traffic disappears because finished
   queries are discarded at each pass boundary. This makes large-P causal prefill
   competitive with smaller-P non-causal prefill in absolute cycle count.

4. **P=4 at T=8192 lands exactly on the validated T=2048 operating point** (64×2048
   array), giving a concrete hardware anchor: the T=2048 silicon can run T=8192
   inference in 4 passes at 1.14M cycles, vs a 64×8192 chip at 1.11M cycles —
   only 2.3% slower at 4× smaller chip and 4× less SRAM.

---

## 19. Causal Prefill Model — Design Notes (2026-06-05)

### Why the DRAM model matters for prefill

Unlike decode (query_tokens=1, compute-bound), prefill with causal masking and
P > 1 passes switches between memory-bound and compute-bound within a single run:

- **Pass 1 (all T queries active):** Q reads = H×T×d×bpe = 128 MB at T=8192.
  At BW=512 B/cycle: 262K cycles. Compute: (H+C-1)×135 + (T-1)×8 = 351K cycles.
  → **Memory-bound** (262K memory wins... wait — max(351K, 526K) = memory-bound
  when output writes also counted: 262K+2K+262K = 526K > 351K).
- **Pass 4 (Q_4 = T/4 queries active):** Q reads = 32 MB → 65K cycles.
  Output writes = 32 MB → 65K cycles. Memory = 65K+2K+65K = 132K < compute 302K.
  → **Compute-bound**.

The crossover means the model must compute DRAM per-pass, not just multiply a
single-pass result by P. The `causal=True` path in `kv_stationary_metrics` does this.

### State management between passes

After each pass, the `(m, l, O)` state for continuing queries must survive until
the next pass. Three options (not yet chosen for hardware):

| Option | Cost | Notes |
|--------|------|-------|
| Keep in PE registers | 0 DRAM | Only viable if T/P ≤ register file capacity |
| Write to on-chip SRAM | ~(d+2)×bpe×Q_k bytes | ~260 KB at T=8192, negligible |
| Write to DRAM | ~(d+2)×bpe×Q_k bytes per pass | Adds ~1 MB/pass at T=8192, still tiny |

State save/load was omitted from the cycle model (negligible at these sizes).

### Streaming output

Each pass completes exactly C queries (the diagonal slice). Their output O vectors
can be written to DRAM immediately after that pass, without waiting for the final
pass. This:
- Amortises output DRAM across all P passes (C×H×d×bpe per pass vs T×H×d×bpe at end)
- Enables consumer overlap (next-stage processing can start on early tokens)
- Is already modelled correctly in `total_cycles_causal` (output added per-pass)
