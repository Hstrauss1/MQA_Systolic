# KV-Stationary MQA Accelerator

**Course:** CSEN 318 — High-Performance Computer Architecture, SCU Spring 2026  
**Authors:** Hudson Strauss & Dhruv Patel

---

## Overview

Analytical performance model of a custom 2D systolic array for Multi-Query
Attention (MQA) inference. Keys and Values are loaded once into array columns
and remain stationary; Query vectors stream left-to-right computing an inline
online softmax at every column. Validated against SCALE-Sim cycle-accurate
simulation for the Q·K compute stage.

**Central finding:** Under the SCALE-Sim–validated wavefront fill model,
prefill is DRAM-bound at 512 B/cycle. KV-stationary and fused FlashAttention
hit the same 273 MB DRAM floor and achieve roughly equal cycle counts on equal
silicon. A multi-pass silicon re-use scheme (P passes over T/P columns) trades
P× latency overhead for P× smaller chip and P²× better area-normalised
throughput, with sub-linear latency cost at large T.

---

## Prerequisites

```bash
# 1. Install this repo as a package (provides the scalesim import)
pip3 install -e /path/to/MQA_SCALE_SIM

# 2. Python dependencies
pip3 install numpy matplotlib

# 3. LaTeX (to recompile the report)
#    brew install --cask mactex   (macOS)
```

---

## Repository Structure

```
MQA_SCALE_SIM/
│
├── analysis/                     # All project Python — run scripts from here
│   ├── kv_stationary_model.py    # Core analytical pipeline model (wavefront fill)
│   ├── kv_reuse_model.py         # Multi-pass silicon re-use extension
│   ├── baseline_mqa_model.py     # Roofline baseline (fused / unfused FlashAttention)
│   ├── mqa_correctness.py        # Numerical correctness check for online softmax
│   │
│   ├── validate_lower_mac_sweep.py   # SCALE-Sim Q·K GEMM validation (29 configs)
│   ├── validate_sweep_scalesim.py    # Wavefront fill validation (40 configs)
│   ├── prefill_element_streaming.py  # Per-config cycle breakdowns
│   ├── reuse_full_sweep.py           # 750-row BW × lmc × T × P analytical sweep
│   ├── generate_scalesim_topology.py # Write SCALE-Sim topology/config files
│   │
│   ├── plot_prefill_corrected.py  # figs 1–3, 5, 7
│   ├── make_corrected_figs.py     # figs 4, 6, 13–16
│   ├── make_reuse_figs.py         # figs 7–12 (re-use model)
│   ├── make_reuse_plots.py        # plots A, B (decode sweep)
│   ├── make_prefill_plots.py      # plots C–F (causal prefill)
│   └── make_results_figs.py       # figs 17–20 (core results)
│
├── results/
│   ├── data/                      # Validated CSV outputs
│   │   ├── validate_lower_mac_sweep.csv    # 29-config SCALE-Sim vs analytical
│   │   ├── validate_sweep_scalesim.csv     # 40-run wavefront fill validation
│   │   ├── prefill_element_streaming.csv   # Cycle breakdowns per config
│   │   └── reuse_full_sweep.csv            # 750-row full sweep
│   ├── figures/                   # All publication figures (26 PNGs)
│   └── rundir-accelergy/          # Accelergy energy estimation output
│
├── report/
│   ├── report.tex                 # LaTeX source (14 pages)
│   ├── bib.bib                    # Bibliography
│   └── report.pdf                 # Compiled PDF
│
├── documentation/
│   ├── kv_stationary_architecture.html   # Interactive architecture diagram
│   └── resources/mqa-designs/            # Architecture diagram PNGs
│
├── old_dwell_model/               # Archived superseded work (dwell fill model)
│   └── old_plots/                 # Old figures from dwell-model era
│
├── scale/                         # Python virtual environment (SCALE-Sim)
├── scalesim/                      # SCALE-Sim package source (editable install)
└── scalesim_base/                 # All other SCALE-Sim infrastructure
    ├── configs/    topologies/    layouts/
    ├── scripts/    submodules/    code-examples/
    ├── CPP__Simulation/           verify_outputs/
    └── scalesim_artifacts/        # Base repo READMEs, Makefile, shell scripts
```

---

## Reproducing the Results

All scripts are run from the `analysis/` directory.

```bash
cd analysis/
```

### Step 0 — Verify online-softmax correctness
```bash
python3 mqa_correctness.py
# Expected: All correctness checks passed
```

### Step 1 — SCALE-Sim lower-MAC validation (Table 1 in report)
```bash
python3 validate_sweep_scalesim.py
# Writes ../results/data/validate_sweep_scalesim.csv
# Runtime: ~5–15 min (40 GEMM configs, requires SCALE-Sim installed)
```

### Step 2 — Full BW × lmc × T × P sweep
```bash
python3 reuse_full_sweep.py
# Writes ../results/data/reuse_full_sweep.csv (750 rows, analytical only)
# Runtime: < 10 seconds
```

### Step 3 — Generate all figures
```bash
python3 plot_prefill_corrected.py   # figs 1–3, 5, 7
python3 make_corrected_figs.py      # figs 4, 6, 13–16
python3 make_reuse_figs.py          # figs 7–12
python3 make_reuse_plots.py         # plots A, B
python3 make_prefill_plots.py       # plots C–F
python3 make_results_figs.py        # figs 17–20
# All output → ../results/figures/
```

### Step 4 — Compile the report
```bash
cd ../report
pdflatex report.tex && biber report && pdflatex report.tex && pdflatex report.tex
# Output: report.pdf (14 pages)
```

---

## Key Results

| Metric | Value |
|--------|-------|
| SCALE-Sim fill (measured) | ~2.5 cycles/column |
| Prefill bottleneck at 512 B/cycle | DRAM-bound (both KV-stat and Flash) |
| KV-stat vs fused Flash, equal silicon | ~1× (same DRAM floor) |
| KV-stat vs unfused baseline (T=8192) | ~65× (score-matrix DRAM eliminated) |
| Best per-MAC speedup (compute-bound) | ~3.5× (lmc ≤ BW/128, large T, large P) |
| Multi-pass latency overhead (P=4, T=8192) | +2.3% vs full array |
| Energy vs 2-GEMM baseline (decode) | 3.2× lower (Accelergy + CACTI, analytical) |

---

## Hardware Parameters

| Parameter | Value |
|-----------|-------|
| Query heads H | 64 |
| Head dimension d | 128 |
| Precision | FP16 (2 B/element) |
| DRAM bandwidth | 512 B/cycle (commodity); sweep covers 512–2048 |
| Upper PE MAC width | 128 |
| Exp lookup latency | 4 cycles |
| lmc balance point | ≈ BW/128 (= 4 at 512 B/cycle) |

---

## Model Boundaries

- **SCALE-Sim validates only the lower-MAC Q·K GEMM** at infinite bandwidth.
  Full pipeline timing (DRAM service, upper PE) is roofline-analytical.
- **Upper PE (exp unit, V accumulation) is not hardware-validated.**
  The 4-cycle exp latency is an architectural assumption.
- **Decode is not SCALE-Sim validated.** Analytical decode counts d MAC cycles
  per token step; SCALE-Sim measures the full T-column pipeline traversal
  (~130× larger). All decode numbers are analytical projections only.
- **Energy estimation is analytical** (Accelergy + CACTI at 40 nm node).
  No silicon or RTL validation.
