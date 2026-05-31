# MQA SCALE-Sim: Native Decode Modeling for Baseline and KV-Stationary Multi-Query Attention

This repository is a research fork of SCALE-Sim that adds a native workflow for modeling decode-time Multi-Query Attention (MQA) with two comparable execution modes: a conventional baseline mapping and a custom KV-stationary dataflow. The project combines a canonical workload description, decode-specific stage modeling, explicit online-softmax timing, and a shared memory/reporting bridge so that both designs can be evaluated under the same reporting pipeline.[cite:245][cite:246][cite:247]

## Repository purpose

The goal of this fork is to turn MQA decode from an external analytical side-model into a first-class simulation flow that can be run, swept, compared, and validated inside a consistent SCALE-Sim-oriented environment. The active workflow now centers on `mqa_scalesim/`, the modified `scalesim/` integration points, root-level benchmark drivers, and the `validation/` suite, while older exploratory material has been moved into `legacy/` for reference only.[cite:245][cite:246]

## Directory layout

The current top-level layout is intentionally organized around active code, validation, and archived material.[cite:245]

| Path | Role |
|---|---|
| `README.md` | Unified project documentation and operational guide.[cite:245] |
| `PLAN.MD` | Implementation plan and milestone history for the project.[cite:245] |
| `run_sweep.py` | Main benchmark driver; creates a timestamped run under `outputs/` and writes `sweep_results.csv`, JSON, and summary artifacts.[cite:245] |
| `compare.py` | Loads sweep results, compares baseline and KV-stationary runs, and writes `comparison.csv` and `comparison.json` into the selected run directory.[cite:245] |
| `plot_results.py` | Reads sweep and comparison outputs and writes plots into `outputs/<timestamp>/plots/`.[cite:245] |
| `mqa_scalesim/` | Canonical MQA workload, simulators, memory bridge, result schema, softmax timing, and validation adapters.[cite:245][cite:246] |
| `scalesim/` | Modified SCALE-Sim core integration points used by the MQA flow.[cite:245][cite:247] |
| `validation/` | Phase validation scripts and their artifacts, updated to run from the new `validation/` layout.[cite:245][cite:244] |
| `legacy/` | Archived exploratory models, prior outputs, original configs, documentation assets, and non-active scripts retained for reference.[cite:245][cite:236] |

A practical mental model is: `scalesim/` provides the shared simulation substrate, `mqa_scalesim/` provides the MQA-specific execution layer, the root scripts drive experiments, `validation/` guards correctness and regression behavior, and `legacy/` preserves prior work without cluttering the active repository surface.[cite:245][cite:246]

## Why `mqa_scalesim` exists

SCALE-Sim was originally structured around convolution and GEMM-style workloads, which is not enough to describe decode-time MQA as a staged process with online softmax state, KV reuse semantics, and a KV-stationary streaming schedule. `mqa_scalesim/` was created to provide a canonical workload object, decode-aware simulators, shared result schemas, and a translation layer that lets MQA-specific execution concepts plug into a common memory and reporting backend rather than living in disconnected one-off scripts.[cite:246][cite:247]

The package exists for three reasons. First, it provides a single source of truth for MQA experiments through `MQAWorkload`, which captures the mode, sequence length, head structure, array shape, SRAM sizes, decode token count, bandwidth settings, and softmax/pipeline metadata in one validated object.[cite:246] Second, it separates architectural policy from experiment orchestration: the root scripts build workloads and collect outputs, while the actual execution semantics live in the package.[cite:245][cite:246] Third, it ensures that baseline and KV-stationary modes can be flattened into the same reporting format through `result_schema.py` and `validation_bridge.py`, making side-by-side comparison practical and fair.[cite:246][cite:247]

## `mqa_scalesim` architecture

The active MQA package is organized as a layered architecture rather than a monolithic script collection.[cite:245][cite:246]

### Workload layer

`mqa_scalesim/workload.py` defines `MQAWorkload`, the canonical workload specification for all active MQA experiments. It validates mode selection, head relationships, decode-token settings, bandwidth mode, array sizes, KV grouping constraints, and optional overrides such as KV block sizing, stream grouping, and softmax-state precision.[cite:246]

This workload object is the contract between user intent and simulator behavior. Every active execution path in this repository starts by constructing one or more `MQAWorkload` instances and then passing them to the appropriate simulator mode.[cite:245][cite:246]

### Baseline decode path

`mqa_scalesim/baseline_decode.py` implements `BaselineMQADecodeSimulator`, which models a conventional decode schedule as the stage sequence `score_gemm`, `softmax_reduce`, `value_gemm`, and `writeback`. It uses explicit online-softmax cost estimation and stage-level accounting to build a structured result, and it supports optional memory-model application via `run_memory_model=True` so the same execution can be evaluated with the shared memory bridge enabled.[cite:246][cite:247]

This path is important because it converts what would otherwise be treated as a plain GEMM proxy into a decode-aware baseline with explicit stage semantics. That makes the baseline comparable to the KV-stationary design at the stage, cycle, and traffic levels rather than only at an aggregated total.[cite:246][cite:247]

### KV-stationary decode path

`mqa_scalesim/kv_stationary_decode.py` implements `KVStationaryMQADecodeSimulator`, which models the custom streaming dataflow through the stages `kv_preload`, `query_stream`, `online_softmax_accum`, `final_normalize`, and `writeback`. The implementation explicitly accounts for softmax state bytes, KV preload behavior, streaming row costs, and final normalization, again with optional memory-model application through the same simulation result path used by the baseline mode.[cite:247]

This is the core architectural contribution of the repository. Instead of treating KV-stationary execution as a separate spreadsheet or post-hoc estimate, the design is represented as a simulator with named stages, derived resource quantities, and a result structure that can be consumed by the same comparison and plotting pipeline as the baseline path.[cite:246][cite:247]

### Shared timing and softmax layer

`mqa_scalesim/softmax_ops.py` provides explicit estimators for online-softmax cost and streaming-softmax row/step costs, while `mqa_scalesim/pe_timing.py` provides timing helpers shared across execution paths. This layer exists because softmax and reduction timing were a key technical gap in earlier approximations, and the active repository models them explicitly rather than hiding them in opaque totals.[cite:246][cite:247]

### Result, bridge, and reporting layer

`mqa_scalesim/result_schema.py` defines the structured simulation result and tracks whether the memory model has actually been applied, while `mqa_scalesim/validation_bridge.py` converts simulation outputs into flattened dictionaries suitable for CSV rows, validation payloads, and downstream plotting. The bridge records fields such as stage names and `memory_model_applied`, which are critical for regression checks and benchmark reporting.[cite:247]

### Memory integration layer

`mqa_scalesim/memory_bridge.py` provides `MQAMemoryBridge`, the point where MQA stage-level behavior is connected to the shared memory accounting path. This is what makes the project more than a pair of isolated analytical decoders: both baseline and KV-stationary simulations can pass through a common memory-model application path so SRAM, DRAM, stall, and utilization metrics are reported under the same infrastructure.[cite:246][cite:247]

## How SCALE-Sim was modified

The project does not replace SCALE-Sim; it extends it. The active implementation adds MQA-aware integration points in the existing SCALE-Sim code so decode workloads can be described, routed, and reported without forcing the entire project into an external side tool.[cite:245][cite:247]

The modifications are visible in the active codebase through new workload-related fields and dispatch points in `scalesim/scale_config.py`, `scalesim/topology_utils.py`, and `scalesim/simulator.py`. The search results show MQA-aware handling for `softmax_variant`, staged topology naming such as `softmax_reduce`, and simulator integration points that route MQA parameters into the active simulation flow.[cite:247]

At a high level, the project changes SCALE-Sim in four ways:
- It extends configuration and topology plumbing so MQA workloads carry decode-specific parameters rather than only legacy GEMM assumptions.[cite:247]
- It allows staged decode semantics, including softmax-specific stages, to appear in the workload/reporting pipeline.[cite:247]
- It integrates MQA simulation outputs with the shared memory path so both decode modes can claim memory-model-backed metrics using the same infrastructure.[cite:247]
- It keeps the active project anchored to the repository’s SCALE-Sim foundation rather than splitting execution into unrelated scripts and post-processing notebooks.[cite:245][cite:247]

## Active workflow

The active benchmark flow is straightforward and centered on three root-level drivers.[cite:245]

1. `run_sweep.py` constructs a sweep over sequence lengths, decode token counts, and array sizes, then emits a new timestamped run directory under `outputs/` for every invocation.[cite:245]
2. `compare.py` reads `sweep_results.csv` from a selected or auto-detected run directory, matches baseline and KV rows by their join keys, and writes comparison artifacts into the same run directory.[cite:245] 
3. `plot_results.py` reads both `sweep_results.csv` and `comparison.csv` and writes summary figures under `outputs/<timestamp>/plots/`.[cite:245]

Both `compare.py` and `plot_results.py` auto-detect the latest timestamped run directory under `outputs/` when no `--run-dir` is provided, which makes the default sweep-to-plot workflow convenient for repeated benchmarking on new machines or clean clones.[cite:245]

## Step-by-step setup in an arbitrary environment

The repository is designed to run in a standard Python environment without machine-specific assumptions beyond normal package installation and a working interpreter. The current documentation and scripts assume a plain repository checkout with Python dependencies installed from `requirements.txt`.[cite:245]

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd MQA_Systolic
```

### 2. Create and activate a Python environment

A virtual environment is the safest default in arbitrary environments because it avoids dependency collisions and makes the run instructions reproducible.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

The repository keeps runtime dependencies in `requirements.txt`, and the active root-level scripts depend on that environment being installed before use.[cite:245]

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 4. Optional: run validation

The validation suite now lives under `validation/` and has been patched so its artifact paths work from the new location.[cite:244][cite:245]

```bash
python3 validation/phase6_validation.py --verbose
```

### 5. Run a sweep

The sweep driver creates a fresh timestamped directory under `outputs/` on every run and writes `sweep_results.csv` together with JSON and summary artifacts there.[cite:245][cite:251]

```bash
python3 run_sweep.py --verbose --progress
```

If `tqdm` is available, `--progress` enables progress bars; otherwise the script still works, and `--verbose` gives per-workload logging.[cite:245]

### 6. Compare the two execution modes

The comparison script defaults to the latest run directory under `outputs/`, reads `sweep_results.csv`, and writes `comparison.csv` plus `comparison.json` into that same run directory.[cite:245]

```bash
python3 compare.py --verbose --progress
```

To target a specific prior run instead of the latest one:

```bash
python3 compare.py --run-dir outputs/<timestamp> --verbose --progress
```

### 7. Generate plots

The plotting script also defaults to the latest run directory and writes figures into `outputs/<timestamp>/plots/`.[cite:245]

```bash
python3 plot_results.py
```

To target a specific run explicitly:

```bash
python3 plot_results.py --run-dir outputs/<timestamp>
```

## Changing sweep parameters

The sweep is intentionally parameterized at the command line so new machines and new experiments do not require code edits for ordinary changes. The active script exposes sequence lengths, decode-token counts, array sizes, batch size, head structure, precision, stress inclusion, output root, verbosity, and progress-bar behavior through CLI flags.[cite:245]

The most common parameter changes are handled directly in the command line.

### Example: change sequence lengths

```bash
python3 run_sweep.py --sequence-lengths 64,128,256,512 --verbose --progress
```

### Example: change decode-token counts

```bash
python3 run_sweep.py --decode-tokens 1,2,4,8 --verbose --progress
```

### Example: change array sizes

```bash
python3 run_sweep.py --array-sizes 8x8,16x16,32x32 --verbose --progress
```

### Example: change architectural parameters

```bash
python3 run_sweep.py \
  --batch-size 1 \
  --query-heads 8 \
  --kv-heads 2 \
  --head-dim 64 \
  --precision int8 \
  --verbose --progress
```

### Example: include the stress preset

```bash
python3 run_sweep.py --include-stress --verbose --progress
```

These flags are all translated into `MQAWorkload` objects before execution, which means the command line is not bypassing the architecture layer; it is configuring it through the canonical workload interface.[cite:245][cite:246]

## Changing the default sweep in code

For recurring benchmark campaigns, the default sweep can also be changed directly in `run_sweep.py`. The script defines defaults for sequence lengths, decode tokens, array sizes, query heads, KV heads, head dimension, batch size, and precision near the top of the file, then uses those defaults to build the workload list before execution.[cite:245]

That design supports two styles of use:
- For one-off experiments, pass parameters on the command line.[cite:245]
- For team-wide benchmark baselines, update the default constants in `run_sweep.py` so a plain `python3 run_sweep.py` produces the desired standard campaign.[cite:245]

## Validation layout

The validation suite has been separated from the root benchmark flow to make the repository easier to navigate. The active validation entry points live in `validation/`, and their artifact directories now live alongside them under `validation/phase*_validation_artifacts/` rather than at repo root.[cite:236][cite:244]

This layout makes validation runs easier to audit because each phase’s script and artifacts are colocated. It also keeps the root of the repository focused on the active operational path: sweep, compare, plot, and the implementation code that backs those commands.[cite:236][cite:244]

## Legacy material

`legacy/` is a deliberate archive, not a trash directory. It contains earlier analytical models, prior outputs, reference assets, original configs, archived tests, and historical documentation that remain useful for provenance, debugging, or reconstructing earlier milestones, but they are no longer part of the active benchmark path described in this README.[cite:245][cite:236]

This split is important for publishable quality. Active users can understand the repository quickly from the root-level files and the `mqa_scalesim/` and `validation/` directories, while researchers who need prior context still have access to the archived material without having it interfere with normal use.[cite:245][cite:236]

## Recommended usage pattern

For a clean end-to-end run on a fresh machine, the recommended sequence is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 validation/phase6_validation.py --verbose
python3 run_sweep.py --verbose --progress
python3 compare.py --verbose --progress
python3 plot_results.py
```

This sequence validates the current implementation, generates a timestamped run directory, produces a normalized comparison artifact, and then renders figures for inspection or inclusion in reports.[cite:244][cite:245]

## Summary of the active design

This repository is best understood as a layered MQA extension of SCALE-Sim rather than as a collection of scripts. `scalesim/` provides the shared simulation substrate, `mqa_scalesim/` provides the decode-specific architecture and canonical workload layer, the root drivers execute reproducible benchmark campaigns into timestamped `outputs/` directories, `validation/` protects milestone correctness, and `legacy/` preserves the historical and exploratory record of the project.[cite:245][cite:246][cite:247]
