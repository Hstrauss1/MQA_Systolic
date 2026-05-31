# PLAN: Cycle-Accurate MQA Integration into SCALE-Sim

This document records both the original implementation plan and the actual steps taken across all phases of the project. Sections marked **[COMPLETED]** reflect what was built; sections marked **[REVISED]** indicate where the implementation diverged from the original plan with a note explaining why.

## End-state architecture

The repository has three clearly separated roles:

- `scalesim/`: the shared simulation substrate, extended with MQA-aware config, topology, and dispatch integration points.
- `mqa_scalesim/`: the MQA-specific execution layer with a canonical workload object, analytical decode simulators, shared timing helpers, a memory bridge, and a validation/reporting adapter.
- `validation/`: phase-by-phase correctness and regression suite for all active simulator paths.
- `legacy/`: archived exploratory scripts, prior outputs, original configs, and historical documentation retained for reference.

The simulator supports two comparable execution modes under identical hardware and memory constraints:

- `baseline_mqa_decode`: conventional GEMM-style systolic execution of MQA decode with explicit cycle accounting for score, normalization, and value-accumulation stages.
- `kv_stationary_mqa_decode`: custom systolic execution with resident K/V state, streaming queries, propagated online-softmax state, and explicit final normalization.

---

## Phase 0: Preserve and baseline the current repo [COMPLETED]

### Step 0.1: Snapshot the current state

The existing analytical model outputs, CSV artifacts, and plotting scripts were preserved. The old `compare.py`, `baseline_mqa_model.py`, `kv_stationary_model.py`, and plotting scripts were retained in place and later archived under `legacy/` once the new active workflow superseded them.

### Step 0.2: Validate the existing SCALE-Sim subtree

The `scalesim/` directory was confirmed to run correctly for existing workloads before any MQA-specific plumbing was added. The existing compute backends and memory system remained untouched by the MQA integration.

---

## Phase 1: Create the MQA integration layer [COMPLETED]

### Step 1.1: Create `mqa_scalesim/` package

Created the following files as the canonical MQA integration layer:

- `mqa_scalesim/__init__.py` — package initialization.
- `mqa_scalesim/workload.py` — `MQAWorkload` dataclass with full validation, covering mode, sequence length, batch size, query/KV heads, head dimension, precision, array shape, SRAM sizes, bandwidth mode, decode-token count, softmax variant, KV block size, stream grouping, pipeline depth override, and KV reuse flag.
- `mqa_scalesim/result_schema.py` — `MQASimulationResult` and `MQAStageResult` with total cycles, per-stage cycles, DRAM/SRAM traffic, PE utilization, KV preload bytes, and `memory_model_applied` tracking.
- `mqa_scalesim/pe_timing.py` — shared cycle-level timing parameters for MAC, exp, compare/max, renormalization, and forwarding operations.
- `mqa_scalesim/softmax_ops.py` — explicit online-softmax cost estimators and streaming-softmax step and row cost helpers.
- `mqa_scalesim/baseline_decode.py` — `BaselineMQADecodeSimulator` implementing the four-stage baseline decode schedule: `score_gemm`, `softmax_reduce`, `value_gemm`, `writeback`.
- `mqa_scalesim/kv_stationary_decode.py` — `KVStationaryMQADecodeSimulator` implementing the five-stage streaming schedule: `kv_preload`, `query_stream`, `online_softmax_accum`, `final_normalize`, `writeback`.
- `mqa_scalesim/memory_bridge.py` — `MQAMemoryBridge` connecting both simulators to shared DRAM/SRAM traffic accounting.
- `mqa_scalesim/validation_bridge.py` — `result_to_validation_dict`, `result_to_experiment_row`, and `validate_against_reference` adapters for validation and CSV reporting.

**Design note:** Both simulators were implemented as analytical decode models rather than fully invasive SCALE-Sim compute-backend modifications. This decision was made because the MQA decode stages have shapes and memory access patterns that differ materially from the convolution/GEMM assumptions baked into `single_layer_sim.py` and `operand_matrix.py`. Analytical modeling inside the `mqa_scalesim/` layer, backed by the shared memory bridge, was found to be a cleaner and more maintainable approach than modifying the inner compute backends.

---

## Phase 2: Extend SCALE-Sim config, topology, and simulator dispatch [COMPLETED]

### Step 2.1: Modify `scalesim/scale_config.py`

Added MQA-aware workload fields to the config object: `workload_type`, `sequence_length`, `batch_size`, `query_heads`, `kv_heads`, `head_dim`, `precision`, `decode_tokens`, `decode_step`, `softmax_variant`, and `exp_variant`. Preserved full backward compatibility with existing GEMM/convolution workloads.

### Step 2.2: Modify `scalesim/topology_utils.py`

Extended topology handling to represent MQA decode stages as named pseudo-layers, including `softmax_reduce` stage naming so the topology pipeline could carry MQA-specific stage structure.

### Step 2.3: Modify `scalesim/simulator.py`

Added top-level workload dispatch so the simulator detects MQA workload types from the config object and routes them into the `mqa_scalesim` execution layer while leaving the standard GEMM/convolution path entirely unchanged.

---

## Phase 3: Baseline analytical decode and stage validation [COMPLETED]

### Step 3.1: Implement baseline decode with rich result structure

The `BaselineMQADecodeSimulator` was implemented with explicit stage-level accounting. Each of the four decode stages produces a structured `MQAStageResult` with cycles, MACs, DRAM reads/writes, and SRAM reads/writes, enabling stage-level comparison rather than only aggregated totals.

### Step 3.2: Validate baseline correctness

`validation/phase3_validation.py` was written to verify that:
- The baseline simulator returns the correct four-stage sequence.
- Score and value stages carry positive MAC counts.
- Larger sequence lengths produce more cycles.
- Larger array sizes reduce or preserve total cycles.
- Simulator dispatch still routes correctly with the real backend.
- The validation bridge produces normalized, required-field dictionaries.

---

## Phase 4: KV-stationary streaming decode and validation [COMPLETED]

### Step 4.1: Implement KV-stationary simulator

The `KVStationaryMQADecodeSimulator` was implemented with explicit stage accounting for the five-stage streaming schedule. The implementation tracks softmax state bytes, KV preload bytes, streaming row costs, and final normalization cycles, all derived from the canonical `MQAWorkload` object.

### Step 4.2: Validate KV-stationary correctness

`validation/phase4_validation.py` was written to verify that:
- KV-stationary returns the correct five-stage sequence.
- Sequence length scaling increases both total cycles and KV preload bytes.
- Array size scaling has the expected effect on total cycles.
- The validation bridge produces normalized outputs.
- The baseline simulator still passes structural checks after Phase 4 changes.

---

## Phase 5: Sweep, compare, and plot pipeline integration [COMPLETED]

### Step 5.1: Implement `run_sweep.py`

Created the canonical benchmark sweep driver. On each invocation, the script:
- Accepts sequence lengths, decode-token counts, array sizes, batch size, head structure, precision, and stress-preset inclusion as CLI parameters.
- Constructs `MQAWorkload` objects for both execution modes for every parameter combination.
- Writes results to a timestamped run directory under `outputs/` as `sweep_results.csv`, a full JSON artifact, and a summary JSON.

### Step 5.2: Implement `compare.py`

Created the comparison driver. The script:
- Accepts `--run-dir` for explicit targeting or auto-detects the latest timestamped run under `outputs/`.
- Reads `sweep_results.csv`, matches baseline and KV rows by their parameter join keys.
- Computes speedup, DRAM-traffic ratios, SRAM-traffic ratios, and PE utilization delta.
- Writes `comparison.csv` and `comparison.json` into the same run directory.

### Step 5.3: Implement `plot_results.py`

Created the plotting driver. The script:
- Accepts `--run-dir` or auto-detects the latest run.
- Reads both `sweep_results.csv` and `comparison.csv`.
- Writes figures into `outputs/<timestamp>/plots/`.

### Step 5.4: Validate the pipeline

`validation/phase5_validation.py` was written to confirm that sweep outputs contain all required columns, comparison outputs are structurally correct, and plot generation completes without errors.

---

## Phase 6: End-to-end memory bridge validation [COMPLETED]

### Step 6.1: Implement `validation/phase6_validation.py`

Created a top-level end-to-end validation script that:
- Runs both `BaselineMQADecodeSimulator` and `KVStationaryMQADecodeSimulator` through the full `MQAMemoryBridge` with `run_memory_model=True`.
- Verifies that result dicts contain `memory_stall_cycles` and `kv_preload_bandwidth_cycles`.
- Checks that all numeric fields are finite.
- Confirms KV-stationary total cycles are >= baseline total cycles.
- Calls `validate_against_reference()` for structural key checks.
- Supports `--verbose` and `--csv` flags for detailed output.

**Design note:** Rather than modifying `scalesim/memory/` read and write buffer internals as originally planned (steps 6.1–6.4 of the original plan), the memory model was integrated through the `MQAMemoryBridge` class in `mqa_scalesim/memory_bridge.py`. This kept the existing SCALE-Sim memory subsystem stable while providing the DRAM/SRAM traffic and stall-cycle accounting required by the project.

---

## Phase 7: Reporting, sweep output naming, and path modernization [COMPLETED]

### Step 7.1: Standardize output paths to `outputs/`

The sweep, compare, and plot scripts were all updated to use a unified `outputs/<timestamp>/` run-directory layout. This replaced the older flat `phase7_outputs/...` paths that had been the default in prior milestones.

### Step 7.2: Rename sweep CSV

The main sweep output CSV was renamed from the `phase7_sweep_results.csv` convention to the cleaner `sweep_results.csv`, consistent with the non-phase-specific naming used in `compare.py` and `plot_results.py`.

### Step 7.3: Add run-directory auto-detection

Both `compare.py` and `plot_results.py` were updated to accept an explicit `--run-dir` argument or, when none is provided, automatically detect the latest timestamped directory under `outputs/`. This makes re-running the full pipeline on any environment a single-command operation.

---

## Phase 8: Validation path migration [COMPLETED]

**Note:** This phase was not part of the original plan. It emerged from the fact that phase validation scripts had been written at repo root and needed to be migrated into the new `validation/` directory without breaking their path logic.

### Step 8.1: Migrate validation scripts to `validation/`

All phase validation scripts (`phase2_validation.py` through `phase6_validation.py`) were moved from repo root into `validation/`.

### Step 8.2: Patch validation script paths

Each validation script was updated to introduce `VALIDATION_ROOT = Path(__file__).resolve().parent` and derive `REPO_ROOT = VALIDATION_ROOT.parent`. Artifact directories were updated to point to `validation/phase*_validation_artifacts/` instead of repo root. The Phase 5 validation script was updated to invoke `legacy/phase5_sweep.py` instead of the old root-level location.

---

## Phase 9: Repository cleanup, documentation consolidation, and README unification [COMPLETED]

**Note:** This phase replaced the original Phase 8 documentation and cleanup plan and went further in scope.

### Step 9.1: Archive legacy material

All exploratory scripts, prior outputs, original configs, historical documentation, archived tests, and non-active plotting scripts were moved into `legacy/` to clean the active root surface. This included:
- `baseline_mqa_model.py`, `kv_stationary_model.py`, `phase5_sweep.py`, `mqa_correctness.py`
- All `configs/`, `layouts/`, `plots/`, `scripts/`, `topologies/`, `rundir-accelergy/`, `code-examples/`, `CPP__Simulation/`, `documentation/` trees
- `CHANGELOG.md`, `CONTRIBUTING.md`, `METADATA.md`, `README_Sparsity.md`, `README_ramulator.md`, `Makefile`, `LICENSE`
- Prior output directories: `phase5_outputs/`, `phase7_outputs/`
- Old `dist/` wheel artifacts

### Step 9.2: Consolidate README files

The separate `README.md` (SCALE-Sim fork overview) and `README_MQA.md` (MQA-specific guide) were merged into a single unified `README.md`. The new README documents:
- The current repository directory layout with roles for every top-level path.
- Why `mqa_scalesim/` exists and its layered architecture.
- How SCALE-Sim was modified (config, topology, and simulator dispatch extension points).
- Step-by-step environment setup for any arbitrary machine.
- How to run the full sweep/compare/plot pipeline.
- How to change sweep parameters at the CLI or in code.
- The validation layout and recommended usage pattern.

`README_MQA.md` was deleted after the merge.

### Step 9.3: README audit

The README was audited against the active validation script inventory and current CLI naming conventions. One stale reference to `phase7_sweep_results.json` was corrected. Validation entry points were confirmed against the actual files in `validation/`.

### Step 9.4: PLAN.MD update

This document was rewritten to reflect what was actually built across all phases, with design notes where the implementation diverged from the original plan, and with Phase 8 (validation migration) and Phase 9 (cleanup and documentation) added as new phases that were not in the original plan.

---

## Recommended execution order (final)

For a clean end-to-end run on any machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 validation/phase6_validation.py --verbose
python3 run_sweep.py --verbose --progress
python3 compare.py --verbose --progress
python3 plot_results.py
```

To change sweep parameters, pass them as CLI flags to `run_sweep.py`:

```bash
python3 run_sweep.py \
  --sequence-lengths 64,128,256,512 \
  --decode-tokens 1,2,4 \
  --array-sizes 8x8,16x16,32x32 \
  --verbose --progress
```

For persistent benchmark campaigns, update the default constants at the top of `run_sweep.py` directly.

---

## Divergences from the original plan

| Original plan | What was actually done | Reason |
|---|---|---|
| Modify `scalesim/single_layer_sim.py` and `operand_matrix.py` for MQA stages | Implemented analytical simulators in `mqa_scalesim/` with a shared memory bridge | MQA decode stages differ materially from GEMM/conv assumptions; the integration layer approach was cleaner and less brittle |
| Modify `scalesim/memory/` read/write buffers directly | Integrated memory accounting through `MQAMemoryBridge` | Avoided destabilizing the existing SCALE-Sim memory subsystem |
| Refactor `generate_scalesim_topology.py` as front-end compiler | Created `run_sweep.py` as the new front-end driver | `run_sweep.py` is a cleaner separation of experiment orchestration from simulation logic |
| Keep `compare.py` and `plot_results.py` at flat output paths | Updated both to use `outputs/<timestamp>/` layout with auto-detection | Better reproducibility and cleaner run isolation |
| Update both `README.md` and `README_MQA.md` | Merged both into a single unified `README.md` | One entry point is better for a publishable repository |
| Phase 8 was documentation and cleanup only | Phases 8 and 9 were split: Phase 8 handled validation path migration, Phase 9 handled cleanup and documentation | The validation migration was substantial enough to warrant its own phase |
