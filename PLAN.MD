# PLAN: Cycle-Accurate MQA Integration into SCALE-Sim

This document is the implementation plan for refactoring the repository so that both the conventional GEMM-style MQA baseline and the KV-stationary MQA systolic design are simulated cycle-accurately inside SCALE-Sim.

The target end state is a single performance backend: all final cycle counts, memory traffic numbers, utilization metrics, and comparison plots should come from SCALE-Sim internals rather than from a split workflow that mixes SCALE-Sim with an external analytical model.

## End-state architecture

After this refactor, the repository should have three clearly separated roles:

- `scalesim/`: the source of truth for cycle-accurate simulation.
- `mqa_scalesim/`: a new integration layer that expresses MQA decode workloads and bridges them into SCALE-Sim.
- `CPP__Simulation/attention/`: correctness and algorithm-reference code only, not the source of final performance numbers.

The simulator must support two comparable execution modes under identical hardware and memory constraints:

- `baseline_mqa_decode`: conventional GEMM-style systolic execution of MQA decode with explicit cycle accounting for score, normalization, and value-accumulation stages.
- `kv_stationary_mqa_decode`: custom systolic execution with resident K/V state, streaming queries, propagated online-softmax state, and explicit final normalization.

## Phase 0: Preserve and baseline the current repo

### Step 0.1: Snapshot the current state

Before making architectural changes, preserve the current behavior and outputs of the repository.

Tasks:
- Record the current command flow for `mqa_correctness.py`, `generate_scalesim_topology.py`, `compare.py`, and plotting scripts.
- Save a copy of current CSV outputs and plots produced by the old baseline vs analytical workflow.
- Note the current README language that says KV-stationary results are not natively modeled in SCALE-Sim; this will be updated later after the new implementation is complete.

Deliverables:
- A known-good baseline branch or commit.
- Saved old outputs for regression sanity checks.

### Step 0.2: Validate the existing SCALE-Sim subtree

Confirm that the newly added `scalesim/` directory runs correctly for ordinary workloads before adding MQA-specific logic.

Tasks:
- Run one known-good general systolic simulation through `scalesim/scale_sim.py` or the current front-end workflow.
- Confirm the existing compute backends (`systolic_compute_os.py`, `systolic_compute_is.py`, `systolic_compute_ws.py`) and memory system produce stable outputs.
- Verify report generation still works through `scalesim/utilities/scalesim_report.py`.

Deliverables:
- A confirmed working baseline SCALE-Sim installation inside the repo.

## Phase 1: Create the MQA integration layer

### Step 1.1: Create a new `mqa_scalesim/` package

Create a dedicated package at the repository root for all MQA-specific simulation plumbing.

Create these files:
- `mqa_scalesim/__init__.py`
- `mqa_scalesim/workload.py`
- `mqa_scalesim/baseline_decode.py`
- `mqa_scalesim/kv_stationary_decode.py`
- `mqa_scalesim/softmax_ops.py`
- `mqa_scalesim/pe_timing.py`
- `mqa_scalesim/result_schema.py`
- `mqa_scalesim/validation_bridge.py`

Purpose of each file:

#### `mqa_scalesim/workload.py`
Define the canonical workload object for MQA decode.

Fields should include:
- sequence length
- batch size
- query head count
- KV head count
- head dimension
- precision
- array rows and columns
- SRAM sizes / memory constraints
- DRAM bandwidth mode
- decode token count or decode step index if needed
- selected simulation mode (`baseline_mqa_decode` or `kv_stationary_mqa_decode`)

#### `mqa_scalesim/result_schema.py`
Define the common simulation result object.

Fields should include:
- total cycles
- per-stage cycles
- DRAM bytes / words read and written
- SRAM accesses
- PE utilization
- on-chip storage requirement
- KV preload traffic
- per-token decode latency
- any stall-cycle categories exposed by SCALE-Sim

#### `mqa_scalesim/pe_timing.py`
Define cycle-level timing parameters that both simulator modes will share.

This file should define or parameterize:
- MAC latency
- compare/max latency
- exp approximation latency
- renormalization latency
- weighted-value accumulation latency
- forwarding latency between PEs
- final normalization/divide latency

#### `mqa_scalesim/softmax_ops.py`
Define the reduction and online-softmax micro-ops so that normalization is part of the cycle model rather than an analytical side calculation.

#### `mqa_scalesim/baseline_decode.py`
Define the baseline decode schedule using SCALE-Sim-managed stages.

#### `mqa_scalesim/kv_stationary_decode.py`
Define the custom KV-stationary decode schedule and its interface into SCALE-Sim.

#### `mqa_scalesim/validation_bridge.py`
Provide adapters that compare simulator outputs and assumptions against `mqa_correctness.py` and `CPP__Simulation/attention/*` reference logic.

Deliverables:
- New `mqa_scalesim/` package scaffold checked into the repository.

### Step 1.2: Add tests for the new package

Create dedicated tests for workload schema, schedule construction, and correctness validation.

Create these files:
- `test/test_mqa_workload.py`
- `test/test_baseline_decode_cycles.py`
- `test/test_kv_stationary_cycles.py`
- `test/test_softmax_microops.py`
- `test/test_mqa_vs_reference.py`

Deliverables:
- Test stubs with initial imports and minimal scaffolding.

## Phase 2: Extend SCALE-Sim configuration and dispatch

### Step 2.1: Modify `scalesim/scale_config.py`

Extend the SCALE-Sim configuration object so it can represent MQA decode workloads, not only conventional array dimensions, dataflow, topology path, and memory settings.

Modify:
- `scalesim/scale_config.py`

Add support for:
- workload type field with at least `gemm`, `baseline_mqa_decode`, `kv_stationary_mqa_decode`
- sequence length
- batch size
- query head count
- KV head count
- head dimension
- precision
- optional softmax mode / exp approximation mode
- optional decode-step metadata if needed

Tasks:
- Add setter/getter methods or extend existing config-list plumbing.
- Ensure configuration can still be serialized and read from config files if that is part of the existing flow.
- Keep backward compatibility for normal SCALE-Sim workloads.

Deliverables:
- MQA-aware config object with no regressions for normal runs.

### Step 2.2: Modify `scalesim/topology_utils.py`

Extend topology handling to support MQA decode workloads.

Modify:
- `scalesim/topology_utils.py`

Tasks:
- Add a representation for baseline MQA decode as either staged pseudo-layers or explicit sub-operations.
- Add a representation for KV-stationary decode metadata such as resident KV count, query stream length, and output accumulation shape.
- Preserve existing GEMM/convolution parsing behavior.

Deliverables:
- Topology ingestion that can carry MQA decode information into the simulator pipeline.

### Step 2.3: Modify `scalesim/simulator.py`

Add top-level dispatch for the new workload types.

Modify:
- `scalesim/simulator.py`

Tasks:
- Detect the workload type from the config or topology object.
- Route standard workloads to the current path unchanged.
- Route `baseline_mqa_decode` to the new baseline schedule path.
- Route `kv_stationary_mqa_decode` to the new KV-stationary schedule path.

Deliverables:
- Simulator-level branching that can launch either MQA execution mode.

## Phase 3: Refactor the front-end MQA scripts around the new workload object

### Step 3.1: Modify `generate_scalesim_topology.py`

Convert this script from a one-off baseline topology generator into the front-end workload compiler for both MQA modes.

Modify:
- `generate_scalesim_topology.py`

Tasks:
- Parse MQA workload parameters into `mqa_scalesim.workload.MQAWorkload`.
- Generate inputs for both `baseline_mqa_decode` and `kv_stationary_mqa_decode`.
- Eliminate assumptions that only the baseline path needs SCALE-Sim topology generation.
- Keep compatibility with existing CLI arguments where practical.

Deliverables:
- A single front-end script that can instantiate either MQA simulator path.

### Step 3.2: Modify `baseline_mqa_model.py`

Convert this file into a thin wrapper around the new baseline decode simulation path.

Modify:
- `baseline_mqa_model.py`

Tasks:
- Replace any standalone timing logic with calls into `mqa_scalesim/baseline_decode.py`.
- Ensure output is emitted using the unified result schema.

Deliverables:
- Baseline wrapper using SCALE-Sim as the sole timing source.

### Step 3.3: Modify `kv_stationary_model.py`

Retain the file name if needed, but stop using it as the source of final cycle counts.

Modify:
- `kv_stationary_model.py`

Tasks:
- Replace or wrap the old analytical model with calls into `mqa_scalesim/kv_stationary_decode.py`.
- Optionally preserve the old analytical equations behind a debug or validation flag only.
- Make it explicit in comments and docs that the analytical path is deprecated for final results.

Deliverables:
- KV-stationary wrapper using SCALE-Sim-backed execution.

### Step 3.4: Modify `compare.py`

Convert comparison logic so it compares two SCALE-Sim-backed runs under identical conditions.

Modify:
- `compare.py`

Tasks:
- Build one canonical workload object.
- Run baseline SCALE-Sim decode.
- Run KV-stationary SCALE-Sim decode.
- Emit a unified CSV row format for downstream plots.

Deliverables:
- Apples-to-apples comparison driver for the two execution modes.

## Phase 4: Implement the cycle-accurate baseline decode path

### Step 4.1: Modify `scalesim/single_layer_sim.py`

This file will likely be the central place for staged execution of attention decode within the existing simulator flow.

Modify:
- `scalesim/single_layer_sim.py`

Tasks:
- Allow one logical simulation job to consist of multiple ordered sub-stages rather than a single conventional layer operation.
- Implement or call into the baseline decode schedule from `mqa_scalesim/baseline_decode.py`.
- Ensure cycle accumulation includes stage boundaries, intermediate buffering, and writeback timing.

Deliverables:
- Per-job staged execution support integrated into SCALE-Sim.

### Step 4.2: Modify `scalesim/compute/operand_matrix.py`

Extend operand generation to support attention decode stages, not only conventional GEMM/convolution operands.

Modify:
- `scalesim/compute/operand_matrix.py`

Tasks:
- Add score-stage operand generation for `QK^T`.
- Add value-stage operand generation for `PV`.
- Add any intermediate-shape support required by normalization or stage chaining.

Deliverables:
- Operand generation for baseline MQA decode stages.

### Step 4.3: Reuse or lightly modify existing compute backends for the baseline

For the baseline path, use existing OS/IS/WS backends wherever possible instead of creating a new one, because the baseline is still conventional systolic GEMM-style execution.

Inspect and potentially modify:
- `scalesim/compute/systolic_compute_os.py`
- `scalesim/compute/systolic_compute_is.py`
- `scalesim/compute/systolic_compute_ws.py`

Tasks:
- Choose the baseline dataflow you want to evaluate as the reference mapping.
- Ensure it can be invoked stage-by-stage by the attention decode scheduler.
- Avoid embedding hidden analytical shortcuts in the baseline path.

Deliverables:
- Baseline path executing with existing SCALE-Sim compute semantics.

### Step 4.4: Create `scalesim/compute/softmax_pipeline.py`

Normalization and reduction stages must be cycle-accounted inside SCALE-Sim, not analytically outside it.

Create:
- `scalesim/compute/softmax_pipeline.py`

Tasks:
- Model max reduction timing.
- Model exp approximation timing.
- Model denominator accumulation timing.
- Model final normalization timing.
- Provide an interface usable by both baseline and KV-stationary execution paths.

Deliverables:
- Shared cycle-accurate softmax/reduction pipeline model.

## Phase 5: Implement the cycle-accurate KV-stationary path

### Step 5.1: Create `scalesim/compute/systolic_compute_kv_mqa.py`

This is the core new simulator backend for the custom dataflow.

Create:
- `scalesim/compute/systolic_compute_kv_mqa.py`

Responsibilities:
- Represent resident K/V state in each PE.
- Advance query data through the array cycle by cycle.
- Perform local score MAC operations.
- Update online-softmax state (`m`, `l`) in flight.
- Accumulate weighted output state (`O`) in flight.
- Track forwarding dependencies and bubbles/stalls.
- Report utilization and timing in the same style as other compute backends.

Deliverables:
- A new SCALE-Sim compute backend dedicated to KV-stationary MQA decode.

### Step 5.2: Modify `scalesim/single_layer_sim.py` again for KV-stationary integration

Once the custom compute backend exists, integrate it into the single-layer or staged-execution orchestration.

Modify:
- `scalesim/single_layer_sim.py`

Tasks:
- Add a path that instantiates and drives `systolic_compute_kv_mqa.py`.
- Connect it with `softmax_pipeline.py` where appropriate.
- Ensure final normalization and output writeback are cycle-accounted.

Deliverables:
- End-to-end KV-stationary execution path inside SCALE-Sim.

### Step 5.3: Implement `mqa_scalesim/kv_stationary_decode.py`

This file should define the high-level schedule and interfaces used by the simulator for the KV-stationary mode.

Tasks:
- Translate the architectural design from the presentation and report into explicit simulation stages and state machines.
- Define how resident KV state is initialized.
- Define how one decode query is injected and drained.
- Define how per-token cycles are computed and reported.

Deliverables:
- High-level KV-stationary decode schedule connected to the custom compute backend.

## Phase 6: Extend memory modeling so both paths are compared fairly

### Step 6.1: Modify `scalesim/memory/double_buffered_scratchpad_mem.py`

Memory timing must be shared and fair across both execution modes.

Modify:
- `scalesim/memory/double_buffered_scratchpad_mem.py`

Tasks:
- Add support for KV preload or residency accounting.
- Distinguish one-time setup traffic from recurrent decode-step traffic.
- Model intermediate buffering for staged baseline attention if required.
- Preserve normal SCALE-Sim behavior for non-MQA jobs.

Deliverables:
- MQA-aware scratchpad and DRAM interaction model.

### Step 6.2: Modify `scalesim/memory/read_buffer.py`

Modify:
- `scalesim/memory/read_buffer.py`

Tasks:
- Support query-stream read timing.
- Support resident-K/V refill behavior or preload behavior.
- Ensure bandwidth-related stalls are exposed accurately for both modes.

Deliverables:
- Correct decode-time read behavior and timing.

### Step 6.3: Modify `scalesim/memory/write_buffer.py`

Modify:
- `scalesim/memory/write_buffer.py`

Tasks:
- Support per-token output writeback timing.
- Support intermediate-stage writes if the baseline path requires them.
- Preserve compatibility with existing workloads.

Deliverables:
- Correct decode-time write behavior and timing.

### Step 6.4: Review other memory helper files if needed

Inspect and update only if required:
- `scalesim/memory/read_port.py`
- `scalesim/memory/write_port.py`
- `scalesim/memory/read_buffer_estimate_bw.py`

Tasks:
- Ensure no hidden assumptions break MQA read/write patterns or bandwidth estimation.

Deliverables:
- Consistent low-level memory timing for the new execution modes.

## Phase 7: Reporting, validation, and comparison

### Step 7.1: Modify `scalesim/utilities/scalesim_report.py`

Update reporting so the simulator emits MQA-specific metrics rather than only conventional layer outputs.

Modify:
- `scalesim/utilities/scalesim_report.py`

Add support for:
- total decode cycles
- score-stage cycles
- normalization cycles
- value-stage cycles
- KV preload cycles / traffic
- DRAM and SRAM traffic by category
- PE utilization
- on-chip storage requirement
- latency per generated token

Deliverables:
- SCALE-Sim reports that expose the exact metrics required by the project plan.

### Step 7.2: Modify `mqa_correctness.py`

Keep this as the numerical oracle.

Modify:
- `mqa_correctness.py`

Tasks:
- Add validation entry points for baseline and KV-stationary simulator outputs.
- Compare final output vectors or key intermediate states against the existing correctness logic.

Deliverables:
- Numerical validation hooked into the new simulation flow.

### Step 7.3: Use `CPP__Simulation/attention/*` as validation-only reference code

Leave these mostly intact:
- `CPP__Simulation/attention/baseline.py`
- `CPP__Simulation/attention/streaming.py`
- `CPP__Simulation/attention/block_streaming.py`
- `CPP__Simulation/attention/benchmark.py`
- `CPP__Simulation/attention/utils.py`

Tasks:
- Add small adapters only if necessary.
- Do not use these files for final cycle counts.

Deliverables:
- Reference math path remains available without competing with SCALE-Sim as the performance backend.

### Step 7.4: Modify `compare.py` and plotting scripts

Modify:
- `compare.py`
- `plot_results.py`
- `make_autoregressive_plots.py`
- `make_plot11.py`

Tasks:
- Standardize CSV output from the two new simulator modes.
- Update plots to consume the new schema.
- Ensure metrics align with those described in the report and presentation: total cycles, latency, throughput, KV-cache traffic reduction, PE utilization, and storage.

Deliverables:
- End-to-end comparison and plotting flow driven entirely by SCALE-Sim outputs.

## Phase 8: Documentation and cleanup

### Step 8.1: Modify `README.md`

Update the fork overview and modeling claims.

Modify:
- `README.md`

Tasks:
- Remove the statement that KV-stationary is not natively modeled by SCALE-Sim once the new path is working.
- Document the new architecture at a high level.
- Update quick-start instructions.

Deliverables:
- Main README consistent with the new implementation.

### Step 8.2: Modify `README_MQA.md`

This should become the operational guide for the new workflow.

Modify:
- `README_MQA.md`

Tasks:
- Describe the workload schema.
- Describe both simulator modes.
- Document validation steps.
- Document sweep commands and plot generation.

Deliverables:
- Focused MQA guide for running the new simulator.

### Step 8.3: Update packaging if needed

Modify if required:
- `requirements.txt`
- `setup.py`

Tasks:
- Ensure the new `mqa_scalesim` package is installed or importable.
- Add any test dependencies if needed.

Deliverables:
- Clean imports and reproducible environment setup.

## Recommended execution order

The recommended implementation order is:

1. Preserve the current baseline outputs and confirm ordinary SCALE-Sim still works.
2. Create the new `mqa_scalesim/` package and test scaffolding.
3. Extend `scale_config.py`, `topology_utils.py`, and `simulator.py` for new workload types.
4. Refactor `generate_scalesim_topology.py` around the canonical workload object.
5. Implement the baseline decode path first using existing compute backends plus a new shared softmax pipeline.
6. Implement the KV-stationary compute backend and integrate it into `single_layer_sim.py`.
7. Extend memory modeling and reporting.
8. Update comparison scripts, plots, and documentation.

## File creation summary

Create these new files:

- `PLAN.MD`
- `mqa_scalesim/__init__.py`
- `mqa_scalesim/workload.py`
- `mqa_scalesim/baseline_decode.py`
- `mqa_scalesim/kv_stationary_decode.py`
- `mqa_scalesim/softmax_ops.py`
- `mqa_scalesim/pe_timing.py`
- `mqa_scalesim/result_schema.py`
- `mqa_scalesim/validation_bridge.py`
- `scalesim/compute/systolic_compute_kv_mqa.py`
- `scalesim/compute/softmax_pipeline.py`
- `test/test_mqa_workload.py`
- `test/test_baseline_decode_cycles.py`
- `test/test_kv_stationary_cycles.py`
- `test/test_softmax_microops.py`
- `test/test_mqa_vs_reference.py`

## File modification summary

Modify these existing files:

- `generate_scalesim_topology.py`
- `baseline_mqa_model.py`
- `kv_stationary_model.py`
- `compare.py`
- `mqa_correctness.py`
- `plot_results.py`
- `make_autoregressive_plots.py`
- `make_plot11.py`
- `README.md`
- `README_MQA.md`
- `requirements.txt` (if needed)
- `setup.py` (if needed)
- `scalesim/scale_config.py`
- `scalesim/topology_utils.py`
- `scalesim/simulator.py`
- `scalesim/single_layer_sim.py`
- `scalesim/compute/operand_matrix.py`
- `scalesim/compute/systolic_compute_os.py` (only if baseline path needs light adaptation)
- `scalesim/compute/systolic_compute_is.py` (only if baseline path needs light adaptation)
- `scalesim/compute/systolic_compute_ws.py` (only if baseline path needs light adaptation)
- `scalesim/memory/double_buffered_scratchpad_mem.py`
- `scalesim/memory/read_buffer.py`
- `scalesim/memory/write_buffer.py`
- `scalesim/memory/read_port.py` (only if needed)
- `scalesim/memory/write_port.py` (only if needed)
- `scalesim/memory/read_buffer_estimate_bw.py` (only if needed)
- `scalesim/utilities/scalesim_report.py`

## Ground rules for implementation

The following implementation rules should be enforced throughout the project:

- Final cycle counts must come from SCALE-Sim, not from side calculations.
- The baseline and KV-stationary designs must use the same memory model and reporting infrastructure so the comparison is fair.
- `CPP__Simulation/attention/*` remains for correctness/reference only, not for performance claims.
- The old analytical `kv_stationary_model.py` logic may be preserved only as a debug oracle, not as the main evaluation path.
- Softmax/reduction timing must be explicitly modeled inside the simulator, because the report identified this as the main technical challenge and previous analytical approximation is no longer acceptable.

## First step to execute next

The first implementation step should be to scaffold the new `mqa_scalesim/` package and the associated test files, because those provide the shared workload schema and result format that all later SCALE-Sim modifications will depend on.