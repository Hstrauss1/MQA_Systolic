# Project Metadata

## Repo Context

- Repository: `MQA_SCALE_SIM`
- Base project: SCALE-Sim
- Purpose of fork: evaluate baseline GEMM-style Multi-Query Attention (MQA)
  against a custom KV-stationary analytical model for a computer architecture
  class project

## Work Completed

### 1. Added MQA analysis and verification scripts

Added these repo-root files:

- `mqa_correctness.py`
- `baseline_mqa_model.py`
- `kv_stationary_model.py`
- `generate_scalesim_topology.py`
- `compare.py`
- `plot_results.py`
- `README_MQA.md`

### 2. Implemented MQA numerical correctness flow

- Added `reference_mqa(Q, K, V)` using dense attention:
  `scores = Q @ K.T`, `attn = softmax(scores)`, `out = attn @ V`
- Added `kv_stationary_mqa(Q, K, V)` using streaming online softmax
- Added a random-input test and max-absolute-error reporting

### 3. Implemented analytical performance models

- Added baseline two-GEMM MQA model in `baseline_mqa_model.py`
- Added KV-stationary analytical model in `kv_stationary_model.py`
- Kept the modeling boundary explicit:
  - SCALE-Sim is for the baseline GEMM path
  - KV-stationary is a separate analytical extension
  - KV-stationary estimates are not claimed to be cycle-accurate

### 4. Implemented baseline SCALE-Sim topology generation

- Added `generate_scalesim_topology.py`
- Generates `mqa_baseline_gemm.csv` with:
  - `QK_scores, H, T, d`
  - `AV_output, H, d, T`
- Updated the generator to match SCALE-Sim GEMM CSV expectations by writing a
  trailing empty column, because the SCALE-Sim GEMM parser drops the final CSV
  field on each row

### 5. Implemented comparison and plotting flow

- Added `compare.py`
- Sweeps `T = [128, 512, 1024, 2048, 4096, 8192]`
- Uses defaults:
  - `H = 32`
  - `d = 128`
  - `array_rows = 64`
  - `array_cols = 64`
  - `bytes_per_element = 2`
  - `memory_bandwidth_bytes_per_cycle = 512`
- Writes `results.csv`
- Added `plot_results.py`
- Plots cycles, DRAM traffic, and arithmetic intensity vs sequence length
- Forces matplotlib `Agg` backend to avoid macOS AppKit crashes in headless runs

### 6. Updated documentation

- Added `README_MQA.md` for the MQA workflow
- Updated `README.md` with a fork-specific section describing:
  - what changed from vanilla SCALE-Sim
  - the MQA scripts
  - the analytical KV-stationary boundary
  - a quick-start flow
- Updated `requirements.txt` comments to note that the listed dependencies also
  cover the added MQA scripts

### 7. Recentered git remote configuration

- Set:
  - `origin` -> `https://github.com/Hstrauss1/MQA_SCALE_SIM.git`
  - `upstream` -> `https://github.com/scalesim-project/SCALE-Sim.git`

### 8. Fixed a base SCALE-Sim runtime bug

File changed:

- `scalesim/memory/double_buffered_scratchpad_mem.py`

Bug:

- Base SCALE-Sim runs were failing in:
  `self.total_cycles = int(max(ofmap_serviced_cycles))`
- This broke built-in convolution, built-in GEMM, and generated MQA GEMM runs
  because `max()` over NumPy scalar/array entries could return a non-scalar
  object

Fix:

- Replaced the broken total-cycle assignment with:
  `self.total_cycles = int(self.ofmap_trace_matrix[-1][0])`

## Verification Performed

### MQA scripts

Verified in this repo with the sibling virtualenv Python:

- `../scalesim/bin/python -m py_compile mqa_correctness.py baseline_mqa_model.py kv_stationary_model.py generate_scalesim_topology.py compare.py plot_results.py`
- `../scalesim/bin/python mqa_correctness.py`
- `../scalesim/bin/python generate_scalesim_topology.py`
- `../scalesim/bin/python compare.py`
- `../scalesim/bin/python plot_results.py`

Observed result for correctness:

- `H=32, T=512, d=128, seed=0`
- `Max absolute error: 1.11460686e-05`
- `Allclose(rtol=1e-4, atol=1e-4): True`

Generated artifacts:

- `mqa_baseline_gemm.csv`
- `results.csv`
- `results.png`

### Base SCALE-Sim operations

After the memory-system fix, verified these runs complete:

- `../scalesim/bin/python -m scalesim.scale -c configs/scale.cfg -t topologies/conv_nets/Resnet_test.csv -p verify_outputs/conv`
- `../scalesim/bin/python -m scalesim.scale -c configs/scale.cfg -t topologies/GEMM_mnk/vit_s.csv -i gemm -p verify_outputs/gemm`
- `../scalesim/bin/python -m scalesim.scale -c configs/scale.cfg -t mqa_baseline_gemm.csv -i gemm -p verify_outputs/mqa_gemm`

Observed completion notes:

- Built-in convolution run completed
- Built-in GEMM run completed
- Generated MQA baseline GEMM run completed

## Important Modeling Notes

- The correctness model verifies numerical equivalence of the KV-stationary
  online-softmax algorithm against standard NumPy attention.
- It does not compare against SCALE-Sim outputs.
- SCALE-Sim is used for baseline GEMM-style modeling only.
- The KV-stationary architecture is evaluated analytically and is not presented
  as natively supported or cycle-accurate inside SCALE-Sim.

## Current Repo Notes

- The working tree currently includes the real source fix in
  `scalesim/memory/double_buffered_scratchpad_mem.py`.
- There are also generated `__pycache__` updates from local verification runs.
- A top-level `LICENSE` file is currently untracked in this repo copy.
