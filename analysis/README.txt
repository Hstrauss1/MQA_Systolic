Analysis — KV-Stationary MQA Accelerator
==========================================

All project Python files live here. Run scripts from this directory.

Core Models (imported by scripts)
----------------------------------
  kv_stationary_model.py      Main analytical + pipeline model. Key functions:
                              simulate_2d_kv_stationary_array(), kv_stationary_metrics()
  kv_reuse_model.py           Multi-pass silicon re-use extension (num_passes, causal)
  baseline_mqa_model.py       Roofline baseline: fused FlashAttention + unfused 2-GEMM
  mqa_correctness.py          Numerical correctness checks (softmax, online update)

Validation Scripts (require SCALE-Sim installed)
--------------------------------------------------
  validate_lower_mac_sweep.py  29-config Q·K GEMM sweep → ../results/data/validate_lower_mac_sweep.csv
  validate_sweep_scalesim.py   40-run wavefront fill validation → ../results/data/validate_sweep_scalesim.csv
  prefill_element_streaming.py  Cycle breakdown per config → ../results/data/prefill_element_streaming.csv
  generate_scalesim_topology.py  Helper: write SCALE-Sim topology/config files

Sweep Scripts
-------------
  reuse_full_sweep.py   750-row BW×n×lmc×T×P sweep → ../results/data/reuse_full_sweep.csv

Plotting Scripts (write to ../results/figures/)
------------------------------------------------
  plot_prefill_corrected.py   figs 1–3, 5, 7  (reads prefill_element_streaming.csv)
  make_corrected_figs.py      figs 4, 6, 13–16 (reads validate_sweep + reuse_full_sweep)
  make_reuse_figs.py          figs 7–12 (multi-pass re-use model)
  make_reuse_plots.py         plots A, B (decode sweep)
  make_prefill_plots.py       plots C–F (causal prefill sweep)
  make_results_figs.py        figs 17–20 (wavefront model, core results section)

Usage
-----
  cd MQA_SCALE_SIM/analysis
  python3 plot_prefill_corrected.py   # regenerate figures 1-3, 5, 7
  python3 make_reuse_figs.py          # regenerate figures 7-12

All figures → ../results/figures/
All data    → ../results/data/
Report      → ../report/report.tex   (compile with: cd ../report && pdflatex report.tex && biber report && pdflatex report.tex && pdflatex report.tex)
