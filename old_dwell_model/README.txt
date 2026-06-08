OLD / DWELL MODEL — SUPERSEDED
===============================
These files were produced under the original "dwell" pipeline-fill model
(A model: charges column_dwell=135 cycles per pipeline step).

The dwell model was ruled out by SCALE-Sim in June 2026, which measured
~2.5 cycles/column fill — off by ~50x from the dwell prediction.
The correct model is the "wavefront" model (B) implemented in
kv_stationary_model.py (wavefront_fill=True, the default).

Files here should NOT be used for new analysis. They are kept for audit
trail and comparison purposes only.

KEY FILES (current, correct model):
  kv_stationary_model.py        -- wavefront model (default)
  kv_reuse_model.py             -- multi-pass re-use extension
  prefill_element_streaming.py  -- compute-only element-streaming model
  validate_lower_mac_sweep.py   -- SCALE-Sim validated results
  reuse_full_sweep.py           -- 750-row BW x lmc x T x P sweep
  plot_prefill_corrected.py     -- publication figures
