# MQA KV-Stationary Study

This mini-project adds a simple comparison flow for Multi-Query Attention
decode inside the SCALE-Sim repository.

Important modeling boundary:

- SCALE-Sim is used for the baseline GEMM-style MQA mapping through an MNK
  topology CSV and the `-i gemm` option.
- The KV-stationary architecture is evaluated with a separate analytical model.
- The KV-stationary model is not cycle-accurate.
- The correctness script checks numerical equivalence of the online-softmax
  algorithm against NumPy attention. It does not compare against SCALE-Sim
  output.

## Files

- `mqa_correctness.py`: NumPy reference attention and online-softmax
  KV-stationary correctness check
- `baseline_mqa_model.py`: simple analytical two-GEMM baseline model
- `kv_stationary_model.py`: simple analytical KV-stationary model
- `generate_scalesim_topology.py`: writes a SCALE-Sim GEMM topology CSV
- `compare.py`: sweeps sequence length and saves `results.csv`
- `plot_results.py`: plots the saved comparison results using matplotlib

## How to run

Install dependencies first:

```bash
pip3 install -r requirements.txt
```

```bash
python3 mqa_correctness.py
python3 generate_scalesim_topology.py
python3 compare.py
python3 plot_results.py
```

If your environment provides `python` instead of `python3`, the same commands
work with `python`.

## SCALE-Sim baseline usage

Generate the baseline GEMM topology:

```bash
python3 generate_scalesim_topology.py --heads 32 --seq-len 2048 --head-dim 128
```

Then run SCALE-Sim with GEMM input mode:

```bash
python3 -m scalesim.scale -c configs/scale.cfg -t mqa_baseline_gemm.csv -i gemm
```
