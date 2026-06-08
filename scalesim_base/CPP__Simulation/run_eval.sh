#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp32}"
SEQ_LEN="${SEQ_LEN:-128}"
BENCHMARK_OUT="${BENCHMARK_OUT:-$ROOT_DIR/benchmark.csv}"

# PyTorch wheel index can be overridden for your cluster or driver setup.
# For V100 systems, cu118 is a conservative default that works well on many
# servers with slightly older NVIDIA drivers.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch torchvision torchaudio}"
EXTRA_PACKAGES="${EXTRA_PACKAGES:-pytest}"

echo "==> Repo root: $ROOT_DIR"
echo "==> Creating virtual environment at: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "==> Activating environment"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing PyTorch from: $TORCH_INDEX_URL"
python -m pip install --index-url "$TORCH_INDEX_URL" $TORCH_PACKAGES

echo "==> Installing test dependencies"
python -m pip install $EXTRA_PACKAGES

if [[ "$DEVICE" == "cuda" ]]; then
  echo "==> Checking CUDA availability"
  python - <<'PY'
import torch
print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA device requested but torch.cuda.is_available() is False")
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
PY
else
  echo "==> Using CPU evaluation"
fi

echo "==> Running pytest correctness suite"
pytest "$ROOT_DIR/attention/tests.py"

echo "==> Running single-pass sanity checks"
python "$ROOT_DIR/run.py" --mode baseline --seq-len "$SEQ_LEN" --device "$DEVICE" --dtype "$DTYPE" --causal
python "$ROOT_DIR/run.py" --mode streaming --seq-len "$SEQ_LEN" --device "$DEVICE" --dtype "$DTYPE" --causal
python "$ROOT_DIR/run.py" --mode integer --seq-len "$SEQ_LEN" --device "$DEVICE" --dtype "$DTYPE" --causal

echo "==> Running full benchmark sweep"
python -m attention.benchmark > "$BENCHMARK_OUT"

echo "==> Benchmark CSV written to: $BENCHMARK_OUT"
echo "==> Finished successfully"
