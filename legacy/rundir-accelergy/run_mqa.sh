#!/bin/bash
# Full MQA → SCALE-Sim → Accelergy pipeline
#
# Usage: ./run_mqa.sh
#        from inside rundir-accelergy/
#
# Outputs:
#   output/scale_sim_output_mqa_decode_64x64_os/  — SCALE-Sim reports
#   output/accelergy_output_mqa_decode_64x64_os/  — Accelergy energy breakdown

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MQA_ROOT="$(dirname "$SCRIPT_DIR")"
SCALE_SIM_ROOT="/Users/hudsons/Code/ScaleSim/SCALE-Sim"
VENV="$SCALE_SIM_ROOT/scale"
PYTHON="$VENV/bin/python3"
ACCELERGY="$VENV/bin/accelergy"

CFG="$MQA_ROOT/configs/mqa_accelergy.cfg"
TOPOLOGY="$MQA_ROOT/topologies/mqa_decode_H64_T8192_d128.csv"
SCALESIM_LOG="$MQA_ROOT/rundir-accelergy/scalesim_output"
OUTDIR="$SCRIPT_DIR/output"

mkdir -p "$SCALESIM_LOG"
mkdir -p "$OUTDIR"
mkdir -p "$SCRIPT_DIR/accelergy_input"
mkdir -p "$SCRIPT_DIR/accelergy_input/components"

# ── Step 1: Generate architecture.yaml from SCALE-Sim config ─────────────────
echo ""
echo "=== Step 1: Preprocessing config → Accelergy architecture.yaml ==="
cd "$SCRIPT_DIR"
$PYTHON preprocess.py -c "$CFG" -t "$TOPOLOGY" -p "$SCALESIM_LOG" -o "$OUTDIR"

# ── Step 2: Run SCALE-Sim (gemm mode, output-stationary) ─────────────────────
echo ""
echo "=== Step 2: Running SCALE-Sim ==="
cd "$MQA_ROOT"
PYTHONPATH="$PYTHONPATH:$MQA_ROOT" $PYTHON -m scalesim.scale \
    -c "$CFG" \
    -t "$TOPOLOGY" \
    -p "$SCALESIM_LOG" \
    -i gemm

# ── Step 3: Extract action counts from SCALE-Sim output ──────────────────────
echo ""
echo "=== Step 3: Extracting action counts ==="
cd "$SCRIPT_DIR"
RUN_NAME=$(grep "run_name" "$CFG" | awk -F'=' '{print $2}' | tr -d ' ')

$PYTHON create_action_count.py \
    --saved_folder "$SCALESIM_LOG" \
    --run_name "$RUN_NAME" \
    --arch_name "systolic_array" \
    --SRAM_row_size 2 \
    --DRAM_row_size 2 \
    --config "$CFG"

cp "$SCALESIM_LOG/$RUN_NAME/action_count.yaml" "$SCRIPT_DIR/accelergy_input/action_count.yaml"
echo "action_count.yaml written to accelergy_input/"

# ── Step 4: Run Accelergy ─────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Running Accelergy ==="
cd "$SCRIPT_DIR"
ACCEL_OUT="$SCRIPT_DIR/accelergy_output/$RUN_NAME"
mkdir -p "$ACCEL_OUT"

$ACCELERGY \
    accelergy_input/*.yaml \
    accelergy_input/components/*.yaml \
    -o "$ACCEL_OUT" \
    -v 1

# ── Step 5: Copy outputs to final dir ────────────────────────────────────────
echo ""
echo "=== Step 5: Collecting outputs ==="
SCALE_OUT_DEST="$OUTDIR/scale_sim_output_$RUN_NAME"
ACCEL_OUT_DEST="$OUTDIR/accelergy_output_$RUN_NAME"

mkdir -p "$SCALE_OUT_DEST"
cp -r "$SCALESIM_LOG/$RUN_NAME/"* "$SCALE_OUT_DEST/"

mkdir -p "$ACCEL_OUT_DEST"
cp -r "$ACCEL_OUT/"* "$ACCEL_OUT_DEST/"

echo ""
echo "=== Done ==="
echo "SCALE-Sim output : $SCALE_OUT_DEST"
echo "Accelergy output : $ACCEL_OUT_DEST"
echo ""
echo "Energy breakdown:"
if [ -f "$ACCEL_OUT_DEST/energy_estimation.yaml" ]; then
    $PYTHON "$SCRIPT_DIR/summarize_energy.py" "$ACCEL_OUT_DEST/energy_estimation.yaml"
else
    echo "  (energy_estimation.yaml not found — check accelergy logs)"
fi
