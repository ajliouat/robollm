#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# Run RoboLLM full training pipeline on AWS T4 (g4dn.xlarge)
#
# Usage:
#   chmod +x training/run_aws.sh
#   ./training/run_aws.sh
#
# This script:
#   1. Installs dependencies
#   2. Trains all primitives (pick, move_to, place, color_pick)
#   3. Runs the full benchmark including trained SAC policies
#   4. Saves results to evaluation/results/
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

echo "=== RoboLLM AWS Training Pipeline ==="
echo ""

# ── Config ────────────────────────────────────────────────────────
STEPS=500000
DEVICE="cuda"
SEED=42
CKPT_DIR="checkpoints"

# ── 1. Install dependencies ───────────────────────────────────────
echo ">>> Installing dependencies..."
pip install -e ".[dev]" --quiet
pip install tensorboard --quiet

# ── 2. Train primitives ───────────────────────────────────────────
echo ""
echo ">>> Training pick primitive (${STEPS} steps)..."
python -m training.train_all \
    --primitives pick,move_to,place \
    --steps ${STEPS} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --log-dir ${CKPT_DIR}

# ── 3. Run benchmark with trained policies ────────────────────────
echo ""
echo ">>> Running benchmark with SAC policies..."
python -m evaluation.benchmark \
    --episodes 100 \
    --seed ${SEED} \
    --sac-pick ${CKPT_DIR}/pick/best.pt \
    --sac-move-to ${CKPT_DIR}/move_to/best.pt

# ── 4. Summary ────────────────────────────────────────────────────
echo ""
echo "=== Training complete ==="
echo "Checkpoints: ${CKPT_DIR}/"
echo "Results:     evaluation/results/"
