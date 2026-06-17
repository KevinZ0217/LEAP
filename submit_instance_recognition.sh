#!/bin/bash
#SBATCH --job-name=retr_oxparis
#SBATCH --output=slurm_logs/slurm_instance_retrieval_%j.out
#SBATCH --error=slurm_logs/slurm_instance_retrieval_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --mem=64G
#SBATCH --partition=gpus

# ============================================================================
# Revisited Oxford / Paris instance retrieval (eval_instance_retrieval.py).
# ============================================================================
# Default: --from-official (Oxford/VGG tgz + revisitop gnd.pkl, no Hugging Face
# `datasets`). HF galilai-group/revisitop fails on datasets>=3.
#
# Student backbone size (must match training):
#   STUDENT_SIZE=small  (default) ViT-S    STUDENT_SIZE=tiny  ViT-T
#   (for 'tiny' the model is built directly, so MODEL_NAME is ignored)
#
# Usage:
#   TEST_DATASET=roxford5k CHECKPOINT=/path/exported_last.pt sbatch submit_instance_recognition.sh
#   TEST_DATASET=rparis6k  CHECKPOINT=/path/exported_last.pt sbatch submit_instance_recognition.sh
#   STUDENT_SIZE=tiny TEST_DATASET=roxford5k CHECKPOINT=... sbatch submit_instance_recognition.sh
#
# Data already prepared (no download):
#   FROM_PREPARE=0 TEST_DATASET=rparis6k RUN_DIR=... sbatch submit_instance_recognition.sh
# ============================================================================

set -euo pipefail

mkdir -p slurm_logs

# Override for your own environment:
#   PYTHON_BIN=/path/python REPO_DIR=/path/LEAP REVISIT_ROOT=/path/revisitop \
#     CHECKPOINT=/path/model.ckpt sbatch submit_instance_recognition.sh
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="${REPO_DIR:-/path/to/LEAP}"
SCRIPT_PATH="$SCRIPT_DIR/eval_instance_retrieval.py"

RUN_DIR="${RUN_DIR:-/path/to/output/leap_baseline}"
RUN_DIR="${RUN_DIR%/}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/exported_models/exported_last.pt}"
# REVISIT_ROOT: parent dir where roxford5k / rparis6k live (or get downloaded
# to with FROM_OFFICIAL=1). Point this wherever you want the data.
REVISIT_ROOT="${REVISIT_ROOT:-/path/to/revisitop}"
TEST_DATASET="${TEST_DATASET:-roxford5k}" # roxford5k | rparis6k
FROM_PREPARE="${FROM_PREPARE:-1}"          # 0 = skip --from-official if data exists
FROM_OFFICIAL="${FROM_OFFICIAL:-1}"        # 1 = download from Oxford/VGG + revisitop
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR}"

STUDENT_SIZE="${STUDENT_SIZE:-small}"
# MODEL_NAME selects the timm backbone when STUDENT_SIZE=small. When
# STUDENT_SIZE=tiny the ViT-T architecture is built directly in
# eval_linear.load_model and MODEL_NAME is ignored.
MODEL_NAME="${MODEL_NAME:-vit_small_patch14_dinov2.lvd142m}"
PATCH_SIZE="${PATCH_SIZE:-}"
MODEL_IMG_SIZE="${MODEL_IMG_SIZE:-}"
ADAPT_PATCH_SIZE="${ADAPT_PATCH_SIZE:-}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
FEATURE_TYPE="${FEATURE_TYPE:-cls}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OVERWRITE_CACHE="${OVERWRITE_CACHE:-0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python not found: $PYTHON_BIN (set PYTHON_BIN=/path/to/python)"
    exit 1
fi
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: $SCRIPT_PATH not found"
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$SCRIPT_DIR"

CMD=(
    "$PYTHON_BIN" "$SCRIPT_PATH"
    --checkpoint "$CHECKPOINT"
    --test-dataset "$TEST_DATASET"
    --revisit-root "$REVISIT_ROOT"
    --output-dir "$OUTPUT_DIR"
    --student-size "$STUDENT_SIZE"
    --model-name "$MODEL_NAME"
    --feature-type "$FEATURE_TYPE"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
)
if [ -n "$PATCH_SIZE" ]; then CMD+=(--patch-size "$PATCH_SIZE"); fi
if [ -n "$MODEL_IMG_SIZE" ]; then CMD+=(--model-img-size "$MODEL_IMG_SIZE"); fi
if [ -n "$ADAPT_PATCH_SIZE" ]; then CMD+=(--adapt-patch-size "$ADAPT_PATCH_SIZE"); fi
if [ -n "$IMAGE_SIZE" ]; then CMD+=(--image-size "$IMAGE_SIZE"); fi
if [ "$FROM_PREPARE" = "1" ] || [ "$FROM_PREPARE" = "true" ]; then
    if [ "$FROM_OFFICIAL" = "1" ] || [ "$FROM_OFFICIAL" = "true" ]; then
        CMD+=(--from-official)
    fi
fi

# shellcheck disable=SC2206
CMD+=($EXTRA_ARGS)
if [ "$OVERWRITE_CACHE" = "1" ] || [ "$OVERWRITE_CACHE" = "true" ]; then
    CMD+=(--overwrite-hf-cache)
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
