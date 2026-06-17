#!/bin/bash
#SBATCH --job-name=leap_train
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --mem=100G
#SBATCH --partition=qed

# ============================================================================
# LEAP: Distill DINOv2 ViT-G -> ViT-S (SLURM)
# ============================================================================
# The student is always ViT-S/14 built explicitly at img_size=224 (the "fair"
# setup) with a single teacher target block (last-n = 1).
#
# Dataset:
#   DATASET=mini-imagenet   /path/to/mini-imagenet
#   DATASET=imagenet-1k     /path/to/imagenet
#   DATASET=/custom/path    any custom path (must contain a train/ subdir)
#
# Student backbone size:
#   STUDENT_SIZE=small      (default) ViT-S (embed=384, depth=12, heads=6)
#   STUDENT_SIZE=tiny       ViT-T (embed=192, depth=12, heads=3, ~5.6M params)
#
# Multi-GPU:
#   LightlyTrain auto-detects all GPUs allocated by SLURM and uses DDP.
#
# Usage:
#   # Baseline distillation (CLS supervision):
#   DATASET=imagenet-1k EPOCHS=150 BATCH_SIZE=2048 LR=3.15 \
#     CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 sbatch submit_train.sh
#
#   # Curriculum block-skipping distillation (CLS supervision):
#   CURRICULUM=1 DATASET=imagenet-1k CKA_THRESHOLD=0.8 EMA_ALPHA=1 \
#     MAX_EPOCHS_BLOCK=5 EPOCHS=100 BATCH_SIZE=2048 LR=3.15 \
#     CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 sbatch submit_train.sh
# ============================================================================

EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
PRECISION="${PRECISION:-16-mixed}"
LR="${LR:-}"

# Student always built explicitly at 224 (fair) with a single teacher target.
MODE="fair"
LAST_N=1

# Student backbone size: 'small' (ViT-S, default) or 'tiny' (ViT-T).
STUDENT_SIZE="${STUDENT_SIZE:-small}"

# Teacher model: any name supported by lightly_train (default ViT-G/14 w/ regs).
TEACHER_MODEL="${TEACHER_MODEL:-dinov2/vitg14}"

# Override the auto-derived OUT_DIR with a custom path.
OUT_DIR_OVERRIDE="${OUT_DIR_OVERRIDE:-}"

LR_FLAG=""
if [ -n "$LR" ]; then
    LR_FLAG="--lr $LR"
fi

# Curriculum block-skipping distillation.
CURRICULUM_FLAG=""
if [ "${CURRICULUM:-0}" = "1" ]; then
    CURRICULUM_FLAG="--curriculum --cka-threshold ${CKA_THRESHOLD:-0.85} --ema-alpha ${EMA_ALPHA:-0.05} --max-epochs-per-block ${MAX_EPOCHS_BLOCK:-50}"
fi

# CLS token supervision.
CLS_LOSS_FLAG=""
if [ "${CLS_LOSS:-0}" = "1" ]; then
    CLS_LOSS_FLAG="--cls-loss --cls-loss-weight ${CLS_LOSS_WEIGHT:-1.0}"
fi

# ---- Dataset selection ----
# Point these at wherever you downloaded the data. Three ways to override:
#   1. DATA_DIR=/abs/path/to/train OUT_BASE=/abs/out   (most explicit)
#   2. IMAGENET_DIR=/abs/imagenet  MINI_IMAGENET_DIR=/abs/mini-imagenet
#      (dataset root; the script appends /train automatically)
#   3. DATASET=/abs/path           (custom root containing a train/ subdir)
DATASET="${DATASET:-imagenet-1k}"
if [ "$DATASET" = "imagenet-1k" ]; then
    DATA_DIR="${DATA_DIR:-${IMAGENET_DIR:-/path/to/imagenet}/train}"
    OUT_BASE="${OUT_BASE:-/path/to/output}"
elif [ "$DATASET" = "mini-imagenet" ]; then
    DATA_DIR="${DATA_DIR:-${MINI_IMAGENET_DIR:-/path/to/mini-imagenet}/train}"
    OUT_BASE="${OUT_BASE:-/path/to/output}"
else
    DATA_DIR="${DATA_DIR:-$DATASET/train}"
    OUT_BASE="${OUT_BASE:-/path/to/output}"
fi

if [ "${CURRICULUM:-0}" = "1" ]; then
    OUT_DIR="$OUT_BASE/leap_curriculum"
else
    OUT_DIR="$OUT_BASE/leap_baseline"
fi
if [ -n "$OUT_DIR_OVERRIDE" ]; then
    OUT_DIR="$OUT_DIR_OVERRIDE"
fi

echo "=========================================="
echo "LEAP Distillation (SLURM)"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $SLURM_NODELIST"
echo "  GPUs:       ${SLURM_GPUS_ON_NODE:-1} (allocated by SLURM)"
echo "  Dataset:    $DATASET"
echo "  Student:    $STUDENT_SIZE (small=ViT-S / tiny=ViT-T)"
echo "  Curriculum: ${CURRICULUM:-0}"
echo "  Teacher:    $TEACHER_MODEL"
echo "  Epochs:     $EPOCHS"
echo "  Batch:      $BATCH_SIZE (global, divided across GPUs by LightlyTrain)"
echo "  LR:         ${LR:-auto}"
echo "  CLS loss:   ${CLS_LOSS:-0} (weight=${CLS_LOSS_WEIGHT:-1.0})"
echo "  Precision:  $PRECISION"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "  Start:      $(date)"
echo "=========================================="

mkdir -p "$OUT_BASE"

# When multiple GPUs are allocated, Lightning's SLURMEnvironment expects
# SLURM to have spawned one process per GPU. Since we use --ntasks=1 and let
# Lightning manage DDP internally, disable SLURM auto-detection by setting the
# job name to "interactive" (official Lightning workaround).
NUM_ALLOC_GPUS="${SLURM_GPUS_ON_NODE:-1}"
if [ "$NUM_ALLOC_GPUS" -gt 1 ]; then
    export SLURM_JOB_NAME="interactive"
fi

# Python interpreter and repo location (override for your own environment):
#   PYTHON=/path/to/venv/bin/python REPO_DIR=/path/to/LEAP sbatch submit_train.sh
PYTHON="${PYTHON:-python}"
REPO_DIR="${REPO_DIR:-/path/to/LEAP}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: Python not found: $PYTHON (set PYTHON=/path/to/python)"
    exit 1
fi

cd "$REPO_DIR"

echo "Python:      $PYTHON"
echo "PyTorch:     $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA:        $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"
echo "LightlyTrain:$($PYTHON -c 'import lightly_train; print(lightly_train.__version__)')"
echo "=========================================="

$PYTHON train_distill.py \
    --data "$DATA_DIR" \
    --out "$OUT_DIR" \
    --mode "$MODE" \
    --student-size "$STUDENT_SIZE" \
    --last-n "$LAST_N" \
    --teacher-model "$TEACHER_MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --precision "$PRECISION" \
    --num-workers 8 \
    $LR_FLAG \
    $CURRICULUM_FLAG \
    $CLS_LOSS_FLAG

echo "=========================================="
echo "End Time: $(date)"
echo "Distillation complete!"
echo "=========================================="
