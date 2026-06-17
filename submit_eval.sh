#!/bin/bash
#SBATCH --job-name=leap_linear
#SBATCH --output=slurm_logs/slurm_eval_%j.out
#SBATCH --error=slurm_logs/slurm_eval_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --mem=40G
#SBATCH --partition=gpus

# ============================================================================
# Linear Probing Evaluation for LEAP Students (SLURM)
# ============================================================================
# Evaluates a "fair" ViT-S/14 student (patch14, img_size=224) with a linear
# probe on top of frozen features.
#
# Feature type:
#   FEAT_TYPE=cls       (default) use CLS token
#   FEAT_TYPE=avgpool   use avg-pooled patch tokens
#
# Student backbone size (must match training):
#   STUDENT_SIZE=small  (default) ViT-S    STUDENT_SIZE=tiny  ViT-T
#
# Dataset:
#   DATASET=mini-imagenet   evaluate on mini-ImageNet
#   DATASET=imagenet-1k     (default) evaluate on ImageNet-1K
#   DATASET=/custom/path    custom dataset root
#
# Usage:
#   CKPT=/path/to/exported_last.pt sbatch submit_eval.sh
#   DATASET=imagenet-1k CKPT=/path/to/exported_last.pt sbatch submit_eval.sh
#   STUDENT_SIZE=tiny CKPT=... sbatch submit_eval.sh
#   FEAT_TYPE=avgpool CKPT=... sbatch submit_eval.sh
# ============================================================================

# ---- Dataset selection ----
# Linear probing needs the dataset ROOT (containing train/ and val/). Override
# for your own machine via DATA_DIR / IMAGENET_DIR / MINI_IMAGENET_DIR / OUT_BASE,
# or pass DATASET=/abs/path.
DATASET="${DATASET:-imagenet-1k}"
if [ "$DATASET" = "imagenet-1k" ]; then
    DATA_DIR="${DATA_DIR:-${IMAGENET_DIR:-/path/to/imagenet}}"
    OUT_BASE="${OUT_BASE:-/path/to/output}"
elif [ "$DATASET" = "mini-imagenet" ]; then
    DATA_DIR="${DATA_DIR:-${MINI_IMAGENET_DIR:-/path/to/mini-imagenet}}"
    OUT_BASE="${OUT_BASE:-/path/to/output}"
else
    DATA_DIR="${DATA_DIR:-$DATASET}"
    OUT_BASE="${OUT_BASE:-/path/to/output}"
fi

CKPT="${CKPT:-$OUT_BASE/leap_baseline/exported_models/exported_last.pt}"
FEAT_TYPE="${FEAT_TYPE:-cls}"
STUDENT_SIZE="${STUDENT_SIZE:-small}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
COMMON="--data-dir $DATA_DIR --feature-type $FEAT_TYPE --student-size $STUDENT_SIZE --probe-epochs 100 --probe-lr 0.1 --probe-batch-size 256"

echo "=========================================="
echo "LEAP Linear Probe Evaluation (SLURM)"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $SLURM_NODELIST"
echo "  Dataset:    $DATASET"
echo "  Student:    $STUDENT_SIZE (small=ViT-S / tiny=ViT-T)"
echo "  Feature:    $FEAT_TYPE"
echo "  Checkpoint: $CKPT"
echo "  Data:       $DATA_DIR"
echo "  Start:      $(date)"
echo "=========================================="

mkdir -p "$OUT_BASE"

# Override for your own environment: PYTHON=/path/python REPO_DIR=/path/LEAP
PYTHON="${PYTHON:-python}"
REPO_DIR="${REPO_DIR:-/path/to/LEAP}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: Python not found: $PYTHON (set PYTHON=/path/to/python)"
    exit 1
fi

cd "$REPO_DIR"

# shellcheck disable=SC2086
$PYTHON eval_linear.py \
    --checkpoint "$CKPT" \
    --image-size 224 \
    --model-img-size 224 \
    $COMMON \
    $EXTRA_EVAL_ARGS

echo "=========================================="
echo "End Time: $(date)"
echo "Evaluation complete!"
echo "=========================================="
