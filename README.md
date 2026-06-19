# LEAP: Layer-skipping Efficiency via Adaptive Progression for Vision Transformer Distillation

[Project Page](https://kevinz0217.github.io/LEAP_page/)
[Paper](https://arxiv.org/abs/2606.19483)
## Introduction

Vision Foundation Models (VFMs) with Vision Transformer (ViT) backbones, such as DINOv2, have become essential for downstream tasks like object recognition and instance retrieval. The immense computational requirements of large teachers often necessitate distillation into compact students for edge deployment. Feature-based knowledge distillation (KD) is a strong paradigm for ViTs, but a small student can struggle to imitate a large teacher's complex feature maps in a single step due to the teacher-student capacity gap.

In this work, we propose **LEAP**: **L**ayer-skipping **E**fficiency via **A**daptive **P**rogression, a training curriculum for ViT feature-based KD. Rather than supervising the student against a fixed teacher block from the start, LEAP advances the supervisory target through the teacher's feature maps shallow-to-deep based on online CKA alignment, allowing the student to build a foundational representation before tackling higher-level abstractions. We distill DINOv2 ViT-G/14 into ViT-S/14 (or ViT-T/14) students using [LightlyTrain](https://github.com/lightly-ai/lightly-train), with optional CLS-token supervision.

## Get Started

Please install the required packages:

```bash
pip install -r requirements.txt
```

All scripts use generic `/path/to/...` placeholders. Set your own paths via environment variables before launching jobs (no need to edit the Python files). If you use SLURM, also edit the `#SBATCH` header lines (`--partition`, `--gres`, `--mem`) in the submit scripts to match your cluster.

| Variable | Meaning | Default |
| --- | --- | --- |
| `PYTHON` / `PYTHON_BIN` | Python interpreter | `python` |
| `REPO_DIR` | Path to this LEAP checkout | `/path/to/LEAP` |
| `IMAGENET_DIR` | ImageNet-1K root (script appends `/train` for training) | `/path/to/imagenet` |
| `MINI_IMAGENET_DIR` | mini-ImageNet root | `/path/to/mini-imagenet` |
| `OUT_BASE` | Output directory for checkpoints and logs | `/path/to/output` |
| `REVISIT_ROOT` | Parent dir for Revisited Oxford / Paris datasets | `/path/to/revisitop` |

Training data should follow an `ImageFolder` layout:

```
<dataset-root>/
  train/
    n01440764/ *.JPEG
    ...
  val/
    ...
```

We provide distilled ViT-S checkpoints on Hugging Face:

| Model | Dataset | Download |
| --- | --- | --- |
| LEAP Distilled ViT-S | ImageNet-100 | [Download](https://huggingface.co/Kevin-Z/LEAP_Distilled_ViT) |
| LEAP Distilled ViT-Tiny | ImageNet-100 | [Download](https://huggingface.co/Kevin-Z/LEAP_Distilled_ViT) |
| LEAP Distilled ViT-S | ImageNet-1K | [Download](https://huggingface.co/Kevin-Z/LEAP_Distilled_ViT) |
| LEAP Distilled ViT-Tiny | ImageNet-1K | [Download](https://huggingface.co/Kevin-Z/LEAP_Distilled_ViT) |


## ImageNet-1K Experiments

### Baseline distillation

Distill DINOv2 ViT-G → ViT-S/14 with CLS supervision:

```bash
IMAGENET_DIR=/path/to/imagenet OUT_BASE=/path/to/output \
  DATASET=imagenet-1k EPOCHS=100 BATCH_SIZE=256 LR=6 \
  CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 sbatch submit_train.sh
```

Checkpoints are saved to `$OUT_BASE/leap_baseline/exported_models/exported_last.pt`.

### LEAP curriculum distillation

Distill with the adaptive layer-skipping curriculum (CKA-based block progression):

```bash
IMAGENET_DIR=/path/to/imagenet OUT_BASE=/path/to/output \
  CURRICULUM=1 DATASET=imagenet-1k CKA_THRESHOLD=0.8 EMA_ALPHA=1 \
  MAX_EPOCHS_BLOCK=5 EPOCHS=100 BATCH_SIZE=256 LR=6 \
  CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 sbatch submit_train.sh
```

Checkpoints are saved to `$OUT_BASE/leap_curriculum/exported_models/exported_last.pt`.

### Student backbone size

Both training scripts accept `STUDENT_SIZE`:

- `STUDENT_SIZE=small` (default) — ViT-S (embed=384, depth=12, heads=6)
- `STUDENT_SIZE=tiny` — ViT-T (embed=192, depth=12, heads=3)

```bash
STUDENT_SIZE=tiny IMAGENET_DIR=/path/to/imagenet OUT_BASE=/path/to/output \
  DATASET=imagenet-1k EPOCHS=100 BATCH_SIZE=256 LR=6 \
  CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 sbatch submit_train.sh
```

You can also call `train_distill.py` directly:

```bash
python train_distill.py \
  --data /path/to/imagenet/train \
  --out /path/to/output/leap_baseline \
  --mode fair \
  --student-size small \
  --last-n 1 \
  --teacher-model dinov2/vitg14 \
  --epochs 100 \
  --batch-size 256 \
  --lr 6 \
  --cls-loss \
  --cls-loss-weight 0.05
```

## mini-ImageNet / ImageNet-100 Experiments

For smaller-scale runs, set `DATASET=mini-imagenet` and point `MINI_IMAGENET_DIR` at your dataset root:

```bash
MINI_IMAGENET_DIR=/path/to/mini-imagenet OUT_BASE=/path/to/output \
  DATASET=mini-imagenet EPOCHS=100 BATCH_SIZE=256 LR=6 \
  CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 sbatch submit_train.sh
```

For the LEAP curriculum on mini-ImageNet:

```bash
MINI_IMAGENET_DIR=/path/to/mini-imagenet OUT_BASE=/path/to/output \
  CURRICULUM=1 DATASET=mini-imagenet CKA_THRESHOLD=0.85 EMA_ALPHA=1 \
  MAX_EPOCHS_BLOCK=5 EPOCHS=100 BATCH_SIZE=256 LR=6 \
  CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 sbatch submit_train.sh
```

## Evaluation

### Linear probing

Evaluate a frozen-backbone linear probe on ImageNet validation set:

```bash
IMAGENET_DIR=/path/to/imagenet DATASET=imagenet-1k \
  CKPT=/path/to/output/leap_baseline/exported_models/exported_last.pt \
  sbatch submit_eval.sh
```

Use `FEAT_TYPE=avgpool` for avg-pooled patch tokens instead of the CLS token. You can also run `eval_linear.py` directly:

```bash
python eval_linear.py \
  --checkpoint /path/to/output/leap_baseline/exported_models/exported_last.pt \
  --data-dir /path/to/imagenet \
  --image-size 224 \
  --model-img-size 224 \
  --feature-type cls \
  --student-size small
```

### Instance retrieval (Revisited Oxford / Paris)

Evaluate on Revisited Oxford-5k or Paris-6k. With `FROM_OFFICIAL=1` (default), the datasets are downloaded automatically into `REVISIT_ROOT`:

```bash
REVISIT_ROOT=/path/to/revisitop TEST_DATASET=roxford5k \
  CHECKPOINT=/path/to/output/leap_baseline/exported_models/exported_last.pt \
  sbatch submit_instance_recognition.sh

REVISIT_ROOT=/path/to/revisitop TEST_DATASET=rparis6k \
  CHECKPOINT=/path/to/output/leap_baseline/exported_models/exported_last.pt \
  sbatch submit_instance_recognition.sh
```

If the data is already prepared locally, pass `FROM_PREPARE=0` to skip the download step.

## Acknowledgement

Our codebase builds on [LightlyTrain](https://github.com/lightly-ai/lightly-train) for distillation training and [DINOv2](https://github.com/facebookresearch/dinov2) as the teacher backbone. Instance retrieval evaluation uses the [Revisitop](https://github.com/AndreBiedermann/Revisitop) benchmark protocol.

If you find this repo helpful, please consider giving it a star ⭐.
