# LEAP

Distillation of DINOv2 ViT-G into a small student (ViT-S/14 or ViT-T/14), with
an optional curriculum block-skipping schedule. The student is always built
explicitly at `img_size=224` and distilled against a single
teacher target block.

## Layout

```
train_distill.py              # core distillation training entry point
submit_train.sh               # SLURM launcher for training (baseline / curriculum)

eval_linear.py                # linear probing evaluation
submit_eval.sh                # SLURM launcher for linear probing

eval_instance_retrieval.py    # Revisited Oxford / Paris instance retrieval
revisitop_toolkit/            # revisitop dataset + mAP helpers
submit_instance_recognition.sh# SLURM launcher for retrieval

requirements.txt              # pinned Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

All paths in the scripts are generic `/path/to/...` placeholders. Set them to
your own locations via environment variables (no need to edit the scripts);
also edit the `#SBATCH` header lines (`--partition`, `--gres`, `--mem`, and the
`slurm_logs/` output paths) to match your cluster.

| Variable    | Meaning                               | Default        |
| ----------- | ------------------------------------- | -------------- |
| `PYTHON`    | Python interpreter (train/eval)       | `python`       |
| `PYTHON_BIN`| Python interpreter (retrieval script) | `python`       |
| `REPO_DIR`  | Path to this LEAP checkout            | `/path/to/LEAP`|

```bash
PYTHON=/path/to/venv/bin/python REPO_DIR=/path/to/LEAP \
  DATA_DIR=/path/to/imagenet/train OUT_BASE=/path/to/output \
  EPOCHS=150 BATCH_SIZE=2048 LR=3.15 CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 \
  sbatch submit_train.sh
```

## Student backbone size

All scripts accept `STUDENT_SIZE` to choose the student architecture (it must
match between training and evaluation):

- `STUDENT_SIZE=small` (default) — ViT-S (embed=384, depth=12, heads=6)
- `STUDENT_SIZE=tiny` — ViT-T (embed=192, depth=12, heads=3, ~5.6M params)

```bash
# train a ViT-T student
STUDENT_SIZE=tiny DATASET=imagenet-1k EPOCHS=150 BATCH_SIZE=2048 LR=3.15 \
  CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 sbatch submit_train.sh
```

## Datasets — where to put them

Download ImageNet-1K / mini-ImageNet (and the retrieval datasets) **anywhere you
like**, then tell the scripts where they are via environment variables.

### Pretraining data (ImageNet-1K / mini-ImageNet)

Training expects an `ImageFolder`-style layout, i.e. a `train/` directory with
one subfolder per class:

```
<your-imagenet>/
  train/
    n01440764/ *.JPEG
    n01443537/ *.JPEG
    ...
```

There are three ways to point the scripts at your copy (any one works):

| How | Example |
| --- | ------- |
| **Explicit train dir** (most direct) | `DATA_DIR=/data/imagenet/train sbatch submit_train.sh` |
| **Dataset root** (script appends `/train`) | `IMAGENET_DIR=/data/imagenet sbatch submit_train.sh` <br> `MINI_IMAGENET_DIR=/data/mini-imagenet DATASET=mini-imagenet sbatch submit_train.sh` |
| **Custom path** as the `DATASET` value | `DATASET=/data/my-set sbatch submit_train.sh` (uses `/data/my-set/train`) |

For **linear probing** (`submit_eval.sh`) the same variables apply, but they
point at the dataset **root** (containing both `train/` and `val/`), e.g.
`DATA_DIR=/data/imagenet` or `IMAGENET_DIR=/data/imagenet`.

The built-in `DATASET=imagenet-1k` / `DATASET=mini-imagenet` shortcuts default
to `/path/to/...` placeholders, so set
`DATA_DIR`/`IMAGENET_DIR`/`MINI_IMAGENET_DIR` on your machine.

### Output location (checkpoints / logs)

Set `OUT_BASE=/your/output/dir`. The run folder under it is named
`leap_baseline` or `leap_curriculum`; override the whole run folder with
`OUT_DIR_OVERRIDE=/abs/path`. Exported checkpoints land in
`$OUT_DIR/exported_models/exported_last.pt`. (SLURM stdout/stderr go to
`slurm_logs/` next to where you submit — `mkdir -p slurm_logs` first, or edit
the `#SBATCH --output/--error` header lines.)

### Retrieval data (Revisited Oxford / Paris)

Set `REVISIT_ROOT=/your/revisitop` — the parent directory where the datasets
live (or will be downloaded to). Each dataset is expected at
`$REVISIT_ROOT/datasets/<TEST_DATASET>/` with the images plus
`gnd_<TEST_DATASET>.pkl`.

| Variable       | Meaning                                   | Default              |
| -------------- | ----------------------------------------- | -------------------- |
| `REVISIT_ROOT` | parent dir for the datasets               | `/path/to/revisitop` |
| `TEST_DATASET` | `roxford5k` (Oxford) / `rparis6k` (Paris) | `roxford5k`          |

With `FROM_OFFICIAL=1` (default) the data is downloaded automatically from the
Oxford/VGG mirror + revisitop into `REVISIT_ROOT`. If you already have it, pass
`FROM_PREPARE=0` to skip the download. Results are written to `OUTPUT_DIR`
(defaults to `RUN_DIR`).

```bash
REVISIT_ROOT=/data/revisitop TEST_DATASET=roxford5k \
  CHECKPOINT=/path/to/exported_last.pt sbatch submit_instance_recognition.sh
```

## Training

Set `IMAGENET_DIR` (or `DATA_DIR`) and `OUT_BASE` to your own locations.

Baseline distillation (CLS supervision):

```bash
IMAGENET_DIR=/data/imagenet OUT_BASE=/data/leap_out \
  DATASET=imagenet-1k EPOCHS=150 BATCH_SIZE=2048 LR=3.15 \
  CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 sbatch submit_train.sh
```

Curriculum block-skipping distillation (CLS supervision):

```bash
IMAGENET_DIR=/data/imagenet OUT_BASE=/data/leap_out \
  CURRICULUM=1 DATASET=imagenet-1k CKA_THRESHOLD=0.8 EMA_ALPHA=1 \
  MAX_EPOCHS_BLOCK=5 EPOCHS=100 BATCH_SIZE=2048 LR=3.15 \
  CLS_LOSS=1 CLS_LOSS_WEIGHT=0.05 sbatch submit_train.sh
```

The mode is fixed to `fair` and the teacher target depth to `last-n=1`.
The result is at `$OUT_BASE/leap_baseline` (or `leap_curriculum`).

## Evaluation

Linear probing (point `CKPT` at your trained checkpoint and `IMAGENET_DIR` at
the dataset root):

```bash
IMAGENET_DIR=/data/imagenet DATASET=imagenet-1k \
  CKPT=/data/leap_out/leap_baseline/exported_models/exported_last.pt \
  sbatch submit_eval.sh
```

Instance retrieval (Revisited Oxford / Paris):

```bash
REVISIT_ROOT=/data/revisitop TEST_DATASET=roxford5k \
  CHECKPOINT=/path/to/exported_last.pt sbatch submit_instance_recognition.sh
REVISIT_ROOT=/data/revisitop TEST_DATASET=rparis6k \
  CHECKPOINT=/path/to/exported_last.pt sbatch submit_instance_recognition.sh
```
