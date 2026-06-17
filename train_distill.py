"""
Baseline Distillation using LightlyTrain

Distills from a frozen DINOv2 ViT-G/14 teacher into a ViT-S student
using LightlyTrain's distillation method (MSE loss with strong augmentations).

Student modes:
  --mode default    ViT-S/14, 224x224 input (standard LightlyTrain setup, timm default
                    img_size=518 → pos_embed interpolated at runtime)
  --mode fair       ViT-S/14, 224x224 input, explicitly built with img_size=224
                    (pos_embed exactly matched to training resolution, fair comparison
                    against the dual-res 112/patch7 setup)
  --mode lowres     ViT-S/14, 112x112 input, both student and teacher at 112x112
                    (8x8 = 64 tokens, faster teacher inference)
  --mode custom     ViT-S with patch_size=7, 112x112 input (matches kd_distill setup)

LightlyTrain handles:
  - Teacher loading and freezing
  - Feature projection (student → teacher dim)
  - Strong shared augmentations (crop, flip, color jitter, blur, grayscale)
  - Mixed-precision training
  - Multi-GPU support
  - Checkpointing and logging

Usage:
  python train_distill.py                              # default mode (224x224, patch14, timm default img_size=518)
  python train_distill.py --mode fair                  # fair mode (224x224, patch14, img_size=224)
  python train_distill.py --mode custom                # custom mode (112x112, patch7)
  python train_distill.py --epochs 200 --batch-size 256
  python train_distill.py --precision 32-true          # disable AMP
"""

import argparse
from typing import List, Optional

import timm
import torch

import lightly_train
from lightly_train._methods.distillationv2.distillationv2 import DistillationV2
from lightly_train._methods.method import TrainingStepResult

import gc


def _get_raw_model(emb_model):
    """Unwrap LightlyTrain's EmbeddingModel / ModelWrapper to the underlying nn.Module."""
    if hasattr(emb_model, "wrapped_model"):
        return emb_model.wrapped_model.get_model()
    if hasattr(emb_model, "get_model"):
        return emb_model.get_model()
    return emb_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="LightlyTrain baseline: distill DINOv2 ViT-G → ViT-S"
    )

    parser.add_argument("--data", type=str, default="/path/to/imagenet/train",
                        help="Path to training images (ImageFolder structure)")
    parser.add_argument("--out", type=str, default="/path/to/output/leap_baseline",
                        help="Output directory for checkpoints, logs, and exports")

    # Student mode
    parser.add_argument("--mode", type=str, default="default",
                        choices=["default", "fair", "lowres", "custom"],
                        help="'default': ViT-S/14 at 224x224, timm default img_size=518 (pos_embed interpolated). "
                             "'fair': ViT-S/14 at 224x224, explicitly img_size=224 (pos_embed matched). "
                             "'lowres': ViT-S/14 at low resolution (default 112, set with --image-size). "
                             "'custom': ViT-S patch7 at 112x112 (matches kd_distill).")

    parser.add_argument("--image-size", type=int, default=None,
                        help="Override image resolution (only used with --mode lowres). "
                             "Must be divisible by 14. Examples: 56, 70, 84, 98, 112, 140, 168.")

    # Student model size (architecture, independent of --mode which controls
    # patch_size / img_size). 'small' (default) is the existing ViT-S
    # backbone. 'tiny' is a DeiT-Tiny-shape ViT (embed=192, depth=12,
    # heads=3, ~5.6M params), useful for distilling ViT-G into a much
    # smaller student.
    parser.add_argument("--student-size", type=str, default="small",
                        choices=["small", "tiny"],
                        help="Student backbone size. 'small' = ViT-S "
                             "(embed=384, depth=12, heads=6, ~21M params). "
                             "'tiny' = ViT-T (embed=192, depth=12, heads=3, "
                             "~5.6M params). Patch size and image size are "
                             "still controlled by --mode / --image-size.")

    # Teacher model
    parser.add_argument("--teacher-model", type=str, default="dinov2/vitg14",
                        help="Teacher model for distillation")
    parser.add_argument(
        "--last-n",
        type=int,
        default=2,
        help="Number of final teacher blocks to concatenate for patch-token "
             "distillation. The same value is used for CLS supervision when "
             "--cls-loss is enabled (default: 2).",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (lightly recommends 100-3000)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Per-GPU batch size (lightly recommends 128-2048)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Base learning rate for LARS optimizer. "
                             "LightlyTrain default is 9.0 (scaled by sqrt(batch/1536)). "
                             "For ImageNet-1K with large batch, try 1.5-3.0.")
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["16-mixed", "bf16-mixed", "32-true"],
                        help="Training precision (16-mixed recommended for speed)")
    parser.add_argument("--matmul-precision", type=str, default="medium",
                        choices=["highest", "high", "medium"],
                        help="Float32 matmul precision. 'medium' enables TF32 tensor "
                             "cores for ~2x speedup on Ampere+ GPUs (A6000, L40S).")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers per GPU")

    # Distillation method
    parser.add_argument("--method", type=str, default="distillation",
                        choices=["distillation", "distillationv1"],
                        help="Distillation method (v2 is faster and more accurate)")

    # Teacher early exit (feature skipping)
    parser.add_argument(
        "--teacher-block", type=int, default=None,
        help="Extract features from this block index only (0-based) and prune "
             "all subsequent teacher blocks. For ViT-G/14 (40 blocks), e.g. "
             "--teacher-block 4 keeps only blocks 0-4, saving ~87%% of teacher compute. "
             "Default: use last 2 blocks (LightlyTrain default).",
    )

    # Conv2d patch embedding loss
    parser.add_argument(
        "--embed-loss",
        action="store_true",
        help="Add normalized MSE loss between teacher and student conv2d "
        "patch embeddings (with a linear projection layer). "
        "Total loss = model_loss + embed_loss_weight * embed_loss.",
    )
    parser.add_argument(
        "--embed-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the embedding loss (default: 1.0).",
    )

    # Curriculum block-skipping distillation
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum block-skipping: student progressively targets "
        "deeper teacher blocks based on CKA convergence.",
    )
    parser.add_argument(
        "--cka-threshold",
        type=float,
        default=0.85,
        help="CKA EMA threshold to trigger block level-up (default: 0.85).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.05,
        help="EMA smoothing for per-batch CKA (default: 0.05).",
    )
    parser.add_argument(
        "--max-epochs-per-block",
        type=int,
        default=50,
        help="Max epochs on one block before forced level-up (default: 50).",
    )
    parser.add_argument(
        "--curriculum-lr-reset",
        action="store_true",
        help="Reset LR to peak on every curriculum block level-up. "
        "No warmup — starts directly at peak and cosine-decays over the "
        "remaining training steps. Resets on each block transition. "
        "Only valid with --curriculum.",
    )
    parser.add_argument(
        "--curriculum-lr-decay-peak",
        action="store_true",
        help="When used with --curriculum-lr-reset, the reset peak follows "
        "the global cosine envelope instead of jumping to the full peak. "
        "E.g. a reset at 50%% progress uses ~50%% of the original peak LR. "
        "Prevents loss explosions from late-training LR spikes.",
    )
    parser.add_argument(
        "--curriculum-lr-reset-after-warmup",
        action="store_true",
        help="When used with --curriculum-lr-reset, keep the initial warmup "
        "phase, then apply LR resets only after warmup completes. "
        "Each curriculum level-up after warmup resets LR to global peak and "
        "cosine-decays until the next reset.",
    )
    parser.add_argument(
        "--curriculum-lr-decay-peak-gap",
        action="store_true",
        help="When used with --curriculum-lr-reset, each reset peak is "
        "lr_before_reset + envelope(step) * (peak - lr_before_reset). "
        "Every reset is a real bump up (never below current LR), but the "
        "bump amount shrinks over training (late resets ≈ no bump). "
        "Mutually exclusive with --curriculum-lr-decay-peak.",
    )
    parser.add_argument(
        "--curriculum-lr-no-decay",
        action="store_true",
        help="When used with --curriculum-lr-reset, LR stays flat at peak "
        "after warmup (no cosine decay, no resets needed). If combined "
        "with --curriculum-lr-reset-after-warmup, warmup is preserved; "
        "otherwise LR starts at peak from step 0. Mutually exclusive with "
        "--curriculum-lr-decay-peak and --curriculum-lr-decay-peak-gap.",
    )
    parser.add_argument(
        "--curriculum-lr-rising-gap",
        action="store_true",
        help="When used with --curriculum-lr-reset, each reset peak is "
        "lr_before + scale * (1 - envelope(step)) * (peak - lr_before). "
        "Early resets give a small bump up; late resets give a larger bump. "
        "Mutually exclusive with --curriculum-lr-decay-peak, "
        "--curriculum-lr-decay-peak-gap, and --curriculum-lr-no-decay.",
    )
    parser.add_argument(
        "--curriculum-lr-rising-gap-scale",
        type=float,
        default=0.5,
        help="Max bump fraction for --curriculum-lr-rising-gap. "
        "Default: 0.5 (late resets bump up to ~50%% of the gap-to-peak). "
        "Use 1.0 to allow late resets to bump all the way to peak. "
        "Lower values reduce all resets uniformly.",
    )
    parser.add_argument(
        "--curriculum-lr-constant-warmup",
        action="store_true",
        help="When used with --curriculum-lr-reset and "
        "--curriculum-lr-reset-after-warmup, the warmup phase uses a "
        "constant LR at global peak instead of a linear ramp from 0 to peak. "
        "Only changes warmup behavior; post-warmup schedule is unaffected.",
    )
    parser.add_argument(
        "--curriculum-lr-disable-resets",
        action="store_true",
        help="When used with --curriculum-lr-reset, curriculum block "
        "transitions do NOT trigger LR resets. Useful for getting pure "
        "cosine decay after warmup while still running curriculum training.",
    )
    parser.add_argument(
        "--curriculum-lr-segment-warmup",
        action="store_true",
        help="When used with --curriculum-lr-reset, every segment starts at "
        "(segment_start_ratio * peak), linearly ramps to peak over "
        "--curriculum-lr-segment-warmup-steps, then cosine-decays until the "
        "next block transition (or end of training). Replaces the initial "
        "warmup entirely — the first segment's ramp IS the initial warmup. "
        "Mutually exclusive with --curriculum-lr-decay-peak, "
        "--curriculum-lr-decay-peak-gap, --curriculum-lr-no-decay, "
        "--curriculum-lr-rising-gap, --curriculum-lr-reset-after-warmup, "
        "and --curriculum-lr-constant-warmup.",
    )
    parser.add_argument(
        "--curriculum-lr-segment-warmup-steps",
        type=int,
        default=2000,
        help="Number of steps for each segment's ramp-up (low → peak). "
        "Used only with --curriculum-lr-segment-warmup. Default: 2000.",
    )
    parser.add_argument(
        "--curriculum-lr-segment-start-ratio",
        type=float,
        default=0.01,
        help="Fraction of peak LR that each segment begins at. "
        "E.g. 0.01 means each segment starts at 1%% of peak and ramps up. "
        "Used only with --curriculum-lr-segment-warmup. Default: 0.01.",
    )
    parser.add_argument(
        "--curriculum-lr-segment-peak-decay",
        action="store_true",
        help="When used with --curriculum-lr-segment-warmup, each segment's "
        "target peak is reduced multiplicatively from the previous segment's "
        "peak by --curriculum-lr-segment-peak-decay-rate. "
        "E.g. rate=0.05 gives peaks 1.00, 0.95, 0.9025, ... Clamped so peak "
        "never falls below segment_start_ratio.",
    )
    parser.add_argument(
        "--curriculum-lr-segment-peak-decay-rate",
        type=float,
        default=0.05,
        help="Multiplicative decay rate applied to the segment peak on every "
        "reset. peak_k = peak_{k-1} * (1 - rate). Must be in [0, 1]. "
        "Default: 0.05 (5%% off per segment). "
        "Used only with --curriculum-lr-segment-peak-decay.",
    )
    parser.add_argument(
        "--cka-sample-images",
        type=int,
        default=0,
        help="If >0, compute curriculum CKA on only this many images per step "
        "(sampled from the current batch). Decouples CKA estimate size from "
        "training global batch size.",
    )
    parser.add_argument(
        "--curriculum-snap-to-last",
        action="store_true",
        help="Curriculum paradigm: train with no warmup (LR starts at peak "
             "and cosine-decays over the full run, no resets), and as soon "
             "as the curriculum reaches block --curriculum-snap-to-last-at, "
             "jump straight to the LAST teacher block and stay there for the "
             "rest of training. Implies --curriculum-lr-reset and "
             "--curriculum-lr-disable-resets internally; mutually exclusive "
             "with all other --curriculum-lr-* schedule modes. Requires "
             "--curriculum.",
    )
    parser.add_argument(
        "--curriculum-snap-to-last-at",
        type=int,
        default=30,
        help="1-indexed teacher block at (or beyond) which the curriculum "
             "snaps to the last teacher block. Used only with "
             "--curriculum-snap-to-last. Default: 30.",
    )
    parser.add_argument(
        "--curriculum-sync-rank",
        action="store_true",
        help="Synchronize curriculum controller state across DDP ranks "
             "(rank 0 decides level-up and broadcasts to all ranks). "
             "If disabled, each rank progresses independently (original behavior).",
    )
    # Static concat-last-two distillation (no curriculum).
    parser.add_argument(
        "--concat-last-two",
        action="store_true",
        default=False,
        help="Static distillation against the CONCATENATION of (teacher's "
             "second-to-last block, teacher's last block) along the feature "
             "dim, throughout the entire training (no curriculum, no CKA, "
             "no level-up, no calibration). A widened projector "
             "Linear(D_s -> 2*D_t) is installed and used for both patch and "
             "CLS tokens. Mutually exclusive with --curriculum, --cls-loss, "
             "--all-layers, --teacher-block, and "
             "--curriculum-concat-teacher-last.",
    )
    parser.add_argument(
        "--concat-last-two-cls-weight",
        type=float,
        default=0.0,
        help="CLS-token loss weight added to the loss when "
             "--concat-last-two is on (uses the same widened projection "
             "head; target is [t_second_last_cls; t_last_cls]). "
             "Default 0.0 (CLS off). Typical value: 0.05.",
    )

    parser.add_argument(
        "--curriculum-reset-projector",
        action="store_true",
        default=False,
        help="When --curriculum is on, re-initialize the student projection "
             "head (and, if --curriculum-concat-teacher-last is on, also the "
             "widened concat projection head) to its INITIAL random state "
             "every time the curriculum advances to a new teacher block. "
             "A snapshot of the head's state_dict is captured once at init "
             "time and reloaded on each block transition. Optimizer state "
             "(LARS momentum, etc.) is NOT cleared. Requires --curriculum.",
    )
    parser.add_argument(
        "--curriculum-metric",
        type=str,
        default="cka",
        choices=["cka", "cosine"],
        help="Similarity metric driving curriculum progression. "
             "'cka' (default): Linear CKA between flattened projected student "
             "patch tokens and target-block teacher patch tokens. "
             "'cosine': mean of per-token cosine similarities between the "
             "(D_t-dimensional) projected student patch tokens and the "
             "matching teacher patch tokens. Cosine range is [-1, 1]; tune "
             "--curriculum-cosine-threshold accordingly. Only valid with "
             "--curriculum.",
    )
    parser.add_argument(
        "--curriculum-cosine-threshold",
        type=float,
        default=0.6,
        help="EMA threshold to trigger block level-up when "
             "--curriculum-metric=cosine. Default: 0.6. Range: [-1, 1].",
    )
    parser.add_argument(
        "--curriculum-concat-teacher-last",
        action="store_true",
        default=False,
        help="When --curriculum is on, supervise the student's final feature "
             "map against the CONCATENATION of (teacher target block, "
             "teacher last block) along the feature dim. A widened projector "
             "Linear(D_s -> 2*D_t) is installed and used for both patch and "
             "CLS tokens. The curriculum still advances on CKA(target block) "
             "as before — the target half of the projection is used for CKA. "
             "Requires --curriculum; incompatible with --cls-loss and "
             "--all-layers.",
    )
    parser.add_argument(
        "--curriculum-concat-cls-weight",
        type=float,
        default=0.0,
        help="CLS-token loss weight added to the curriculum loss when "
             "--curriculum-concat-teacher-last is on (uses the same widened "
             "projection head; target is [t_target_cls; t_last_cls]). "
             "Default 0.0 (CLS off). Typical value: 0.05.",
    )

    # ------------------------------------------------------------------
    # All-layers naive distillation (baseline against curriculum)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Naive layer-wise distillation: align EVERY student block with "
             "the SAME-INDEX teacher block via a separate per-layer Linear "
             "projector (s_dim -> t_dim). Total loss = sum_i MSE_i. Requires "
             "an equal number of student and teacher blocks. Mutually "
             "exclusive with --curriculum, --teacher-block, "
             "--cls-loss, --embed-loss, --sigreg.",
    )
    parser.add_argument(
        "--all-layers-mean",
        action="store_true",
        help="When --all-layers is set, average the per-layer MSE losses "
             "instead of summing them. Useful for keeping the loss scale "
             "comparable to the single-target curriculum loss.",
    )
    parser.add_argument(
        "--all-layers-cls-weight",
        type=float,
        default=0.0,
        help="Weight on the per-layer CLS-token MSE term in the all-layers "
             "loss. The same per-layer Linear projector is applied to both "
             "patch tokens and the CLS token. Final loss is "
             "R(patch_loss_i) + cls_weight * R(cls_loss_i) where R is "
             "mean if --all-layers-mean, else sum. Default 0.0 (disabled).",
    )

    # CLS token supervision
    parser.add_argument(
        "--cls-loss",
        action="store_true",
        help="Add MSE loss on the CLS token (student vs teacher). "
        "Uses the same projection head as patch tokens by default. "
        "Total loss += cls_loss_weight * MSE(proj(student_cls), teacher_cls).",
    )
    parser.add_argument(
        "--separate-cls-head",
        action="store_true",
        help="Use a separate linear projection head for CLS supervision "
             "(instead of sharing student_projection_head with patch tokens).",
    )
    parser.add_argument(
        "--cls-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the CLS token loss (default: 1.0).",
    )
    parser.add_argument(
        "--cls-loss-last-n",
        type=int,
        default=None,
        help="[Deprecated] Use --last-n instead. If set, it must match --last-n.",
    )

    # SigReg regularization
    parser.add_argument(
        "--sigreg",
        action="store_true",
        help="Add Strong SIGReg loss (from LeJEPA) to regularize student features "
        "toward an isotropic Gaussian. Prevents representation collapse. "
        "Total loss = MSE + sigreg_weight * sigreg_loss.",
    )
    parser.add_argument(
        "--sigreg-weight",
        type=float,
        default=0.01,
        help="Weight for the SIGReg loss (default: 0.01).",
    )
    parser.add_argument(
        "--sigreg-sketch-dim",
        type=int,
        default=1024,
        help="Sketch dimension for SIGReg random projection (default: 1024).",
    )

    # Learning rate schedule
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=None,
        help="Minimum LR as a fraction of peak LR at the end of cosine decay. "
        "Default (None) uses LightlyTrain's default of 0.001 (LR decays to "
        "0.1%% of peak). Set higher (e.g. 0.1) for gentler decay.",
    )

    # Resume / overwrite
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted training from last checkpoint")
    parser.add_argument("--lr-override-on-resume", action="store_true",
                        help="When resuming, force the scheduler/optimizer to "
                             "use the freshly-built peak LR derived from --lr "
                             "instead of the base_lrs stored in the checkpoint "
                             "(which Lightning restores by default). Requires "
                             "--resume and --lr.")

    return parser.parse_args()


def _prune_teacher_blocks(teacher, block_idx: int) -> int:
    """Truncate a teacher model to keep only blocks 0..block_idx.

    Handles both DINOv2's DinoVisionTransformer (chunked blocks with
    nn.Identity padding) and timm's VisionTransformer.
    Returns the original total number of blocks.
    """
    import torch.nn as nn

    if hasattr(teacher, "chunked_blocks") and teacher.chunked_blocks:
        # DINOv2's DinoVisionTransformer: flatten BlockChunks, skip Identity
        flat_blocks = []
        for chunk in teacher.blocks:
            for blk in chunk:
                if not isinstance(blk, nn.Identity):
                    flat_blocks.append(blk)
        total_blocks = len(flat_blocks)
        if block_idx < 0 or block_idx >= total_blocks:
            raise ValueError(
                f"--teacher-block {block_idx} out of range [0, {total_blocks - 1}]"
            )
        teacher.blocks = nn.ModuleList(flat_blocks[: block_idx + 1])
        teacher.chunked_blocks = False
    elif hasattr(teacher, "prune_intermediate_layers"):
        # timm VisionTransformer
        total_blocks = len(teacher.blocks)
        if block_idx < 0 or block_idx >= total_blocks:
            raise ValueError(
                f"--teacher-block {block_idx} out of range [0, {total_blocks - 1}]"
            )
        teacher.prune_intermediate_layers(
            indices=[block_idx], prune_norm=False, prune_head=True,
        )
    else:
        # Generic fallback
        total_blocks = len(teacher.blocks)
        if block_idx < 0 or block_idx >= total_blocks:
            raise ValueError(
                f"--teacher-block {block_idx} out of range [0, {total_blocks - 1}]"
            )
        teacher.blocks = teacher.blocks[: block_idx + 1]

    return total_blocks


def patch_teacher_early_exit(block_idx: int) -> None:
    """Prune teacher to stop at a specific block for faster inference.

    Monkey-patches DistillationV2.__init__ so that after the teacher model
    is created, blocks after ``block_idx`` are permanently removed.
    This frees GPU memory AND avoids running those blocks during inference.

    Combined with ``n_teacher_blocks=1`` in method_args, the distillation
    loss is computed against the single specified block's features.
    """
    original_init = DistillationV2.__init__

    def patched_init(self, method_args, optimizer_args, embedding_model,
                     global_batch_size, num_input_channels):
        original_init(self, method_args, optimizer_args, embedding_model,
                      global_batch_size, num_input_channels)

        total_blocks = _prune_teacher_blocks(
            self.teacher_embedding_model, block_idx
        )
        remaining = len(self.teacher_embedding_model.blocks)
        removed = total_blocks - remaining
        print(f"  [Teacher early exit] Using block {block_idx}: "
              f"kept {remaining}/{total_blocks} blocks, "
              f"removed {removed} ({removed * 100.0 / total_blocks:.0f}% compute saved)")
        gc.collect()

    DistillationV2.__init__ = patched_init  # type: ignore[assignment]
    print(f"  [Patch] Teacher will be pruned to block {block_idx} (early exit)")


def patch_embed_loss(weight: float) -> None:
    """Add normalized MSE loss between conv2d patch embeddings.

    Patches DistillationV2 to:
    1. Create embed_proj (Linear s_dim -> t_dim), embed_norm_teacher,
       embed_norm_student (LayerNorm) after normal init.
    2. Include those modules in trainable_modules().
    3. Replace training_step_impl to compute:
       embed_loss = MSE(LN(proj(student_embed)), LN(teacher_embed))
       total_loss = model_loss + weight * embed_loss

    Must be applied AFTER teacher early exit and BEFORE pretrain().
    Not compatible with --ibot-mask (raises ValueError in main).
    """
    import torch.nn as nn
    import torch.nn.functional as F

    current_init = DistillationV2.__init__

    def patched_init(self, method_args, optimizer_args, embedding_model,
                     global_batch_size, num_input_channels):
        current_init(self, method_args, optimizer_args, embedding_model,
                     global_batch_size, num_input_channels)
        s_dim = self.student_embedding_model.embed_dim
        t_dim = self.teacher_embedding_model.embed_dim
        self.embed_proj = nn.Linear(s_dim, t_dim)
        self.embed_norm_teacher = nn.LayerNorm(t_dim)
        self.embed_norm_student = nn.LayerNorm(t_dim)
        print(f"  [Embed loss] Proj: {s_dim} -> {t_dim}, weight={weight}")

    DistillationV2.__init__ = patched_init  # type: ignore[assignment]

    current_trainable = DistillationV2.trainable_modules

    def patched_trainable(self):
        result = current_trainable(self)
        extra = [self.embed_proj, self.embed_norm_teacher, self.embed_norm_student]
        result.modules = list(result.modules) + extra
        return result

    DistillationV2.trainable_modules = patched_trainable  # type: ignore[assignment]

    def patched_training_step_impl(
        self: DistillationV2, batch, batch_idx: int
    ) -> TrainingStepResult:
        views = batch["views"][0]
        views = self._mixup_data(views)

        teacher_raw = _get_raw_model(self.teacher_embedding_model)
        student_raw = _get_raw_model(self.student_embedding_model)
        with torch.no_grad():
            t_embed = teacher_raw.patch_embed.proj(views)
        s_embed = student_raw.patch_embed.proj(views)

        t_flat = t_embed.flatten(2).transpose(1, 2)
        s_flat = s_embed.flatten(2).transpose(1, 2)
        s_proj = self.embed_proj(s_flat)

        embed_loss_val = F.mse_loss(
            self.embed_norm_student(s_proj),
            self.embed_norm_teacher(t_flat),
        )

        x_teacher, (th, tw) = self._forward_teacher(views)
        x_student = self._forward_student(
            views, teacher_features_h=th, teacher_features_w=tw
        )
        model_loss = self.criterion(
            teacher_features=x_teacher, student_features=x_student
        )

        total_loss = model_loss + weight * embed_loss_val
        log = {
            "model_loss": model_loss.detach(),
            "embed_loss": embed_loss_val.detach(),
        }
        return TrainingStepResult(loss=total_loss, log_dict=log)

    DistillationV2.training_step_impl = patched_training_step_impl  # type: ignore[assignment]
    print(f"  [Patch] Embed loss enabled (weight={weight})")


# ---------------------------------------------------------------------------
# Curriculum block-skipping distillation
# ---------------------------------------------------------------------------

def minibatch_mean_cosine(
    features_x: torch.Tensor, features_y: torch.Tensor
) -> float:
    """Mean cosine similarity over paired feature rows.

    Both tensors must have the same shape (N, D). Computes cosine
    similarity between ``features_x[i]`` and ``features_y[i]`` for each
    i, then averages over N. The mean is reduced as a Python float
    (detached) so it can be EMA'd by the curriculum controller. The
    computation is forced to float32 with autocast disabled to avoid
    numerical issues under mixed precision.

    Args:
        features_x: (N, D) tensor -- e.g. flattened projected student
            patch tokens (already in the teacher feature dim D_t).
        features_y: (N, D) tensor -- matching flattened teacher patch
            tokens. Must align row-wise with ``features_x``.

    Returns:
        Mean cosine similarity in [-1, 1] as a Python float.
    """
    import torch.nn.functional as F  # noqa: F401  (local; module has no top-level F)
    with torch.cuda.amp.autocast(enabled=False):
        x = features_x.float()
        y = features_y.float()
        x_n = F.normalize(x, dim=-1, eps=1e-12)
        y_n = F.normalize(y, dim=-1, eps=1e-12)
        cos = (x_n * y_n).sum(dim=-1)  # (N,)
        return float(cos.mean().detach().cpu())


def minibatch_linear_cka(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Compute Linear CKA using the feature-space formulation (no N×N Gram matrix).

    Uses dot products on the feature dimensions (X.T @ Y) instead of the
    batch-dimension Gram matrices, avoiding O(N^2) memory for large batches.

    All computation is forced to float32 with autocast disabled to prevent
    overflow when running inside a mixed-precision training step.

    Args:
        features_x: (N, D1) tensor -- e.g. flattened student features
        features_y: (N, D2) tensor -- e.g. flattened teacher features

    Returns:
        CKA score as a Python float (detached).
    """
    with torch.cuda.amp.autocast(enabled=False):
        X = features_x.float() - features_x.float().mean(0, keepdim=True)
        Y = features_y.float() - features_y.float().mean(0, keepdim=True)

        # Feature-space cross-covariance: (D1, D2), (D1, D1), (D2, D2)
        XtY = X.T @ Y
        XtX = X.T @ X
        YtY = Y.T @ Y

        # CKA = ||X^T Y||_F^2 / (||X^T X||_F · ||Y^T Y||_F)
        hsic_xy = (XtY * XtY).sum()
        hsic_xx = (XtX * XtX).sum()
        hsic_yy = (YtY * YtY).sum()
        denom = torch.sqrt(hsic_xx * hsic_yy)
        if denom < 1e-12:
            return 0.0
        return float((hsic_xy / denom).detach().cpu())


class CurriculumController:
    """State-machine managing which teacher block the student targets."""

    STATE_FILENAME = "curriculum_state.json"

    def __init__(
        self,
        total_teacher_blocks: int,
        cka_threshold: float = 0.85,
        ema_alpha: float = 0.05,
        max_epochs_per_block: int = 50,
        out_dir: Optional[str] = None,
        snap_to_last_at: Optional[int] = None,
    ):
        self.total_teacher_blocks = total_teacher_blocks
        self.cka_threshold = cka_threshold
        self.ema_alpha = ema_alpha
        self.max_epochs_per_block = max_epochs_per_block
        self.out_dir = out_dir
        # When set, as soon as the curriculum reaches this 1-indexed block
        # (or beyond), the controller jumps to `total_teacher_blocks` and
        # stays there for the remainder of training. None = disabled.
        self.snap_to_last_at = snap_to_last_at

        self.current_target_block = 1   # 1-indexed (block 1 = shallowest)
        self.ema_cka = 0.0
        self.epochs_on_current_block = 0
        self._lr_state: Optional["_CurriculumLRState"] = None

    def _state_path(self) -> Optional[str]:
        if self.out_dir is None:
            return None
        import os
        return os.path.join(self.out_dir, self.STATE_FILENAME)

    def save(self) -> None:
        import json
        path = self._state_path()
        if path is None:
            return
        state = {
            "current_target_block": self.current_target_block,
            "ema_cka": self.ema_cka,
            "epochs_on_current_block": self.epochs_on_current_block,
            "total_teacher_blocks": self.total_teacher_blocks,
        }
        if self._lr_state is not None:
            state["lr_segment_start"] = self._lr_state._segment_start
            state["lr_segment_peak"] = self._lr_state._segment_peak
        # Round-robin decider rotation counter (sync_rank only). Stored so that
        # a resumed run continues the rotation from the exact same position.
        decider_rank = getattr(self, "_decider_rank", None)
        if decider_rank is not None:
            state["decider_rank"] = int(decider_rank[0])
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self) -> bool:
        """Load saved state. Returns True if state was restored."""
        import json, os
        path = self._state_path()
        if path is None or not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)
        self.current_target_block = state["current_target_block"]
        self.ema_cka = state["ema_cka"]
        self.epochs_on_current_block = state["epochs_on_current_block"]
        self.total_teacher_blocks = state.get(
            "total_teacher_blocks", self.total_teacher_blocks
        )
        if self._lr_state is not None and "lr_segment_start" in state:
            self._lr_state._segment_start = state["lr_segment_start"]
            if "lr_segment_peak" in state:
                self._lr_state._segment_peak = float(state["lr_segment_peak"])
            print(f"  [Curriculum] Restored LR segment_start={state['lr_segment_start']}"
                  f", segment_peak={self._lr_state._segment_peak:.4f}")
        decider_rank = getattr(self, "_decider_rank", None)
        if decider_rank is not None and "decider_rank" in state:
            decider_rank[0] = int(state["decider_rank"])
            print(f"  [Curriculum] Restored decider_rank={decider_rank[0]}")
        print(f"  [Curriculum] Restored state from {path}: "
              f"block={self.current_target_block}, ema_cka={self.ema_cka:.4f}, "
              f"epochs_on_block={self.epochs_on_current_block}")
        return True

    def update_cka(self, batch_cka_score: float) -> None:
        self.ema_cka = (
            self.ema_alpha * batch_cka_score
            + (1 - self.ema_alpha) * self.ema_cka
        )

    def step_epoch(self) -> None:
        self.epochs_on_current_block += 1
        self.save()

    def _maybe_snap_to_last(self) -> bool:
        """If snap_to_last_at is configured and current block has reached it,
        jump straight to the last teacher block. Returns True if the snap
        happened on this call."""
        if (
            self.snap_to_last_at is None
            or self.current_target_block < self.snap_to_last_at
            or self.current_target_block >= self.total_teacher_blocks
        ):
            return False
        old = self.current_target_block
        self.current_target_block = self.total_teacher_blocks
        print(
            f"  [Curriculum] Snap-to-last! Block {old} → "
            f"{self.current_target_block} (snap_to_last_at={self.snap_to_last_at}, "
            f"total_blocks={self.total_teacher_blocks})"
        )
        self.ema_cka = 0.0
        self.epochs_on_current_block = 0
        self.save()
        return True

    def check_for_level_up(self) -> bool:
        if self.current_target_block >= self.total_teacher_blocks:
            return False
        # Honor snap-to-last even if we landed at/past the threshold from
        # calibration or a resumed checkpoint without a normal +1 transition.
        if self._maybe_snap_to_last():
            return True
        if (self.ema_cka >= self.cka_threshold
                or self.epochs_on_current_block >= self.max_epochs_per_block):
            old = self.current_target_block
            self.current_target_block += 1
            print(
                f"  [Curriculum] Level up! Block {old} → {self.current_target_block} "
                f"(ema_cka={self.ema_cka:.4f}, epochs={self.epochs_on_current_block})"
            )
            self.ema_cka = 0.0
            self.epochs_on_current_block = 0
            self.save()
            # After a normal level-up, check if we just crossed the snap line.
            self._maybe_snap_to_last()
            return True
        return False


class _CurriculumLRState:
    """Callable for ``LambdaLR`` that implements a resettable cosine decay.

    No warmup — starts at multiplier 1.0 (peak LR) and cosine-decays to
    ``min_lr_ratio`` over the remaining training steps.  On each curriculum
    level-up, ``reset(current_step)`` restarts the cosine from a new peak.

    Two modes controlled by ``decaying_peak``:

    * **False** (default): reset to the global peak (multiplier 1.0) every
      time.  Can cause loss explosions if the model has already converged
      to a lower LR regime.
    * **True**: the reset peak follows the *global* cosine envelope.  At
      step *s* out of *T* total steps, the envelope value is::

          env(s) = min_ratio + (1 - min_ratio) * 0.5 * (1 + cos(pi * s / T))

      So a reset at 20% progress uses ~0.90× peak, at 50% → ~0.50× peak,
      etc.  The local segment then cosine-decays from that reduced peak
      down to ``min_lr_ratio`` over the remaining steps.
    """

    def __init__(self, min_lr_ratio: float = 0.001, total_steps: int = 1,
                 decaying_peak: bool = False, decaying_peak_gap: bool = False,
                 no_decay: bool = False, rising_gap: bool = False,
                 rising_gap_scale: float = 0.5,
                 constant_warmup: bool = False,
                 warmup_steps: int = 0,
                 segment_warmup: bool = False,
                 segment_warmup_steps: int = 0,
                 segment_start_ratio: float = 0.01,
                 segment_peak_decay: bool = False,
                 segment_peak_decay_rate: float = 0.05):
        self.min_lr_ratio = min_lr_ratio
        self.total_steps = total_steps
        self.decaying_peak = decaying_peak
        self.decaying_peak_gap = decaying_peak_gap
        self.no_decay = no_decay
        self.rising_gap = rising_gap
        self.rising_gap_scale = max(0.0, min(1.0, float(rising_gap_scale)))
        self.constant_warmup = constant_warmup
        self.warmup_steps = max(0, int(warmup_steps))
        self.segment_warmup = segment_warmup
        self.segment_warmup_steps = max(1, int(segment_warmup_steps))
        self.segment_start_ratio = max(0.0, min(1.0, float(segment_start_ratio)))
        self.segment_peak_decay = segment_peak_decay
        self.segment_peak_decay_rate = max(0.0, min(1.0, float(segment_peak_decay_rate)))
        self._segment_start = 0
        # Tracks the explicit reset peak; used by decay-peak-gap, rising-gap,
        # and segment-warmup+peak-decay modes. Defaults to 1.0 so that
        # pre-reset behavior matches "full peak".
        self._segment_peak = 1.0

    def _envelope(self, step: int) -> float:
        import math
        gp = min(1.0, step / max(1, self.total_steps))
        return (
            self.min_lr_ratio
            + (1.0 - self.min_lr_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * gp))
        )

    def _segment_lr(self, step: int, peak: float) -> float:
        import math
        remaining = max(1, self.total_steps - self._segment_start)
        local = step - self._segment_start
        progress = min(1.0, max(0.0, local / remaining))
        return (
            self.min_lr_ratio
            + (peak - self.min_lr_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * progress))
        )

    def reset(self, current_step: int) -> None:
        # In segment-warmup mode, every reset re-triggers a per-segment warmup
        # from `segment_start_ratio * peak` up to peak. No initial-warmup gate.
        if self.segment_warmup:
            self._segment_start = current_step
            if self.segment_peak_decay:
                # Geometric decay of the peak across segments.
                # peak_k = peak_{k-1} * (1 - rate). Clamp so the peak never
                # falls below the start ratio (otherwise the ramp would go
                # downward instead of upward).
                new_peak = self._segment_peak * (1.0 - self.segment_peak_decay_rate)
                self._segment_peak = max(new_peak, self.segment_start_ratio)
            return
        if current_step < self.warmup_steps:
            # Absorbed during warmup.
            self._segment_start = max(current_step, self.warmup_steps)
            return
        if self.decaying_peak_gap:
            lr_before = self._segment_lr(current_step, self._segment_peak)
            env = self._envelope(current_step)
            self._segment_start = current_step
            self._segment_peak = lr_before + env * (1.0 - lr_before)
        elif self.rising_gap:
            lr_before = self._segment_lr(current_step, self._segment_peak)
            env = self._envelope(current_step)
            bump = self.rising_gap_scale * max(0.0, 1.0 - env)
            self._segment_start = current_step
            self._segment_peak = lr_before + bump * (1.0 - lr_before)
        else:
            self._segment_start = current_step

    def __call__(self, step: int) -> float:
        import math
        # Per-segment warmup mode: every segment starts low and ramps up to
        # peak over `segment_warmup_steps`, then cosine-decays. This replaces
        # any initial `warmup_steps`.
        if self.segment_warmup:
            # Target peak for this segment. With peak-decay enabled each reset
            # reduces `_segment_peak` by `segment_peak_decay_rate` (see reset).
            # Without peak-decay, `_segment_peak` stays at 1.0.
            target_peak = self._segment_peak
            local = max(0, step - self._segment_start)
            if local < self.segment_warmup_steps:
                p = local / max(1, self.segment_warmup_steps)
                return self.segment_start_ratio + (target_peak - self.segment_start_ratio) * p
            # Post-warmup cosine decay within the segment. Use the remaining
            # global horizon (segment_end = total_steps) so the shape is stable
            # regardless of when the next reset arrives.
            ramp_end = self._segment_start + self.segment_warmup_steps
            remaining = max(1, self.total_steps - ramp_end)
            decay_progress = min(1.0, max(0.0, (step - ramp_end) / remaining))
            return (
                self.min_lr_ratio
                + (target_peak - self.min_lr_ratio)
                * 0.5
                * (1.0 + math.cos(math.pi * decay_progress))
            )

        if self.warmup_steps > 0 and step < self.warmup_steps:
            if self.constant_warmup:
                return 1.0
            warmup_progress = min(1.0, max(0.0, step / max(1, self.warmup_steps)))
            return warmup_progress

        if self.no_decay:
            return 1.0

        if self.decaying_peak_gap or self.rising_gap:
            segment_peak = self._segment_peak
        elif self.decaying_peak:
            segment_peak = self._envelope(self._segment_start)
        else:
            segment_peak = 1.0

        return self._segment_lr(step, segment_peak)


def patch_curriculum_distillation(
    cka_threshold: float = 0.85,
    ema_alpha: float = 0.05,
    max_epochs_per_block: int = 50,
    embed_loss_weight: float = 0.0,
    out_dir: Optional[str] = None,
    lr_reset: bool = False,
    min_lr_ratio: float = 0.001,
    decay_peak: bool = False,
    decay_peak_gap: bool = False,
    no_decay: bool = False,
    rising_gap: bool = False,
    rising_gap_scale: float = 0.5,
    constant_warmup: bool = False,
    disable_resets: bool = False,
    reset_after_warmup: bool = False,
    segment_warmup: bool = False,
    segment_warmup_steps: int = 0,
    segment_start_ratio: float = 0.01,
    segment_peak_decay: bool = False,
    segment_peak_decay_rate: float = 0.05,
    cka_sample_images: int = 0,
    sync_rank: bool = False,
    snap_to_last_at: Optional[int] = None,
    concat_teacher_last: bool = False,
    concat_cls_weight: float = 0.0,
    curriculum_metric: str = "cka",
    cosine_threshold: float = 0.6,
    reset_projector: bool = False,
) -> CurriculumController:
    """Monkey-patch DistillationV2 for curriculum block-skipping distillation.

    The student progressively targets deeper teacher blocks.  When the EMA
    of the per-batch Linear CKA reaches ``cka_threshold`` *or* the student
    has spent ``max_epochs_per_block`` epochs on the current block, the
    curriculum advances to the next deeper block.

    The same ``student_projection_head`` (Linear 384 → 1536) is reused across
    block transitions -- its weights are NOT re-initialised.

    Composable with ``patch_embed_loss`` (call embed_loss first).
    Must be used with ``n_teacher_blocks=1`` in method_args.
    """
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F

    if curriculum_metric not in ("cka", "cosine"):
        raise ValueError(
            f"[Curriculum] Unknown metric '{curriculum_metric}'. "
            f"Expected 'cka' or 'cosine'."
        )

    _effective_threshold = (
        float(cosine_threshold) if curriculum_metric == "cosine"
        else float(cka_threshold)
    )

    controller = CurriculumController(
        total_teacher_blocks=1,  # overwritten in patched __init__
        cka_threshold=_effective_threshold,
        ema_alpha=ema_alpha,
        max_epochs_per_block=max_epochs_per_block,
        out_dir=out_dir,
        snap_to_last_at=snap_to_last_at,
    )
    controller._metric_name = curriculum_metric  # type: ignore[attr-defined]

    # Shared LR state for resettable cosine (None if lr_reset is disabled)
    lr_state: Optional[_CurriculumLRState] = (
        _CurriculumLRState(
            min_lr_ratio=min_lr_ratio,
            decaying_peak=decay_peak,
            decaying_peak_gap=decay_peak_gap,
            no_decay=no_decay,
            rising_gap=rising_gap,
            rising_gap_scale=rising_gap_scale,
            constant_warmup=constant_warmup,
            warmup_steps=0,
            segment_warmup=segment_warmup,
            segment_warmup_steps=segment_warmup_steps,
            segment_start_ratio=segment_start_ratio,
            segment_peak_decay=segment_peak_decay,
            segment_peak_decay_rate=segment_peak_decay_rate,
        )
        if lr_reset else None
    )
    controller._lr_state = lr_state

    def _dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _world_size() -> int:
        if not _dist_ready():
            return 1
        return int(dist.get_world_size())

    def _global_rank(method_instance) -> int:
        if not _dist_ready():
            return 0
        return int(getattr(method_instance, "global_rank", 0))

    # Round-robin decider rank: on each block transition we advance by 1
    # (mod world_size) so decisions rotate across GPUs. Mutable single-element
    # list so closures can mutate it. Kept in sync across ranks because it only
    # changes on block_changed events, which are themselves synchronized.
    decider_rank: List[int] = [0]
    controller._decider_rank = decider_rank  # type: ignore[attr-defined]

    def _is_rank0(method_instance) -> bool:
        # Retained for rank-0-only side-effects like stdout printing.
        return _global_rank(method_instance) == 0

    def _is_decider_rank(method_instance) -> bool:
        if not _dist_ready():
            return True
        return _global_rank(method_instance) == decider_rank[0] % _world_size()

    def _sync_controller_state(method_instance, device: torch.device) -> None:
        """Broadcast curriculum controller state from the current decider
        rank to all ranks. The decider rotates 0→1→…→W-1→0 on every block
        transition."""
        if not sync_rank:
            return
        if not _dist_ready():
            return
        src = decider_rank[0] % _world_size()
        if _global_rank(method_instance) == src:
            state = torch.tensor(
                [
                    float(controller.current_target_block),
                    float(controller.epochs_on_current_block),
                    float(controller.ema_cka),
                ],
                device=device,
                dtype=torch.float32,
            )
        else:
            state = torch.zeros(3, device=device, dtype=torch.float32)
        dist.broadcast(state, src=src)
        controller.current_target_block = max(1, int(round(float(state[0].item()))))
        controller.epochs_on_current_block = max(0, int(round(float(state[1].item()))))
        controller.ema_cka = float(state[2].item())

    # ---- Patch __init__: detect teacher block count, de-chunk if needed ----
    current_init = DistillationV2.__init__

    def patched_init(self, method_args, optimizer_args, embedding_model,
                     global_batch_size, num_input_channels):
        current_init(self, method_args, optimizer_args, embedding_model,
                     global_batch_size, num_input_channels)

        teacher = self.teacher_embedding_model
        if hasattr(teacher, "chunked_blocks") and teacher.chunked_blocks:
            flat = []
            for chunk in teacher.blocks:
                for blk in chunk:
                    if not isinstance(blk, nn.Identity):
                        flat.append(blk)
            teacher.blocks = nn.ModuleList(flat)
            teacher.chunked_blocks = False
            total = len(flat)
        else:
            total = len(teacher.blocks)

        controller.total_teacher_blocks = total
        print(f"  [Curriculum] Teacher has {total} blocks, "
              f"starting at block {controller.current_target_block}")

        # Snapshot the projector's initial random state so we can restore it
        # on every curriculum level-up. Tensors are deeply cloned and detached
        # so the snapshot is immutable to training updates.
        if reset_projector:
            self._curriculum_initial_projection_state = {
                k: v.detach().clone()
                for k, v in self.student_projection_head.state_dict().items()
            }
            print(
                f"  [Curriculum] Projector reset enabled: snapshotted "
                f"student_projection_head state "
                f"({sum(v.numel() for v in self._curriculum_initial_projection_state.values())} params)."
            )

        if concat_teacher_last:
            student_raw = _get_raw_model(self.student_embedding_model)
            s_dim = int(getattr(student_raw, "embed_dim", 0))
            t_dim = int(getattr(teacher, "embed_dim", 0))
            if s_dim <= 0 or t_dim <= 0:
                raise RuntimeError(
                    f"[Curriculum-CTL] Could not infer dims "
                    f"(s_dim={s_dim}, t_dim={t_dim})."
                )
            self.student_projection_head_concat = nn.Linear(s_dim, 2 * t_dim)
            self._curriculum_concat_t_dim = t_dim
            self._curriculum_concat_s_dim = s_dim
            # Reserve the actual last teacher block as the concat target so it
            # is always DISTINCT from the curriculum's intermediate target.
            # Concretely: cap curriculum at block (total - 1) so the deepest
            # "intermediate" target is the second-to-last actual block; the
            # final actual block (total) is appended to every concat target.
            self._curriculum_true_last_idx = int(total - 1)  # 0-indexed
            if total >= 2:
                controller.total_teacher_blocks = int(total - 1)
            else:
                raise RuntimeError(
                    "[Curriculum-CTL] Teacher has fewer than 2 blocks; "
                    "concat-teacher-last needs at least 2."
                )
            # Clamp any saved/calibrated target that already crossed the cap.
            if controller.current_target_block > controller.total_teacher_blocks:
                controller.current_target_block = controller.total_teacher_blocks
            print(
                f"  [Curriculum-CTL] Concat-teacher-last enabled: installed "
                f"Linear({s_dim} -> {2 * t_dim}) for both patch & CLS. "
                f"cls_weight={concat_cls_weight}. Curriculum capped at block "
                f"{total - 1}/{total}; block {total} is the fixed concat target."
            )
            # Snapshot the concat projector for projector-reset too.
            if reset_projector:
                self._curriculum_initial_projection_state_concat = {
                    k: v.detach().clone()
                    for k, v in self.student_projection_head_concat.state_dict().items()
                }

    DistillationV2.__init__ = patched_init  # type: ignore[assignment]

    def _do_reset_projector(method_instance, *, reason: str) -> None:
        """Restore the student projection head(s) to the snapshotted initial
        random state captured in patched_init. Called on each curriculum
        level-up when ``reset_projector`` is enabled."""
        snap = getattr(method_instance, "_curriculum_initial_projection_state", None)
        if snap is not None:
            method_instance.student_projection_head.load_state_dict(snap)
        snap_concat = getattr(
            method_instance,
            "_curriculum_initial_projection_state_concat",
            None,
        )
        if snap_concat is not None and hasattr(
            method_instance, "student_projection_head_concat"
        ):
            method_instance.student_projection_head_concat.load_state_dict(
                snap_concat
            )
        if _is_rank0(method_instance):
            tag = (
                "patch+concat" if snap_concat is not None
                else "patch"
            )
            print(
                f"  [Curriculum] Projector reset to initial state "
                f"(block={controller.current_target_block}, "
                f"head={tag}, {reason})"
            )

    # ---- When concat_teacher_last is on, extend trainable_modules ----------
    if concat_teacher_last:
        _orig_trainable = DistillationV2.trainable_modules

        def _patched_trainable_concat(self):
            result = _orig_trainable(self)
            result.modules = list(result.modules) + [
                self.student_projection_head_concat
            ]
            return result

        DistillationV2.trainable_modules = _patched_trainable_concat  # type: ignore[assignment]

    # ---- Patch configure_optimizers for LR reset ----
    if lr_state is not None:
        _orig_configure = DistillationV2.configure_optimizers

        def patched_configure_optimizers(self):
            from torch.optim.lr_scheduler import LambdaLR

            optimizers, scheduler_cfgs = _orig_configure(self)
            total_steps = int(self.trainer.estimated_stepping_batches)
            lr_state.total_steps = total_steps
            warmup_steps = 0
            if reset_after_warmup and scheduler_cfgs and not segment_warmup:
                original_sched = scheduler_cfgs[0].get("scheduler", None)
                # NOTE: LightlyTrain stores warmup count in *steps* under the
                # attribute name `warmup_epochs` (their comment in
                # _methods/method.py:117 confirms: "The arguments are called
                # 'epochs' but they can also be steps."). Use it as-is.
                warmup_steps_raw = int(getattr(original_sched, "warmup_epochs", 0))
                warmup_steps = max(0, min(warmup_steps_raw, total_steps - 1))
            lr_state.warmup_steps = warmup_steps
            # In segment-warmup mode, the very first segment begins at step 0
            # so the initial ramp replaces any global warmup. Otherwise the
            # first segment begins at the end of the (optional) warmup.
            lr_state._segment_start = 0 if segment_warmup else lr_state.warmup_steps

            new_sched = LambdaLR(optimizers[0], lr_lambda=lr_state)
            scheduler_cfgs[0]["scheduler"] = new_sched

            peak_lr = optimizers[0].param_groups[0]["lr"]
            if reset_after_warmup:
                print(
                    f"  [Curriculum LR Reset] Warmup then reset mode, "
                    f"peak={peak_lr:.4f}, warmup_steps={warmup_steps}, "
                    f"min_ratio={min_lr_ratio}, total_steps={total_steps}"
                )
            else:
                print(
                    f"  [Curriculum LR Reset] No warmup, peak={peak_lr:.4f}, "
                    f"min_ratio={min_lr_ratio}, total_steps={total_steps}"
                )
            return optimizers, scheduler_cfgs

        DistillationV2.configure_optimizers = patched_configure_optimizers  # type: ignore[assignment]

    # ---- Patch training_step_impl ----
    def _calibrate_starting_block(self, views):
        """One-time scan: compute the curriculum metric (CKA or cosine)
        between the (random) student and every teacher block, then set the
        starting block to the one with highest score."""
        teacher = self.teacher_embedding_model
        n_skip = 1 + getattr(teacher, "num_register_tokens", 0)
        _sim_fn = (
            minibatch_mean_cosine if curriculum_metric == "cosine"
            else minibatch_linear_cka
        )

        with torch.no_grad():
            # Student features (random init)
            s_out = self.student_embedding_model(views, pool=False)
            s_proj = self.student_projection_head(s_out)
            B, H, W, D = s_proj.shape
            s_flat = s_proj.reshape(B * H * W, D)

            # Run teacher through all blocks, compute metric at each.
            # ``best_score`` initialised below -inf so any cosine score
            # (which can be negative at random init) is accepted.
            x = teacher.prepare_tokens_with_masks(views)
            best_cka, best_block = float("-inf"), 1
            cka_scores = []
            for i, blk in enumerate(teacher.blocks):
                x = blk(x)
                t_normed = teacher.norm(x)
                t_patch = t_normed[:, n_skip:, :]
                score = _sim_fn(s_flat, t_patch.reshape(-1, D))
                cka_scores.append(score)
                if score > best_cka:
                    best_cka = score
                    best_block = i + 1  # 1-indexed

        # Print compact summary (5 per line)
        _metric_tag = "cosine" if curriculum_metric == "cosine" else "CKA"
        print(
            f"  [Curriculum] Initial {_metric_tag} scan "
            f"(student vs each teacher block):"
        )
        for row_start in range(0, len(cka_scores), 5):
            row = cka_scores[row_start : row_start + 5]
            parts = []
            for j, s in enumerate(row):
                idx = row_start + j + 1
                marker = " *" if idx == best_block else "  "
                parts.append(f"B{idx:02d}={s:.3f}{marker}")
            print("    " + " | ".join(parts))

        controller.current_target_block = best_block
        controller.ema_cka = 0.0
        controller.epochs_on_current_block = 0
        # If calibration already lands at/past the snap line, jump straight
        # to the last block so the rest of training distills from the tip.
        controller._maybe_snap_to_last()
        print(
            f"  [Curriculum] Starting at block {best_block}/{len(cka_scores)} "
            f"({_metric_tag}={best_cka:.4f})"
        )

    def patched_training_step_impl(
        self: DistillationV2, batch, batch_idx: int
    ) -> TrainingStepResult:
        views = batch["views"][0]
        views = self._mixup_data(views)

        # One-time calibration on the very first batch (skipped if state restored)
        if not getattr(self, "_curriculum_calibrated", False):
            self._curriculum_calibrated = True  # type: ignore[attr-defined]
            if not controller.load():
                _calibrate_starting_block(self, views)
            else:
                print(f"  [Curriculum] Skipping calibration, resumed at block "
                      f"{controller.current_target_block}")
                # A resumed run may have been launched with a *new* snap
                # threshold; honor it immediately so we don't keep distilling
                # from a mid block.
                controller._maybe_snap_to_last()
            if sync_rank:
                _sync_controller_state(self, views.device)

        target_idx = controller.current_target_block - 1  # 0-indexed
        teacher = self.teacher_embedding_model
        n_skip = 1 + getattr(teacher, "num_register_tokens", 0)

        if concat_teacher_last:
            # ---- Teacher: forward ALL blocks; cache target + last -----------
            # last_idx is the TRUE last teacher block (curriculum cap keeps
            # target_idx <= last_idx - 1, so the two are always distinct).
            last_idx = int(self._curriculum_true_last_idx)
            # Defensive clamp: if a stale resume left target at/past last, cap
            # it to last-1 so we never concat the same features twice.
            target_idx = min(target_idx, last_idx - 1)
            with torch.no_grad():
                x = teacher.prepare_tokens_with_masks(views)
                t_target_patch = None
                t_target_cls = None
                t_last_patch = None
                t_last_cls = None
                for i, blk in enumerate(teacher.blocks):
                    x = blk(x)
                    if i == target_idx:
                        x_n = teacher.norm(x)
                        t_target_patch = x_n[:, n_skip:, :].detach()
                        t_target_cls = x_n[:, 0, :].detach()
                    if i == last_idx:
                        x_n = teacher.norm(x)
                        t_last_patch = x_n[:, n_skip:, :].detach()
                        t_last_cls = x_n[:, 0, :].detach()
                        break
                if (t_target_patch is None) or (t_last_patch is None):
                    raise RuntimeError(
                        f"[Curriculum-CTL] Failed to capture teacher features "
                        f"(target_idx={target_idx}, last_idx={last_idx}, "
                        f"total={controller.total_teacher_blocks})."
                    )
                t_dim_local = int(self._curriculum_concat_t_dim)
                # Concatenate along feature dim.
                t_concat_patch = torch.cat(
                    [t_target_patch, t_last_patch], dim=-1
                )  # (B, N, 2*D_t)
                t_concat_cls = torch.cat(
                    [t_target_cls, t_last_cls], dim=-1
                )  # (B, 2*D_t)

            # ---- Student: raw forward (need patch + CLS tokens at final norm)
            student_raw = _get_raw_model(self.student_embedding_model)
            x_s = (
                student_raw.prepare_tokens_with_masks(views)
                if hasattr(student_raw, "prepare_tokens_with_masks")
                else student_raw.forward_features(views)
            )
            if hasattr(student_raw, "prepare_tokens_with_masks"):
                for blk in student_raw.blocks:
                    x_s = blk(x_s)
                x_s = student_raw.norm(x_s)
            n_skip_s = (
                int(student_raw.num_prefix_tokens)
                if hasattr(student_raw, "num_prefix_tokens")
                else 1 + int(getattr(student_raw, "num_register_tokens", 0))
            )
            s_patch = x_s[:, n_skip_s:, :]  # (B, N, D_s)
            s_cls = x_s[:, 0, :]            # (B, D_s)
            B, N, D_s = s_patch.shape

            # ---- Project through widened head, MSE losses --------------------
            s_proj_patch = self.student_projection_head_concat(s_patch)
            # (B, N, 2*D_t)
            s_proj_cls = self.student_projection_head_concat(s_cls)
            # (B, 2*D_t)
            patch_loss = F.mse_loss(s_proj_patch, t_concat_patch)
            log = {"model_loss": patch_loss.detach()}
            total_loss = patch_loss
            if concat_cls_weight > 0.0:
                cls_loss_val = F.mse_loss(s_proj_cls, t_concat_cls)
                total_loss = total_loss + concat_cls_weight * cls_loss_val
                log["cls_loss"] = cls_loss_val.detach()

            # ---- Surrogates for CKA monitoring: target half only ------------
            s_feat = s_proj_patch[..., :t_dim_local].contiguous()
            t_feat = t_target_patch
            D = t_dim_local
        else:
            # Teacher features with early exit: only run blocks 0..target_idx
            with torch.no_grad():
                x = teacher.prepare_tokens_with_masks(views)
                for blk in teacher.blocks[: target_idx + 1]:
                    x = blk(x)
                x = teacher.norm(x)
                t_feat = x[:, n_skip:, :]  # (B, N_patches, D_teacher)

            # Student features through the LightlyTrain wrapper
            student_out = self.student_embedding_model(views, pool=False)
            # student_out: (B, D_s, H, W)
            student_proj = self.student_projection_head(student_out)
            # student_proj: (B, H, W, D_teacher)
            B, H, W, D = student_proj.shape
            s_feat = student_proj.reshape(B, H * W, D)  # (B, N, D_teacher)

            # L2 / MSE loss between projected student output and teacher block
            model_loss = F.mse_loss(s_feat, t_feat)
            total_loss = model_loss

            log = {"model_loss": model_loss.detach()}

        # Optional patch-embedding loss (composable with --embed-loss)
        if hasattr(self, "embed_proj"):
            teacher_raw = self.teacher_embedding_model
            student_raw = _get_raw_model(self.student_embedding_model)
            with torch.no_grad():
                t_embed = teacher_raw.patch_embed.proj(views)
            s_embed = student_raw.patch_embed.proj(views)

            t_flat = t_embed.flatten(2).transpose(1, 2)
            s_flat = s_embed.flatten(2).transpose(1, 2)
            s_proj_embed = self.embed_proj(s_flat)
            embed_loss_val = F.mse_loss(
                self.embed_norm_student(s_proj_embed),
                self.embed_norm_teacher(t_flat),
            )
            total_loss = total_loss + embed_loss_weight * embed_loss_val
            log["embed_loss"] = embed_loss_val.detach()

        # Curriculum metric (detached, no grad). Optionally use only a
        # fixed-size image subset to decouple variance from batch size.
        # ``cka`` here is a generic metric value (Linear CKA or mean cosine
        # depending on ``curriculum_metric``); keeping the variable/log
        # names for backward compat with curriculum_state.json and plotting.
        _sim_fn = (
            minibatch_mean_cosine if curriculum_metric == "cosine"
            else minibatch_linear_cka
        )
        with torch.no_grad():
            s_for_cka = s_feat.detach()
            t_for_cka = t_feat
            if cka_sample_images > 0 and B > cka_sample_images:
                idx = torch.randperm(B, device=s_for_cka.device)[:cka_sample_images]
                s_for_cka = s_for_cka.index_select(0, idx)
                t_for_cka = t_for_cka.index_select(0, idx)
            cka = _sim_fn(
                s_for_cka.reshape(-1, D),
                t_for_cka.reshape(-1, D),
            )
            if sync_rank and _dist_ready():
                # All-reduce + average the metric so every rank updates its
                # local ema_cka with the *same* value (keeps replicas in
                # sync). The decider-rank rotation below only determines
                # which rank's controller is treated as authoritative for
                # the broadcast and level-up decision; outcomes are
                # identical across ranks.
                cka_tensor = torch.tensor([cka], device=s_for_cka.device, dtype=torch.float32)
                dist.all_reduce(cka_tensor, op=dist.ReduceOp.SUM)
                cka = float(cka_tensor.item() / float(dist.get_world_size()))

        prev_block = controller.current_target_block
        controller.update_cka(cka)
        if sync_rank:
            if _is_decider_rank(self):
                _ = controller.check_for_level_up()
            _sync_controller_state(self, s_for_cka.device)
        else:
            _ = controller.check_for_level_up()
        block_changed = controller.current_target_block != prev_block
        if block_changed and reset_projector:
            _do_reset_projector(self, reason=f"step {self.global_step}")
        if (
            block_changed
            and lr_state is not None
            and not disable_resets
            and self.global_step >= lr_state.warmup_steps
        ):
            lr_state.reset(self.global_step)
            if _is_rank0(self):
                current_lr = self.optimizers().param_groups[0]["lr"]
                decided_by = decider_rank[0] % max(1, _world_size())
                print(f"  [Curriculum LR Reset] LR reset at step {self.global_step} "
                      f"(block {controller.current_target_block}, "
                      f"decided by rank {decided_by}, lr={current_lr:.4f})")
        if sync_rank and block_changed:
            # Rotate the decider for the next transition. This change is
            # deterministic per-step and identical on every rank, because
            # `block_changed` is evaluated on synced state.
            decider_rank[0] = (decider_rank[0] + 1) % max(1, _world_size())
        log["cka"] = cka
        log["target_block"] = float(controller.current_target_block)
        log["ema_cka"] = controller.ema_cka

        return TrainingStepResult(loss=total_loss, log_dict=log)

    DistillationV2.training_step_impl = patched_training_step_impl  # type: ignore[assignment]

    # ---- Patch on_train_epoch_end for curriculum stepping ----
    _original_epoch_end = getattr(DistillationV2, "on_train_epoch_end", None)

    def patched_epoch_end(self):
        if _original_epoch_end is not None:
            _original_epoch_end(self)
        prev_block = controller.current_target_block
        if sync_rank:
            if _is_decider_rank(self):
                controller.step_epoch()
                _ = controller.check_for_level_up()
            rank_device = torch.device(getattr(self, "device", "cpu"))
            _sync_controller_state(self, rank_device)
        else:
            controller.step_epoch()
            _ = controller.check_for_level_up()
        block_changed = controller.current_target_block != prev_block
        if block_changed and reset_projector:
            _do_reset_projector(self, reason="epoch end")
        if (
            block_changed
            and lr_state is not None
            and not disable_resets
            and self.global_step >= lr_state.warmup_steps
        ):
            lr_state.reset(self.global_step)
        if sync_rank and block_changed:
            decider_rank[0] = (decider_rank[0] + 1) % max(1, _world_size())

    DistillationV2.on_train_epoch_end = patched_epoch_end  # type: ignore[assignment]

    print(f"  [Patch] Curriculum distillation enabled "
          f"(metric={curriculum_metric}, "
          f"threshold={_effective_threshold} "
          f"({'cosine' if curriculum_metric == 'cosine' else 'cka'}), "
          f"ema_alpha={ema_alpha}, "
          f"max_epochs/block={max_epochs_per_block}, "
          f"reset_projector={reset_projector}, "
          f"lr_reset={lr_reset}, "
          f"decay_peak={decay_peak}, "
          f"decay_peak_gap={decay_peak_gap}, "
          f"no_decay={no_decay}, "
          f"rising_gap={rising_gap}, "
          f"disable_resets={disable_resets}, "
          f"reset_after_warmup={reset_after_warmup}, "
          f"segment_warmup={segment_warmup}"
          f"(steps={segment_warmup_steps}, start_ratio={segment_start_ratio}, "
          f"peak_decay={segment_peak_decay}, peak_decay_rate={segment_peak_decay_rate}), "
          f"cka_sample_images={cka_sample_images if cka_sample_images > 0 else 'full-batch'}, "
          f"sync_rank={sync_rank})")

    return controller


# ---------------------------------------------------------------------------
# Static concat-last-two distillation (NO curriculum)
# ---------------------------------------------------------------------------

def patch_concat_last_two_distillation(cls_weight: float = 0.0) -> None:
    """Monkey-patch DistillationV2 for static distillation against the
    concatenation of the teacher's two deepest blocks.

    Target per step:
        t_target_patch = [teacher.norm(block_{N-2}(x)).patches ;
                          teacher.norm(block_{N-1}(x)).patches]   shape (B, N_pat, 2*D_t)
        t_target_cls   = [teacher.norm(block_{N-2}(x)).cls ;
                          teacher.norm(block_{N-1}(x)).cls]       shape (B, 2*D_t)

    Student raw forward gives final patch + CLS tokens; both go through the
    same widened projector ``Linear(D_s -> 2*D_t)``.

    Loss = MSE(s_proj_patch, t_target_patch)
           + cls_weight * MSE(s_proj_cls, t_target_cls)

    Differences vs ``patch_curriculum_distillation(concat_teacher_last=True)``:
      - No curriculum, no CKA, no level-up, no calibration, no controller.
      - The intermediate target is FIXED at block ``N-2`` for the entire
        training (not advancing through earlier blocks).
      - Teacher is still forwarded through all blocks (we need the second-to-
        last AND last block outputs); early-exit at block ``N-1`` is used so
        no extra cost beyond the original baseline.

    Must be used with ``n_teacher_blocks=1`` in method_args. Not composable
    with --curriculum, --cls-loss, --all-layers, --ibot-mask, --teacher-block,
    or --curriculum-concat-teacher-last.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    current_init = DistillationV2.__init__

    def patched_init(self, method_args, optimizer_args, embedding_model,
                     global_batch_size, num_input_channels):
        current_init(self, method_args, optimizer_args, embedding_model,
                     global_batch_size, num_input_channels)

        teacher = self.teacher_embedding_model
        if hasattr(teacher, "chunked_blocks") and teacher.chunked_blocks:
            flat = []
            for chunk in teacher.blocks:
                for blk in chunk:
                    if not isinstance(blk, nn.Identity):
                        flat.append(blk)
            teacher.blocks = nn.ModuleList(flat)
            teacher.chunked_blocks = False
            total = len(flat)
        else:
            total = len(teacher.blocks)

        if total < 2:
            raise RuntimeError(
                f"[ConcatLastTwo] Teacher must have >= 2 blocks "
                f"(got {total})."
            )

        student_raw = _get_raw_model(self.student_embedding_model)
        s_dim = int(getattr(student_raw, "embed_dim", 0))
        t_dim = int(getattr(teacher, "embed_dim", 0))
        if s_dim <= 0 or t_dim <= 0:
            raise RuntimeError(
                f"[ConcatLastTwo] Could not infer dims "
                f"(s_dim={s_dim}, t_dim={t_dim})."
            )
        self.student_projection_head_concat = nn.Linear(s_dim, 2 * t_dim)
        self._concat_last_two_t_dim = t_dim
        self._concat_last_two_second_last_idx = int(total - 2)  # 0-indexed
        self._concat_last_two_last_idx = int(total - 1)         # 0-indexed
        print(
            f"  [ConcatLastTwo] Enabled: teacher has {total} blocks, "
            f"target = concat(block_{total - 1}, block_{total}). "
            f"Installed Linear({s_dim} -> {2 * t_dim}) for both patch & CLS. "
            f"cls_weight={cls_weight}"
        )

    DistillationV2.__init__ = patched_init  # type: ignore[assignment]

    _orig_trainable = DistillationV2.trainable_modules

    def _patched_trainable(self):
        result = _orig_trainable(self)
        result.modules = list(result.modules) + [
            self.student_projection_head_concat
        ]
        return result

    DistillationV2.trainable_modules = _patched_trainable  # type: ignore[assignment]

    def patched_training_step_impl(
        self: DistillationV2, batch, batch_idx: int
    ) -> TrainingStepResult:
        views = batch["views"][0]
        views = self._mixup_data(views)

        teacher = self.teacher_embedding_model
        n_skip = 1 + getattr(teacher, "num_register_tokens", 0)
        second_last_idx = int(self._concat_last_two_second_last_idx)
        last_idx = int(self._concat_last_two_last_idx)

        # Teacher forward (no grad); early-exit at last_idx.
        with torch.no_grad():
            x = teacher.prepare_tokens_with_masks(views)
            t_second_last_patch = None
            t_second_last_cls = None
            t_last_patch = None
            t_last_cls = None
            for i, blk in enumerate(teacher.blocks):
                x = blk(x)
                if i == second_last_idx:
                    x_n = teacher.norm(x)
                    t_second_last_patch = x_n[:, n_skip:, :].detach()
                    t_second_last_cls = x_n[:, 0, :].detach()
                if i == last_idx:
                    x_n = teacher.norm(x)
                    t_last_patch = x_n[:, n_skip:, :].detach()
                    t_last_cls = x_n[:, 0, :].detach()
                    break
            if (t_second_last_patch is None) or (t_last_patch is None):
                raise RuntimeError(
                    f"[ConcatLastTwo] Failed to capture teacher features "
                    f"(second_last_idx={second_last_idx}, last_idx={last_idx})."
                )
            t_concat_patch = torch.cat(
                [t_second_last_patch, t_last_patch], dim=-1
            )  # (B, N, 2*D_t)
            t_concat_cls = torch.cat(
                [t_second_last_cls, t_last_cls], dim=-1
            )  # (B, 2*D_t)

        # Student raw forward (need patch + CLS tokens at final norm).
        student_raw = _get_raw_model(self.student_embedding_model)
        x_s = (
            student_raw.prepare_tokens_with_masks(views)
            if hasattr(student_raw, "prepare_tokens_with_masks")
            else student_raw.forward_features(views)
        )
        if hasattr(student_raw, "prepare_tokens_with_masks"):
            for blk in student_raw.blocks:
                x_s = blk(x_s)
            x_s = student_raw.norm(x_s)
        n_skip_s = (
            int(student_raw.num_prefix_tokens)
            if hasattr(student_raw, "num_prefix_tokens")
            else 1 + int(getattr(student_raw, "num_register_tokens", 0))
        )
        s_patch = x_s[:, n_skip_s:, :]  # (B, N, D_s)
        s_cls = x_s[:, 0, :]            # (B, D_s)

        s_proj_patch = self.student_projection_head_concat(s_patch)
        s_proj_cls = self.student_projection_head_concat(s_cls)

        patch_loss = F.mse_loss(s_proj_patch, t_concat_patch)
        log = {"model_loss": patch_loss.detach()}
        total_loss = patch_loss
        if cls_weight > 0.0:
            cls_loss_val = F.mse_loss(s_proj_cls, t_concat_cls)
            total_loss = total_loss + cls_weight * cls_loss_val
            log["cls_loss"] = cls_loss_val.detach()

        return TrainingStepResult(loss=total_loss, log_dict=log)

    DistillationV2.training_step_impl = patched_training_step_impl  # type: ignore[assignment]

    print(
        f"  [Patch] ConcatLastTwo enabled (cls_weight={cls_weight}, "
        f"target = concat(teacher second-to-last block, teacher last block))"
    )


# ---------------------------------------------------------------------------
# Naive all-layer distillation (baseline against curriculum)
# ---------------------------------------------------------------------------

def patch_all_layers_distillation(
    use_mean: bool = False,
    cls_weight: float = 0.0,
) -> None:
    """Monkey-patch DistillationV2 for naive all-layer feature alignment.

    For a teacher with N blocks and a student with N blocks (must match),
    install N independent ``Linear(s_dim, t_dim)`` projectors and compute,
    for every block i in [0, N):

      - patch_loss_i = MSE( proj_i(s_patch_i), t_patch_i )
      - cls_loss_i   = MSE( proj_i(s_cls_i),   t_cls_i )      (if cls_weight > 0)

    using the SAME ``proj_i = Linear(s_dim, t_dim)`` for both patch tokens
    and the CLS token (parameter-efficient; the projector lives in
    feature space, not token space). The reduction across the N layers is
    controlled by ``use_mean``:

      total_loss = R_i(patch_loss_i) + cls_weight * R_i(cls_loss_i)
        with R = mean if use_mean else sum.

    Both student and teacher are forwarded *manually* through their blocks
    so we can capture the per-block features. Teacher features pass
    through ``teacher.norm`` (matching the convention used by the
    curriculum implementation in :func:`patch_curriculum_distillation`)
    so the two methods are directly comparable. Register tokens are
    stripped from the teacher before computing the loss; CLS is at
    position 0 on both sides.

    Must be applied with ``n_teacher_blocks=1`` in method_args (the
    LightlyTrain projection head is unused — we install our own
    per-layer projectors). Not composable with curriculum, ibot-mask,
    teacher-block early exit, embed-loss, cls-loss, or sigreg.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    current_init = DistillationV2.__init__

    def patched_init(self, method_args, optimizer_args, embedding_model,
                     global_batch_size, num_input_channels):
        current_init(self, method_args, optimizer_args, embedding_model,
                     global_batch_size, num_input_channels)

        teacher = self.teacher_embedding_model
        if hasattr(teacher, "chunked_blocks") and teacher.chunked_blocks:
            flat = []
            for chunk in teacher.blocks:
                for blk in chunk:
                    if not isinstance(blk, nn.Identity):
                        flat.append(blk)
            teacher.blocks = nn.ModuleList(flat)
            teacher.chunked_blocks = False

        student_raw = _get_raw_model(self.student_embedding_model)
        n_t_blocks = len(teacher.blocks)
        n_s_blocks = len(student_raw.blocks)
        if n_t_blocks != n_s_blocks:
            raise ValueError(
                f"--all-layers requires the student and teacher to have the "
                f"same number of blocks. Got student={n_s_blocks}, "
                f"teacher={n_t_blocks}. Pick a matching teacher (e.g. "
                f"dinov2/vits14, 12 blocks) and student (--student-size "
                f"tiny/small, 12 blocks)."
            )

        s_dim = student_raw.embed_dim
        t_dim = teacher.embed_dim
        self.layer_projectors = nn.ModuleList(
            [nn.Linear(s_dim, t_dim) for _ in range(n_t_blocks)]
        )
        self._all_layers_n_blocks = n_t_blocks
        self._all_layers_use_mean = use_mean
        self._all_layers_cls_weight = cls_weight
        print(f"  [All-layers] {n_t_blocks} per-layer projectors: "
              f"Linear({s_dim} -> {t_dim}); "
              f"reduction={'mean' if use_mean else 'sum'}, "
              f"cls_weight={cls_weight}")

    DistillationV2.__init__ = patched_init  # type: ignore[assignment]

    current_trainable = DistillationV2.trainable_modules

    def patched_trainable(self):
        result = current_trainable(self)
        result.modules = list(result.modules) + [self.layer_projectors]
        return result

    DistillationV2.trainable_modules = patched_trainable  # type: ignore[assignment]

    def _student_prefix_tokens(student_raw) -> int:
        # timm VisionTransformer exposes num_prefix_tokens (CLS + dist
        # tokens). DINOv2 students set it implicitly via num_register_tokens.
        if hasattr(student_raw, "num_prefix_tokens"):
            return int(student_raw.num_prefix_tokens)
        return 1 + int(getattr(student_raw, "num_register_tokens", 0))

    def _student_forward_blocks(student_raw, views: torch.Tensor):
        """Run a timm-style ViT student through its blocks, collecting
        per-block (patch_features, cls_feature) pairs. Compatible with
        both timm VisionTransformer (used here for ViT-T) and the
        DINOv2-derived DinoVisionTransformer (used for ViT-S).

        Returns:
            patch_feats: list[(B, N_patches, D_s)], length = n_blocks
            cls_feats:   list[(B, D_s)],            length = n_blocks
        """
        if hasattr(student_raw, "prepare_tokens_with_masks"):
            # DinoVisionTransformer path
            x = student_raw.prepare_tokens_with_masks(views)
            n_skip = 1 + int(getattr(student_raw, "num_register_tokens", 0))
        else:
            # timm VisionTransformer path
            x = student_raw.patch_embed(views)
            x = student_raw._pos_embed(x)
            if hasattr(student_raw, "patch_drop"):
                x = student_raw.patch_drop(x)
            if hasattr(student_raw, "norm_pre"):
                x = student_raw.norm_pre(x)
            n_skip = _student_prefix_tokens(student_raw)
        patch_feats = []
        cls_feats = []
        for blk in student_raw.blocks:
            x = blk(x)
            patch_feats.append(x[:, n_skip:, :])
            # CLS is always at position 0 (timm: CLS only; DINOv2: CLS then registers).
            cls_feats.append(x[:, 0, :])
        return patch_feats, cls_feats

    def patched_training_step_impl(
        self: DistillationV2, batch, batch_idx: int
    ) -> TrainingStepResult:
        views = batch["views"][0]
        views = self._mixup_data(views)

        teacher = self.teacher_embedding_model
        n_skip_t = 1 + int(getattr(teacher, "num_register_tokens", 0))

        # Per-block teacher (patch, cls) features (no grad, frozen teacher).
        teacher_patch_feats: List[torch.Tensor] = []
        teacher_cls_feats: List[torch.Tensor] = []
        with torch.no_grad():
            x_t = teacher.prepare_tokens_with_masks(views)
            for blk in teacher.blocks:
                x_t = blk(x_t)
                t_n = teacher.norm(x_t)
                teacher_patch_feats.append(t_n[:, n_skip_t:, :])
                # CLS sits at position 0 even with registers (registers occupy
                # positions [1, 1+num_register_tokens)).
                teacher_cls_feats.append(t_n[:, 0, :])

        # Per-block student (patch, cls) features (with grad).
        student_raw = _get_raw_model(self.student_embedding_model)
        student_patch_feats, student_cls_feats = _student_forward_blocks(
            student_raw, views
        )

        if len(student_patch_feats) != len(teacher_patch_feats):
            raise RuntimeError(
                f"All-layers feature count mismatch at runtime: "
                f"student={len(student_patch_feats)}, "
                f"teacher={len(teacher_patch_feats)}."
            )

        cls_weight_local = float(self._all_layers_cls_weight)
        per_layer_patch_losses: List[torch.Tensor] = []
        per_layer_cls_losses: List[torch.Tensor] = []
        for i in range(self._all_layers_n_blocks):
            proj = self.layer_projectors[i]
            s_patch_proj = proj(student_patch_feats[i])
            per_layer_patch_losses.append(
                F.mse_loss(s_patch_proj, teacher_patch_feats[i])
            )
            if cls_weight_local > 0.0:
                s_cls_proj = proj(student_cls_feats[i])
                per_layer_cls_losses.append(
                    F.mse_loss(s_cls_proj, teacher_cls_feats[i])
                )

        patch_stacked = torch.stack(per_layer_patch_losses)
        if self._all_layers_use_mean:
            patch_term = patch_stacked.mean()
        else:
            patch_term = patch_stacked.sum()

        if cls_weight_local > 0.0:
            cls_stacked = torch.stack(per_layer_cls_losses)
            if self._all_layers_use_mean:
                cls_term = cls_stacked.mean()
            else:
                cls_term = cls_stacked.sum()
            total_loss = patch_term + cls_weight_local * cls_term
        else:
            cls_stacked = None
            cls_term = torch.zeros((), device=patch_term.device)
            total_loss = patch_term

        log = {
            "model_loss": total_loss.detach(),
            "patch_loss": patch_term.detach(),
            "cls_loss": cls_term.detach(),
            "patch_loss_mean": patch_stacked.mean().detach(),
        }
        if cls_stacked is not None:
            log["cls_loss_mean"] = cls_stacked.mean().detach()
        for i, l in enumerate(per_layer_patch_losses):
            log[f"patch_layer_{i:02d}_loss"] = l.detach()
        if cls_weight_local > 0.0:
            for i, l in enumerate(per_layer_cls_losses):
                log[f"cls_layer_{i:02d}_loss"] = l.detach()

        return TrainingStepResult(loss=total_loss, log_dict=log)

    DistillationV2.training_step_impl = patched_training_step_impl  # type: ignore[assignment]

    print(f"  [Patch] All-layers naive distillation enabled "
          f"(reduction={'mean' if use_mean else 'sum'}, "
          f"cls_weight={cls_weight})")


def sigreg_weak_loss(
    x: torch.Tensor,
    global_step: int,
    sketch_dim: int = 1024,
) -> torch.Tensor:
    """Weak SIGReg: forces Covariance(x) ≈ Identity.

    Only enforces decorrelation and unit variance (2nd moment), not full
    Gaussianity.  Less likely to conflict with MSE distillation because
    it doesn't constrain the distribution shape — just prevents feature
    collapse and redundancy.

    DDP-correct: random sketch is seeded by ``global_step`` so all ranks
    use the same projection, and the covariance is all-reduced.

    Args:
        x: (N_local, C) features (per-GPU batch).
        global_step: training step (used as seed for synced projections).
        sketch_dim: sketch dimension for random projection.

    Returns:
        Scalar loss tensor (differentiable).
    """
    import torch.distributed as dist

    N_local, C = x.size()
    dev = dict(device=x.device)

    # Sketch to lower dimension if C > sketch_dim
    if C > sketch_dim:
        g = torch.Generator(**dev)
        g.manual_seed(global_step)
        S = torch.randn(sketch_dim, C, generator=g, **dev) / (C ** 0.5)
        x = x @ S.T  # (N_local, sketch_dim)
        d = sketch_dim
    else:
        d = C

    # Center across batch
    x = x - x.mean(dim=0, keepdim=True)

    # Covariance: (d, d)
    cov = (x.T @ x) / (N_local - 1 + 1e-6)

    # All-reduce covariance across GPUs
    if dist.is_initialized():
        dist.all_reduce(cov, op=dist.ReduceOp.AVG)

    # Frobenius distance to identity
    target = torch.eye(d, device=x.device)
    return torch.norm(cov - target, p="fro")


def patch_sigreg(weight: float, sketch_dim: int = 1024,
                 include_cls: bool = False) -> None:
    """Add Weak SIGReg loss to DistillationV2's training step.

    Regularizes the student's raw backbone features so that their
    covariance approximates the identity matrix (decorrelation + unit
    variance).  Unlike Strong SIGReg (Epps-Pulley), this does not
    enforce full Gaussianity — only 2nd-moment structure — making it
    compatible with MSE distillation where the teacher dictates a
    specific (non-Gaussian) feature distribution.

    Uses a forward hook on the student backbone's final LayerNorm to
    capture features from the existing forward pass (no second pass).

    When ``include_cls`` is True, SigReg is also applied to the CLS
    token features (separately from patch tokens) so that the CLS
    representation is also decorrelated and unit-variance.

    DDP-correct: sketch is seeded by ``global_step`` and the covariance
    is all-reduced across ranks.

    Wraps whatever training_step_impl is currently installed (default,
    embed_loss, curriculum, or cls_loss) — apply this LAST.
    """
    current_training_step = DistillationV2.training_step_impl
    _cached: List[Optional[torch.Tensor]] = [None]

    def _capture_hook(module, inp, out):
        _cached[0] = out

    def patched_training_step_impl(
        self: DistillationV2, batch, batch_idx: int
    ) -> TrainingStepResult:
        raw_student = _get_raw_model(self.student_embedding_model)
        hook = raw_student.norm.register_forward_hook(_capture_hook)

        result = current_training_step(self, batch, batch_idx)

        hook.remove()
        feat = _cached[0]
        _cached[0] = None

        if feat is not None:
            # feat shape: (B, N+1, D) from backbone LayerNorm (N patches + CLS)
            if feat.dim() == 3:
                s_pooled = feat[:, 1:, :].mean(dim=1)  # avg-pool patch tokens -> (B, D)
            else:
                s_pooled = feat.mean(dim=1)

            with torch.amp.autocast("cuda", enabled=False):
                sig_loss = sigreg_weak_loss(
                    s_pooled.float(),
                    global_step=self.global_step,
                    sketch_dim=sketch_dim,
                )
                if include_cls and feat.dim() == 3:
                    s_cls = feat[:, 0, :]  # (B, D)
                    sig_loss = sig_loss + sigreg_weak_loss(
                        s_cls.float(),
                        global_step=self.global_step,
                        sketch_dim=sketch_dim,
                    )

            total_loss = result.loss + weight * sig_loss
            log = result.log_dict if result.log_dict is not None else {}
            log["sigreg_loss"] = sig_loss.detach()
            return TrainingStepResult(loss=total_loss, log_dict=log)

        return result

    DistillationV2.training_step_impl = patched_training_step_impl  # type: ignore[assignment]
    cls_str = "+CLS" if include_cls else ""
    print(f"  [Patch] Weak SIGReg enabled (weight={weight}, sketch_dim={sketch_dim}, "
          f"cov→I on raw backbone features{cls_str})")


def patch_cls_loss(
    weight: float,
    teacher_cls_last_n: int = 1,
    separate_head: bool = False,
) -> None:
    """Add CLS token supervision: MSE(proj(student_cls), teacher_cls_target).

    By default this uses the same ``student_projection_head`` as patch tokens
    (CLS embedding is reshaped to (B, D_s, 1, 1) so the projection head can
    process it identically).

    If ``separate_head=True``, a dedicated linear layer
    ``student_cls_projection_head`` (D_s -> D_t_concat) is added and used only
    for CLS supervision.

    By default the target is the last teacher CLS token. If
    ``teacher_cls_last_n > 1``, this concatenates the CLS tokens from the
    last N executed teacher blocks to match multi-block distillation settings.

    Wraps whatever ``training_step_impl`` is currently installed (default,
    embed_loss, or curriculum).  Apply after embed_loss / curriculum but
    before SigReg.
    """
    import torch.nn.functional as F

    if separate_head:
        original_init = DistillationV2.__init__

        def patched_init(self, method_args, optimizer_args, embedding_model,
                         global_batch_size, num_input_channels):
            original_init(
                self, method_args, optimizer_args, embedding_model,
                global_batch_size, num_input_channels
            )
            self.student_cls_projection_head = nn.Linear(
                embedding_model.embed_dim,
                self.teacher_embedding_dim,
            )

        DistillationV2.__init__ = patched_init  # type: ignore[assignment]

    current_training_step = DistillationV2.training_step_impl
    _student_cached: List[Optional[torch.Tensor]] = [None]
    _teacher_cached: List[Optional[torch.Tensor]] = [None]

    def _student_hook(module, inp, out):
        _student_cached[0] = out

    def _teacher_hook(module, inp, out):
        _teacher_cached[0] = out

    def patched_training_step_impl(
        self: DistillationV2, batch, batch_idx: int
    ) -> TrainingStepResult:
        raw_student = _get_raw_model(self.student_embedding_model)
        s_hook = raw_student.norm.register_forward_hook(_student_hook)
        teacher_block_outputs: List[torch.Tensor] = []

        def _teacher_block_hook(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(x):
                teacher_block_outputs.append(x)
                if len(teacher_block_outputs) > teacher_cls_last_n:
                    teacher_block_outputs.pop(0)

        t_hooks = []
        teacher_blocks = getattr(self.teacher_embedding_model, "blocks", None)
        if teacher_blocks is not None:
            for blk in teacher_blocks:
                t_hooks.append(blk.register_forward_hook(_teacher_block_hook))
        else:
            # Fallback path for unknown teacher wrappers: only final CLS.
            t_hooks.append(
                self.teacher_embedding_model.norm.register_forward_hook(_teacher_hook)
            )

        result = current_training_step(self, batch, batch_idx)

        s_hook.remove()
        for hook in t_hooks:
            hook.remove()

        s_feat = _student_cached[0]
        t_feat = _teacher_cached[0]
        _student_cached[0] = None
        _teacher_cached[0] = None

        teacher_has_signal = (t_feat is not None) or bool(teacher_block_outputs)
        if s_feat is not None and teacher_has_signal:
            s_cls = s_feat[:, 0, :]  # (B, D_s)

            if separate_head:
                s_cls_proj = self.student_cls_projection_head(s_cls)  # (B, D_proj)
            else:
                s_cls_proj = self.student_projection_head(
                    s_cls.unsqueeze(-1).unsqueeze(-1)  # (B, D_s, 1, 1)
                ).squeeze(-2).squeeze(-2)  # (B, D_proj)

            t_cls = None
            if teacher_block_outputs:
                # Build target from last N executed teacher blocks.
                with torch.no_grad():
                    teacher_dim = self.teacher_embedding_model.embed_dim
                    max_by_proj = max(1, s_cls_proj.shape[-1] // teacher_dim)
                    n_use = min(
                        teacher_cls_last_n, len(teacher_block_outputs), max_by_proj
                    )
                    if n_use < teacher_cls_last_n and not getattr(
                        self, "_cls_last_n_warned", False
                    ):
                        print(
                            f"  [CLS loss] Requested last_n={teacher_cls_last_n}, "
                            f"using {n_use} to match projection dim/executed blocks."
                        )
                        self._cls_last_n_warned = True  # type: ignore[attr-defined]
                    cls_chunks = []
                    for block_out in teacher_block_outputs[-n_use:]:
                        cls_chunks.append(
                            self.teacher_embedding_model.norm(block_out)[:, 0, :].detach()
                        )
                    t_cls = torch.cat(cls_chunks, dim=-1)
            elif t_feat is not None:
                t_cls = t_feat[:, 0, :].detach()  # (B, D_t)

            if t_cls is None:
                return result

            # Handle mild dimensional mismatch robustly by tail-alignment.
            if s_cls_proj.shape[-1] > t_cls.shape[-1]:
                s_cls_proj = s_cls_proj[:, -t_cls.shape[-1]:]
            elif s_cls_proj.shape[-1] < t_cls.shape[-1]:
                t_cls = t_cls[:, -s_cls_proj.shape[-1]:]

            cls_loss_val = F.mse_loss(s_cls_proj, t_cls)
            total_loss = result.loss + weight * cls_loss_val
            log = result.log_dict if result.log_dict is not None else {}
            log["cls_loss"] = cls_loss_val.detach()
            return TrainingStepResult(loss=total_loss, log_dict=log)

        return result

    DistillationV2.training_step_impl = patched_training_step_impl  # type: ignore[assignment]
    head_desc = "separate_linear_head" if separate_head else "shared_head"
    print(
        f"  [Patch] CLS token loss enabled (weight={weight}, "
        f"last_n={teacher_cls_last_n}, head={head_desc})"
    )


def patch_lr_override_on_resume() -> None:
    """Force the scheduler/optimizer to adopt a new base LR on resume.

    PyTorch Lightning restores the LR scheduler's full state on resume,
    including ``base_lrs`` — which overrides any new LR passed via
    ``optim_args={"lr": ...}``. This patch captures the freshly-built
    ``base_lrs`` in ``configure_optimizers`` (which runs BEFORE the
    checkpoint load), stashes them on the module, and re-applies them in
    ``on_train_start`` (which runs AFTER the checkpoint load). The net
    effect is: the resumed run uses the new peak LR, while preserving
    cosine progress (``last_epoch`` / ``_step_count``), momentum buffers,
    and curriculum state.
    """
    original_configure = DistillationV2.configure_optimizers
    original_on_train_start = getattr(DistillationV2, "on_train_start", None)

    def patched_configure(self):
        optimizers, scheduler_cfgs = original_configure(self)
        # Stash the fresh base_lrs (built from the new --lr) BEFORE Lightning
        # restores the checkpoint's scheduler state.
        self._lr_override_base_lrs = []
        self._lr_override_pg_lrs = []
        for sched_cfg in scheduler_cfgs:
            sched = sched_cfg["scheduler"]
            self._lr_override_base_lrs.append(
                list(getattr(sched, "base_lrs", []))
            )
        for opt in optimizers:
            self._lr_override_pg_lrs.append(
                [pg.get("lr") for pg in opt.param_groups]
            )
        return optimizers, scheduler_cfgs

    def patched_on_train_start(self):
        if original_on_train_start is not None:
            original_on_train_start(self)
        stash_base = getattr(self, "_lr_override_base_lrs", None)
        stash_pg = getattr(self, "_lr_override_pg_lrs", None)
        if not stash_base:
            return
        try:
            sched_cfgs = self.trainer.lr_scheduler_configs
        except AttributeError:
            sched_cfgs = []
        try:
            optimizers = self.trainer.optimizers
        except AttributeError:
            optimizers = []

        for sched_cfg, new_base in zip(sched_cfgs, stash_base):
            sched = sched_cfg.scheduler
            old_base = list(getattr(sched, "base_lrs", []))
            if hasattr(sched, "base_lrs") and new_base:
                sched.base_lrs = list(new_base)
                print(f"  [LR Override on Resume] scheduler.base_lrs "
                      f"{old_base} -> {list(sched.base_lrs)}")

        for opt, new_pg_lrs in zip(optimizers, stash_pg):
            # Recompute the current step's LR from the fresh base_lrs via
            # the scheduler so optimizer param_groups match the new curve.
            if sched_cfgs:
                sched = sched_cfgs[0].scheduler
                if hasattr(sched, "get_last_lr"):
                    try:
                        current_lrs = sched.get_last_lr()
                    except Exception:
                        current_lrs = new_pg_lrs
                else:
                    current_lrs = new_pg_lrs
            else:
                current_lrs = new_pg_lrs
            # Align length defensively (some setups might have more
            # param_groups than scheduler base_lrs).
            while len(current_lrs) < len(opt.param_groups):
                current_lrs = list(current_lrs) + [current_lrs[-1]]
            old_pg = [pg.get("lr") for pg in opt.param_groups]
            for pg, new_lr in zip(opt.param_groups, current_lrs):
                pg["lr"] = float(new_lr)
            new_pg = [pg.get("lr") for pg in opt.param_groups]
            print(f"  [LR Override on Resume] optimizer param_group lr "
                  f"{old_pg} -> {new_pg}")

    DistillationV2.configure_optimizers = patched_configure  # type: ignore[assignment]
    DistillationV2.on_train_start = patched_on_train_start  # type: ignore[assignment]
    print("  [Patch] LR override on resume enabled "
          "(scheduler.base_lrs + optimizer param_group lr will be rewritten "
          "after checkpoint restore).")


def patch_lr_schedule(min_lr_ratio: float) -> None:
    """Override the cosine scheduler's end_value to control LR decay severity.

    By default LightlyTrain uses end_value=0.001 (LR decays to 0.1% of peak).
    Setting min_lr_ratio higher (e.g. 0.1) keeps the final LR at 10% of peak.
    """
    from lightly.utils.scheduler import CosineWarmupScheduler

    original_configure = DistillationV2.configure_optimizers

    def patched_configure(self):
        # Returns ([optimizer], [scheduler_dict])
        optimizers, scheduler_cfgs = original_configure(self)
        for sched_cfg in scheduler_cfgs:
            sched = sched_cfg["scheduler"]
            if isinstance(sched, CosineWarmupScheduler):
                new_sched = CosineWarmupScheduler(
                    optimizer=sched.optimizer,
                    warmup_epochs=sched.warmup_epochs,
                    max_epochs=sched.max_epochs,
                    end_value=min_lr_ratio,
                )
                sched_cfg["scheduler"] = new_sched
                print(f"  [LR schedule] Cosine end_value={min_lr_ratio} "
                      f"(final LR = {min_lr_ratio:.1%} of peak)")
                break
        return optimizers, scheduler_cfgs

    DistillationV2.configure_optimizers = patched_configure  # type: ignore[assignment]
    print(f"  [Patch] LR schedule: min_lr_ratio={min_lr_ratio}")


def _build_vit_student(size: str, img_size: int, patch_size: int, label: str):
    """Build a ViT student at a given size / img_size / patch_size.

    `size`:
      * "small" — ViT-S/DINOv2 shape via timm's
        `vit_small_patch14_dinov2.lvd142m` config (embed=384, depth=12,
        heads=6). Uses the existing pretrained config layout but starts
        from random weights (`pretrained=False`).
      * "tiny" — DeiT-Tiny shape (embed=192, depth=12, heads=3) constructed
        directly from `timm.models.vision_transformer.VisionTransformer`
        because DINOv2 does not publish a tiny config. mlp_ratio=4,
        qkv_bias=True, num_classes=0 to expose features.
    """
    if size == "small":
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
        )
        size_label = "ViT-S"
    elif size == "tiny":
        from timm.models.vision_transformer import VisionTransformer
        model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=0,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
        )
        size_label = "ViT-T"
    else:
        raise ValueError(f"Unknown student size '{size}' (expected 'small' or 'tiny').")

    grid_h = img_size // patch_size
    grid_w = img_size // patch_size
    tokens = grid_h * grid_w
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {label} student: {size_label}, patch_size={patch_size}, "
          f"img_size={img_size}, grid={grid_h}x{grid_w}, tokens={tokens}, "
          f"params={n_params:,}")
    return model


def build_fair_student(size: str = "small"):
    """
    Build a ViT/14 student explicitly at img_size=224.

    When passing the model as a string to LightlyTrain, timm uses its default
    img_size=518 for vit_small_patch14_dinov2.lvd142m, giving pos_embed [1,1370,d]
    which gets interpolated at runtime for 224x224 inputs. This function builds the
    model with img_size=224 directly so pos_embed is exactly [1,257,d], matching
    the actual training resolution with no interpolation.
    """
    return _build_vit_student(size, img_size=224, patch_size=14, label="Fair")


def build_lowres_student(img_size: int = 112, size: str = "small"):
    """Build a ViT/14 student at a given low resolution.

    Both student and teacher process the same resolution with patch_size=14.
    img_size must be divisible by 14.
    """
    if img_size % 14 != 0:
        raise ValueError(
            f"--image-size must be divisible by 14 for patch_size=14, got {img_size}"
        )
    return _build_vit_student(size, img_size=img_size, patch_size=14, label="Lowres")


def build_custom_student(size: str = "small"):
    """
    Build a ViT student with patch_size=7 for 112x112 input.
    This gives 112/7 = 16x16 = 256 tokens, matching the teacher's
    224/14 = 16x16 = 256 tokens grid.
    """
    return _build_vit_student(size, img_size=112, patch_size=7, label="Custom")


def main():
    args = parse_args()
    if args.last_n < 1:
        raise ValueError("--last-n must be >= 1.")
    if args.cls_loss_last_n is not None and args.cls_loss_last_n < 1:
        raise ValueError("--cls-loss-last-n must be >= 1.")
    if args.cls_loss_last_n is not None and args.cls_loss_last_n != args.last_n:
        raise ValueError(
            "--cls-loss-last-n must match --last-n. "
            "Use --last-n as the single source of truth."
        )

    if args.curriculum and args.teacher_block is not None:
        raise ValueError(
            "--curriculum and --teacher-block cannot be used together "
            "(curriculum manages teacher blocks dynamically)."
        )
    if args.curriculum and args.last_n != 1:
        raise ValueError(
            "--curriculum currently supports --last-n=1 only "
            "(curriculum targets one teacher block at a time)."
        )
    if args.teacher_block is not None and args.last_n != 1:
        raise ValueError(
            "--teacher-block requires --last-n=1 because the teacher is pruned "
            "to a single block."
        )
    if args.all_layers:
        # Build a small list of incompatible flags so the error message
        # explains *why* this is rejected.
        forbidden = []
        if args.curriculum:
            forbidden.append("--curriculum")
        if args.teacher_block is not None:
            forbidden.append("--teacher-block")
        if args.embed_loss:
            forbidden.append("--embed-loss")
        if args.cls_loss:
            forbidden.append("--cls-loss")
        if args.sigreg:
            forbidden.append("--sigreg")
        if forbidden:
            raise ValueError(
                "--all-layers is mutually exclusive with: "
                + ", ".join(forbidden)
                + ". The all-layers patch installs its own training_step_impl "
                "and per-layer projectors, so it cannot compose with the "
                "other patch_* helpers in this file."
            )
        if args.last_n != 1:
            raise ValueError(
                "--all-layers requires --last-n=1 (the LightlyTrain "
                "projection head is bypassed; per-layer projectors handle "
                "the dim mismatch)."
            )
    if args.all_layers_mean and not args.all_layers:
        raise ValueError(
            "--all-layers-mean requires --all-layers."
        )
    if args.all_layers_cls_weight != 0.0 and not args.all_layers:
        raise ValueError(
            "--all-layers-cls-weight requires --all-layers."
        )
    if args.all_layers_cls_weight < 0.0:
        raise ValueError(
            "--all-layers-cls-weight must be >= 0."
        )
    if args.curriculum_lr_reset and not args.curriculum:
        raise ValueError(
            "--curriculum-lr-reset requires --curriculum."
        )
    if args.curriculum_lr_decay_peak and not args.curriculum_lr_reset:
        raise ValueError(
            "--curriculum-lr-decay-peak requires --curriculum-lr-reset."
        )
    if args.curriculum_lr_reset_after_warmup and not args.curriculum_lr_reset:
        raise ValueError(
            "--curriculum-lr-reset-after-warmup requires --curriculum-lr-reset."
        )
    if args.curriculum_lr_decay_peak_gap and not args.curriculum_lr_reset:
        raise ValueError(
            "--curriculum-lr-decay-peak-gap requires --curriculum-lr-reset."
        )
    if args.curriculum_lr_decay_peak_gap and args.curriculum_lr_decay_peak:
        raise ValueError(
            "--curriculum-lr-decay-peak-gap is mutually exclusive with "
            "--curriculum-lr-decay-peak."
        )
    if args.curriculum_lr_no_decay and not args.curriculum_lr_reset:
        raise ValueError(
            "--curriculum-lr-no-decay requires --curriculum-lr-reset."
        )
    if args.curriculum_lr_no_decay and (
        args.curriculum_lr_decay_peak or args.curriculum_lr_decay_peak_gap
    ):
        raise ValueError(
            "--curriculum-lr-no-decay is mutually exclusive with "
            "--curriculum-lr-decay-peak and --curriculum-lr-decay-peak-gap."
        )
    if args.curriculum_lr_rising_gap and not args.curriculum_lr_reset:
        raise ValueError(
            "--curriculum-lr-rising-gap requires --curriculum-lr-reset."
        )
    if args.curriculum_lr_rising_gap and (
        args.curriculum_lr_decay_peak
        or args.curriculum_lr_decay_peak_gap
        or args.curriculum_lr_no_decay
    ):
        raise ValueError(
            "--curriculum-lr-rising-gap is mutually exclusive with "
            "--curriculum-lr-decay-peak, --curriculum-lr-decay-peak-gap, "
            "and --curriculum-lr-no-decay."
        )
    if args.curriculum_lr_segment_warmup and not args.curriculum_lr_reset:
        raise ValueError(
            "--curriculum-lr-segment-warmup requires --curriculum-lr-reset."
        )
    if args.curriculum_lr_segment_warmup and (
        args.curriculum_lr_decay_peak
        or args.curriculum_lr_decay_peak_gap
        or args.curriculum_lr_no_decay
        or args.curriculum_lr_rising_gap
        or args.curriculum_lr_reset_after_warmup
        or args.curriculum_lr_constant_warmup
    ):
        raise ValueError(
            "--curriculum-lr-segment-warmup is mutually exclusive with "
            "--curriculum-lr-decay-peak, --curriculum-lr-decay-peak-gap, "
            "--curriculum-lr-no-decay, --curriculum-lr-rising-gap, "
            "--curriculum-lr-reset-after-warmup, and "
            "--curriculum-lr-constant-warmup."
        )
    if args.curriculum_lr_segment_warmup and args.curriculum_lr_segment_warmup_steps < 1:
        raise ValueError(
            "--curriculum-lr-segment-warmup-steps must be >= 1."
        )
    if args.curriculum_lr_segment_warmup and not (
        0.0 <= args.curriculum_lr_segment_start_ratio <= 1.0
    ):
        raise ValueError(
            "--curriculum-lr-segment-start-ratio must be in [0.0, 1.0]."
        )
    if args.curriculum_lr_segment_peak_decay and not args.curriculum_lr_segment_warmup:
        raise ValueError(
            "--curriculum-lr-segment-peak-decay requires "
            "--curriculum-lr-segment-warmup."
        )
    if args.curriculum_lr_segment_peak_decay and not (
        0.0 <= args.curriculum_lr_segment_peak_decay_rate <= 1.0
    ):
        raise ValueError(
            "--curriculum-lr-segment-peak-decay-rate must be in [0.0, 1.0]."
        )
    if args.cka_sample_images < 0:
        raise ValueError("--cka-sample-images must be >= 0.")

    # --- snap-to-last paradigm ---------------------------------------------
    if args.curriculum_concat_teacher_last and not args.curriculum:
        raise ValueError(
            "--curriculum-concat-teacher-last requires --curriculum."
        )
    if args.curriculum_concat_teacher_last and args.cls_loss:
        raise ValueError(
            "--curriculum-concat-teacher-last handles CLS internally via "
            "--curriculum-concat-cls-weight; do not pass --cls-loss."
        )
    if args.curriculum_concat_teacher_last and args.all_layers:
        raise ValueError(
            "--curriculum-concat-teacher-last is incompatible with --all-layers."
        )
    if args.curriculum_concat_cls_weight < 0.0:
        raise ValueError(
            "--curriculum-concat-cls-weight must be non-negative."
        )
    if (
        args.curriculum_concat_cls_weight > 0.0
        and not args.curriculum_concat_teacher_last
    ):
        raise ValueError(
            "--curriculum-concat-cls-weight has no effect without "
            "--curriculum-concat-teacher-last; remove the flag or enable "
            "the toggle."
        )

    if args.concat_last_two:
        if args.curriculum:
            raise ValueError(
                "--concat-last-two is incompatible with --curriculum."
            )
        if args.cls_loss:
            raise ValueError(
                "--concat-last-two is incompatible with --cls-loss "
                "(use --concat-last-two-cls-weight instead)."
            )
        if args.all_layers:
            raise ValueError(
                "--concat-last-two is incompatible with --all-layers."
            )
        if args.teacher_block is not None:
            raise ValueError(
                "--concat-last-two is incompatible with --teacher-block."
            )
        if args.curriculum_concat_teacher_last:
            raise ValueError(
                "--concat-last-two is incompatible with "
                "--curriculum-concat-teacher-last (pick one)."
            )
    if args.concat_last_two_cls_weight < 0.0:
        raise ValueError(
            "--concat-last-two-cls-weight must be non-negative."
        )
    if (
        args.concat_last_two_cls_weight > 0.0
        and not args.concat_last_two
    ):
        raise ValueError(
            "--concat-last-two-cls-weight has no effect without "
            "--concat-last-two; remove the flag or enable the toggle."
        )

    if args.curriculum_reset_projector and not args.curriculum:
        raise ValueError(
            "--curriculum-reset-projector requires --curriculum."
        )
    if args.curriculum_metric == "cosine" and not args.curriculum:
        raise ValueError(
            "--curriculum-metric=cosine requires --curriculum."
        )
    if not (-1.0 <= args.curriculum_cosine_threshold <= 1.0):
        raise ValueError(
            f"--curriculum-cosine-threshold must be in [-1, 1], got "
            f"{args.curriculum_cosine_threshold}."
        )

    if args.curriculum_snap_to_last and not args.curriculum:
        raise ValueError(
            "--curriculum-snap-to-last requires --curriculum."
        )
    if args.curriculum_snap_to_last and args.curriculum_snap_to_last_at < 1:
        raise ValueError(
            "--curriculum-snap-to-last-at must be >= 1."
        )
    if args.curriculum_snap_to_last and (
        args.curriculum_lr_decay_peak
        or args.curriculum_lr_decay_peak_gap
        or args.curriculum_lr_no_decay
        or args.curriculum_lr_rising_gap
        or args.curriculum_lr_reset_after_warmup
        or args.curriculum_lr_constant_warmup
        or args.curriculum_lr_segment_warmup
        or args.curriculum_lr_segment_peak_decay
    ):
        raise ValueError(
            "--curriculum-snap-to-last is mutually exclusive with all other "
            "--curriculum-lr-* schedule modes (decay-peak, decay-peak-gap, "
            "no-decay, rising-gap, reset-after-warmup, constant-warmup, "
            "segment-warmup, segment-peak-decay). It defines its own LR "
            "paradigm: no warmup, peak-LR start, cosine decay, no resets."
        )
    # The paradigm forces a specific LR mode: no warmup, no curriculum LR
    # resets, pure cosine decay from peak. Implement by auto-enabling the
    # equivalent existing flags.
    if args.curriculum_snap_to_last:
        if not args.curriculum_lr_reset:
            args.curriculum_lr_reset = True
        if not args.curriculum_lr_disable_resets:
            args.curriculum_lr_disable_resets = True
        print(
            f"  [Curriculum] snap-to-last paradigm enabled: "
            f"snap_at=block {args.curriculum_snap_to_last_at}, "
            f"LR mode = no-warmup + cosine decay + no resets "
            f"(implies --curriculum-lr-reset --curriculum-lr-disable-resets)."
        )

    if args.lr_override_on_resume and not args.resume:
        raise ValueError(
            "--lr-override-on-resume requires --resume."
        )
    if args.lr_override_on_resume and args.lr is None:
        raise ValueError(
            "--lr-override-on-resume requires --lr to specify the new base LR."
        )

    force_custom_model = False

    # Determine student model, image size, and patch size based on mode.
    # `--student-size` is orthogonal to mode: it picks ViT-S vs ViT-T while
    # mode controls patch / image size and any custom-module wrapping. When
    # student_size != 'small' we always build the model explicitly (no
    # string-based timm default), since the default LightlyTrain string path
    # only resolves to ViT-S.
    size_label_short = "ViT-S" if args.student_size == "small" else "ViT-T"
    needs_custom_build = force_custom_model or args.student_size != "small"
    if args.mode == "custom":
        student_model = build_custom_student(size=args.student_size)
        image_size = [112, 112]
        student_patch_size = 7
        mode_desc = f"custom ({size_label_short}, patch7, 112x112)"
    elif args.mode == "lowres":
        lowres_size = args.image_size if args.image_size is not None else 112
        student_model = build_lowres_student(
            img_size=lowres_size, size=args.student_size
        )
        image_size = [lowres_size, lowres_size]
        student_patch_size = 14
        grid = lowres_size // 14
        mode_desc = (f"lowres ({size_label_short}/14, {lowres_size}x{lowres_size}, "
                     f"both student+teacher at {grid}x{grid}={grid*grid} tokens)")
    elif args.mode == "fair":
        if needs_custom_build:
            student_model = build_fair_student(size=args.student_size)
            mode_desc = (f"fair ({size_label_short}/14, 224x224, "
                         f"explicit module)")
        else:
            student_model = "timm/vit_small_patch14_dinov2.lvd142m"
            mode_desc = "fair (ViT-S/14, 224x224, img_size=224 via model_args)"
        image_size = [224, 224]
        student_patch_size = 14
    else:
        if needs_custom_build:
            if args.student_size == "small":
                student_model = timm.create_model(
                    "vit_small_patch14_dinov2.lvd142m", pretrained=False,
                )
            else:
                student_model = build_fair_student(size=args.student_size)
            mode_desc = (f"default ({size_label_short}/14, 224x224, "
                         f"explicit module)")
        else:
            student_model = "timm/vit_small_patch14_dinov2.lvd142m"
            mode_desc = "default (ViT-S/14, 224x224, timm default img_size=518)"
        image_size = [224, 224]
        student_patch_size = 14

    print("=" * 60)
    print("LightlyTrain Distillation Baseline")
    print(f"  Mode:    {mode_desc}")
    print(f"  Student size: {args.student_size}")
    print(f"  Teacher: {args.teacher_model}")
    print(f"  Data:    {args.data}")
    print(f"  Output:  {args.out}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch_size}")
    print(f"  Precision: {args.precision}")
    print(f"  Method:  {args.method}")
    print(f"  Image:   {image_size[0]}x{image_size[1]}")
    print(f"  LR:      {args.lr if args.lr is not None else 'auto (9.0)'}")
    print(f"  Embed loss: {args.embed_loss} (weight={args.embed_loss_weight})")
    print(f"  All-layers: {args.all_layers} "
          f"(reduction={'mean' if args.all_layers_mean else 'sum'}, "
          f"cls_weight={args.all_layers_cls_weight})")
    print(
        f"  CLS loss: {args.cls_loss} "
        f"(weight={args.cls_loss_weight}, last_n={args.last_n}, "
        f"separate_head={args.separate_cls_head})"
    )
    print(
        f"  ConcatLastTwo: {args.concat_last_two} "
        f"(cls_weight={args.concat_last_two_cls_weight})"
    )
    print(f"  SIGReg: {args.sigreg} (weight={args.sigreg_weight}, sketch_dim={args.sigreg_sketch_dim})")
    print(f"  Min LR ratio: {args.min_lr_ratio if args.min_lr_ratio is not None else '0.001 (default)'}")
    print(f"  Curriculum: {args.curriculum} "
          f"(metric={args.curriculum_metric}, "
          f"cka_thr={args.cka_threshold}, "
          f"cos_thr={args.curriculum_cosine_threshold}, "
          f"ema={args.ema_alpha}, "
          f"max_ep/blk={args.max_epochs_per_block}, "
          f"lr_reset={args.curriculum_lr_reset}, "
          f"decay_peak={args.curriculum_lr_decay_peak}, "
          f"decay_peak_gap={args.curriculum_lr_decay_peak_gap}, "
          f"no_decay={args.curriculum_lr_no_decay}, "
          f"rising_gap={args.curriculum_lr_rising_gap}"
          f"(scale={args.curriculum_lr_rising_gap_scale}), "
          f"constant_warmup={args.curriculum_lr_constant_warmup}, "
          f"disable_resets={args.curriculum_lr_disable_resets}, "
          f"reset_after_warmup={args.curriculum_lr_reset_after_warmup}, "
          f"segment_warmup={args.curriculum_lr_segment_warmup}"
          f"(steps={args.curriculum_lr_segment_warmup_steps}, "
          f"start_ratio={args.curriculum_lr_segment_start_ratio}, "
          f"peak_decay={args.curriculum_lr_segment_peak_decay}, "
          f"peak_decay_rate={args.curriculum_lr_segment_peak_decay_rate}), "
          f"snap_to_last={args.curriculum_snap_to_last}"
          f"(at_block={args.curriculum_snap_to_last_at}), "
          f"sync_rank={args.curriculum_sync_rank})")
    if args.curriculum:
        print(f"  Curriculum CKA sample images: "
              f"{args.cka_sample_images if args.cka_sample_images > 0 else 'full batch'}")
    if args.teacher_block is not None:
        print(f"  Teacher block: {args.teacher_block} (early exit, 1 block)")
    elif args.curriculum:
        print(f"  Teacher blocks: curriculum (progressive, starting at 1)")
    else:
        print(f"  Teacher blocks: last {args.last_n}")
    print("=" * 60)

    # Teacher early exit: prune teacher to a single block
    if args.teacher_block is not None:
        patch_teacher_early_exit(args.teacher_block)

    # Conv2d patch embedding loss
    if args.embed_loss:
        patch_embed_loss(args.embed_loss_weight)

    # LR schedule override (skip if curriculum-lr-reset handles it)
    if args.min_lr_ratio is not None and not args.curriculum_lr_reset:
        patch_lr_schedule(args.min_lr_ratio)

    # LR override on resume (must be applied BEFORE curriculum patch, so
    # patch_curriculum_distillation's own configure_optimizers wrapper
    # composes with our freshly-captured base_lrs).
    if args.lr_override_on_resume:
        patch_lr_override_on_resume()

    # Curriculum block-skipping (applied last — replaces training_step_impl)
    if args.curriculum:
        effective_min_lr = args.min_lr_ratio if args.min_lr_ratio is not None else 0.001
        patch_curriculum_distillation(
            cka_threshold=args.cka_threshold,
            ema_alpha=args.ema_alpha,
            max_epochs_per_block=args.max_epochs_per_block,
            embed_loss_weight=args.embed_loss_weight if args.embed_loss else 0.0,
            out_dir=args.out,
            lr_reset=args.curriculum_lr_reset,
            min_lr_ratio=effective_min_lr,
            decay_peak=args.curriculum_lr_decay_peak,
            decay_peak_gap=args.curriculum_lr_decay_peak_gap,
            no_decay=args.curriculum_lr_no_decay,
            rising_gap=args.curriculum_lr_rising_gap,
            rising_gap_scale=args.curriculum_lr_rising_gap_scale,
            constant_warmup=args.curriculum_lr_constant_warmup,
            disable_resets=args.curriculum_lr_disable_resets,
            reset_after_warmup=args.curriculum_lr_reset_after_warmup,
            segment_warmup=args.curriculum_lr_segment_warmup,
            segment_warmup_steps=args.curriculum_lr_segment_warmup_steps,
            segment_start_ratio=args.curriculum_lr_segment_start_ratio,
            segment_peak_decay=args.curriculum_lr_segment_peak_decay,
            segment_peak_decay_rate=args.curriculum_lr_segment_peak_decay_rate,
            cka_sample_images=args.cka_sample_images,
            sync_rank=args.curriculum_sync_rank,
            snap_to_last_at=(
                args.curriculum_snap_to_last_at
                if args.curriculum_snap_to_last
                else None
            ),
            concat_teacher_last=args.curriculum_concat_teacher_last,
            concat_cls_weight=args.curriculum_concat_cls_weight,
            curriculum_metric=args.curriculum_metric,
            cosine_threshold=args.curriculum_cosine_threshold,
            reset_projector=args.curriculum_reset_projector,
        )

    # CLS token supervision (wraps whatever training step is active)
    if args.cls_loss:
        patch_cls_loss(
            args.cls_loss_weight,
            args.last_n,
            separate_head=args.separate_cls_head,
        )

    # SIGReg regularization (applied last — wraps whatever training step is active)
    if args.sigreg:
        patch_sigreg(args.sigreg_weight, args.sigreg_sketch_dim,
                     include_cls=args.cls_loss)

    # Naive all-layer alignment (replaces training_step_impl; not composable
    # with the other patches above — guarded in the validation block).
    if args.all_layers:
        patch_all_layers_distillation(
            use_mean=args.all_layers_mean,
            cls_weight=args.all_layers_cls_weight,
        )

    # Static concat-last-two (replaces training_step_impl; not composable with
    # curriculum/cls-loss/all-layers/ibot/teacher-block; guarded above).
    if args.concat_last_two:
        patch_concat_last_two_distillation(
            cls_weight=args.concat_last_two_cls_weight,
        )

    method_args_dict = {"teacher": args.teacher_model}
    if args.teacher_block is not None:
        method_args_dict["n_teacher_blocks"] = 1
    elif args.curriculum:
        method_args_dict["n_teacher_blocks"] = 1
    elif args.all_layers:
        # We bypass the LightlyTrain projection head; n_teacher_blocks=1 keeps
        # the head's input dim small (teacher_dim) so init is cheap.
        method_args_dict["n_teacher_blocks"] = 1
    elif args.concat_last_two:
        # Same rationale as all-layers / curriculum: our own widened projector
        # replaces the LightlyTrain head entirely.
        method_args_dict["n_teacher_blocks"] = 1
    else:
        method_args_dict["n_teacher_blocks"] = args.last_n

    pretrain_kwargs = {
        "out": args.out,
        "data": args.data,
        "model": student_model,
        "method": args.method,
        "method_args": method_args_dict,
        "transform_args": {
            "image_size": image_size,
        },
        "callbacks": {
            "model_checkpoint": {
                "every_n_epochs": 5,
                "save_top_k": -1,
            },
        },
        "loggers": {
            "jsonl": {},
            "tensorboard": None,
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "num_workers": args.num_workers,
        "float32_matmul_precision": args.matmul_precision,
        "overwrite": args.overwrite,
        "resume_interrupted": args.resume,
    }
    if args.lr is not None:
        pretrain_kwargs["optim_args"] = {"lr": args.lr}
    if args.mode == "fair" and not force_custom_model:
        pretrain_kwargs["model_args"] = {"img_size": 224, "pretrained": False}

    lightly_train.pretrain(**pretrain_kwargs)

    print("=" * 60)
    print("Training complete!")
    print(f"  Exported model: {args.out}/exported_models/exported_last.pt")
    print(f"  Checkpoints:    {args.out}/checkpoints/")
    print("=" * 60)


if __name__ == "__main__":
    main()
