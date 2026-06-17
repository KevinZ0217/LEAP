"""
Linear Probing Evaluation for LightlyTrain Distilled Student

Loads the exported ViT-S model from LightlyTrain, freezes it, extracts
CLS token features, and trains a linear classifier — same protocol as
kd_distill/eval_linear.py for fair comparison.

Usage:
  python eval_linear.py --checkpoint /path/to/output/leap_baseline/exported_models/exported_last.pt
"""

import argparse
import os
import time
import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear probing evaluation for LightlyTrain distilled student"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to exported .pt or Lightning .ckpt file")
    parser.add_argument("--data-dir", type=str, default="/path/to/imagenet",
                        help="Dataset root (must have train/ and val/ subdirs)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Image size for evaluation")
    parser.add_argument("--model-name", type=str, default="vit_small_patch14_dinov2.lvd142m",
                        help="TIMM model name (architecture only, weights loaded from checkpoint). "
                             "Ignored when --student-size=tiny (a DeiT-Tiny shape is built directly).")
    parser.add_argument("--student-size", type=str, default="small",
                        choices=["small", "tiny"],
                        help="Student backbone size. 'small' = ViT-S (embed=384, "
                             "depth=12, heads=6) — uses --model-name. 'tiny' = "
                             "ViT-T (embed=192, depth=12, heads=3) — built "
                             "directly from timm's VisionTransformer to match "
                             "the architecture used during training. Must match "
                             "the student that produced the checkpoint.")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Override patch size (e.g. 7 for dual-res student). "
                             "If None, uses the model default.")
    parser.add_argument("--model-img-size", type=int, default=None,
                        help="Override model img_size for architecture instantiation "
                             "(e.g. 112 for dual-res student). If None, uses the model default.")
    parser.add_argument("--adapt-patch-size", type=int, default=None,
                        help="Interpolate patch_embed kernel from the checkpoint's patch size "
                             "to this target size before evaluation (e.g. --patch-size 7 "
                             "--adapt-patch-size 14 to upsample a patch7 student to patch14). "
                             "pos_embed is reused as-is (grid size must match).")

    # Feature extraction
    parser.add_argument("--feature-type", type=str, default="cls",
                        choices=["cls", "avgpool"],
                        help="Feature type: 'cls' or 'avgpool' over patch tokens")
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    # Linear probe
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-lr", type=float, default=0.1)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-momentum", type=float, default=0.9)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0)

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to same dir as checkpoint)")

    return parser.parse_args()


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("LinearProbe-Lightly")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(output_dir, "eval_linear_lightly.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def get_eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(
            int(image_size * 256 / 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def adapt_patch_embed(state_dict: dict, target_patch_size: int, logger) -> dict:
    """Interpolate patch_embed.proj.weight to a new kernel size.

    Useful for zero-shot evaluation of a patch7 student at patch14/224x224:
    the conv kernel [384, 3, 7, 7] is bicubic-resized to [384, 3, 14, 14].
    pos_embed is left unchanged — this only works when the token grid size is
    the same (e.g. patch7/112 and patch14/224 both give a 16x16 grid).
    """
    old_weight = state_dict["patch_embed.proj.weight"]
    src_patch_size = old_weight.shape[-1]

    if src_patch_size == target_patch_size:
        logger.info(f"  adapt-patch-size: already {target_patch_size}, no interpolation needed")
        return state_dict

    new_weight = F.interpolate(
        old_weight.float(),
        size=(target_patch_size, target_patch_size),
        mode="bicubic",
        align_corners=False,
    )
    state_dict = dict(state_dict)
    state_dict["patch_embed.proj.weight"] = new_weight
    logger.info(
        f"  adapt-patch-size: patch_embed.proj.weight "
        f"{list(old_weight.shape)} → {list(new_weight.shape)} (bicubic)"
    )

    pos_embed = state_dict["pos_embed"]
    logger.info(f"  adapt-patch-size: pos_embed {list(pos_embed.shape)} unchanged "
                f"(grid size preserved)")
    return state_dict


CKPT_PREFIX = "student_embedding_model.wrapped_model._model."


def _extract_student_sd(checkpoint_path: str, logger) -> dict:
    """Extract student state_dict from a Lightning .ckpt file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    student_sd = {}
    for k, v in state.items():
        if k.startswith(CKPT_PREFIX):
            student_sd[k[len(CKPT_PREFIX):]] = v
    if not student_sd:
        raise ValueError(
            f"No keys with prefix '{CKPT_PREFIX}' found in checkpoint. "
            f"Available prefixes: {set(k.split('.')[0] for k in state.keys())}"
        )
    logger.info(f"  Extracted {len(student_sd)} student keys from .ckpt")
    return student_sd


def _load_state_dict(checkpoint_path: str, logger) -> dict:
    """Load state_dict from either .pt or .ckpt file."""
    if checkpoint_path.endswith(".ckpt"):
        return _extract_student_sd(checkpoint_path, logger)
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def _build_arch(
    student_size: str,
    model_name: str,
    img_size: int,
    patch_size: int,
    logger,
):
    """Construct a frozen-architecture model matching the training student.

    For 'small' we go through timm.create_model with the supplied
    `model_name` (default ViT-S/DINOv2). For 'tiny' we build a DeiT-Tiny
    shape directly from `timm.models.vision_transformer.VisionTransformer`,
    mirroring `_build_vit_student("tiny", ...)` in train_distill.py:
    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True.
    """
    if student_size == "tiny":
        from timm.models.vision_transformer import VisionTransformer
        # ViT-T was trained with patch14 to match the teacher's token grid;
        # default to 14 if the caller didn't pass --patch-size explicitly.
        # This mirrors the default of all 'fair' / 'lowres' / 'default'
        # student modes in train_distill.py.
        effective_patch = patch_size if patch_size is not None else 14
        model = VisionTransformer(
            img_size=img_size,
            patch_size=effective_patch,
            in_chans=3,
            num_classes=0,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
        )
        logger.info(
            f"  Built ViT-T directly: img_size={img_size}, "
            f"patch_size={effective_patch}, embed=192, depth=12, heads=3"
        )
        return model

    arch_kwargs = {"patch_size": patch_size} if patch_size is not None else {}
    arch_kwargs["img_size"] = img_size
    return timm.create_model(
        model_name=model_name,
        pretrained=False,
        dynamic_img_size=True,
        **arch_kwargs,
    )


def load_model(
    checkpoint_path: str,
    model_name: str,
    img_size: int,
    device: torch.device,
    logger,
    patch_size: int = None,
    model_img_size: int = None,
    adapt_patch_size: int = None,
    student_size: str = "small",
):
    """Load model and restore weights from .pt or .ckpt checkpoint.

    Supports both exported .pt files (raw state_dict) and Lightning .ckpt
    files (student weights extracted automatically).

    For custom architectures (e.g. patch7, 112x112 dual-res student), pass
    patch_size and model_img_size to match the architecture used during training.

    Pass adapt_patch_size to interpolate patch_embed before loading.

    Set student_size='tiny' to build a DeiT-Tiny shape (matches
    --student-size tiny in train_distill.py).
    """
    arch_kwargs = {}
    if patch_size is not None:
        arch_kwargs["patch_size"] = patch_size
    if model_img_size is not None:
        arch_kwargs["img_size"] = model_img_size

    # Image size used when constructing the model architecture. Falls back to
    # the eval img_size if no explicit override.
    arch_img_size = model_img_size if model_img_size is not None else img_size
    arch_label = (
        "ViT-T (built directly)" if student_size == "tiny" else model_name
    )

    if adapt_patch_size is not None:
        logger.info(
            f"Loading model: {arch_label}, eval img_size={img_size}"
            + (f", source arch: {arch_kwargs}" if arch_kwargs else "")
            + f", adapt patch_size → {adapt_patch_size}"
        )
        state_dict = _load_state_dict(checkpoint_path, logger)
        state_dict = adapt_patch_embed(state_dict, adapt_patch_size, logger)

        # After adapt-patch, the model is built at the *target* patch size
        # (the source patch_embed has already been interpolated).
        model = _build_arch(
            student_size=student_size,
            model_name=model_name,
            img_size=img_size,
            patch_size=adapt_patch_size,
            logger=logger,
        )
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys:
            logger.warning(f"  Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning(f"  Unexpected keys: {incompatible.unexpected_keys}")
    else:
        logger.info(f"Loading model: {arch_label}, eval img_size={img_size}"
                    + (f", arch overrides: {arch_kwargs}" if arch_kwargs else ""))

        # Architecture build: route through `_build_arch` so 'tiny' takes
        # the VisionTransformer construction path. For 'small' we still go
        # via timm.create_model with the configured `model_name`.
        arch_patch = patch_size if patch_size is not None else None

        if checkpoint_path.endswith(".ckpt"):
            state_dict = _extract_student_sd(checkpoint_path, logger)
            model = _build_arch(
                student_size=student_size,
                model_name=model_name,
                img_size=arch_img_size,
                patch_size=arch_patch,
                logger=logger,
            )
            # 'small' historically tolerates strict=False (head is dropped);
            # 'tiny' requires strict matching since we built it ourselves
            # — surface any unexpected mismatch loudly.
            strict = (student_size == "tiny")
            incompatible = model.load_state_dict(state_dict, strict=strict)
            if not strict:
                if incompatible.missing_keys:
                    logger.warning(f"  Missing keys: {incompatible.missing_keys}")
                if incompatible.unexpected_keys:
                    logger.warning(f"  Unexpected keys: {incompatible.unexpected_keys}")
        else:
            if student_size == "tiny":
                # .pt path: load state_dict, then build + strict-load.
                state_dict = _load_state_dict(checkpoint_path, logger)
                model = _build_arch(
                    student_size=student_size,
                    model_name=model_name,
                    img_size=arch_img_size,
                    patch_size=arch_patch,
                    logger=logger,
                )
                model.load_state_dict(state_dict, strict=True)
            else:
                # timm defaults (e.g. img_size=518 for ViT-S/14 DINOv2) must not override
                # fair-224 checkpoints; always set architecture img_size to match eval
                # unless --model-img-size is passed (already in arch_kwargs).
                timm_kwargs = {"img_size": arch_img_size}
                timm_kwargs.update(arch_kwargs)
                model = timm.create_model(
                    model_name=model_name,
                    pretrained=False,
                    dynamic_img_size=True,
                    checkpoint_path=checkpoint_path,
                    **timm_kwargs,
                )

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Loaded and frozen ({n_params:,} params)")
    return model


@torch.no_grad()
def extract_features(model, dataloader, device, feature_type, desc):
    all_features, all_labels = [], []
    for images, labels in tqdm(dataloader, desc=desc):
        images = images.to(device)
        out = model.forward_features(images)

        if feature_type == "cls":
            if hasattr(out, "shape") and out.dim() == 3:
                features = out[:, 0]  # CLS token
            else:
                features = out
        else:
            if hasattr(out, "shape") and out.dim() == 3:
                features = out[:, 1:].mean(dim=1)  # avg pool patch tokens
            else:
                features = out

        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features, 0), torch.cat(all_labels, 0)


def accuracy(output, target, topk=(1, 5)):
    maxk = min(max(topk), output.shape[1])
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        if k > output.shape[1]:
            results.append(0.0)
        else:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.item() * 100.0 / target.size(0))
    return results


def train_linear_probe(
    train_features, train_labels, val_features, val_labels,
    embed_dim, num_classes, args, device, logger,
):
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    nn.init.zeros_(classifier.weight)
    nn.init.zeros_(classifier.bias)

    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=args.probe_lr,
        momentum=args.probe_momentum, weight_decay=args.probe_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.probe_epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(train_features, train_labels),
                              batch_size=args.probe_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_labels),
                            batch_size=args.probe_batch_size, shuffle=False)

    logger.info(f"Training linear probe: dim={embed_dim}, classes={num_classes}, "
                f"epochs={args.probe_epochs}, lr={args.probe_lr}")

    best_top1, best_top5, best_epoch = 0.0, 0.0, 0
    best_state = None

    for epoch in range(args.probe_epochs):
        classifier.train()
        train_loss, n_batches = 0.0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            loss = criterion(classifier(feats), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        classifier.eval()
        all_logits, all_labels_val = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                all_logits.append(classifier(feats.to(device)).cpu())
                all_labels_val.append(labels)
        all_logits = torch.cat(all_logits, 0)
        all_labels_val = torch.cat(all_labels_val, 0)
        top1, top5 = accuracy(all_logits, all_labels_val, topk=(1, 5))

        if top1 > best_top1:
            best_top1, best_top5, best_epoch = top1, top5, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.probe_epochs:
            logger.info(f"  Epoch {epoch+1:3d}/{args.probe_epochs} | "
                        f"Loss: {train_loss/n_batches:.4f} | "
                        f"Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
    classifier.load_state_dict(best_state)
    logger.info(f"Best: Top-1={best_top1:.2f}%, Top-5={best_top5:.2f}% (epoch {best_epoch})")
    return {
        "best_top1": best_top1,
        "best_top5": best_top5,
        "best_epoch": best_epoch,
        "state_dict": best_state,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)

    logger = setup_logger(args.output_dir)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    model = load_model(
        args.checkpoint, args.model_name, args.image_size, device, logger,
        patch_size=args.patch_size, model_img_size=args.model_img_size,
        adapt_patch_size=args.adapt_patch_size,
        student_size=args.student_size,
    )
    transform = get_eval_transforms(args.image_size)

    logger.info(f"Loading datasets from: {args.data_dir}")
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=transform)
    num_classes = len(train_dataset.classes)
    logger.info(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Classes: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.extract_batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.extract_batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logger.info("Extracting train features...")
    t0 = time.time()
    train_features, train_labels = extract_features(
        model, train_loader, device, args.feature_type, "Extract train")
    logger.info(f"  Train features: {train_features.shape} ({time.time()-t0:.1f}s)")

    logger.info("Extracting val features...")
    t0 = time.time()
    val_features, val_labels = extract_features(
        model, val_loader, device, args.feature_type, "Extract val")
    logger.info(f"  Val features: {val_features.shape} ({time.time()-t0:.1f}s)")

    embed_dim = train_features.shape[1]
    results = train_linear_probe(
        train_features, train_labels, val_features, val_labels,
        embed_dim, num_classes, args, device, logger,
    )

    results_path = os.path.join(args.output_dir, "eval_linear_lightly.pt")
    torch.save(
        {
            "checkpoint": args.checkpoint,
            "feature_type": args.feature_type,
            "image_size": args.image_size,
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            **results,
            "top1": results["best_top1"],
            "top5": results["best_top5"],
        },
        results_path,
    )
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
