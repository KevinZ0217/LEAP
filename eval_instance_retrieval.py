"""
Instance retrieval on Revisited Oxford (ROxford-5k) and Paris (RParis-6k).

Prepares the revisitop layout from Hugging Face (`galilai-group/revisitop` by default),
or uses an existing `datasets/` tree as in fast_dinov2
https://github.com/KevinZ0217/fast_dinov2 (dinov2/eval/instance_recog.py).

Evaluation protocol (mAP / mP@k for Easy, Medium, Hard) follows filipradenovic/revisitop.

Queries are cropped with each entry's ``bbx`` (ROI) before the eval transform,
matching the revisitop reference
https://github.com/filipradenovic/revisitop/blob/master/python/example_process_images.py
(full-frame queries are not standard for this benchmark).

Hugging Face ``datasets>=3`` no longer runs script-based datasets (you get
``Dataset scripts are no longer supported``). Use ``--from-official`` (default in
``submit_instance_recognition.sh``) to download Oxford/VGG archives + revisitop
``.pkl`` with no ``datasets`` dependency, or pin ``pip install 'datasets<3'`` for
``--from-hf``.

Example:
  pip install datasets  # if needed

  python eval_instance_retrieval.py \\
    --checkpoint /path/to/exported_last.pt \\
    --test-dataset roxford5k \\
    --revisit-root /data/revisitop \\
    --from-official
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.request
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eval_linear import get_eval_transforms, load_model
from revisitop_toolkit.dataset import configdataset
from revisitop_toolkit.evaluate import compute_map


def _setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("lightly.instance_retrieval")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(output_dir, "instance_retrieval.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _flush_log(logger: logging.Logger) -> None:
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass


def _parse_args():
    p = argparse.ArgumentParser(description="ROxford / RParis instance retrieval")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Student exported_*.pt checkpoint. Required unless --teacher is set.")
    p.add_argument(
        "--test-dataset",
        type=str,
        choices=["roxford5k", "rparis6k"],
        required=True,
    )
    p.add_argument(
        "--teacher",
        action="store_true",
        help="Evaluate the frozen DINOv2 teacher directly (skips --checkpoint). "
             "With --teacher-blocks, emits one result JSON per teacher block.",
    )
    p.add_argument(
        "--teacher-model",
        type=str,
        default="dinov2/vitg14",
        help="Teacher model name (LightlyTrain alias).",
    )
    p.add_argument(
        "--teacher-weights-img-size",
        type=int,
        default=None,
        help="Architecture img_size used to load teacher weights (auto: 518 for vitg14).",
    )
    p.add_argument(
        "--teacher-blocks",
        type=int,
        nargs="+",
        default=None,
        help="Block indices (0-based) at which to extract features and run retrieval. "
             "Defaults to the last block only when --teacher is set. "
             "Example for last 10 of 40: --teacher-blocks 30 31 32 33 34 35 36 37 38 39",
    )
    p.add_argument(
        "--revisit-root",
        type=str,
        default="/path/to/revisitop",
        help="Root directory containing datasets/roxford5k or datasets/rparis6k (fast_dinov2 layout).",
    )
    p.add_argument(
        "--from-official",
        action="store_true",
        help="Download Oxford/Paris image tgz (VGG) + revisitop gnd.pkl — works without Hugging Face `datasets`.",
    )
    p.add_argument(
        "--from-hf",
        action="store_true",
        help="Materialize via Hugging Face (requires datasets<3 for galilai-group/revisitop).",
    )
    p.add_argument(
        "--hf-dataset-id",
        type=str,
        default="galilai-group/revisitop",
        help="Hugging Face dataset id (needs `pip install datasets`).",
    )
    p.add_argument("--model-name", type=str, default="vit_small_patch14_dinov2.lvd142m")
    p.add_argument("--student-size", type=str, default="small", choices=["small", "tiny"])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--model-img-size", type=int, default=None)
    p.add_argument("--adapt-patch-size", type=int, default=None)
    p.add_argument("--feature-type", type=str, default="cls", choices=["cls", "avgpool"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--l2-normalize", action="store_true", default=True)
    p.add_argument("--no-l2-normalize", action="store_false", dest="l2_normalize")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--overwrite-hf-cache", action="store_true")
    return p.parse_args()


def _stem_from_example(ex) -> str:
    fn = ex.get("filename")
    if fn is None:
        raise KeyError("Hugging Face example missing 'filename'")
    base = os.path.basename(str(fn))
    return os.path.splitext(base)[0]


def _as_index_array(key: str, ex) -> "np.ndarray":
    v = ex.get(key)
    if v is None:
        return np.array([], dtype=np.int64)
    if hasattr(v, "tolist"):
        v = v.tolist()
    return np.asarray(list(v), dtype=np.int64)


def _as_bbx_array(ex) -> Optional["np.ndarray"]:
    """Query ROI [x1, y1, x2, y2] in pixel coords (revisited protocol)."""
    v = ex.get("bbx")
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64).ravel()
    return arr if arr.size >= 4 else None


def _crop_query_roi(img: Image.Image, bbx: "np.ndarray") -> Image.Image:
    """Crop query to building ROI before resize / eval (revisitop protocol)."""
    bbx = np.asarray(bbx, dtype=np.float64).ravel()
    if bbx.size < 4:
        return img
    w, h = img.size
    x1, y1, x2, y2 = (int(round(float(bbx[0]))), int(round(float(bbx[1]))), int(round(float(bbx[2]))), int(round(float(bbx[3]))))
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


# Same URLs as galilai-group/revisitop/revisitop.py (no Hugging Face `datasets` needed).
_OFFICIAL_REVISIT = {
    "roxford5k": {
        "archives": [
            "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz",
        ],
        "gnd": "http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl",
    },
    "rparis6k": {
        "archives": [
            "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1-v1.tgz",
            "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2-v1.tgz",
        ],
        "gnd": "http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl",
    },
}


def _download_file(url: str, dest: str, logger: logging.Logger, timeout: int = 600) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    logger.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (eval_instance_retrieval)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _extract_jpg_from_tgz(tgz_path: str, jpg_dir: str, logger: logging.Logger, overwrite: bool) -> int:
    """Extract every .jpg/.jpeg from archive into jpg_dir/<basename-stem>.jpg."""
    n = 0
    with tarfile.open(tgz_path, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if not name.lower().endswith((".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(os.path.basename(name))[0]
            out_path = os.path.join(jpg_dir, stem + ".jpg")
            if not overwrite and os.path.isfile(out_path):
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    logger.info("  Extracted %d images from %s", n, os.path.basename(tgz_path))
    return n


def prepare_revisitop_official(
    revisit_root: str,
    test_dataset: str,
    logger: logging.Logger,
    overwrite: bool = False,
) -> None:
    """Download revisitop gnd pickle + Oxford/Paris tgz archives into fast_dinov2 layout."""
    if test_dataset not in _OFFICIAL_REVISIT:
        raise ValueError(f"No official URLs for {test_dataset}")

    ds_dir = os.path.join(revisit_root, "datasets", test_dataset)
    jpg_dir = os.path.join(ds_dir, "jpg")
    gnd_path = os.path.join(ds_dir, f"gnd_{test_dataset}.pkl")
    os.makedirs(jpg_dir, exist_ok=True)

    if not overwrite and os.path.isfile(gnd_path) and os.path.isdir(jpg_dir):
        try:
            with open(gnd_path, "rb") as f:
                cfg = pickle.load(f)
            n_stems = len([x for x in os.listdir(jpg_dir) if x.lower().endswith(".jpg")])
            if len(cfg.get("imlist", [])) > 0 and n_stems >= len(cfg["imlist"]):
                logger.info("Found existing official data %s (%d jpgs); skip download.", gnd_path, n_stems)
                return
        except Exception:
            pass

    spec = _OFFICIAL_REVISIT[test_dataset]
    with tempfile.TemporaryDirectory(prefix="revisitop_dl_") as tmp:
        gnd_tmp = os.path.join(tmp, f"gnd_{test_dataset}.pkl")
        _download_file(spec["gnd"], gnd_tmp, logger)
        shutil.copyfile(gnd_tmp, gnd_path)
        logger.info("Wrote %s", gnd_path)

        for url in spec["archives"]:
            arc = os.path.join(tmp, os.path.basename(url.split("?")[0]))
            if not os.path.isfile(arc):
                _download_file(url, arc, logger)
            _extract_jpg_from_tgz(arc, jpg_dir, logger, overwrite=overwrite)

    with open(gnd_path, "rb") as f:
        cfg = pickle.load(f)
    n_jpg = len([x for x in os.listdir(jpg_dir) if x.lower().endswith(".jpg")])
    logger.info("Official prepare done: %d database entries in gnd, %d jpgs on disk", len(cfg["imlist"]), n_jpg)
    if n_jpg < len(cfg["imlist"]):
        logger.warning(
            "Fewer jpgs (%d) than imlist (%d) — check downloads or re-run with --overwrite-hf-cache.",
            n_jpg,
            len(cfg["imlist"]),
        )


def _hf_example_to_pil(ex: dict) -> Image.Image:
    """Decode HF ``image`` field whether it is PIL, ndarray, or path (older loaders)."""
    im = ex.get("image")
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if isinstance(im, np.ndarray):
        return Image.fromarray(im).convert("RGB")
    if isinstance(im, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(im)).convert("RGB")
    if isinstance(im, str) and os.path.isfile(im):
        return Image.open(im).convert("RGB")
    if isinstance(im, dict):
        p = im.get("path")
        if p and os.path.isfile(str(p)):
            return Image.open(str(p)).convert("RGB")
    raise TypeError(f"Cannot decode image field type={type(im)!r} keys={getattr(im, 'keys', lambda: [])()}")


def prepare_revisitop_from_hf(
    revisit_root: str,
    test_dataset: str,
    hf_dataset_id: str,
    logger: logging.Logger,
    overwrite: bool = False,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The `datasets` package is required for --from-hf. Install with: pip install datasets"
        ) from e

    ds_dir = os.path.join(revisit_root, "datasets", test_dataset)
    jpg_dir = os.path.join(ds_dir, "jpg")
    gnd_path = os.path.join(ds_dir, f"gnd_{test_dataset}.pkl")
    os.makedirs(jpg_dir, exist_ok=True)

    if not overwrite and os.path.isfile(gnd_path) and os.path.isdir(jpg_dir):
        try:
            with open(gnd_path, "rb") as f:
                cfg = pickle.load(f)
            n_stems = len([x for x in os.listdir(jpg_dir) if x.lower().endswith(".jpg")])
            gnd_list = cfg.get("gnd") or []
            has_bbx = bool(
                len(gnd_list) > 0 and "bbx" in gnd_list[0] and gnd_list[0]["bbx"] is not None
            )
            if len(cfg.get("imlist", [])) > 0 and n_stems >= len(cfg["imlist"]):
                if has_bbx:
                    logger.info("Found existing %s (%d jpgs); skip HF download.", gnd_path, n_stems)
                    return
                logger.warning(
                    "Existing %s has no query bbx (retrieval would be protocol-wrong). "
                    "Re-downloading from Hugging Face (use --overwrite-hf-cache to replace jpgs too).",
                    gnd_path,
                )
        except Exception:
            pass

    logger.info("Loading HF %s config=%s (imlist + qimlist)...", hf_dataset_id, test_dataset)
    imlist_ds = load_dataset(
        hf_dataset_id,
        name=test_dataset,
        split="imlist",
        trust_remote_code=True,
    )
    q_ds = load_dataset(
        hf_dataset_id,
        name=test_dataset,
        split="qimlist",
        trust_remote_code=True,
    )

    imlist: List[str] = []
    for ex in tqdm(imlist_ds, desc="imlist (save jpg)"):
        stem = _stem_from_example(ex)
        imlist.append(stem)
        fp = os.path.join(jpg_dir, stem + ".jpg")
        if overwrite or not os.path.isfile(fp):
            _hf_example_to_pil(ex).save(fp, quality=95)

    qimlist: List[str] = []
    gnd = []
    for ex in tqdm(q_ds, desc="qimlist (save jpg + gnd)"):
        stem = _stem_from_example(ex)
        qimlist.append(stem)
        fp = os.path.join(jpg_dir, stem + ".jpg")
        if overwrite or not os.path.isfile(fp):
            _hf_example_to_pil(ex).save(fp, quality=95)
        gnd.append(
            {
                "easy": _as_index_array("easy", ex),
                "hard": _as_index_array("hard", ex),
                "junk": _as_index_array("junk", ex),
                "bbx": _as_bbx_array(ex),
            }
        )

    payload = {"imlist": imlist, "qimlist": qimlist, "gnd": gnd}
    with open(gnd_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Wrote %d database + %d query images under %s", len(imlist), len(qimlist), jpg_dir)
    logger.info("Wrote ground truth: %s", gnd_path)


class _OxfordParisJpgDataset(Dataset):
    """Database images: full frame (revisited protocol)."""

    def __init__(self, stems: List[str], jpg_dir: str, transform):
        self.stems = stems
        self.jpg_dir = jpg_dir
        self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index: int):
        path = os.path.join(self.jpg_dir, self.stems[index] + ".jpg")
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class _OxfordParisQueryDataset(Dataset):
    """Query images: crop with gnd[i]['bbx'] then eval transform (revisitop)."""

    def __init__(self, qimlist: List[str], gnd_raw: list, jpg_dir: str, transform):
        self.stems = qimlist
        self.gnd_raw = gnd_raw
        self.jpg_dir = jpg_dir
        self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index: int):
        path = os.path.join(self.jpg_dir, self.stems[index] + ".jpg")
        img = Image.open(path).convert("RGB")
        gi = self.gnd_raw[index]
        bbx = gi.get("bbx")
        if bbx is not None:
            bbx = np.asarray(bbx, dtype=np.float64).ravel()
            if bbx.size >= 4:
                img = _crop_query_roi(img, bbx)
        return self.transform(img)


def _resolve_teacher_weights_img_size(teacher_name: str, eval_image_size: int) -> int:
    """ViT-G/14 weights use a 518x518 positional grid; smaller teachers match eval crop."""
    key = teacher_name.split("/")[-1].lower().replace("-", "_")
    if "vitg14" in key or "giant" in key:
        return 518
    return eval_image_size


def _build_teacher(teacher_name: str, weights_img_size: int, device: torch.device,
                   logger: logging.Logger):
    """Load + freeze the DINOv2 teacher via LightlyTrain's cached weights."""
    from lightly_train._models import package_helpers
    logger.info("Building teacher: %s (architecture img_size=%s for weight load)",
                teacher_name, weights_img_size)
    wrapped = package_helpers.get_wrapped_model(
        model=teacher_name,
        num_input_channels=3,
        model_args={"img_size": weights_img_size},
    )
    model = wrapped.get_model().to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "chunked_blocks") and model.chunked_blocks:
        import torch.nn as nn
        flat = []
        for chunk in model.blocks:
            for blk in chunk:
                if not isinstance(blk, nn.Identity):
                    flat.append(blk)
        model.blocks = nn.ModuleList(flat)
        model.chunked_blocks = False
    n_blocks = len(model.blocks)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  Loaded teacher: %d blocks, embed_dim=%s, params=%s (frozen)",
                n_blocks, getattr(model, "embed_dim", "?"), f"{n_params:,}")
    return model


@torch.no_grad()
def _extract_matrices_teacher_multiblock(
    teacher,
    loader: DataLoader,
    device: torch.device,
    feature_type: str,
    l2_normalize: bool,
    use_amp: bool,
    block_indices,
    desc: str,
):
    """Single teacher pass per batch; collects (B,D) features at each requested block.

    Returns ``{block_idx: torch.Tensor[N, D]}`` (CPU, float32).
    """
    n_skip = 1 + int(getattr(teacher, "num_register_tokens", 0))
    deepest = max(block_indices)
    target_set = set(block_indices)
    feats_buf = {b: [] for b in block_indices}
    use_cuda_amp = device.type == "cuda" and use_amp
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device, non_blocking=True)
        ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_cuda_amp else nullcontext()
        with ctx:
            x = teacher.prepare_tokens_with_masks(batch)
            for i, blk in enumerate(teacher.blocks):
                x = blk(x)
                if i in target_set:
                    x_n = teacher.norm(x)
                    if feature_type == "cls":
                        feats = x_n[:, 0]
                    else:
                        feats = x_n[:, n_skip:, :].mean(dim=1)
                    if l2_normalize:
                        feats = F.normalize(feats, dim=-1)
                    feats_buf[i].append(feats.float().cpu())
                if i == deepest:
                    break
    return {b: torch.cat(feats_buf[b], 0) for b in block_indices}


@torch.no_grad()
def _extract_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_type: str,
    l2_normalize: bool,
    use_amp: bool,
    desc: str,
) -> torch.Tensor:
    feats_list = []
    use_cuda_amp = device.type == "cuda" and use_amp
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device, non_blocking=True)
        if use_cuda_amp:
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            ctx = nullcontext()
        with ctx:
            out = model.forward_features(batch)
            if feature_type == "cls":
                if hasattr(out, "shape") and out.dim() == 3:
                    feats = out[:, 0]
                else:
                    feats = out
            else:
                if hasattr(out, "shape") and out.dim() == 3:
                    feats = out[:, 1:].mean(dim=1)
                else:
                    feats = out
            if l2_normalize:
                feats = F.normalize(feats, dim=-1)
        feats_list.append(feats.float().cpu())
    return torch.cat(feats_list, dim=0)


def _eval_protocols(ranks: np.ndarray, gnd_raw: list, ks: list):
    """ranks: (n_db, n_q) — same convention as revisitop example_evaluate."""

    def run(gnd_t):
        return compute_map(ranks, gnd_t, ks)

    gnd_t = []
    for i in range(len(gnd_raw)):
        gi = gnd_raw[i]
        g = {
            "ok": np.asarray(gi["easy"], dtype=np.int64).ravel(),
            "junk": np.concatenate(
                [np.asarray(gi["junk"], dtype=np.int64).ravel(), np.asarray(gi["hard"], dtype=np.int64).ravel()]
            ),
        }
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = run(gnd_t)

    gnd_t = []
    for i in range(len(gnd_raw)):
        gi = gnd_raw[i]
        g = {
            "ok": np.concatenate(
                [np.asarray(gi["easy"], dtype=np.int64).ravel(), np.asarray(gi["hard"], dtype=np.int64).ravel()]
            ),
            "junk": np.asarray(gi["junk"], dtype=np.int64).ravel(),
        }
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = run(gnd_t)

    gnd_t = []
    for i in range(len(gnd_raw)):
        gi = gnd_raw[i]
        g = {
            "ok": np.asarray(gi["hard"], dtype=np.int64).ravel(),
            "junk": np.concatenate(
                [np.asarray(gi["junk"], dtype=np.int64).ravel(), np.asarray(gi["easy"], dtype=np.int64).ravel()]
            ),
        }
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = run(gnd_t)

    return {
        "mAP_E": float(mapE),
        "mAP_M": float(mapM),
        "mAP_H": float(mapH),
        "mP_at_k_E": mprE.tolist(),
        "mP_at_k_M": mprM.tolist(),
        "mP_at_k_H": mprH.tolist(),
        "ks": ks,
    }


def main():
    args = _parse_args()
    if not args.teacher and not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --teacher is set.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.output_dir:
        out_dir = args.output_dir
    elif args.checkpoint:
        out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    else:
        out_dir = os.getcwd()
    logger = _setup_logger(out_dir)
    try:
        logger.info("Device: %s", device)
        if args.teacher:
            logger.info("Teacher mode: %s (blocks=%s)",
                        args.teacher_model, args.teacher_blocks)
        else:
            logger.info("Checkpoint: %s", args.checkpoint)
        _flush_log(logger)

        if args.from_official:
            prepare_revisitop_official(
                args.revisit_root,
                args.test_dataset,
                logger,
                overwrite=args.overwrite_hf_cache,
            )
            _flush_log(logger)
        elif args.from_hf:
            try:
                prepare_revisitop_from_hf(
                    args.revisit_root,
                    args.test_dataset,
                    args.hf_dataset_id,
                    logger,
                    overwrite=args.overwrite_hf_cache,
                )
            except Exception as hf_exc:
                logger.warning(
                    "Hugging Face dataset prepare failed (%s). "
                    "This often happens with ``datasets>=3`` (script datasets disabled). "
                    "Falling back to official Oxford/VGG + revisitop downloads.",
                    hf_exc,
                )
                prepare_revisitop_official(
                    args.revisit_root,
                    args.test_dataset,
                    logger,
                    overwrite=args.overwrite_hf_cache,
                )
            _flush_log(logger)

        datasets_root = os.path.join(args.revisit_root, "datasets")
        gnd_path = os.path.join(datasets_root, args.test_dataset, f"gnd_{args.test_dataset}.pkl")
        jpg_dir = os.path.join(datasets_root, args.test_dataset, "jpg")
        if not os.path.isfile(gnd_path):
            logger.error(
                "Missing %s — run once with --from-official (recommended) or --from-hf (needs datasets<3).",
                gnd_path,
            )
            sys.exit(1)
        if not os.path.isdir(jpg_dir):
            logger.error("Missing image dir %s", jpg_dir)
            sys.exit(1)

        cfg = configdataset(args.test_dataset, datasets_root)
        imlist = cfg["imlist"]
        qimlist = cfg["qimlist"]
        gnd_raw = cfg["gnd"]

        n_q = len(qimlist)
        n_bbx = sum(
            1
            for g in gnd_raw
            if g.get("bbx") is not None and np.asarray(g["bbx"]).ravel().size >= 4
        )
        if n_bbx < n_q:
            logger.warning(
                "Only %d/%d queries have non-empty 'bbx'; missing queries use full frame (not standard revisitop).",
                n_bbx,
                n_q,
            )

        transform = get_eval_transforms(args.image_size)
        pin = device.type == "cuda"

        db_loader = DataLoader(
            _OxfordParisJpgDataset(imlist, jpg_dir, transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
            drop_last=False,
        )
        q_loader = DataLoader(
            _OxfordParisQueryDataset(qimlist, gnd_raw, jpg_dir, transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
            drop_last=False,
        )

        use_amp = args.amp and device.type == "cuda"
        ks = [1, 5, 10]

        if args.teacher:
            weights_img_size = (
                args.teacher_weights_img_size
                or _resolve_teacher_weights_img_size(args.teacher_model, args.image_size)
            )
            teacher = _build_teacher(args.teacher_model, weights_img_size, device, logger)
            _flush_log(logger)
            n_blocks = len(teacher.blocks)
            block_indices = sorted(args.teacher_blocks) if args.teacher_blocks else [n_blocks - 1]
            for b in block_indices:
                if not (0 <= b < n_blocks):
                    raise SystemExit(
                        f"--teacher-blocks contains out-of-range index {b} "
                        f"(teacher has {n_blocks} blocks: valid 0..{n_blocks-1})."
                    )
            logger.info("Extracting teacher features at blocks: %s", block_indices)
            t0 = time.time()
            db_dict = _extract_matrices_teacher_multiblock(
                teacher, db_loader, device, args.feature_type,
                args.l2_normalize, use_amp, block_indices, "database features",
            )
            q_dict = _extract_matrices_teacher_multiblock(
                teacher, q_loader, device, args.feature_type,
                args.l2_normalize, use_amp, block_indices, "query features",
            )
            logger.info("Teacher extraction done (%.1fs)", time.time() - t0)

            teacher_tag = args.teacher_model.replace("/", "_")
            for b in block_indices:
                db_feats = db_dict[b]
                q_feats = q_dict[b]
                X = db_feats.t().numpy()
                Q = q_feats.t().numpy()
                sim = np.dot(X.T, Q)
                ranks = np.argsort(-sim, axis=0)
                metrics = _eval_protocols(ranks, gnd_raw, ks)
                logger.info(
                    ">> %s [%s block %d/%d feat=%s]: mAP E: %.2f, M: %.2f, H: %.2f",
                    args.test_dataset, teacher_tag, b, n_blocks - 1,
                    args.feature_type,
                    metrics["mAP_E"] * 100, metrics["mAP_M"] * 100, metrics["mAP_H"] * 100,
                )
                out_path = os.path.join(
                    out_dir,
                    f"instance_retrieval_{args.test_dataset}_teacher_{teacher_tag}_block{b:02d}_{args.feature_type}.json",
                )
                payload = {
                    "teacher_model": args.teacher_model,
                    "teacher_block": b,
                    "teacher_n_blocks": n_blocks,
                    "test_dataset": args.test_dataset,
                    "revisit_root": args.revisit_root,
                    "from_official": args.from_official,
                    "from_hf": args.from_hf,
                    "hf_dataset_id": args.hf_dataset_id if args.from_hf else None,
                    "image_size": args.image_size,
                    "feature_type": args.feature_type,
                    "l2_normalize": args.l2_normalize,
                    "queries_with_bbox": n_bbx,
                    "num_queries": n_q,
                    **metrics,
                }
                with open(out_path, "w") as f:
                    json.dump(payload, f, indent=2)
                logger.info("Wrote %s", out_path)
            return

        model = load_model(
            args.checkpoint,
            args.model_name,
            args.image_size,
            device,
            logger,
            patch_size=args.patch_size,
            model_img_size=args.model_img_size,
            adapt_patch_size=args.adapt_patch_size,
            student_size=args.student_size,
        )
        _flush_log(logger)

        t0 = time.time()
        db_feats = _extract_matrix(
            model, db_loader, device, args.feature_type, args.l2_normalize, use_amp, "database features",
        )
        q_feats = _extract_matrix(
            model, q_loader, device, args.feature_type, args.l2_normalize, use_amp, "query features",
        )
        logger.info(
            "Features db=%s query=%s (%.1fs)",
            tuple(db_feats.shape),
            tuple(q_feats.shape),
            time.time() - t0,
        )

        X = db_feats.t().numpy()
        Q = q_feats.t().numpy()
        sim = np.dot(X.T, Q)
        ranks = np.argsort(-sim, axis=0)

        metrics = _eval_protocols(ranks, gnd_raw, ks)

        logger.info(
            ">> %s: mAP E: %.2f, M: %.2f, H: %.2f",
            args.test_dataset,
            metrics["mAP_E"] * 100,
            metrics["mAP_M"] * 100,
            metrics["mAP_H"] * 100,
        )
        logger.info(
            ">> %s: mP@k%s E: %s, M: %s, H: %s",
            args.test_dataset,
            np.array(ks),
            np.around(np.array(metrics["mP_at_k_E"]) * 100, decimals=2),
            np.around(np.array(metrics["mP_at_k_M"]) * 100, decimals=2),
            np.around(np.array(metrics["mP_at_k_H"]) * 100, decimals=2),
        )

        out_path = os.path.join(
            out_dir,
            f"instance_retrieval_{args.test_dataset}_{os.path.basename(args.checkpoint).replace('.pt', '')}.json",
        )
        payload = {
            "checkpoint": args.checkpoint,
            "test_dataset": args.test_dataset,
            "revisit_root": args.revisit_root,
            "from_official": args.from_official,
            "from_hf": args.from_hf,
            "hf_dataset_id": args.hf_dataset_id if args.from_hf else None,
            "image_size": args.image_size,
            "feature_type": args.feature_type,
            "l2_normalize": args.l2_normalize,
            "queries_with_bbox": n_bbx,
            "num_queries": n_q,
            **metrics,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote %s", out_path)
    except Exception:
        logger.error("instance_retrieval failed:\n%s", traceback.format_exc())
        _flush_log(logger)
        raise


if __name__ == "__main__":
    main()