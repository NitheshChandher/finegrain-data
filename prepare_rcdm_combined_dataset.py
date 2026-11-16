#!/usr/bin/env python3
"""
Prepare a leakage-audited fine-grained pre-training dataset from:
  * Bird-11K[-CUB] (locally reconstructed; class-folder layout)
  * CompCars web-nature (manually downloaded; make/model/year hierarchy)
  * Open Images V7 aircraft/airplane detections (optional FiftyOne download)
  * Military Aircraft Detection crops (locally downloaded; class-folder layout)

Creates a flat combined image dataset (no class folders, no class names in
file names):
   OUT/train/<unique_image_name>.jpg
   OUT/val/<unique_image_name>.jpg
   OUT/test/<unique_image_name>.jpg
   OUT/manifest.csv
   OUT/summary.json

plus, for each requested encoder, a flat image + representation dataset:
   OUT_DINOv2/train/<unique_image_name>.jpg
   OUT_DINOv2/DINOv2/train/<unique_image_name>.npy
   (and similarly for DINOv3 / CLIP)

Images are real files: by default they are copied (--link-mode copy). Symlinks
are never created. Representations are saved as float32 NumPy arrays.

IMPORTANT:
- Bird-11K and CompCars must be obtained according to their terms.
- This script does not redistribute or bypass dataset access restrictions.
- Pass all three downstream dataset roots to --exclude-root. The script uses
  exact SHA-256 and optional perceptual-hash exclusion.
- This is image-level leakage prevention. It does not exclude all bird species,
  aircraft models, or car models that may overlap semantically with downstream
  benchmarks.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class Record:
    path: Path
    domain: str
    class_name: str
    source: str
    source_id: str


def safe_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "unknown"


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def image_files(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    for p in root.rglob("*"):
        if is_image(p):
            yield p


def sha256_file(path: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_exclusion_hashes(roots: List[Path], use_phash: bool):
    """Build exact hashes and (optionally) perceptual hashes of downstream data."""
    exact = set()
    perceptual = set()

    if use_phash:
        try:
            import imagehash
        except ImportError as exc:
            raise RuntimeError(
                "Install ImageHash for perceptual exclusion: pip install ImageHash"
            ) from exc

    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Exclusion root not found: {root}")
        files = list(image_files(root))
        print(f"[leakage] Hashing {len(files):,} images in {root}")
        for p in files:
            try:
                exact.add(sha256_file(p))
                if use_phash:
                    with Image.open(p) as im:
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        perceptual.add(str(imagehash.phash(im)))
            except Exception as e:
                print(f"[warn] Could not hash exclusion image {p}: {e}")
    return exact, perceptual


def is_excluded(path: Path, exact_hashes: set, perceptual_hashes: set,
                use_phash: bool, phash_threshold: int) -> Tuple[bool, str]:
    try:
        digest = sha256_file(path)
        if digest in exact_hashes:
            return True, "sha256"
        if use_phash:
            import imagehash
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                ph = imagehash.phash(im)
            # Conservative: only exclude if an existing downstream pHash is
            # within the configured Hamming distance.
            for existing in perceptual_hashes:
                if ph - imagehash.hex_to_hash(existing) <= phash_threshold:
                    return True, "phash"
    except Exception as e:
        return True, f"unreadable:{e}"
    return False, ""


def discover_class_folder_dataset(root: Path, domain: str, source: str,
                                  class_depth: int = 1) -> List[Record]:
    """
    Read images from class-folder datasets. The class is inferred from the
    directory `class_depth` levels below root (1 = immediate parent folder).
    Bird-11K: root should point to its CUB-excluded image split/root.
    Military aircraft: root should point to the pre-cropped class folders.
    """
    records = []
    for p in image_files(root):
        rel = p.relative_to(root)
        parents = rel.parts[:-1]
        if not parents:
            continue
        idx = min(max(class_depth, 1), len(parents))
        cls = safe_name(parents[idx - 1])
        records.append(Record(p, domain, cls, source, str(rel)))
    return records


def discover_compcars(root: Path, class_depth: int = 2) -> List[Record]:
    """
    CompCars web-nature typically has image/<make>/<model>/<year>/<file>.
    Point --compcars-root at the `image` directory. By default the class is
    make_model (first two path components), not year.
    """
    records = []
    for p in image_files(root):
        rel = p.relative_to(root)
        parts = rel.parts[:-1]
        if not parts:
            continue
        use = parts[:max(1, class_depth)]
        cls = safe_name("_".join(use))
        records.append(Record(p, "cars", cls, "compcars_web", str(rel)))
    return records


def download_open_images_aircraft(cache_dir: Path, max_images: int,
                                  seed: int) -> List[Record]:
    """
    Download aircraft/airplane detections from Open Images V7 using FiftyOne.
    Each detection is cropped from its source image and written to cache_dir.
    The image-level sample is deduplicated before returning.
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError(
            "Open Images download requires FiftyOne. Install with: pip install fiftyone"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[OpenImages] Loading Open Images V7 aircraft/airplane detections.")
    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Aircraft", "Airplane"],
        max_samples=max_images,
        shuffle=True,
        seed=seed,
        dataset_name=f"rcdm_open_images_aircraft_{seed}",
    )

    out = []
    seen = set()
    for sample in ds:
        try:
            source_path = Path(sample.filepath)
            dets = sample.detections.detections if sample.detections else []
            with Image.open(source_path) as im0:
                im = ImageOps.exif_transpose(im0).convert("RGB")
                w, h = im.size
                for j, det in enumerate(dets):
                    if det.label.lower() not in {"aircraft", "airplane"}:
                        continue
                    x, y, bw, bh = det.bounding_box
                    # Skip extremely small boxes and malformed boxes.
                    if bw <= 0 or bh <= 0 or bw * w < 32 or bh * h < 32:
                        continue
                    left = max(0, int(x * w))
                    top = max(0, int(y * h))
                    right = min(w, int((x + bw) * w))
                    bottom = min(h, int((y + bh) * h))
                    if right <= left or bottom <= top:
                        continue
                    crop = im.crop((left, top, right, bottom))
                    key = (sample.id, j)
                    if key in seen:
                        continue
                    seen.add(key)
                    name = f"oid_aircraft_{safe_name(sample.id)}_{j:03d}.jpg"
                    dst = cache_dir / name
                    if not dst.exists():
                        crop.save(dst, quality=95)
                    out.append(Record(
                        dst, "aircraft", "aircraft", "open_images_v7",
                        f"{sample.id}:{j}:{det.label}"
                    ))
        except Exception as e:
            print(f"[warn] Open Images sample skipped ({sample.id}): {e}")

    print(f"[OpenImages] Extracted {len(out):,} aircraft crops.")
    return out


def balanced_sample(records: List[Record], target: int,
                    rng: random.Random) -> List[Record]:
    """Sample as evenly as possible across classes; fail if target unavailable."""
    by_class = defaultdict(list)
    for r in records:
        by_class[r.class_name].append(r)

    available = sum(len(v) for v in by_class.values())
    if available < target:
        counts = sorted(
            ((k, len(v)) for k, v in by_class.items()),
            key=lambda x: x[1], reverse=True
        )
        raise RuntimeError(
            f"Insufficient images for {records[0].domain if records else 'domain'}: "
            f"requested {target:,}, found {available:,}. "
            f"Top classes: {counts[:10]}"
        )

    classes = sorted(by_class)
    for values in by_class.values():
        rng.shuffle(values)

    chosen = []
    # Round-robin sampling prevents a few large classes dominating.
    cursor = {c: 0 for c in classes}
    while len(chosen) < target:
        progressed = False
        for c in classes:
            i = cursor[c]
            if i < len(by_class[c]):
                chosen.append(by_class[c][i])
                cursor[c] += 1
                progressed = True
                if len(chosen) == target:
                    break
        if not progressed:
            break
    return chosen


def stratified_split(records: List[Record], val_frac: float,
                     test_frac: float, rng: random.Random):
    """Split each class independently; at least one train image per class."""
    grouped = defaultdict(list)
    for r in records:
        grouped[r.class_name].append(r)

    split_records = {s: [] for s in SPLIT_NAMES}
    for cls, items in grouped.items():
        rng.shuffle(items)
        n = len(items)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        # Keep at least one train image per class.
        while n_val + n_test >= n and (n_val > 0 or n_test > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        split_records["val"].extend(items[:n_val])
        split_records["test"].extend(items[n_val:n_val + n_test])
        split_records["train"].extend(items[n_val + n_test:])

    for s in SPLIT_NAMES:
        rng.shuffle(split_records[s])
    return split_records


def unique_filename(record: Record, index: int) -> str:
    # Class-free, collision-free name: coarse domain + running index + short
    # hash of (source, source_id). The original file stem is deliberately not
    # used because it often contains the class name. The full mapping back to
    # source/class lives in manifest.csv.
    token = hashlib.sha1(
        f"{record.source}|{record.source_id}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{record.domain}_{index:06d}_{token}{record.path.suffix.lower()}"


def link_or_copy(src: Path, dst: Path, mode: str):
    """Place an image at dst. Never creates symlinks.

    mode="copy" (default): real file copy.
    mode="hardlink": hard link (still a real directory entry, but shares data
    with the source); falls back to a copy if linking is not possible.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -------------------------- Representation models --------------------------

def load_model(name: str, args, device):
    import torch
    if name == "DINOv2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model.eval().to(device)
        return model, None

    if name == "DINOv3":
        if not args.dinov3_repo or not args.dinov3_weights:
            raise ValueError(
                "DINOv3 requires --dinov3-repo and --dinov3-weights"
            )
        model = torch.hub.load(
            args.dinov3_repo,
            "dinov3_vitb16",
            source="local",
            weights=args.dinov3_weights,
        )
        model.eval().to(device)
        return model, None

    if name == "CLIP":
        from transformers import CLIPModel, CLIPProcessor
        model_id = args.clip_model
        model = CLIPModel.from_pretrained(model_id).eval().to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        return model, processor

    raise ValueError(f"Unsupported representation encoder: {name}")


class FlatImageDataset:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        with Image.open(r.path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")


def extract_split_representations(records: List[dict], image_root: Path,
                                  rep_root: Path, model_name: str,
                                  args, device):
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    class RepDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            p = image_root / row["image_relpath"]
            try:
                with Image.open(p) as im:
                    image = ImageOps.exif_transpose(im).convert("RGB")
                return self.transform(image), idx
            except Exception as e:
                raise RuntimeError(f"Cannot read {p}: {e}") from e

    model, processor = load_model(model_name, args, device)
    ds = RepDataset(records)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    out_dir = rep_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.inference_mode():
        for batch, indices in loader:
            batch = batch.to(device, non_blocking=True)
            if model_name == "CLIP":
                # Use the same normalized 224x224 tensor path as DINO models,
                # then feed pixels to CLIP's vision model. CLIP expects its own
                # normalization; instead use the processor on original PILs.
                pil_images = []
                for idx in indices.tolist():
                    row = records[idx]
                    with Image.open(image_root / row["image_relpath"]) as im:
                        pil_images.append(ImageOps.exif_transpose(im).convert("RGB"))
                inputs = processor(images=pil_images, return_tensors="pt").to(device)
                outputs = model.vision_model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]
                features = torch.nn.functional.normalize(features, dim=1)
            else:
                features = model(batch)
                if isinstance(features, dict):
                    features = features.get("x_norm_clstoken", features.get("pooler_output"))
                if isinstance(features, (tuple, list)):
                    features = features[0]
                if features.ndim > 2:
                    features = features[:, 0, :]
            features = features.detach().float().cpu().numpy()

            for feat, idx in zip(features, indices.tolist()):
                row = records[idx]
                np.save(out_dir / row["rep_relpath"], feat.astype(np.float32))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--bird-root", type=Path, required=True,
                   help="Bird-11K[-CUB] root, arranged as class folders")
    p.add_argument("--compcars-root", type=Path, required=True,
                   help="CompCars web-nature `image` directory")
    p.add_argument("--military-root", type=Path, required=True,
                   help="Military aircraft pre-cropped class-folder directory")
    p.add_argument("--out", type=Path, required=True,
                   help="Output dataset root (flat train/val/test), e.g. ./dataset")
    p.add_argument("--oid-cache", type=Path, default=None,
                   help="Where to cache Open Images aircraft crops")
    p.add_argument("--download-oid", action="store_true",
                   help="Download Open Images V7 aircraft/airplane crops using FiftyOne")
    p.add_argument("--oid-max-images", type=int, default=50000,
                   help="Maximum Open Images source images to inspect")
    p.add_argument("--exclude-root", type=Path, action="append", default=[],
                   help="Downstream dataset root to exclude; repeat for Aircraft/CUB/Cars")
    p.add_argument("--phash-exclusion", action="store_true",
                   help="Also exclude perceptual-hash near duplicates")
    p.add_argument("--phash-threshold", type=int, default=4)
    p.add_argument("--birds", type=int, default=100000)
    p.add_argument("--cars", type=int, default=100000)
    p.add_argument("--aircraft", type=int, default=10000)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compcars-class-depth", type=int, default=2,
                   help="Path components used for CompCars class: make/model=2")
    p.add_argument("--bird-class-depth", type=int, default=1)
    p.add_argument("--military-class-depth", type=int, default=1)
    p.add_argument("--link-mode", choices=["copy", "hardlink"],
                   default="copy",
                   help="How images are placed in the output. 'copy' makes real "
                        "files (default). 'hardlink' saves disk space; symlinks "
                        "are never created.")
    p.add_argument("--representations", nargs="+",
                   choices=["DINOv2", "DINOv3", "CLIP"],
                   default=["DINOv2"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dinov3-repo", type=str, default=None)
    p.add_argument("--dinov3-weights", type=str, default=None)
    p.add_argument("--clip-model", type=str,
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--skip-representations", action="store_true")
    args = p.parse_args()

    if not (0 <= args.val_frac < 1 and 0 <= args.test_frac < 1
            and args.val_frac + args.test_frac < 1):
        p.error("val/test fractions must be nonnegative and sum to < 1")

    rng = random.Random(args.seed)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.out}. "
            "Choose a new --out path or move the existing output."
        )
    args.out.mkdir(parents=True, exist_ok=True)

    print("[1/7] Discovering source datasets...")
    candidates = []
    candidates += discover_class_folder_dataset(
        args.bird_root, "birds", "bird11k_cub_excluded",
        args.bird_class_depth
    )
    candidates += discover_compcars(
        args.compcars_root, args.compcars_class_depth
    )
    candidates += discover_class_folder_dataset(
        args.military_root, "aircraft", "military_aircraft",
        args.military_class_depth
    )

    if args.download_oid:
        oid_cache = args.oid_cache or (args.out.parent / "oid_aircraft_cache")
        candidates += download_open_images_aircraft(
            oid_cache, args.oid_max_images, args.seed
        )
    # Ensure all three target benchmark roots are explicitly supplied.
    if len(args.exclude_root) < 3:
        raise ValueError(
            "Pass --exclude-root three times: FGVC-Aircraft, CUB-200-2011, "
            "and Stanford Cars (include all splits)."
        )

    print("[2/7] Building downstream exclusion hashes...")
    exact_hashes, phashes = load_exclusion_hashes(
        args.exclude_root, args.phash_exclusion
    )

    print("[3/7] Removing corrupt images, duplicates and downstream leakage...")
    filtered = []
    seen_hashes = set()
    excluded_counts = defaultdict(int)
    for r in candidates:
        try:
            with Image.open(r.path) as im:
                im.verify()
            digest = sha256_file(r.path)
            if digest in seen_hashes:
                excluded_counts["source_duplicate"] += 1
                continue
            seen_hashes.add(digest)
            excluded, reason = is_excluded(
                r.path, exact_hashes, phashes,
                args.phash_exclusion, args.phash_threshold
            )
            if excluded:
                excluded_counts[reason] += 1
                continue
            filtered.append(r)
        except Exception:
            excluded_counts["corrupt"] += 1

    print(f"Candidates: {len(candidates):,}; retained: {len(filtered):,}")
    print("Excluded:", dict(excluded_counts))

    targets = {
        "birds": args.birds,
        "cars": args.cars,
        "aircraft": args.aircraft,
    }

    print("[4/7] Balanced sampling and stratified splitting...")
    all_splits = {s: [] for s in SPLIT_NAMES}
    for domain, target in targets.items():
        domain_records = [r for r in filtered if r.domain == domain]
        if not domain_records:
            raise RuntimeError(f"No usable records found for domain: {domain}")
        sampled = balanced_sample(domain_records, target, rng)
        splits = stratified_split(
            sampled, args.val_frac, args.test_frac, rng
        )
        for split in SPLIT_NAMES:
            all_splits[split].extend(splits[split])
        print(
            f"{domain}: total={len(sampled):,}, "
            + ", ".join(f"{s}={len(splits[s]):,}" for s in SPLIT_NAMES)
        )

    # Create collision-resistant, class-free file names directly in OUT/<split>/.
    print("[5/7] Writing images and manifest...")
    manifest_rows = []
    rows_by_split = {s: [] for s in SPLIT_NAMES}
    global_index = 0

    for split in SPLIT_NAMES:
        (args.out / split).mkdir(parents=True, exist_ok=True)
        for r in all_splits[split]:
            global_index += 1
            filename = unique_filename(r, global_index)
            image_rel = Path(split) / filename

            link_or_copy(r.path, args.out / image_rel, args.link_mode)

            row = {
                "split": split,
                "domain": r.domain,
                "class_name": r.class_name,
                "source": r.source,
                "source_id": r.source_id,
                "original_path": str(r.path.resolve()),
                "image_relpath": str(image_rel),
                "rep_relpath": str(Path(split) / (Path(filename).stem + ".npy")),
            }
            manifest_rows.append(row)
            rows_by_split[split].append(row)

    write_csv(
        args.out / "manifest.csv",
        manifest_rows,
        ["split", "domain", "class_name", "source", "source_id",
         "original_path", "image_relpath", "rep_relpath"],
    )

    print("[6/7] Creating image datasets for each representation...")
    rep_roots = {}
    for rep in args.representations:
        rep_root = args.out.parent / f"{args.out.name}_{rep}"
        rep_roots[rep] = rep_root
        for split in SPLIT_NAMES:
            for row in rows_by_split[split]:
                src = args.out / row["image_relpath"]
                dst = rep_root / row["image_relpath"]
                link_or_copy(src, dst, args.link_mode)

        # Copy manifest and metadata for traceability.
        shutil.copy2(args.out / "manifest.csv", rep_root / "manifest.csv")

    if not args.skip_representations:
        print("[7/7] Extracting representations...")
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for rep, rep_root in rep_roots.items():
            print(f"\n[representations] {rep} on {device}")
            for split in SPLIT_NAMES:
                extract_split_representations(
                    rows_by_split[split],
                    rep_root,
                    rep_root / rep,
                    rep,
                    args,
                    device,
                )

    # Summary report.
    summary = defaultdict(lambda: defaultdict(int))
    for row in manifest_rows:
        summary[row["domain"]][row["split"]] += 1

    with (args.out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "targets": targets,
            "actual": {d: dict(v) for d, v in summary.items()},
            "representations": args.representations,
            "seed": args.seed,
            "exclusion_roots": [str(x) for x in args.exclude_root],
            "excluded_counts": dict(excluded_counts),
        }, f, indent=2)

    print("\nFinished.")
    print(f"Images: {args.out.resolve()}")
    for rep, root in rep_roots.items():
        print(f"{rep} images + features: {root.resolve()}")


if __name__ == "__main__":
    main()
