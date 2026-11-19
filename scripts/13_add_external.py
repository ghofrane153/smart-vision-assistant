"""Step 9b - Add an external dataset (Open Images, Roboflow, ...) to the TRAINING set only.

- Accepts a folder or a .zip in YOLO format (with a data.yaml): the output of 12_download_openimages.py,
  or a Roboflow export ("YOLOv11" / "YOLOv8" format: train/, valid/, test/ folders).
- Renames the classes to ours (automatic for same names and common synonyms, or with --map).
  Boxes of other classes are ignored. Polygon labels (segmentation) are converted to boxes.
- Images go ONLY to data/dataset/images/train (validation and test stay 100% our photos),
  named ext_<name>_<file>, resized to max 1280 px like our photos.
- Auto-fill (default ON): the pretrained YOLO11 adds person/chair/table/laptop/pc boxes that the
  external dataset did not annotate (same problem as our audit), so they are not learned as background.
- Can be undone: --remove <name> deletes everything that was added under that name.

Run from the repository root:
    Windows : .venv\\Scripts\\python.exe scripts\\13_add_external.py data\\external\\openimages --name openimages
              .venv\\Scripts\\python.exe scripts\\13_add_external.py roboflow_cables.zip --name roboflow --map "Loose Wires=cable"
              .venv\\Scripts\\python.exe scripts\\13_add_external.py --remove roboflow
!! Running 06_build_dataset.py again rebuilds the dataset WITHOUT the external images: add them again after.
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SYNONYMS = {"human": "person", "people": "person", "pedestrian": "person", "man": "person", "woman": "person",
            "wire": "cable", "wires": "cable", "cables": "cable", "loose_wires": "cable", "loose_wire": "cable",
            "cord": "cable", "stair": "stairs", "staircase": "stairs", "steps": "stairs", "doors": "door",
            "monitor": "pc", "computer_monitor": "pc", "computer": "pc", "desk": "table", "tables": "table",
            "chairs": "chair", "notebook": "laptop", "bag": "handbag", "bottles": "bottle"}
COCO_TO_OURS = {"person": "person", "chair": "chair", "dining table": "table", "laptop": "laptop", "tv": "pc"}


def clean(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse_line(parts: list[str]):
    """YOLO box 'c x y w h' or polygon 'c x1 y1 x2 y2 ...' -> (c, x, y, w, h)."""
    c = int(float(parts[0]))
    v = [float(p) for p in parts[1:]]
    if len(v) == 4:
        return c, *v
    if len(v) >= 6 and len(v) % 2 == 0:
        xs, ys = v[0::2], v[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return None


def remove(ds: Path, name: str) -> int:
    manifest = ds / f"external_{name}.txt"
    if not manifest.exists():
        return 0
    n = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        for p in (ds / "images" / "train" / line, (ds / "labels" / "train" / line).with_suffix(".txt")):
            if p.exists():
                p.unlink()
                n += 1
    manifest.unlink()
    return n // 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Add an external YOLO dataset to the training set")
    parser.add_argument("source", nargs="?", help="folder or .zip in YOLO format")
    parser.add_argument("--name", help="short name, e.g. openimages, roboflow")
    parser.add_argument("--map", nargs="*", default=[], help='"Their Name=our_class", e.g. "Loose Wires=cable"')
    parser.add_argument("--max-images", type=int, default=0, help="limit the number of images (0 = all)")
    parser.add_argument("--keep-empty", action="store_true", help="also keep images without any of our classes")
    parser.add_argument("--no-autofill", action="store_true")
    parser.add_argument("--autofill-model", default="yolo11s.pt")
    parser.add_argument("--max-size", type=int, default=1280)
    parser.add_argument("--dataset", default="data/dataset")
    parser.add_argument("--remove", metavar="NAME", help="remove the images added under NAME")
    args = parser.parse_args()

    ds = Path(args.dataset)
    for cache in (ds / "labels").glob("*.cache"):
        cache.unlink()
    if args.remove:
        print(f"Removed {remove(ds, args.remove)} images added as '{args.remove}'.")
        return 0
    if not args.source or not args.name:
        parser.error("give a source and --name (or --remove NAME)")
    classes = (ds / "classes.txt").read_text(encoding="utf-8").split()

    src = Path(args.source)
    tmp = None
    if src.suffix.lower() == ".zip":
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = Path(tmp)
    try:
        yamls = sorted(src.rglob("data.yaml")) + sorted(src.rglob("dataset.yaml"))
        if not yamls:
            print("No data.yaml found: is it a YOLO-format export?")
            return 1
        y = yaml.safe_load(yamls[0].read_text(encoding="utf-8"))
        names = y["names"]
        names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

        user_map = {}
        for m in args.map:
            if "=" not in m:
                parser.error(f'--map needs "Their Name=our_class", got {m!r}')
            a, b = m.split("=", 1)
            user_map[clean(a)] = clean(b)
        mapping = {}
        for i, n in enumerate(names):
            target = user_map.get(clean(n), clean(n))
            target = SYNONYMS.get(target, target)
            if target in classes:
                mapping[i] = classes.index(target)
        print("Class mapping:")
        for i, n in enumerate(names):
            print(f"  {n!s:25s} -> {classes[mapping[i]] if i in mapping else '(ignored)'}")
        if not mapping:
            print("No class matches ours. Use --map, e.g. --map \"Loose Wires=cable\"")
            return 1

        images = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT}
        pairs = []
        for lf in sorted(src.rglob("*.txt")):
            if lf.parent.name != "labels" and "labels" not in lf.parts:
                continue
            if lf.stem in images:
                pairs.append((images[lf.stem], lf))
        print(f"{len(pairs)} labeled images found")

        removed = remove(ds, args.name)
        if removed:
            print(f"(previous '{args.name}' images removed first: {removed})")

        model = None
        if not args.no_autofill:
            from ultralytics import YOLO
            model = YOLO(args.autofill_model)

        out_img, out_lab = ds / "images" / "train", ds / "labels" / "train"
        added_files, counts, filled, skipped = [], Counter(), Counter(), 0
        for img_path, lab_path in pairs:
            boxes = []
            for line in lab_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                parsed = parse_line(parts)
                if parsed and parsed[0] in mapping and parsed[3] > 0 and parsed[4] > 0:
                    c, x, yy, w, h = parsed
                    boxes.append((mapping[c], x, yy, w, h))
            if not boxes and not args.keep_empty:
                skipped += 1
                continue
            if args.max_images and len(added_files) >= args.max_images:
                break

            fname = f"ext_{args.name}_{re.sub(r'[^A-Za-z0-9_.-]', '_', img_path.stem)}.jpg"
            with Image.open(img_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((args.max_size, args.max_size), Image.Resampling.LANCZOS)
                im.save(out_img / fname, "JPEG", quality=90)
                W, H = im.size

            if model is not None:
                r = model.predict(str(out_img / fname), conf=0.5, verbose=False)[0]
                for b in r.boxes:
                    ours = COCO_TO_OURS.get(model.names[int(b.cls)])
                    if ours not in classes:
                        continue
                    ci = classes.index(ours)
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    det = (x1 / W, y1 / H, x2 / W, y2 / H)
                    if any(c == ci and iou(det, (x - w / 2, yy - h / 2, x + w / 2, yy + h / 2)) >= 0.3
                           for c, x, yy, w, h in boxes):
                        continue
                    boxes.append((ci, (det[0] + det[2]) / 2, (det[1] + det[3]) / 2, det[2] - det[0], det[3] - det[1]))
                    filled[ours] += 1

            (out_lab / Path(fname).with_suffix(".txt").name).write_text(
                "".join(f"{c} {x:.6f} {yy:.6f} {w:.6f} {h:.6f}\n" for c, x, yy, w, h in boxes), encoding="utf-8")
            counts.update(classes[c] for c, *_ in boxes)
            added_files.append(fname)

        (ds / f"external_{args.name}.txt").write_text("\n".join(added_files) + "\n", encoding="utf-8")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    total = Counter()
    for f in (ds / "labels" / "train").glob("*.txt"):
        total.update(classes[int(l.split()[0])] for l in f.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"\nAdded to TRAIN: {len(added_files)} images ({skipped} skipped: none of our classes)")
    print("Boxes added:", dict(counts))
    if filled:
        print("  of which auto-filled by YOLO11:", dict(filled))
    print("TRAIN boxes per class now:", {c: total[c] for c in classes})
    print("Validation and test are unchanged (our photos only).")
    print(f"Undo: .venv\\Scripts\\python.exe scripts\\13_add_external.py --remove {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())