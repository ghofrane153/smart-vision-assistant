"""Step 4a - Automatic pre-labeling with the pretrained YOLO11 (model-assisted labeling).

YOLO11 (COCO) already knows 7 of our 8 classes. It draws the boxes for them, so in CVAT we only
CORRECT its mistakes and ADD the missing objects (all cables, PC towers, missed objects).

For each session folder in data/resized it creates a CVAT import file (format "YOLO 1.1"):
    data/prelabels/<session>_prelabels.zip
and a report of the number of objects per class: data/prelabels/class_counts.csv

Run from the repository root:
    Windows : .venv\\Scripts\\python.exe scripts\\05_prelabel.py
    Linux   : .venv/bin/python scripts/05_prelabel.py
"""
from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

# Our classes, in this exact order (same names and order in CVAT and data.yaml)
CLASSES = ["person", "chair", "table", "laptop", "pc", "backpack", "handbag", "cable"]

# COCO name -> our class name ('cable' does not exist in COCO: always labeled by hand)
COCO_TO_PROJECT = {
    "person": "person",
    "chair": "chair",
    "dining table": "table",
    "laptop": "laptop",
    "tv": "pc",            # computer monitors are often detected as 'tv'
    "backpack": "backpack",
    "handbag": "handbag",
}
EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET = 100  # minimum objects per class (course plan)


def yolo11_cvat_zip(zip_path: Path, labels: dict[str, list[str]]) -> None:
    """Write a CVAT 'YOLO 1.1' annotation archive (annotations only, matched by image name)."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(CLASSES) + "\n")
        z.writestr("obj.data", f"classes = {len(CLASSES)}\ntrain = data/train.txt\n"
                               "names = data/obj.names\nbackup = backup/\n")
        z.writestr("train.txt", "".join(f"data/obj_train_data/{name}\n" for name in labels))
        for name, lines in labels.items():
            z.writestr(f"obj_train_data/{Path(name).stem}.txt", "".join(l + "\n" for l in lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-label photos with pretrained YOLO11 for CVAT")
    parser.add_argument("--input", default="data/resized")
    parser.add_argument("--output", default="data/prelabels")
    parser.add_argument("--model", default="yolo11s.pt", help="yolo11s = more accurate pre-labels than yolo11n")
    parser.add_argument("--conf", type=float, default=0.40, help="Higher = fewer wrong boxes to delete")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    root, out = Path(args.input), Path(args.output)
    sessions = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if not sessions:
        print(f"No session folders in {root}. Run 03_resize_photos.py first.")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    keep_ids = {i for i, n in model.names.items() if n in COCO_TO_PROJECT}
    total, rows, n_images = Counter(), [], 0

    for session in sessions:
        images = sorted(p for p in session.iterdir() if p.suffix.lower() in EXTENSIONS)
        labels: dict[str, list[str]] = {}
        counts = Counter()
        for img in images:
            r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz,
                              classes=sorted(keep_ids), verbose=False)[0]
            lines = []
            for box in r.boxes:
                name = COCO_TO_PROJECT[model.names[int(box.cls)]]
                x, y, w, h = box.xywhn[0].tolist()
                lines.append(f"{CLASSES.index(name)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                counts[name] += 1
            labels[img.name] = lines
        yolo11_cvat_zip(out / f"{session.name}_prelabels.zip", labels)
        total.update(counts)
        n_images += len(images)
        rows.append({"session": session.name, "images": len(images), **{c: counts[c] for c in CLASSES}})
        print(f"[{session.name}] {len(images)} images, {sum(counts.values())} boxes -> "
              f"{session.name}_prelabels.zip")

    rows.append({"session": "TOTAL", "images": n_images, **{c: total[c] for c in CLASSES}})
    with open(out / "class_counts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session", "images", *CLASSES])
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"{len(sessions)} sessions, {n_images} images")
    print(f"Objects found automatically (target: at least {TARGET} per class):")
    for c in CLASSES:
        if c == "cable":
            print(f"  {c:9s}    - : not in COCO, count it after labeling by hand in CVAT")
            continue
        status = "OK" if total[c] >= TARGET else f"LOW: about {TARGET - total[c]} more needed"
        print(f"  {c:9s} {total[c]:5d}  {status}")
    print("\nThese numbers are only an estimate: the model misses some objects (especially PC "
          "towers and classroom tables). Real counts come after correction in CVAT.")
    if len(sessions) < 6:
        print(f"NOTE: only {len(sessions)} sessions. For a clean train/val/test split by session, "
              "aim for 8 or more.")
    print(f"Report: {out / 'class_counts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())