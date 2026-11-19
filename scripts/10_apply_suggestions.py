"""Step 7c - Add the missing objects found by the audit DIRECTLY to the dataset labels (no CVAT).

Model-assisted labeling without manual review: the boxes that the pretrained YOLO11 detected with
good confidence (person, chair, table, laptop, pc) and that were missing in our labels are added to
data/dataset/labels/*. Use it when there is no time to review every box in CVAT.

Safety:
    - the original labels are saved once in data/dataset/labels_before_audit/ (restore = copy back)
    - running it twice does not add the same boxes twice (it starts again from the saved labels)

Run from the repository root, AFTER 06_build_dataset.py and 08_audit_labels.py:
    Windows : .venv\\Scripts\\python.exe scripts\\10_apply_suggestions.py
    Linux   : .venv/bin/python scripts/10_apply_suggestions.py

!! If you run 06_build_dataset.py again later, it rewrites the labels from the CVAT exports:
   run 08_audit_labels.py and this script again afterwards.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

SPLITS = ("train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Add audit suggestions to the dataset labels")
    parser.add_argument("--dataset", default="data/dataset")
    parser.add_argument("--audit", default="runs/audit/missing_labels.csv")
    parser.add_argument("--min-conf", type=float, default=0.5)
    args = parser.parse_args()

    ds = Path(args.dataset)
    labels_dir, backup = ds / "labels", ds / "labels_before_audit"
    if not Path(args.audit).exists():
        print(f"{args.audit} not found. Run scripts/08_audit_labels.py first.")
        return 1
    classes = (ds / "classes.txt").read_text(encoding="utf-8").split()

    saved_audit = backup / "missing_labels_used.csv"
    if backup.exists():
        print(f"Starting again from the saved original labels ({backup})")
        for s in SPLITS:
            for f in (backup / s).glob("*.txt"):
                shutil.copy2(f, labels_dir / s / f.name)
        if saved_audit.exists():
            args.audit = str(saved_audit)   # the audit made on the ORIGINAL labels
            print(f"Using the audit saved with them: {saved_audit}")
    else:
        shutil.copytree(labels_dir, backup, ignore=shutil.ignore_patterns("*.cache"))
        shutil.copy2(args.audit, saved_audit)
        print(f"Original labels and audit saved in {backup}")

    added = {s: Counter() for s in SPLITS}
    with open(args.audit, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["confidence"]) < args.min_conf or r["class"] not in classes:
                continue
            split, stem = r["split"], Path(r["image"]).stem
            img = next((p for p in (ds / "images" / split).glob(f"{stem}.*")), None)
            if img is None:
                continue
            with Image.open(img) as im:
                w, h = im.size
            x1, y1, x2, y2 = map(float, r["box_xyxy"].split())
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            line = (f"{classes.index(r['class'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
            with open(labels_dir / split / f"{stem}.txt", "a", encoding="utf-8") as lf:
                lf.write(line)
            added[split][r["class"]] += 1

    for cache in labels_dir.glob("*.cache"):   # force Ultralytics to read the new labels
        cache.unlink()

    print("\nBoxes added:")
    for s in SPLITS:
        print(f"  {s:6s} {sum(added[s].values()):4d}  {dict(added[s])}")
    totals = Counter()
    for s in SPLITS:
        for f in (labels_dir / s).glob("*.txt"):
            totals.update(classes[int(l.split()[0])] for l in f.read_text(encoding="utf-8").splitlines() if l.strip())
    print("\nBoxes per class now:", {c: totals[c] for c in classes})
    print("Next: run 08_audit_labels.py again (MISSING should be close to 0), then train run3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())