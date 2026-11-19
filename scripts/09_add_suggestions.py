"""Step 7b - Add the missing objects found by the audit to the CVAT annotations, for review in CVAT.

For each CVAT export in data/cvat_exports/, it creates a new archive in data/cvat_imports/ that
contains ALL the existing annotations (all labels, unchanged) PLUS the boxes from
runs/audit/missing_labels.csv (person, chair, table, laptop, pc found by YOLO11 but not labeled).

Then in CVAT: Actions > Upload annotations > YOLO 1.1 > choose the new archive.
The suggested boxes appear like normal boxes: check them (delete the wrong ones), fix the cables.

!! Upload the new archive BEFORE making other changes in CVAT: uploading replaces the annotations
   of the task. Anything changed in CVAT after your last export would be lost.

Run from the repository root (after 08_audit_labels.py):
    Windows : .venv\\Scripts\\python.exe scripts\\09_add_suggestions.py
    Linux   : .venv/bin/python scripts/09_add_suggestions.py
"""
from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge audit suggestions into the CVAT exports")
    parser.add_argument("--exports", default="data/cvat_exports")
    parser.add_argument("--audit", default="runs/audit/missing_labels.csv")
    parser.add_argument("--dataset", default="data/dataset", help="to read the image sizes")
    parser.add_argument("--output", default="data/cvat_imports")
    parser.add_argument("--min-conf", type=float, default=0.5, help="only add suggestions above this confidence")
    args = parser.parse_args()

    if not Path(args.audit).exists():
        print(f"{args.audit} not found. Run scripts/08_audit_labels.py first.")
        return 1

    # image sizes (the dataset images are the same files as in CVAT)
    sizes = {}
    for p in Path(args.dataset, "images").rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            with Image.open(p) as im:
                sizes[p.stem] = im.size  # (w, h)

    suggestions = defaultdict(list)  # stem -> [(class name, x, y, w, h normalized)]
    with open(args.audit, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["confidence"]) < args.min_conf:
                continue
            stem = Path(r["image"]).stem
            if stem not in sizes:
                continue
            w, h = sizes[stem]
            x1, y1, x2, y2 = map(float, r["box_xyxy"].split())
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            suggestions[stem].append((r["class"], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = Counter()
    for zp in sorted(Path(args.exports).glob("*.zip")):
        with zipfile.ZipFile(zp) as zin:
            names_file = next((n for n in zin.namelist() if n.endswith("obj.names")), None)
            if not names_file:
                print(f"SKIP {zp.name}: not a 'YOLO 1.1' export (no obj.names)")
                continue
            labels = [l.strip() for l in zin.read(names_file).decode("utf-8").splitlines() if l.strip()]
            index = {l.lower().replace(" ", "_"): i for i, l in enumerate(labels)}
            out_zip = out_dir / f"{zp.stem}_with_suggestions.zip"
            added, missing_label = Counter(), Counter()
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout:
                for n in zin.namelist():
                    data = zin.read(n)
                    p = Path(n)
                    if p.suffix == ".txt" and p.parent.name.startswith("obj_") and p.stem in suggestions:
                        lines = data.decode("utf-8").rstrip("\n")
                        new = []
                        for cls, x, y, w, h in suggestions[p.stem]:
                            if cls not in index:
                                missing_label[cls] += 1
                                continue
                            new.append(f"{index[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                            added[cls] += 1
                        data = ("\n".join(([lines] if lines else []) + new) + "\n").encode("utf-8")
                    zout.writestr(n, data)
            total.update(added)
            print(f"{zp.name} -> {out_zip.name}: {sum(added.values())} boxes added {dict(added)}")
            if missing_label:
                print(f"   WARNING: labels not in this CVAT task, not added: {dict(missing_label)}")

    print(f"\nTOTAL added: {sum(total.values())} {dict(total)}")
    print(f"Archives to upload in CVAT: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())