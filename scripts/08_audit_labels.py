"""Step 7 - Audit the annotations: find objects that are probably MISSING in the labels.

The pretrained YOLO11 (COCO) already detects person, chair, table, laptop and monitors (pc) well.
For every image of data/dataset, if it detects one of these objects with good confidence and
there is NO box of the same class at that place in our labels, the object is probably missing.

Output:
    runs/audit/missing_labels.csv      one line per suspected missing object (image, class, confidence)
    runs/audit/images_to_fix.txt       image names to correct in CVAT, worst first
    runs/audit/preview/*.jpg           green = our labels, red = probably missing (worst images)
    + a table per annotator (prefix of the file name: dorra_, emna_, ghofrane_)

Run from the repository root:
    Windows : .venv\\Scripts\\python.exe scripts\\08_audit_labels.py
    Linux   : .venv/bin/python scripts/08_audit_labels.py
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class -> our class
COCO_TO_OURS = {"person": "person", "chair": "chair", "dining table": "table", "laptop": "laptop", "tv": "pc"}
SPLITS = ("train", "val", "test")


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def annotator(stem: str) -> str:
    m = re.match(r"([A-Za-z]+)_", stem)
    return m.group(1).lower() if m else "(other)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Find probably missing labels with the pretrained YOLO11")
    parser.add_argument("--dataset", default="data/dataset")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--conf", type=float, default=0.5, help="Only confident COCO detections are checked")
    parser.add_argument("--iou", type=float, default=0.3, help="Overlap needed to count as 'already labeled'")
    parser.add_argument("--preview", type=int, default=40, help="Number of worst images to draw")
    args = parser.parse_args()

    ds = Path(args.dataset)
    classes = (ds / "classes.txt").read_text(encoding="utf-8").split()
    checked = [c for c in COCO_TO_OURS.values() if c in classes]
    print(f"Classes checked automatically: {checked}  (cable, door, stairs cannot be checked: not in COCO)")

    model = YOLO(args.model)
    rows, per_image = [], defaultdict(list)
    stats = defaultdict(lambda: {"images": 0, "empty": 0, "boxes": Counter(), "missing": Counter()})
    images = [(s, p) for s in SPLITS for p in sorted((ds / "images" / s).glob("*")) if p.suffix.lower() in
              {".jpg", ".jpeg", ".png"}]
    print(f"Checking {len(images)} images...")

    for i, (split, img_path) in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        label_file = ds / "labels" / split / f"{img_path.stem}.txt"
        ours = []
        for line in label_file.read_text(encoding="utf-8").splitlines() if label_file.exists() else []:
            c, x, y, bw, bh = line.split()
            x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
            ours.append((classes[int(c)], (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2)))

        who = annotator(img_path.stem)
        st = stats[who]
        st["images"] += 1
        st["empty"] += not ours
        st["boxes"].update(c for c, _ in ours)

        r = model.predict(img, conf=args.conf, verbose=False)[0]
        for box in r.boxes:
            name = COCO_TO_OURS.get(model.names[int(box.cls)])
            if name not in checked:
                continue
            det = tuple(box.xyxy[0].tolist())
            if any(c == name and iou(det, b) >= args.iou for c, b in ours):
                continue
            conf = float(box.conf)
            rows.append({"image": img_path.name, "split": split, "annotator": who, "class": name,
                         "confidence": round(conf, 2),
                         "box_xyxy": " ".join(f"{v:.0f}" for v in det)})
            per_image[(split, img_path)].append((name, det, conf))
            st["missing"][name] += 1
        if i % 100 == 0:
            print(f"  {i}/{len(images)}")

    out = Path("runs/audit")
    shutil.rmtree(out, ignore_errors=True)
    (out / "preview").mkdir(parents=True)
    with open(out / "missing_labels.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["image", "split", "annotator", "class", "confidence", "box_xyxy"])
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: (r["annotator"], r["image"])))

    worst = sorted(per_image.items(), key=lambda kv: -len(kv[1]))
    (out / "images_to_fix.txt").write_text(
        "".join(f"{p.name}\t{len(v)} missing: {', '.join(sorted({n for n, _, _ in v}))}\n" for (_, p), v in worst),
        encoding="utf-8")

    for (split, img_path), missing in worst[:args.preview]:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        label_file = ds / "labels" / split / f"{img_path.stem}.txt"
        for line in label_file.read_text(encoding="utf-8").splitlines():
            c, x, y, bw, bh = line.split()
            x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
            cv2.rectangle(img, (int(x - bw / 2), int(y - bh / 2)), (int(x + bw / 2), int(y + bh / 2)), (0, 200, 0), 2)
            cv2.putText(img, classes[int(c)], (int(x - bw / 2) + 3, int(y - bh / 2) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        for name, (x1, y1, x2, y2), conf in missing:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(img, f"MISSING {name} {conf:.2f}", (int(x1) + 3, int(y2) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(out / "preview" / img_path.name), img)

    # report
    print("\n" + "=" * 86)
    print("Per annotator (from the file name prefix):")
    head = f"{'annotator':10s}{'images':>7s}{'empty':>7s}{'boxes/img':>10s}"
    head += "".join(f"{c[:7]:>8s}" for c in classes) + f"{'MISSING':>9s}{'miss/img':>9s}"
    print(head)
    for who, st in sorted(stats.items()):
        n = st["images"]
        line = f"{who:10s}{n:7d}{st['empty']:7d}{sum(st['boxes'].values()) / n:10.2f}"
        line += "".join(f"{st['boxes'][c] / n:8.2f}" for c in classes)
        miss = sum(st["missing"].values())
        print(line + f"{miss:9d}{miss / n:9.2f}")
    print("(class columns = boxes per image; MISSING = objects YOLO11 sees but that are not labeled)")
    print("\nMissing by class:", dict(sum((s["missing"] for s in stats.values()), Counter())))
    print(f"\n{len(per_image)} images to check, worst first: {out / 'images_to_fix.txt'}")
    print(f"Details: {out / 'missing_labels.csv'}  |  Pictures: {out / 'preview'} (red = missing)")
    print("NOTE: YOLO11 can be wrong too (e.g. a poster of a person, a TV as 'pc'): check before adding a box.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())