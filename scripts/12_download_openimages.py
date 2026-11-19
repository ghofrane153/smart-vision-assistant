"""Step 9a - Download a small subset of Open Images (images + boxes) in YOLO format.

Only standard Python: no extra package to install. It downloads the official Open Images box
annotation files of the validation and test splits (about 25 + 75 MB, once), picks images that
contain the wanted classes (door, stairs by default), downloads only those images and writes
YOLO labels for ALL our classes that Open Images annotated in them (person, chair, table, ...).

Run from the repository root:
    Windows : .venv\\Scripts\\python.exe scripts\\12_download_openimages.py
    Linux   : .venv/bin/python scripts/12_download_openimages.py

Output: data/external/openimages/  (images/, labels/, data.yaml, sources.csv)
Then:   .venv\\Scripts\\python.exe scripts\\13_add_external.py data\\external\\openimages --name openimages

Open Images is published under CC BY 4.0 (annotations) / CC BY 2.0 (images): cite it in the report.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://storage.googleapis.com/openimages/2018_04"
ANNOTATIONS = {s: f"{BASE}/{s}/{s}-annotations-bbox.csv" for s in ("validation", "test")}
CLASS_FILE = f"{BASE}/class-descriptions-boxable.csv"
IMAGE_URLS = [  # tried in this order
    "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg",
    "https://s3.amazonaws.com/open-images-dataset/{split}/{id}.jpg",
]
# Open Images name -> our class name (all of them are written in the labels when present)
OI_TO_OURS = {"Door": "door", "Stairs": "stairs", "Person": "person", "Man": "person", "Woman": "person",
              "Boy": "person", "Girl": "person", "Chair": "chair", "Table": "table", "Desk": "table",
              "Laptop": "laptop", "Computer monitor": "pc", "Backpack": "backpack", "Handbag": "handbag",
              "Bottle": "bottle"}


def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return dest.stat().st_size > 1000
    except Exception:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a small Open Images subset in YOLO format")
    parser.add_argument("--want", nargs="+", default=["Door", "Stairs"], help="Open Images classes to look for")
    parser.add_argument("--per-class", type=int, default=150, help="images to download per wanted class")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--output", default="data/external/openimages")
    parser.add_argument("--cache", default="data/external/_openimages_csv", help="where the CSV files are kept")
    parser.add_argument("--image-url", default="", help="custom image URL template with {split} and {id}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    # 1. class names <-> Open Images ids ------------------------------------------------------
    cls_file = cache / "class-descriptions-boxable.csv"
    if not cls_file.exists():
        print("Downloading the class list ...")
        if not download(CLASS_FILE, cls_file):
            print(f"Could not download {CLASS_FILE}. Check the internet connection.")
            return 1
    mid_to_name = {}
    with open(cls_file, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                mid_to_name[row[0]] = row[1]
    unknown = [w for w in args.want if w not in mid_to_name.values()]
    if unknown:
        print(f"Unknown Open Images classes: {unknown}")
        return 1

    # 2. boxes of the splits --------------------------------------------------------------------
    boxes = defaultdict(list)          # (split, image id) -> [(our class, x, y, w, h)]
    has_wanted = defaultdict(set)      # wanted class -> {(split, id)}
    for split in args.splits:
        ann = cache / f"{split}-annotations-bbox.csv"
        if not ann.exists():
            print(f"Downloading the {split} box annotations (once, may take a few minutes) ...")
            if not download(ANNOTATIONS[split], ann):
                print(f"Could not download {ANNOTATIONS[split]}")
                return 1
        with open(ann, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                name = mid_to_name.get(r["LabelName"])
                if name not in OI_TO_OURS or r.get("IsDepiction") == "1":   # skip drawings/posters
                    continue
                x1, x2, y1, y2 = float(r["XMin"]), float(r["XMax"]), float(r["YMin"]), float(r["YMax"])
                key = (split, r["ImageID"])
                boxes[key].append((OI_TO_OURS[name], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
                if name in args.want:
                    has_wanted[name].add(key)
    for w in args.want:
        print(f"{w}: {len(has_wanted[w])} images available")

    # 3. choose the images ----------------------------------------------------------------------
    rng = random.Random(args.seed)
    chosen = set()
    for w in args.want:
        pool = sorted(has_wanted[w] - chosen)
        rng.shuffle(pool)
        chosen.update(pool[:args.per_class])
    chosen = sorted(chosen)
    print(f"Downloading {len(chosen)} images ...")

    # 4. download images and write labels -----------------------------------------------------
    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    classes = sorted({c for v in boxes.values() for c, *_ in v})
    templates = [args.image_url] if args.image_url else IMAGE_URLS

    def fetch(key):
        split, iid = key
        dest = out / "images" / f"oi_{iid}.jpg"
        for t in templates:
            if download(t.format(split=split, id=iid), dest):
                return key
        return None

    done = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, res in enumerate(pool.map(fetch, chosen), 1):
            if res:
                done.append(res)
            if i % 50 == 0:
                print(f"  {i}/{len(chosen)}  ({len(done)} ok)")
    if not done:
        print("No image could be downloaded. Check the internet connection (or a firewall).")
        return 1

    counts = Counter()
    with open(out / "sources.csv", "w", newline="", encoding="utf-8") as sf:
        wr = csv.writer(sf)
        wr.writerow(["file", "open_images_id", "split", "source"])
        for split, iid in done:
            lines = []
            for c, x, y, w, h in boxes[(split, iid)]:
                lines.append(f"{classes.index(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                counts[c] += 1
            (out / "labels" / f"oi_{iid}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            wr.writerow([f"oi_{iid}.jpg", iid, split, "Open Images V4/V7 (CC BY 4.0 annotations, CC BY 2.0 images)"])
    (out / "data.yaml").write_text("names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)),
                                   encoding="utf-8")

    print(f"\nDone: {len(done)} images in {out}")
    print("Boxes per class:", dict(counts))
    print(f"Next: .venv\\Scripts\\python.exe scripts\\13_add_external.py {out} --name openimages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())