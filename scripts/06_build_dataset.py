"""Step 5 - Build the YOLO dataset from the CVAT exports and split it into train / val / test.

Input:
    data/cvat_exports/*.zip   annotations exported from CVAT ("YOLO 1.1" or "Ultralytics YOLO Detection")
    data/resized/<folder>/    the images that were uploaded to CVAT (matched by file name)

Output:
    data/dataset/images/{train,val,test}/   data/dataset/labels/{train,val,test}/
    data/dataset/data.yaml                  data/dataset/classes.txt
    data/dataset/split_report.csv           runs/check_labels/*.jpg (boxes drawn, to check by eye)

Classes (--classes):
    auto     (default) all labels found in the exports: our 9 classes first, then the others
             by number of boxes. Labels with fewer than --min-boxes boxes are left out,
             and the labels given with --drop (e.g. --drop wall bottle).
    project  only our 9 base classes
    <file>   a text file with one class name per line (fixed list and order)

Split (--split):
    auto     (default) by session if there are at least 6 session folders, otherwise by groups
    session  whole session folders go to train, val or test (no data leakage)
    groups   neighbour photos (same number range, e.g. _0001 to _0012, across all folders) form a
             group; whole groups go to train, val or test, chosen so that every class is spread
             about 70/20/10 (balanced, and near-identical neighbours stay in the same split)
    blocks   each folder is cut in consecutive blocks in file order (70% / 20% / 10%)

Fair comparison after correcting labels: --reuse-split keeps each image in its previous split.

Image names: a Windows copy suffix like "photo (2).jpg" is ignored, so it still matches "photo".

Run from the repository root:
    Windows : .venv\\Scripts\\python.exe scripts\\06_build_dataset.py
    Linux   : .venv/bin/python scripts/06_build_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

BASE_CLASSES = ["person", "chair", "table", "laptop", "pc", "backpack", "handbag", "cable", "bottle"]
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png"}
SKIP_TXT = {"train.txt", "valid.txt", "val.txt", "test.txt", "obj.names"}

Box = tuple[str, float, float, float, float]  # (label name, x, y, w, h), normalized


def natural_key(name: str):
    """img2 before img10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def clean_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def read_export(zip_path: Path) -> tuple[dict[str, list[Box]], list[str]]:
    """Return {image_stem: [boxes with label NAMES]} from one CVAT export, and warnings."""
    warnings = []
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        export_classes = None
        for n in names:
            if n.endswith("obj.names"):
                export_classes = [clean_name(l) for l in z.read(n).decode("utf-8").splitlines() if l.strip()]
            elif n.endswith(("data.yaml", "data.yml")):
                y = yaml.safe_load(z.read(n))
                ns = y.get("names")
                export_classes = [clean_name(ns[k]) for k in sorted(ns)] if isinstance(ns, dict) \
                    else [clean_name(x) for x in ns]
        if not export_classes:
            raise ValueError(f"{zip_path.name}: no obj.names or data.yaml found. Export in 'YOLO 1.1' format.")

        labels: dict[str, list[Box]] = {}
        bad = 0
        for n in names:
            p = Path(n)
            if p.suffix != ".txt" or p.name in SKIP_TXT:
                continue
            boxes = []
            for line in z.read(n).decode("utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                cid = int(parts[0])
                if cid >= len(export_classes):
                    bad += 1
                    continue
                x, y, w, h = map(float, parts[1:5])
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1.001 and 0 < h <= 1.001):
                    bad += 1
                    continue
                boxes.append((export_classes[cid], x, y, min(w, 1.0), min(h, 1.0)))
            labels[p.stem] = boxes
        if bad:
            warnings.append(f"{zip_path.name}: {bad} invalid boxes removed")
    return labels, warnings


def choose_classes(mode: str, counts: Counter, min_boxes: int, drop: set[str]) -> tuple[list[str], list[str]]:
    """Return (classes kept in order, labels left out)."""
    if mode == "project":
        kept = [c for c in BASE_CLASSES if c not in drop]
    elif mode == "auto":
        others = sorted((c for c in counts if c not in BASE_CLASSES), key=lambda c: (-counts[c], c))
        kept = [c for c in BASE_CLASSES + others if counts[c] >= min_boxes and c not in drop]
    else:
        path = Path(mode)
        if not path.exists():
            raise SystemExit(f"--classes: file {path} not found (use auto, project or a .txt file)")
        kept = [clean_name(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    left_out = sorted(c for c in counts if c not in kept)
    return kept, left_out


def split_by_groups(items_by_session: dict[str, list[Path]], ratios, classes_of, group_size: int,
                    seed: int) -> dict[str, list[Path]]:
    """Group neighbour photos by their number, then try many random assignments of whole groups
    and keep the one where every class is closest to the target ratios."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for sess, files in items_by_session.items():
        for i, f in enumerate(sorted(files, key=lambda p: natural_key(p.name))):
            stem = re.sub(r"\s*\(\d+\)$", "", f.stem)   # ignore Windows copy suffix " (2)"
            m = re.search(r"(\d+)$", stem)
            key = f"n{int(m.group(1)) // group_size}" if m else f"{sess}_{i // group_size}"
            groups[key].append(f)
    keys = sorted(groups)
    total_img = sum(len(v) for v in groups.values())
    gcount = {k: Counter(c for f in groups[k] for c in classes_of(f)) for k in keys}
    all_classes = sum(gcount.values(), Counter())

    def score(assign):
        sc = 0.0
        for i, s in enumerate(SPLITS):
            n_img = sum(len(groups[k]) for k in assign[s])
            sc += 2 * abs(n_img / total_img - ratios[i])
            cnt = sum((gcount[k] for k in assign[s]), Counter())
            for c, n in all_classes.items():
                sc += abs(cnt[c] / n - ratios[i]) * min(n, 50) / 50
        return sc

    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(400):
        order = keys[:]
        rng.shuffle(order)
        assign, cum = {s: [] for s in SPLITS}, 0
        for k in order:
            frac = cum / total_img
            s = "train" if frac < ratios[0] else ("val" if frac < ratios[0] + ratios[1] else "test")
            assign[s].append(k)
            cum += len(groups[k])
        sc = score(assign)
        if sc < best_score:
            best, best_score = assign, sc
    print(f"Split BY GROUPS of neighbour photos ({len(keys)} groups of about {group_size} numbers, "
          f"balanced over 400 tries)")
    return {s: [f for k in best[s] for f in groups[k]] for s in SPLITS}


def split_items(items_by_session: dict[str, list[Path]], ratios, mode: str, seed: int,
                classes_of=None, group_size: int = 12) -> dict[str, list[Path]]:
    result = {s: [] for s in SPLITS}
    sessions = sorted(items_by_session)
    if mode == "auto":
        mode = "session" if len(sessions) >= 6 else "groups"
    if mode == "groups":
        return split_by_groups(items_by_session, ratios, classes_of, group_size, seed)
    if mode == "session":
        print(f"Split BY SESSION ({len(sessions)} sessions)")
        total = sum(len(v) for v in items_by_session.values())
        random.Random(seed).shuffle(sessions)
        order = sorted(sessions, key=lambda s: -len(items_by_session[s]))
        targets = {s: r * total for s, r in zip(SPLITS, ratios)}
        filled = {s: 0 for s in SPLITS}
        for sess in order:
            split = max(SPLITS, key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1))
            result[split].extend(items_by_session[sess])
            filled[split] += len(items_by_session[sess])
            print(f"  {sess:35s} -> {split} ({len(items_by_session[sess])} images)")
        for s in ("val", "test"):
            if not result[s]:
                print(f"  WARNING: no session in {s}. Use --split blocks or add sessions.")
    else:
        print(f"Split BY CONSECUTIVE BLOCKS inside each folder ({len(sessions)} folder(s))")
        for sess in sessions:
            files = sorted(items_by_session[sess], key=lambda p: natural_key(p.name))
            n = len(files)
            n_train, n_val = round(n * ratios[0]), round(n * ratios[1])
            result["train"] += files[:n_train]
            result["val"] += files[n_train:n_train + n_val]
            result["test"] += files[n_train + n_val:]
            print(f"  {sess:35s} train {n_train} | val {n_val} | test {n - n_train - n_val}")
    return result


def draw_preview(img_path: Path, lines: list[str], classes: list[str], out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for line in lines:
        c, x, y, bw, bh = line.split()
        x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
        p1, p2 = (int(x - bw / 2), int(y - bh / 2)), (int(x + bw / 2), int(y + bh / 2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(img, classes[int(c)], (p1[0] + 3, max(p1[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and split the YOLO dataset from CVAT exports")
    parser.add_argument("--exports", default="data/cvat_exports")
    parser.add_argument("--images", default="data/resized")
    parser.add_argument("--output", default="data/dataset")
    parser.add_argument("--classes", default="auto", help="auto | project | path/to/classes.txt")
    parser.add_argument("--min-boxes", type=int, default=20,
                        help="auto mode: leave out new labels with fewer boxes than this")
    parser.add_argument("--drop", nargs="*", default=[], help="labels to leave out, e.g. --drop wall bottle")
    parser.add_argument("--split", default="auto", choices=["auto", "session", "groups", "blocks"])
    parser.add_argument("--group-size", type=int, default=12, help="groups split: numbers per group")
    parser.add_argument("--reuse-split", action="store_true",
                        help="keep every image in the split it had in the existing dataset (fair comparison "
                             "between runs after correcting labels); new images are split normally")
    parser.add_argument("--ratios", default="0.7,0.2,0.1", help="train,val,test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=int, default=12, help="Images with boxes to save for checking")
    args = parser.parse_args()
    ratios = [float(r) for r in args.ratios.split(",")]

    # 1. read all CVAT exports -------------------------------------------------------------
    zips = sorted(Path(args.exports).glob("*.zip"))
    if not zips:
        print(f"No .zip in {args.exports}. Export the task from CVAT and put the zip there.")
        return 1
    raw: dict[str, list[Box]] = {}
    for zp in zips:
        lab, warns = read_export(zp)
        for w in warns:
            print("WARNING", w)
        dup = set(lab) & set(raw)
        if dup:
            print(f"WARNING {zp.name}: {len(dup)} images already in another export, the last one is kept")
        raw.update(lab)
        print(f"{zp.name}: {len(lab)} label files, {sum(len(v) for v in lab.values())} boxes")

    # 2. match with the images (by file name) ----------------------------------------------
    images = {}
    for p in Path(args.images).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            stem = re.sub(r"\s*\(\d+\)$", "", p.stem)   # "photo (2)" -> "photo" (Windows copy)
            if stem in images:
                print(f"WARNING: two images named {stem} ({images[stem]} and {p}). The second is not used.")
                continue
            images[stem] = p
    matched = sorted(set(raw) & set(images))
    no_image = sorted(set(raw) - set(images), key=natural_key)
    no_label = sorted(set(images) - set(raw), key=natural_key)

    # 3. classes (counted on the images that are really used) --------------------------------
    counts = Counter(b[0] for stem in (matched or raw) for b in raw[stem])
    drop = {clean_name(d) for d in args.drop}
    classes, left_out = choose_classes(args.classes, counts, args.min_boxes, drop)
    print(f"\nBoxes per label{' (matched images only)' if matched else ''}:")
    for name, n in counts.most_common():
        status = f"class {classes.index(name)}" if name in classes else "LEFT OUT"
        print(f"  {name:20s} {n:6d}  {status}")
    for c in classes:
        if c not in counts:
            print(f"  {c:20s}      0  class {classes.index(c)}  (no box yet)")
    if left_out:
        print(f"Left out ({len(left_out)}): {left_out}"
              + (f"  (fewer than {args.min_boxes} boxes or --drop; change with --min-boxes / --drop)"
                 if args.classes == "auto" else ""))
    print(f"-> {len(classes)} classes\n")

    if no_image:
        print(f"WARNING: {len(no_image)} label files without image, e.g. {no_image[:3]}")
    if no_label:
        print(f"NOTE: {len(no_label)} images not in the exports (not used), e.g. {no_label[:3]}")
    if not matched:
        print("\nNo image matches the labels. The label files are named like "
              f"{sorted(raw, key=natural_key)[:3]}.\nPut the images with THESE names (the files uploaded "
              f"to CVAT) in {args.images}/<folder>/, or use --images <folder>.")
        return 1
    print(f"{len(matched)} images matched with their labels")

    def to_lines(stem: str) -> list[str]:
        return [f"{classes.index(n)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                for n, x, y, w, h in raw[stem] if n in classes]

    stem_of = {images[stem]: stem for stem in matched}   # image path -> clean name used in labels
    by_session = defaultdict(list)
    for stem in matched:
        by_session[images[stem].parent.name].append(images[stem])

    def classes_of(img: Path) -> list[str]:
        return [n for n, *_ in raw[stem_of[img]] if n in classes]

    # 4. split -------------------------------------------------------------------------------
    previous = {}
    if args.reuse_split:
        for s in SPLITS:
            d = Path(args.output) / "images" / s
            for f in d.glob("*") if d.exists() else []:
                if f.suffix.lower() in IMG_EXT:
                    previous[f.stem] = s
        if not previous:
            print("WARNING --reuse-split: no existing dataset found, normal split is used")
    if previous:
        split = {s: [] for s in SPLITS}
        new_items = defaultdict(list)
        for sess, files in by_session.items():
            for f in files:
                if stem_of[f] in previous:
                    split[previous[stem_of[f]]].append(f)
                else:
                    new_items[sess].append(f)
        kept = sum(len(v) for v in split.values())
        print(f"Split REUSED from the existing dataset for {kept} images "
              f"(train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])})")
        if new_items:
            print(f"{sum(len(v) for v in new_items.values())} new images are split normally:")
            extra = split_items(new_items, ratios, args.split, args.seed, classes_of, args.group_size)
            for s in SPLITS:
                split[s] += extra[s]
    else:
        split = split_items(by_session, ratios, args.split, args.seed, classes_of, args.group_size)

    # 5. write the dataset (start from an empty folder) ---------------------------------------
    out = Path(args.output)
    for sub in ("images", "labels"):
        for s in SPLITS:
            d = out / sub / s
            if d.exists():
                for f in d.iterdir():
                    if f.name != ".gitkeep":
                        f.unlink()
            d.mkdir(parents=True, exist_ok=True)

    stats = {s: Counter() for s in SPLITS}
    n_img, n_empty = Counter(), Counter()
    for s in SPLITS:
        for img in split[s]:
            stem = stem_of[img]
            lines = to_lines(stem)
            shutil.copy2(img, out / "images" / s / f"{stem}{img.suffix.lower()}")
            (out / "labels" / s / f"{stem}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
            n_img[s] += 1
            n_empty[s] += not lines
            stats[s].update(classes[int(l.split()[0])] for l in lines)

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(classes)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    with open(out / "split_report.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "images", "images_without_objects", *classes])
        for s in SPLITS:
            wr.writerow([s, n_img[s], n_empty[s], *[stats[s][c] for c in classes]])
        tot = sum(stats.values(), Counter())
        wr.writerow(["total", sum(n_img.values()), sum(n_empty.values()), *[tot[c] for c in classes]])

    # 6. report ------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'id':>3s} {'class':18s}" + "".join(f"{s:>7s}" for s in SPLITS) + f"{'total':>7s}")
    for i, c in enumerate(classes):
        row = [stats[s][c] for s in SPLITS]
        flag = ""
        if sum(row) < 50:
            flag = "  <- very few examples"
        elif sum(row) < 100:
            flag = "  <- below 100"
        if sum(row) and (row[1] == 0 or row[2] == 0):
            flag += "  <- none in val/test"
        print(f"{i:3d} {c:18s}" + "".join(f"{v:7d}" for v in row) + f"{sum(row):7d}{flag}")
    print(f"    {'IMAGES':18s}" + "".join(f"{n_img[s]:7d}" for s in SPLITS) + f"{sum(n_img.values()):7d}")
    print(f"    {'(no object)':18s}" + "".join(f"{n_empty[s]:7d}" for s in SPLITS))
    print(f"\nDataset: {out}  |  {out / 'data.yaml'}  |  {out / 'classes.txt'}  |  {out / 'split_report.csv'}")

    if args.preview:
        prev_dir = Path("runs/check_labels")
        shutil.rmtree(prev_dir, ignore_errors=True)
        pool = [p for p in split["train"] if to_lines(stem_of[p])]
        sample = random.Random(args.seed).sample(pool, min(args.preview, len(pool)))
        for img in sample:
            draw_preview(img, to_lines(stem_of[img]), classes, prev_dir / f"{stem_of[img]}.jpg")
        print(f"Check by eye: {len(sample)} images with boxes in {prev_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())