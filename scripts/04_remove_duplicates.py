"""Step 3c - Find identical and near-identical photos (e.g. two photos taken without moving).

Method: "difference hash" (dHash). Each photo becomes a 64-bit fingerprint of its structure.
Two photos whose fingerprints differ by only a few bits look almost the same.
Resolution, small light changes and JPG compression do not change the fingerprint much.

Comparison is done INSIDE each session (duplicates come from the same moment).
Duplicates are MOVED (not deleted) to data/duplicates/<session>/ so you can check them.

Run from the repository root (after 03_resize_photos.py):
    Windows : .venv\\Scripts\\python.exe scripts\\04_remove_duplicates.py --dry-run
              .venv\\Scripts\\python.exe scripts\\04_remove_duplicates.py
    Linux   : .venv/bin/python scripts/04_remove_duplicates.py --dry-run

Options:
    --dry-run        only show what would be moved
    --threshold 6    max different bits (0 = exact copies only, 10 = very strict cleaning)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(path: Path, size: int = 8) -> int:
    """64-bit fingerprint: is each pixel brighter than its right neighbour?"""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> int:
    parser = argparse.ArgumentParser(description="Move near-duplicate photos out of the dataset")
    parser.add_argument("--input", default="data/resized")
    parser.add_argument("--output", default="data/duplicates")
    parser.add_argument("--threshold", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.input)
    sessions = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if not sessions:
        print(f"No session folders in {root}. Run 03_resize_photos.py first.")
        return 1

    total_kept = total_dup = 0
    for session in sessions:
        files = sorted(p for p in session.iterdir() if p.suffix.lower() in EXTENSIONS)
        kept: list[tuple[Path, int]] = []
        duplicates: list[tuple[Path, Path, int]] = []
        for f in files:
            try:
                h = dhash(f)
            except Exception as ex:  # noqa: BLE001
                print(f"ERROR {f}: {ex}")
                continue
            match = next(((k, d) for k, kh in kept if (d := hamming(h, kh)) <= args.threshold), None)
            if match:
                duplicates.append((f, match[0], match[1]))
            else:
                kept.append((f, h))

        print(f"\n[{session.name}] {len(files)} photos -> keep {len(kept)}, duplicates {len(duplicates)}")
        for dup, original, dist in duplicates:
            print(f"  {dup.name}  ~  {original.name}  (difference: {dist} bits)")
            if not args.dry_run:
                dst = Path(args.output) / session.name / dup.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dup), str(dst))
        total_kept += len(kept)
        total_dup += len(duplicates)

    print("\n" + "=" * 50)
    print(f"TOTAL: keep {total_kept} photos, {total_dup} duplicates "
          + ("found (dry run: nothing moved)" if args.dry_run else f"moved to {args.output}"))
    if not args.dry_run and total_dup:
        print("Check data/duplicates. To restore a photo, move it back to its session folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())