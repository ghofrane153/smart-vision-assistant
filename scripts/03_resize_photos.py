"""Step 3b - Resize all raw photos to the same maximum size, keeping the proportions.

- Reads every session folder in data/raw (phone photos, webcam photos...)
- Applies the phone rotation (EXIF), so portrait photos are not sideways
- Resizes so the LONGEST side is --max-size pixels (never enlarges, never stretches, never crops)
- Saves JPG files in data/resized/<session>/ with the same names. The originals are not changed.

Run from the repository root:
    Windows : .venv\\Scripts\\python.exe scripts\\03_resize_photos.py
    Linux   : .venv/bin/python scripts/03_resize_photos.py
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Resize raw photos (keep proportions)")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/resized")
    parser.add_argument("--max-size", type=int, default=1280, help="Longest side in pixels")
    parser.add_argument("--quality", type=int, default=90, help="JPG quality (1-95)")
    args = parser.parse_args()

    src_root, dst_root = Path(args.input), Path(args.output)
    files = sorted(p for p in src_root.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not files:
        print(f"No images found in {src_root}")
        return 1

    sizes_before, sizes_after, per_session = Counter(), Counter(), Counter()
    skipped = 0
    for i, src in enumerate(files, 1):
        rel = src.relative_to(src_root)
        dst = (dst_root / rel).with_suffix(".jpg")
        if dst.exists():
            skipped += 1
            continue
        try:
            with Image.open(src) as img:
                sizes_before[img.size] += 1
                img = ImageOps.exif_transpose(img)        # fix phone rotation
                img = img.convert("RGB")
                img.thumbnail((args.max_size, args.max_size), Image.Resampling.LANCZOS)  # keeps ratio
                dst.parent.mkdir(parents=True, exist_ok=True)
                img.save(dst, "JPEG", quality=args.quality)
                sizes_after[img.size] += 1
                per_session[rel.parts[0] if len(rel.parts) > 1 else "(no session)"] += 1
        except Exception as ex:  # noqa: BLE001
            print(f"ERROR {src}: {ex}")
        if i % 50 == 0:
            print(f"{i}/{len(files)} images...")

    print(f"\nDone: {sum(per_session.values())} resized, {skipped} already done -> {dst_root}")
    print("Images per session:")
    for session, n in sorted(per_session.items()):
        print(f"  {session:35s} {n}")
    print("Sizes before:", dict(sizes_before.most_common(5)))
    print("Sizes after :", dict(sizes_after.most_common(5)))
    if "(no session)" in per_session:
        print("WARNING: some photos are directly in data/raw. Put them in a session folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())