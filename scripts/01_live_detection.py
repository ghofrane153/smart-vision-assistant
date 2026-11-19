"""Step 2 - Live detection with the PRETRAINED YOLO11 model (baseline, before our training).

Shows what the generic COCO model already detects in our classroom, the position of each
object (left / center / right) and the speed (FPS). Press Q or ESC to quit, S to save a frame.

Run from the repository root:
    Windows : .venv\\Scripts\\python.exe scripts\\01_live_detection.py
    Linux   : .venv/bin/python scripts/01_live_detection.py

Options:
    --source 1                              other camera, video file or phone stream URL
    --conf 0.35                             confidence threshold
    --all                                   show all 80 COCO classes, not only our project classes
    --model models/best.pt                  later: use our own trained model
    --no-show --max-frames 50               test without a window (saves the last frame)
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO classes that match our project classes.
# 'cable' and 'pc' (desktop tower) do not exist in COCO: this is why we train our own model.
COCO_TO_PROJECT = {
    "person": "person",
    "chair": "chair",
    "dining table": "table",
    "laptop": "laptop",
    "tv": "pc",          # COCO 'tv' often detects computer monitors
    "backpack": "backpack",
    "handbag": "handbag",
}
COLORS = {"left": (255, 160, 0), "center": (0, 0, 255), "right": (0, 200, 0)}


def position(x_center: float, width: int) -> str:
    """Split the image into three vertical zones."""
    if x_center < width / 3:
        return "left"
    if x_center > 2 * width / 3:
        return "right"
    return "center"


def main() -> int:
    parser = argparse.ArgumentParser(description="Live YOLO11 detection with left/center/right zones")
    parser.add_argument("--source", default="0", help="Camera index, video file or stream URL")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--all", action="store_true", help="Show all COCO classes")
    parser.add_argument("--no-show", action="store_true", help="No window (for testing)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Could not open the camera/video. Try --source 1, or close Teams/Zoom.")
        return 1

    model = YOLO(args.model)
    own_model = not Path(args.model).name.startswith("yolo11")
    print(f"Model: {args.model} | classes: {list(model.names.values())}")
    print("Q/ESC = quit | S = save frame")

    out_dir = Path("runs/live")
    seen = Counter()          # how many frames each class was detected in
    frames, fps = 0, 0.0
    annotated = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            fps = 0.9 * fps + 0.1 * (1 / (time.perf_counter() - t0)) if frames else 1 / (time.perf_counter() - t0)
            frames += 1

            h, w = frame.shape[:2]
            annotated = frame.copy()
            # zone separators
            for x in (w // 3, 2 * w // 3):
                cv2.line(annotated, (x, 0), (x, h), (200, 200, 200), 1)

            in_frame = set()
            for box in result.boxes:
                coco_name = model.names[int(box.cls)]
                if own_model or args.all:
                    name = coco_name
                elif coco_name in COCO_TO_PROJECT:
                    name = COCO_TO_PROJECT[coco_name]
                else:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                zone = position((x1 + x2) / 2, w)
                color = COLORS[zone]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {float(box.conf):.2f} | {zone}"
                label_y = y1 - 8 if y1 > 50 else y1 + 22   # inside the box if near the top
                cv2.putText(annotated, label, (x1 + 4, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                in_frame.add(name)
            seen.update(in_frame)

            cv2.rectangle(annotated, (0, h - 34), (120, h), (0, 0, 0), -1)   # bottom-left
            cv2.putText(annotated, f"FPS: {fps:.1f}", (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            key = -1
            if not args.no_show:
                cv2.imshow("Step 2 | YOLO11 live detection (Q to quit)", annotated)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"live_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(path), annotated)
                print("Saved", path)
            if key in (ord("q"), 27) or (args.max_frames and frames >= args.max_frames):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if args.no_show and annotated is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "last_frame.jpg"), annotated)

    # Summary: useful for the baseline (Run 0) in the report
    print("\n=== Summary ===")
    print(f"Frames: {frames} | average FPS: {fps:.1f}")
    print("Detected in % of frames:")
    names = sorted(set(COCO_TO_PROJECT.values()) | set(seen)) if not (own_model or args.all) else sorted(seen)
    for name in names:
        pct = 100 * seen[name] / frames if frames else 0
        print(f"  {name:10s} {pct:5.1f}%")
    if not (own_model or args.all):
        print("  cable      not in COCO -> cannot be detected before our training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())