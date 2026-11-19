"""Step 3 - Capture photos for our dataset, organized by SESSION.

One session = one person, one room, one moment (e.g. "ghofrane_room_b204_morning").
Photos of the same session look alike, so later the whole session goes into the SAME split
(train, val or test). This avoids data leakage.

Keys in the window:
    SPACE  save one photo
    A      start / stop automatic capture (one photo every --interval seconds)
    Q/ESC  quit

Run from the repository root:
    Windows : .venv\\Scripts\\python.exe scripts\\02_capture_photos.py --session ghofrane_b204_morning
    Linux   : .venv/bin/python scripts/02_capture_photos.py --session ghofrane_b204_morning

Phone camera (DroidCam / IP Webcam):
    --source http://192.168.1.10:4747/video
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture dataset photos by session")
    parser.add_argument("--session", required=True,
                        help="name_room_moment, e.g. ghofrane_b204_morning")
    parser.add_argument("--source", default="0", help="Camera index, video file or stream URL")
    parser.add_argument("--output", default="data/raw", help="Root folder for the sessions")
    parser.add_argument("--interval", type=float, default=1.5,
                        help="Seconds between photos in automatic mode (avoid near-duplicates)")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--no-show", action="store_true", help="No window: automatic capture (testing)")
    parser.add_argument("--max", type=int, default=0, help="Stop after N photos (0 = no limit)")
    args = parser.parse_args()

    session = re.sub(r"[^a-zA-Z0-9_-]", "_", args.session.strip().lower())
    out_dir = Path(args.output) / session
    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(out_dir.glob("*.jpg")))  # continue numbering if the session exists

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("Could not open the camera. Try --source 1, or close Teams/Zoom.")
        return 1

    auto = args.no_show
    last_auto = 0.0
    flash = 0
    print(f"Session folder: {out_dir}  ({count} photo(s) already there)")
    print("SPACE = photo | A = automatic on/off | Q = quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            save = False
            now = time.time()
            if auto and now - last_auto >= args.interval:
                save, last_auto = True, now

            key = -1
            if not args.no_show:
                display = frame.copy()
                h, w = display.shape[:2]
                if flash:  # white border = photo taken
                    cv2.rectangle(display, (0, 0), (w - 1, h - 1), (255, 255, 255), 12)
                    flash -= 1
                status = f"{session} | photos: {count} | auto: {'ON' if auto else 'off'}"
                cv2.rectangle(display, (0, 0), (w, 36), (0, 0, 0), -1)
                cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 0) if auto else (255, 255, 255), 2)
                cv2.imshow("Step 3 | Capture (SPACE photo, A auto, Q quit)", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    save = True
                elif key in (ord("a"), ord("A")):
                    auto, last_auto = not auto, 0.0
                    print("Automatic capture", "ON" if auto else "OFF")

            if save:
                count += 1
                path = out_dir / f"{session}_{count:04d}.jpg"
                cv2.imwrite(str(path), frame)  # original frame, without the text overlay
                flash = 3
                print("Saved", path)

            if key in (ord("q"), 27) or (args.max and count >= args.max):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nSession '{session}': {count} photo(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())