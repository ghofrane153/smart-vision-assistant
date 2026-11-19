"""Step 8 - Smart Vision Assistant: live camera -> detection -> distance -> spoken description.

Pipeline for every frame:
    1. YOLO11 (our trained model) detects the objects
    2. MiDaS estimates the relative depth of the scene -> each object gets a distance level
       (very close / close / far). MiDaS gives RELATIVE depth, not meters.
    3. Each object gets a direction: on your left / in front of you / on your right
    4. Priority: closest objects, objects in front, dangerous classes (cable, stairs) first
    5. A short sentence is spoken (offline text-to-speech), without repeating the same message

Keys in the window:  Q quit | M mute/unmute | D show depth map | S save a screenshot

Run from the repository root:
    Windows : .venv\\Scripts\\python.exe scripts\\11_assistant.py
    Linux   : .venv/bin/python scripts/11_assistant.py

Options:
    --model models/run3_clean_labels_best.pt   trained model (default: the best run)
    --source 1 | video.mp4 | http://<phone-ip>:4747/video   camera index, video file, phone stream
    --hybrid                                   pretrained YOLO11 for person/chair/laptop/pc + our model for
                                               cable/door/stairs, and BOTH models for table (--both table)
    --flip                                     when testing with the laptop webcam FACING you
    --class-conf stairs=0.7,door=0.5           stricter thresholds against false alarms
    --min-frames 4                             announce an object only if seen 4 frames in a row
    --lang fr                                  French sentences (needs a French Windows voice)
    --no-depth                                 no MiDaS: distance from the box size (faster)
    --record runs/assistant/demo.mp4           save the annotated video (for the presentation)
    --no-voice | --no-show                     for tests
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DANGER = {"cable", "stairs"}                      # announced with "Warning"
TEXT = {
    "en": {"left": "on your left", "center": "in front of you", "right": "on your right",
           "very close": "very close", "close": "close", "far": "further away",
           "warning": "Warning", "nothing": "Path clear",
           "names": {"pc": "computer", "trash_can": "trash can"}},
    "fr": {"left": "à gauche", "center": "devant vous", "right": "à droite",
           "very close": "très proche", "close": "proche", "far": "plus loin",
           "warning": "Attention", "nothing": "Voie libre",
           "names": {"person": "une personne", "chair": "une chaise", "table": "une table",
                     "laptop": "un ordinateur portable", "pc": "un ordinateur", "cable": "un câble",
                     "door": "une porte", "stairs": "des escaliers", "backpack": "un sac à dos",
                     "handbag": "un sac", "bottle": "une bouteille"}},
}
COCO_TO_OURS = {"person": "person", "chair": "chair", "dining table": "table", "laptop": "laptop", "tv": "pc"}
COLORS = {"very close": (0, 0, 255), "close": (0, 165, 255), "far": (0, 200, 0)}


# --------------------------------------------------------------------------- speech (own thread)
class Speaker:
    """Speaks each message in a separate small Python process (reliable on Windows: the SAPI voice
    often stays silent when it is used from a thread). The video never waits for the voice."""

    SCRIPT = (
        "import sys, pyttsx3\n"
        "text, rate, lang = sys.argv[1], int(sys.argv[2]), sys.argv[3]\n"
        "e = pyttsx3.init()\n"
        "e.setProperty('rate', rate)\n"
        "if lang == 'fr':\n"
        "    for v in e.getProperty('voices'):\n"
        "        if 'fr' in (v.id + v.name).lower() or 'french' in v.name.lower():\n"
        "            e.setProperty('voice', v.id); break\n"
        "e.say(text)\n"
        "e.runAndWait()\n"
    )

    def __init__(self, enabled: bool, lang: str, rate: int):
        import subprocess
        import sys
        self.subprocess, self.python = subprocess, sys.executable
        self.enabled, self.muted, self.lang, self.rate = enabled, False, lang, rate
        self.proc = None
        self.ok = False
        if not enabled:
            return
        try:   # check once that a voice exists
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            self.ok = True
            if lang == "fr" and not any("fr" in (v.id + v.name).lower() or "french" in v.name.lower()
                                        for v in voices):
                print("No French voice installed: the default voice reads the French text "
                      "(use --lang en, or install a French voice).")
            engine.stop()
        except Exception as ex:  # noqa: BLE001
            print(f"Text-to-speech not available ({ex.__class__.__name__}): messages are printed only.")

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def say(self, text: str):
        print(f"[{datetime.now():%H:%M:%S}] SAY: {text}")
        if not (self.enabled and self.ok) or self.muted:
            return
        if self.proc is not None and self.proc.poll() not in (None, 0):
            err = self.proc.stderr.read().decode(errors="ignore").strip().splitlines()
            print("Voice error:", err[-1] if err else self.proc.returncode)
        if self.busy():
            return
        self.proc = self.subprocess.Popen([self.python, "-c", self.SCRIPT, text, str(self.rate), self.lang],
                                          stdout=self.subprocess.DEVNULL, stderr=self.subprocess.PIPE)


# --------------------------------------------------------------------------- depth (MiDaS)
class Depth:
    def __init__(self, device: str, repo: str):
        src = "local" if Path(repo).exists() else "github"
        self.model = torch.hub.load(repo, "MiDaS_small", source=src, trust_repo=True).to(device).eval()
        self.transform = torch.hub.load(repo, "transforms", source=src, trust_repo=True).small_transform
        self.device = device

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pred = self.model(self.transform(rgb).to(self.device))
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic",
                                               align_corners=False).squeeze()
        d = pred.cpu().numpy()                      # inverse depth: bigger = closer
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)   # 0 = far, 1 = close (relative)


def box_iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def distance_level(closeness: float) -> str:
    if closeness > 0.70:
        return "very close"
    if closeness > 0.40:
        return "close"
    return "far"


def direction(x_center: float, width: int) -> str:
    if x_center < width / 3:
        return "left"
    if x_center > 2 * width / 3:
        return "right"
    return "center"


def describe(objects: list[dict], lang: str, max_items: int) -> str:
    t = TEXT[lang]
    if not objects:
        return t["nothing"]
    parts = []
    for o in objects[:max_items]:
        name = t["names"].get(o["name"], o["name"].replace("_", " "))
        if lang == "en":
            name = {"stairs": "stairs"}.get(o["name"], "a " + name if name[0] not in "aeiou" else "an " + name)
        prefix = f'{t["warning"]}, ' if o["name"] in DANGER and o["dist"] != "far" else ""
        dist = f', {t[o["dist"]]}' if o["dist"] != "far" else ""
        part = f'{prefix}{name} {t[o["dir"]]}{dist}'
        parts.append(part[0].upper() + part[1:])
    return ". ".join(parts) + "."


# --------------------------------------------------------------------------- main loop
def main() -> int:
    parser = argparse.ArgumentParser(description="Smart Vision Assistant (live)")
    parser.add_argument("--model", default="models/run3_clean_labels_best.pt")
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--cable-conf", type=float, default=0.15, help="lower threshold: never miss a cable")
    parser.add_argument("--class-conf", default="stairs=0.6,door=0.5,person=0.5,table=0.25",
                        help="stricter thresholds for classes with false alarms, e.g. stairs=0.6,door=0.5")
    parser.add_argument("--overlap", type=float, default=0.6,
                        help="two boxes overlapping more than this = same object, keep the most confident label")
    parser.add_argument("--min-frames", type=int, default=4,
                        help="an object is announced only if seen in this many frames in a row")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lang", default="en", choices=["en", "fr"])
    parser.add_argument("--rate", type=int, default=165, help="speech speed")
    parser.add_argument("--interval", type=float, default=3.5, help="min seconds between two messages")
    parser.add_argument("--repeat", type=float, default=10.0, help="repeat an unchanged message after N seconds")
    parser.add_argument("--max-items", type=int, default=2, help="objects per sentence")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--midas-repo", default="intel-isl/MiDaS:master", help="or a local clone folder")
    parser.add_argument("--depth-every", type=int, default=2, help="run MiDaS every N frames")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--hybrid", action="store_true",
                        help="pretrained YOLO11 (COCO) for person/chair/table/laptop/pc + our model for the rest")
    parser.add_argument("--coco-model", default="yolo11s.pt", help="pretrained model used by --hybrid")
    parser.add_argument("--both", default="table",
                        help="hybrid: classes taken from BOTH models (duplicates removed), e.g. table or table,pc")
    parser.add_argument("--flip", action="store_true",
                        help="mirror the image: use it when the laptop webcam FACES you (directions are then correct)")
    parser.add_argument("--record", default="", help="save the annotated video, e.g. runs/assistant/demo.mp4")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not Path(args.model).exists() and not args.model.startswith("yolo"):
        print(f"Model {args.model} not found. Train first (07_train.py) or use --model <file.pt>.")
        return 1
    model = YOLO(args.model)
    names = model.names
    print(f"Model: {args.model} | device: {device} | classes: {list(names.values())}")
    coco, coco_ids = None, []
    if args.hybrid:
        coco = YOLO(args.coco_model)
        coco_ids = [i for i, n in coco.names.items() if n in COCO_TO_OURS]
        print(f"HYBRID: {args.coco_model} for {sorted(set(COCO_TO_OURS.values()))}, "
              f"{args.model} for the other classes")

    depth = None
    if not args.no_depth:
        try:
            depth = Depth(device, args.midas_repo)
            print("MiDaS depth: ON")
        except Exception as ex:  # noqa: BLE001
            print(f"MiDaS not available ({ex.__class__.__name__}: {ex}). Distance from box size instead.")

    speaker = Speaker(not args.no_voice, args.lang, args.rate)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Could not open the camera/video. Try --source 1, a video file or the phone URL.")
        return 1

    writer = None
    depth_map, show_depth = None, False
    thresholds = {"cable": args.cable_conf}
    for item in filter(None, args.class_conf.split(",")):
        k, v = item.split("=")
        thresholds[k.strip()] = float(v)
    streak = {}          # class name -> number of consecutive frames where it was detected
    last_text, last_time, last_key = "", 0.0, None
    frames, fps, t_prev = 0, 0.0, time.perf_counter()
    print("Q quit | M mute | D depth map | S screenshot")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            if args.flip:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 1. detection (low global threshold, then per-class threshold)
            res = model.predict(frame, conf=min(args.conf, min(thresholds.values())), imgsz=args.imgsz,
                                device=0 if device == "cuda" else "cpu", verbose=False)[0]
            # 2. depth
            if depth is not None and (depth_map is None or frames % args.depth_every == 0):
                depth_map = depth(frame)

            dets = []   # (name, conf, xyxy)
            both = {c.strip() for c in args.both.split(",") if c.strip()}
            coco_classes = (set(COCO_TO_OURS.values()) - both) if coco is not None else set()
            for b in res.boxes:
                name = names[int(b.cls)]
                if not (coco_classes and (name in coco_classes or (name in COCO_TO_OURS and COCO_TO_OURS[name] not in both))):   # hybrid: common objects come from COCO
                    dets.append((name, float(b.conf), b.xyxy[0]))
            if coco is not None:
                rc = coco.predict(frame, conf=min(args.conf, min(thresholds.values())), imgsz=args.imgsz, classes=coco_ids,
                                  device=0 if device == "cuda" else "cpu", verbose=False)[0]
                dets += [(COCO_TO_OURS[coco.names[int(b.cls)]], float(b.conf), b.xyxy[0]) for b in rc.boxes]

            # one object = one label: if two boxes cover almost the same area (e.g. "chair" and "person"
            # on the same empty chair), keep only the most confident one, whatever its class
            dets = [d for d in dets if d[1] >= thresholds.get(d[0], args.conf)]
            dets.sort(key=lambda d: -d[1])
            kept = []
            for d in dets:
                b = [float(v) for v in d[2]]
                if all(box_iou(b, [float(v) for v in k[2]]) < args.overlap for k in kept):
                    kept.append(d)
            dets = kept

            objects = []
            for name, conf, xyxy in dets:
                x1, y1, x2, y2 = map(int, xyxy)
                if depth_map is not None:
                    region = depth_map[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
                    closeness = float(np.percentile(region, 75)) if region.size else 0.0
                else:   # fallback: a big box (especially low in the image) is close
                    closeness = min(1.0, ((y2 - y1) / h) * 0.8 + (y2 / h) * 0.3)
                o = {"name": name, "conf": conf, "box": (x1, y1, x2, y2), "closeness": closeness,
                     "dist": distance_level(closeness), "dir": direction((x1 + x2) / 2, w)}
                o["priority"] = closeness + (0.25 if o["dir"] == "center" else 0) + (0.3 if name in DANGER else 0)
                objects.append(o)
            objects.sort(key=lambda o: -o["priority"])
            present = {o["name"] for o in objects}
            for n in set(streak) | present:
                streak[n] = streak.get(n, 0) + 1 if n in present else 0
            for o in objects:
                o["stable"] = streak[o["name"]] >= args.min_frames
            stable_objects = [o for o in objects if o["stable"]]

            # 3. speech: only when the message changes, or to repeat it after a while
            now = time.time()
            if now - last_time >= args.interval and not speaker.busy():
                text = describe(stable_objects, args.lang, args.max_items)
                # what matters: which objects and how close (direction only for dangers)
                key = tuple(sorted((o["name"], o["dist"], o["dir"] if o["name"] in DANGER else "")
                                   for o in stable_objects[:args.max_items]))
                if (key != last_key or now - last_time >= args.repeat) and (stable_objects or last_key):
                    speaker.say(text)                 # (never "path clear" at start, never twice in a row)
                    last_text, last_time, last_key = text, now, key

            # 4. display
            t_now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6) if frames > 1 else 0.0
            t_prev = t_now
            if args.no_show and not args.record:
                if args.max_frames and frames >= args.max_frames:
                    break
                continue
            view = frame.copy()
            if show_depth and depth_map is not None:
                view = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            for x in (w // 3, 2 * w // 3):
                cv2.line(view, (x, 0), (x, h), (200, 200, 200), 1)
            for i, o in enumerate(objects):
                x1, y1, x2, y2 = o["box"]
                color = COLORS[o["dist"]] if o["stable"] else (160, 160, 160)   # grey = not confirmed yet
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 3 if o["stable"] and i < args.max_items else 1)
                label = f'{o["name"]} {o["conf"]:.2f} | {o["dir"]} | {o["dist"]}'
                cv2.putText(view, label, (x1 + 4, y1 + 20 if y1 < 30 else y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            status = f"FPS {fps:.1f} | {'HYBRID' if coco else 'our model'}{' | flip' if args.flip else ''} | depth {'ON' if depth else 'box size'} | voice " + \
                     ("muted" if speaker.muted else "on" if speaker.ok else "off")
            cv2.rectangle(view, (0, h - 56), (w, h), (0, 0, 0), -1)
            cv2.putText(view, status, (8, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(view, last_text[:90], (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 15
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps if src_fps < 61 else 15, (w, h))
                writer.write(view)
            key = -1
            if not args.no_show:
                cv2.imshow("Smart Vision Assistant (Q quit, M mute, D depth, S screenshot)", view)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or (args.max_frames and frames >= args.max_frames):
                break
            if key == ord("m"):
                speaker.muted = not speaker.muted
            elif key == ord("d"):
                show_depth = not show_depth
            elif key == ord("s"):
                out = Path("runs/assistant")
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"screen_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(path), view)
                print("Saved", path)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print("Video saved:", args.record)
        cv2.destroyAllWindows()
    print(f"Frames: {frames} | last FPS: {fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())