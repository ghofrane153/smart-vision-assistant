"""Step 1 - Environment check for the Smart Vision Assistant project.

Run from the repository root with the project virtual environment:
    Windows : .venv\\Scripts\\python.exe scripts\\00_check_environment.py
    Linux   : .venv/bin/python scripts/00_check_environment.py

Options:
    --no-camera   skip the camera test (no webcam available)
    --no-midas    skip the MiDaS download/test (no internet)
    --source 1    use another camera index or a stream URL
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
import time

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, info: str = "") -> None:
    results.append((name, ok, info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {info}" if info else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the project environment")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-midas", action="store_true")
    parser.add_argument("--source", default="0", help="Camera index or stream URL")
    args = parser.parse_args()

    # 1. Python
    section("1. Python")
    v = sys.version_info
    check("Python >= 3.11", v >= (3, 11), f"{platform.python_version()} ({sys.executable})")
    in_venv = sys.prefix != sys.base_prefix
    check("Running inside a virtual environment", in_venv,
          "OK" if in_venv else "use the .venv Python, not the system Python")

    # 2. Packages
    section("2. Packages")
    packages = {
        "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
        "torch": "torch", "torchvision": "torchvision", "ultralytics": "ultralytics",
        "opencv-python": "cv2", "Pillow": "PIL", "timm": "timm", "pyttsx3": "pyttsx3",
    }
    for pip_name, module in packages.items():
        try:
            mod = importlib.import_module(module)
            check(f"import {module}", True, getattr(mod, "__version__", "installed"))
        except Exception as ex:  # noqa: BLE001
            check(f"import {module}", False, f"pip install {pip_name}  ({ex.__class__.__name__})")

    # 3. Hardware
    section("3. Compute device")
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            info = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device, info = "mps", "Apple GPU"
        else:
            device, info = "cpu", "no GPU found: inference OK, use Google Colab for training"
        check("PyTorch device", True, f"{device} - {info}")
    except Exception as ex:  # noqa: BLE001
        check("PyTorch device", False, str(ex))

    # 4. YOLO11
    section("4. YOLO11 inference")
    try:
        import numpy as np
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")  # downloads ~5 MB the first time
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        t = time.perf_counter()
        model.predict(img, imgsz=416, device="cpu", verbose=False)
        ms = (time.perf_counter() - t) * 1000
        check("YOLO11n loads and runs", True, f"{len(model.names)} COCO classes, {ms:.0f} ms/frame (first call)")
    except Exception as ex:  # noqa: BLE001
        check("YOLO11n loads and runs", False, str(ex))

    # 5. MiDaS
    section("5. MiDaS depth estimation")
    if args.no_midas:
        print("[SKIP] MiDaS test skipped (--no-midas)")
    else:
        try:
            import numpy as np
            import torch
            midas = torch.hub.load("intel-isl/MiDaS:master", "MiDaS_small", trust_repo=True)
            midas.eval()
            transforms = torch.hub.load("intel-isl/MiDaS:master", "transforms", trust_repo=True)
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            with torch.no_grad():
                t = time.perf_counter()
                depth = midas(transforms.small_transform(img))
                ms = (time.perf_counter() - t) * 1000
            check("MiDaS_small loads and runs", True, f"output {tuple(depth.shape)}, {ms:.0f} ms/frame")
        except Exception as ex:  # noqa: BLE001
            check("MiDaS_small loads and runs", False, f"{ex.__class__.__name__}: {ex}")

    # 6. Camera
    section("6. Camera")
    if args.no_camera:
        print("[SKIP] Camera test skipped (--no-camera)")
    else:
        try:
            import cv2
            source = int(args.source) if args.source.isdigit() else args.source
            cap = cv2.VideoCapture(source)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            check(f"Camera '{args.source}' gives a frame", ok,
                  f"{frame.shape[1]}x{frame.shape[0]}" if ok else "try --source 1, or close Teams/Zoom")
        except Exception as ex:  # noqa: BLE001
            check("Camera", False, str(ex))

    # 7. Text-to-speech
    section("7. Text-to-speech")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say("Environment check. A chair on your left.")
        engine.runAndWait()
        check("pyttsx3 speaks", True, "you should have heard a sentence")
    except Exception as ex:  # noqa: BLE001
        hint = " (Linux: sudo apt install espeak-ng)" if platform.system() == "Linux" else ""
        check("pyttsx3 speaks", False, f"{ex.__class__.__name__}: {ex}{hint}")

    # Summary
    failed = [name for name, ok, _ in results if not ok]
    print("\n" + "=" * 50)
    print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} checks passed")
    for name in failed:
        print(f"  - FAIL: {name}")
    print("Environment ready." if not failed else "Fix the FAIL lines above, then run again.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
