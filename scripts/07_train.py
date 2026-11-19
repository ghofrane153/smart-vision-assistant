"""Step 6 - Train YOLO11 on our dataset, evaluate it, and record the run.

Each run:
    - trains (fine-tunes) a pretrained YOLO11 model on data/dataset/data.yaml
    - evaluates the best weights on the VALIDATION and TEST sets
    - copies the best weights to models/<name>_best.pt
    - adds one line to runs/experiments.csv (settings + metrics + your notes)
      -> this table is the documentation of the training runs for the report

Run from the repository root:
    Windows : .venv\\Scripts\\python.exe scripts\\07_train.py --name run1 --notes "first training, default settings"
    Linux   : .venv/bin/python scripts/07_train.py --name run1 --notes "first training, default settings"

Quick test first (1 epoch, a few minutes) to check everything works and see the time per epoch:
    .venv\\Scripts\\python.exe scripts\\07_train.py --name test --epochs 1
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


def evaluate(weights: Path, data: str, split: str, imgsz: int, device: str):
    model = YOLO(str(weights))
    m = model.val(data=data, split=split, imgsz=imgsz, device=device, plots=True, verbose=False,
                  project=str(Path("runs/eval").resolve()), name=f"{weights.parent.parent.name}_{split}",
                  exist_ok=True)
    per_class = {model.names[c]: float(m.box.maps[i]) for i, c in enumerate(m.box.ap_class_index)}
    ap50_per_class = {model.names[c]: float(m.box.ap50[i]) for i, c in enumerate(m.box.ap_class_index)}
    return {
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
    }, ap50_per_class, per_class


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate YOLO11 on our dataset")
    parser.add_argument("--name", required=True, help="Run name, e.g. run1, run2_more_data")
    parser.add_argument("--notes", default="", help="What changed in this run and why")
    parser.add_argument("--data", default="data/dataset/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt (fast) or yolo11s.pt (more accurate)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15, help="Stop if no improvement for N epochs")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0 (first GPU)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    # augmentation (change these in a later run and write why in --notes)
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="Brightness variation")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic probability")
    parser.add_argument("--degrees", type=float, default=0.0, help="Random rotation (degrees)")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"{args.data} not found. Run scripts/06_build_dataset.py first.")
        return 1
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device: {device}" + ("" if device != "cpu" else
          "  (CPU: training is slow. Run a 1-epoch test first to see the time per epoch.)"))

    run_dir = Path("runs/train") / args.name
    if run_dir.exists():
        print(f"{run_dir} already exists. Choose another --name (each run must keep its results).")
        return 1

    t0 = time.time()
    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, device=device, workers=args.workers, seed=args.seed,
        project=str(Path("runs/train").resolve()), name=args.name, exist_ok=False, plots=True,
        fliplr=args.fliplr, hsv_v=args.hsv_v, mosaic=args.mosaic, degrees=args.degrees,
    )
    minutes = (time.time() - t0) / 60

    run_dir = Path(model.trainer.save_dir)          # real output folder of this run
    best = run_dir / "weights" / "best.pt"
    Path("models").mkdir(exist_ok=True)
    shutil.copy2(best, Path("models") / f"{args.name}_best.pt")

    val, val_ap50, _ = evaluate(best, args.data, "val", args.imgsz, device)
    test, test_ap50, _ = evaluate(best, args.data, "test", args.imgsz, device)

    # record the run
    log = Path("runs/experiments.csv")
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "run": args.name, "notes": args.notes,
        "model": args.model, "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "fliplr": args.fliplr, "hsv_v": args.hsv_v, "mosaic": args.mosaic, "degrees": args.degrees,
        "train_minutes": round(minutes, 1),
        **{f"val_{k}": round(v, 3) for k, v in val.items()},
        **{f"test_{k}": round(v, 3) for k, v in test.items()},
        **{f"test_AP50_{c}": round(v, 3) for c, v in test_ap50.items()},
    }
    old_rows = []
    if log.exists():
        with open(log, newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
    fields = list(dict.fromkeys([k for r in old_rows for k in r] + list(row)))
    with open(log, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(old_rows + [row])

    print("\n" + "=" * 64)
    print(f"RUN {args.name}  ({minutes:.1f} min)  - {args.notes}")
    print(f"{'':12s}{'precision':>10s}{'recall':>10s}{'mAP50':>10s}{'mAP50-95':>10s}")
    for split, m in (("validation", val), ("test", test)):
        print(f"{split:12s}" + "".join(f"{m[k]:10.3f}" for k in ("precision", "recall", "mAP50", "mAP50-95")))
    print("\nTest AP50 per class (1.0 = perfect):")
    for c, v in sorted(test_ap50.items(), key=lambda x: -x[1]):
        print(f"  {c:15s} {v:.3f}  {'#' * int(v * 30)}")
    print(f"\nWeights : models/{args.name}_best.pt")
    print(f"Curves  : {run_dir}  (results.png, confusion_matrix.png, ...)")
    print(f"Log     : {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())