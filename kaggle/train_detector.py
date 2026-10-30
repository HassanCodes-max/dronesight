"""Experiment 1: which small YOLO model and image size work best on VisDrone?

Runs on a Kaggle GPU. Train on the train split, pick the winner on val only, then
score the winner once on test-dev so the reported number is not tuned to the test set.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

from ultralytics import YOLO  # noqa: E402

OUT = Path("/kaggle/working")
RESULTS = OUT / "experiment1_results.json"
EPOCHS = 30
EXPERIMENTS = [
    {"name": "n_640", "weights": "yolo11n.pt", "imgsz": 640, "batch": 32},
    {"name": "n_960", "weights": "yolo11n.pt", "imgsz": 960, "batch": 16},
    {"name": "s_960", "weights": "yolo11s.pt", "imgsz": 960, "batch": 12},
]


def box_metrics(metrics):
    return {
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_mAP50-95": {metrics.names[c]: float(ap)
                               for c, ap in zip(metrics.box.ap_class_index, metrics.box.ap)},
    }


results = {}
for exp in EXPERIMENTS:
    model = YOLO(exp["weights"])
    model.train(data="VisDrone.yaml", epochs=EPOCHS, imgsz=exp["imgsz"], batch=exp["batch"],
                device=0, workers=2, seed=0, project=str(OUT / "runs"), name=exp["name"],
                exist_ok=True, plots=False)
    best = OUT / "runs" / exp["name"] / "weights" / "best.pt"
    val = YOLO(str(best)).val(data="VisDrone.yaml", split="val", imgsz=exp["imgsz"],
                              batch=exp["batch"], device=0, plots=False)
    results[exp["name"]] = {"config": exp, "epochs": EPOCHS, "val": box_metrics(val),
                            "speed_ms_per_image": val.speed, "weights": str(best)}
    # Saved after every run so a Kaggle timeout cannot lose finished experiments.
    RESULTS.write_text(json.dumps(results, indent=2))

winner = max(results, key=lambda k: results[k]["val"]["mAP50-95"])
exp = results[winner]["config"]
test = YOLO(results[winner]["weights"]).val(data="VisDrone.yaml", split="test", imgsz=exp["imgsz"],
                                            batch=exp["batch"], device=0, plots=False)
results["winner"] = {"name": winner, "test": box_metrics(test)}
RESULTS.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
