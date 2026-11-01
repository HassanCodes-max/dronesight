"""Export the three YOLO models so they take any frame shape, not only a square.

A widescreen video in a square input is about 44% grey padding, and the model works through all of it.
Taking the frame's own shape keeps every video pixel the model saw before and drops the rest.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxslim", "onnxruntime"], check=True)

import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402
from ultralytics import YOLO  # noqa: E402

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")


def mounted(slug):
    """Kaggle mounts attached data at any of these paths, so look in all of them."""
    return next(p for pattern in (slug, f"datasets/*/{slug}", f"notebooks/*/{slug}") for p in INPUT.glob(pattern))


def export(weights, name, size):
    path = OUT / name
    shutil.copy(YOLO(str(weights)).export(format="onnx", imgsz=size, dynamic=True, simplify=True, device="cpu"), path)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for height, width in [(size, size), (384, 640), (640, 384)]:
        output = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, height, width), np.float32)})[0]
        print(name, (height, width), "gives", output.shape, flush=True)
    print(name, path.stat().st_size, flush=True)


def copied(path, name):
    shutil.copy(path, Path("/tmp") / name)  # the mounted folder is read-only and the exporter writes beside its weights
    return Path("/tmp") / name


export(copied(mounted("dronesight-experiment-3-train") / "runs/vh_1280/weights/best.pt", "person.pt"), "person.onnx", 1280)
export("yolo11n.pt", "close.onnx", 640)
export(copied(mounted("dronesight-experiment-4") / "runs/fire/weights/best.pt", "fire.pt"), "fire.onnx", 640)
