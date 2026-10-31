"""Export a small general person detector for the visitor's browser.

The main people model only learned small figures seen from far above, so on real clips it misses people
close to the camera, in crowds or in low-quality video. YOLO11n's COCO weights learned everyday photos,
which covers those. It is not trained here, and is checked by eye on real videos.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxslim"], check=True)

from ultralytics import YOLO  # noqa: E402

OUT = Path("/kaggle/working")
SIZE = 640

shutil.copy(YOLO("yolo11n.pt").export(format="onnx", imgsz=SIZE, simplify=True, device="cpu"), OUT / "close.onnx")
print("exported", (OUT / "close.onnx").stat().st_size)
