"""Export the water and fire models for the visitor's browser, so real videos can be checked for danger."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxslim", "onnxruntime",
                "segmentation-models-pytorch"], check=True)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
WATER_SIZE = (384, 640)  # height and width, both dividing by 32 for the U-Net
FIRE_SIZE = 640


def mounted(slug):
    """Kaggle mounts attached data at any of these paths, so look in all of them."""
    return next(p for pattern in (slug, f"datasets/*/{slug}", f"notebooks/*/{slug}") for p in INPUT.glob(pattern))


water = smp.Unet("resnet34", encoder_weights=None, classes=1)
water.load_state_dict(torch.load(mounted("dronesight-water-mixed") / "water_unet.pt", map_location="cpu"))
water.eval()
blank = torch.zeros(1, 3, *WATER_SIZE)
torch.onnx.export(water, blank, OUT / "water.onnx", input_names=["images"], output_names=["water"], opset_version=17, dynamo=False)
session = ort.InferenceSession(OUT / "water.onnx", providers=["CPUExecutionProvider"])
with torch.inference_mode():
    gap = np.abs(session.run(None, {"images": blank.numpy()})[0] - water(blank).numpy()).max()
print("water output", session.get_outputs()[0].shape, "largest gap from PyTorch", gap, flush=True)
assert gap < 1e-2

best = Path("/tmp/fire.pt")  # export writes next to the weights, and /kaggle/input is read-only
shutil.copy(mounted("dronesight-experiment-4") / "runs/fire/weights/best.pt", best)
fire = YOLO(str(best))
print("fire classes", fire.names, flush=True)
shutil.copy(fire.export(format="onnx", imgsz=FIRE_SIZE, simplify=True, device="cpu"), OUT / "fire.onnx")
for path in ["water.onnx", "fire.onnx"]:
    print(path, round((OUT / path).stat().st_size / 1e6, 1), "MB", flush=True)
