"""Turn the best Experiment 2 model into what the website ships.

Exports ONNX for CPU inference, measures real CPU speed, and builds a browser-playable sample
flight from the VisDrone MOT val sequence where the model finds the most people.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxslim",
                "onnxruntime", "imageio-ffmpeg"], check=True)

import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from ultralytics import YOLO  # noqa: E402

NOTEBOOKS = Path("/kaggle/input/notebooks/hassanagddsbdxfg")
CANDIDATES = [NOTEBOOKS / "dronesight-experiment-2/runs/train_vhc/weights/best.pt",
              NOTEBOOKS / "dronesight-experiment-1/runs/s_960/weights/best.pt"]
FRAMES = Path("/kaggle/input/datasets/mainhatnam/visdrone-2019-mot-val-yolo/images")
OUT = Path("/kaggle/working")
IMGSZ = 960
SOURCE_FPS = 25  # VisDrone-VID frames; the published sequences are roughly this rate
CLIP_SEQUENCES = 3

source = next(p for p in CANDIDATES if p.exists())
# Export writes next to the weights, and /kaggle/input is read-only.
best = Path("/tmp/best.pt")
shutil.copy(source, best)
model = YOLO(str(best))
onnx_path = Path(model.export(format="onnx", imgsz=IMGSZ, simplify=True, device="cpu"))
shutil.copy(onnx_path, OUT / "person.onnx")

session = ort.InferenceSession(str(OUT / "person.onnx"), providers=["CPUExecutionProvider"])
blob = np.zeros((1, 3, IMGSZ, IMGSZ), dtype=np.float32)
session.run(None, {session.get_inputs()[0].name: blob})
start = time.perf_counter()
for _ in range(5):
    session.run(None, {session.get_inputs()[0].name: blob})
cpu_ms = (time.perf_counter() - start) / 5 * 1000

sequences = defaultdict(list)
for frame in sorted(FRAMES.glob("*.jpg")):
    sequences[frame.name.rsplit("_", 1)[0]].append(frame)

scores = {}
for name, frames in sequences.items():
    sample = frames[:: max(1, len(frames) // 8)][:8]
    results = model.predict([str(f) for f in sample], imgsz=IMGSZ, conf=0.25, classes=[0, 1],
                            device=0, verbose=False)
    scores[name] = float(np.mean([len(r.boxes) for r in results]))


busiest = sorted(scores, key=scores.get, reverse=True)[:CLIP_SEQUENCES]
clip = [frame for name in busiest for frame in sequences[name]]
listing = OUT / "frames.txt"
listing.write_text("".join(f"file '{f}'\nduration {1 / SOURCE_FPS}\n" for f in clip))
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "slow", "-crf", "30",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(OUT / "flight.mp4")], check=True)
listing.unlink()

report = {"source_weights": str(source), "imgsz": IMGSZ, "cpu_ms_per_frame": cpu_ms,
          "onnx_mb": (OUT / "person.onnx").stat().st_size / 1e6,
          "video_mb": (OUT / "flight.mp4").stat().st_size / 1e6,
          "sequences": busiest, "frames": len(clip), "mean_people_by_sequence": scores}
(OUT / "package_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
