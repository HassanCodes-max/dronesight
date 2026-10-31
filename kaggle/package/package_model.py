"""Pick the model the website ships, then package it.

Scores the current site model and the Experiment 2 candidates on VisDrone and HERIDAL validation
images and keeps the best. C2A is left out because its people are pasted onto scenes, and test
sets stay out of the choice. The winner is exported to ONNX for CPU inference, its real CPU speed
is measured, and a browser-playable sample flight is built from the busiest VisDrone MOT sequences.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxslim",
                "onnxruntime", "imageio-ffmpeg"], check=True)

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

INPUT = Path("/kaggle/input")
NOTEBOOKS = INPUT / "notebooks/hassanagddsbdxfg"
EXP2_RUNS = NOTEBOOKS / "dronesight-experiment-2/runs"
CANDIDATES = {"site": NOTEBOOKS / "dronesight-experiment-1/runs/s_960/weights/best.pt",
              **{run: EXP2_RUNS / run / "weights/best.pt" for run in ["train_v", "train_vh", "train_vhc"]}}
HERIDAL_VAL_PLACES = {"BLI", "JAS", "MED"}  # from experiment2_results.json
HERIDAL_LONG_SIDE = 1280
EXP2_TEST = {"train_vh": {"mAP50-95": 0.16299, "mAP50": 0.40704}}  # VisDrone test, by Ultralytics
OUT = Path("/kaggle/working")
IMGSZ = 960
SITE_CONF = 0.25
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
SOURCE_FPS = 25  # VisDrone-VID frames; the published sequences are roughly this rate
CLIP_SEQUENCES = 3


def mounted(slug):
    """Kaggle mounts attached datasets at either of these paths, so look in both."""
    return next(p for pattern in (slug, f"datasets/*/{slug}") for p in INPUT.glob(pattern))


HERIDAL = mounted("heridal-4k")
FRAMES = mounted("visdrone-2019-mot-val-yolo") / "images"


def read_boxes(label, shape, keep=None):
    """YOLO label file to pixel xyxy boxes, keeping only source classes in `keep`."""
    height, width = shape[:2]
    boxes = []
    if label.exists():
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and (keep is None or int(float(parts[0])) in keep):
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([(cx - w / 2) * width, (cy - h / 2) * height,
                              (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def load_visdrone(folder):
    images = []
    for path in sorted(Path(folder).glob("*.jpg")):
        image = cv2.imread(str(path))
        head, tail = str(path).rsplit("/images/", 1)
        label = Path(head) / "labels" / Path(tail).with_suffix(".txt")
        images.append((image, read_boxes(label, image.shape, keep={0, 1})))  # pedestrian, people
    return images


def load_heridal_val():
    images = []
    for split in ["train", "valid", "test"]:
        for path in sorted((HERIDAL / split / "images").glob("*.jpg")):
            name = re.sub(r"^(train|valid|test)_", "", path.name)
            if re.match(r"[A-Za-z]+", name).group() not in HERIDAL_VAL_PLACES:
                continue
            image = cv2.imread(str(path))
            scale = HERIDAL_LONG_SIDE / max(image.shape[:2])
            if scale < 1:  # same resize Experiment 2 trained on
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            images.append((image, read_boxes(HERIDAL / split / "labels" / f"{path.stem}.txt", image.shape)))
    return images


def average_precision(hits, total):
    recall = np.cumsum(hits) / max(total, 1)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.interp(np.linspace(0, 1, 101), recall, envelope, right=0).mean())


def score(model, images):
    """One-class mAP, plus precision and recall at the threshold the site really uses.

    NMS ignores class, as the site does, so the 10-class site model counts its pedestrian and
    people outputs as one `person`.
    """
    confidences, hits, total = [], [], 0
    for image, truth in images:
        result = model.predict(image, imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, classes=[0, 1],
                               agnostic_nms=True, device=0, verbose=False)[0]
        conf = result.boxes.conf.cpu()
        order = conf.argsort(descending=True)
        overlap = box_iou(result.boxes.xyxy.cpu()[order], torch.from_numpy(truth)).numpy()
        matched = np.zeros((len(order), len(IOU_THRESHOLDS)), dtype=bool)
        for t, threshold in enumerate(IOU_THRESHOLDS):
            free = np.ones(len(truth), dtype=bool)
            for p in range(len(order)):
                if not free.any():
                    break
                best = (overlap[p] * free).argmax()
                if overlap[p, best] * free[best] >= threshold:
                    matched[p, t] = True
                    free[best] = False
        confidences.append(conf[order].numpy())
        hits.append(matched)
        total += len(truth)
    confidences = np.concatenate(confidences)
    hits = np.concatenate(hits)
    order = np.argsort(-confidences, kind="stable")
    confidences, hits = confidences[order], hits[order]
    per_threshold = [average_precision(hits[:, t], total) for t in range(len(IOU_THRESHOLDS))]
    kept = hits[confidences >= SITE_CONF, 0]
    return {"mAP50-95": float(np.mean(per_threshold)), "mAP50": per_threshold[0],
            "site_precision": float(kept.mean()) if kept.size else 0.0,
            "site_recall": float(kept.sum() / max(total, 1)), "people": total, "images": len(images)}


visdrone = check_det_dataset("VisDrone.yaml")
domains = {"visdrone": load_visdrone(visdrone["val"]), "heridal": load_heridal_val()}
assert all(domains.values()), {name: len(images) for name, images in domains.items()}

# The evaluator is our own, so it must first reproduce a score Ultralytics already reported.
check_model = YOLO(str(CANDIDATES["train_vh"]))
evaluator_check = {"ours": score(check_model, load_visdrone(visdrone["test"])), "ultralytics": EXP2_TEST["train_vh"]}
print(json.dumps(evaluator_check, indent=2))
assert abs(evaluator_check["ours"]["mAP50-95"] - EXP2_TEST["train_vh"]["mAP50-95"]) < 0.01, evaluator_check

validation = {}
for name, weights in CANDIDATES.items():
    model = YOLO(str(weights))
    validation[name] = {domain: score(model, images) for domain, images in domains.items()}
    validation[name]["mean_mAP50-95"] = float(np.mean([validation[name][d]["mAP50-95"] for d in domains]))
    print(name, json.dumps(validation[name]))
winner = max(validation, key=lambda name: validation[name]["mean_mAP50-95"])
print("winner:", winner)

# Export writes next to the weights, and /kaggle/input is read-only.
best = Path("/tmp/best.pt")
shutil.copy(CANDIDATES[winner], best)
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
    results = model.predict([str(f) for f in sample], imgsz=IMGSZ, conf=SITE_CONF, classes=[0, 1],
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
shutil.rmtree(OUT / "datasets", ignore_errors=True)  # Ultralytics downloads here, not to a temp folder

report = {"winner": winner, "source_weights": str(CANDIDATES[winner]), "imgsz": IMGSZ, "validation": validation,
          "evaluator_check": evaluator_check, "cpu_ms_per_frame": cpu_ms,
          "onnx_mb": (OUT / "person.onnx").stat().st_size / 1e6,
          "video_mb": (OUT / "flight.mp4").stat().st_size / 1e6,
          "sequences": busiest, "frames": len(clip), "mean_people_by_sequence": scores}
(OUT / "package_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
