"""Experiment 3, part 1: how far can inference settings alone go on tiny people?

No training. Runs the two best Experiment 2 models whole-frame at 960 and 1280, and on overlapping
tiles of the full-resolution image, on VisDrone and HERIDAL validation images. Tiling makes each
person larger on the model's input but costs one model pass per tile, so passes are recorded.
Also finds the confidence with the best F1. Test sets are not touched.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.ops import nms  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

INPUT = Path("/kaggle/input")
EXP2_RUNS = INPUT / "notebooks/hassanagddsbdxfg/dronesight-experiment-2/runs"
MODELS = ["train_vhc", "train_vh"]
HERIDAL_VAL_PLACES = {"BLI", "JAS", "MED"}  # from experiment2_results.json
OUT = Path("/kaggle/working")
RESULTS = OUT / "experiment3a_results.json"
SITE_CONF = 0.25
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
TILE_OVERLAP = 0.2
MERGE_IOU = 0.5
MAX_DETECTIONS = 1000
# name: (image size, include whole frame, tile columns, tile rows)
ARMS = {"full_960": (960, True, 0, 0),
        "full_1280": (1280, True, 0, 0),
        "tiles_2x2": (960, False, 2, 2),
        "full_plus_2x2": (960, True, 2, 2),
        "tiles_3x2": (960, False, 3, 2)}
EXPECTED = {"train_vhc": {"visdrone": 0.26596}, "train_vh": {"visdrone": 0.29616}}  # full_960, by package_model.py


def mounted(slug):
    """Kaggle mounts attached datasets at either of these paths, so look in both."""
    return next(p for pattern in (slug, f"datasets/*/{slug}") for p in INPUT.glob(pattern))


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
    root = mounted("heridal-4k")
    images = []
    for split in ["train", "valid", "test"]:
        for path in sorted((root / split / "images").glob("*.jpg")):
            name = re.sub(r"^(train|valid|test)_", "", path.name)
            if re.match(r"[A-Za-z]+", name).group() in HERIDAL_VAL_PLACES:
                image = cv2.imread(str(path))  # kept at full resolution, which tiling needs
                images.append((image, read_boxes(root / split / "labels" / f"{path.stem}.txt", image.shape)))
    return images


def tiles(width, height, columns, rows):
    """Overlapping (x1, y1, x2, y2) windows covering the image."""
    tile_w = width / (columns - (columns - 1) * TILE_OVERLAP)
    tile_h = height / (rows - (rows - 1) * TILE_OVERLAP)
    return [(round(c * tile_w * (1 - TILE_OVERLAP)), round(r * tile_h * (1 - TILE_OVERLAP)),
             round(c * tile_w * (1 - TILE_OVERLAP) + tile_w), round(r * tile_h * (1 - TILE_OVERLAP) + tile_h))
            for r in range(rows) for c in range(columns)]


def detect(model, image, imgsz, whole, columns, rows):
    height, width = image.shape[:2]
    windows = ([(0, 0, width, height)] if whole else []) + (tiles(width, height, columns, rows) if columns else [])
    boxes, scores = [], []
    for x1, y1, x2, y2 in windows:
        result = model.predict(image[y1:y2, x1:x2], imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                               classes=[0, 1], agnostic_nms=True, device=0, verbose=False)[0]
        boxes.append(result.boxes.xyxy.cpu() + torch.tensor([x1, y1, x1, y1], dtype=torch.float32))
        scores.append(result.boxes.conf.cpu())
    boxes, scores = torch.cat(boxes), torch.cat(scores)
    keep = nms(boxes, scores, MERGE_IOU)[:MAX_DETECTIONS]  # sorted by score
    return boxes[keep], scores[keep], len(windows)


def average_precision(hits, total):
    recall = np.cumsum(hits) / max(total, 1)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.interp(np.linspace(0, 1, 101), recall, envelope, right=0).mean())


def score(model, images, arm):
    confidences, hits, total, passes = [], [], 0, 0
    for image, truth in images:
        boxes, conf, used = detect(model, image, *arm)
        overlap = box_iou(boxes, torch.from_numpy(truth)).numpy()
        matched = np.zeros((len(boxes), len(IOU_THRESHOLDS)), dtype=bool)
        for t, threshold in enumerate(IOU_THRESHOLDS):
            free = np.ones(len(truth), dtype=bool)
            for p in range(len(boxes)):
                if not free.any():
                    break
                best = (overlap[p] * free).argmax()
                if overlap[p, best] * free[best] >= threshold:
                    matched[p, t] = True
                    free[best] = False
        confidences.append(conf.numpy())
        hits.append(matched)
        total += len(truth)
        passes = used
    confidences = np.concatenate(confidences)
    hits = np.concatenate(hits)
    order = np.argsort(-confidences, kind="stable")
    confidences, hits = confidences[order], hits[order]
    per_threshold = [average_precision(hits[:, t], total) for t in range(len(IOU_THRESHOLDS))]
    kept = hits[confidences >= SITE_CONF, 0]
    found = np.cumsum(hits[:, 0])
    precision = found / np.arange(1, len(found) + 1)
    recall = found / max(total, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(f1.argmax())
    return {"mAP50-95": float(np.mean(per_threshold)), "mAP50": per_threshold[0], "passes_per_frame": passes,
            "site_precision": float(kept.mean()) if kept.size else 0.0,
            "site_recall": float(kept.sum() / max(total, 1)),
            "best_f1": float(f1[best]), "best_f1_conf": float(confidences[best]),
            "best_f1_precision": float(precision[best]), "best_f1_recall": float(recall[best])}


domains = {"visdrone": load_visdrone(check_det_dataset("VisDrone.yaml")["val"]), "heridal": load_heridal_val()}
assert all(domains.values()), {name: len(images) for name, images in domains.items()}
print({name: (len(images), sum(len(t) for _, t in images)) for name, images in domains.items()})

results = {}
for run in MODELS:
    model = YOLO(str(EXP2_RUNS / run / "weights/best.pt"))
    results[run] = {}
    for name, arm in ARMS.items():
        entry = {domain: score(model, images, arm) for domain, images in domains.items()}
        entry["mean_mAP50-95"] = float(np.mean([entry[d]["mAP50-95"] for d in domains]))
        results[run][name] = entry
        print(run, name, json.dumps(entry), flush=True)
        RESULTS.write_text(json.dumps(results, indent=2))
        if name == "full_960":  # must reproduce the numbers the site model was chosen on
            assert abs(entry["visdrone"]["mAP50-95"] - EXPECTED[run]["visdrone"]) < 0.01, entry

shutil.rmtree(OUT / "datasets", ignore_errors=True)  # Ultralytics downloads here, not to a temp folder
print(json.dumps(results, indent=2))
