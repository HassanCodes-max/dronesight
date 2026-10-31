"""Experiment 3, part 3: score the chosen people model once on its test sets.

vh_1280 was chosen on validation data alone, so these images have never steered a decision. It runs
through the site's shrinking and NMS, on VisDrone test-dev and the four HERIDAL places Experiment 2
held out. Nothing is chosen from this result.
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
OUT = Path("/kaggle/working")
HERIDAL_TEST_PLACES = {"BRA", "BRK", "TRS", "ZRI"}  # from experiment2_results.json
IMGSZ = 1280
SITE_CONF = 0.25
SITE_NMS_IOU = 0.5
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def mounted(slug):
    """Kaggle mounts attached data at any of these paths, so look in all of them."""
    return next(p for pattern in (slug, f"datasets/*/{slug}", f"notebooks/*/{slug}") for p in INPUT.glob(pattern))


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


def visdrone_test():
    for path in sorted(Path(check_det_dataset("VisDrone.yaml")["test"]).glob("*.jpg")):
        image = cv2.imread(str(path))
        head, tail = str(path).rsplit("/images/", 1)
        yield image, read_boxes(Path(head) / "labels" / Path(tail).with_suffix(".txt"), image.shape, keep={0, 1})  # pedestrian, people


def heridal_test():
    root = mounted("heridal-4k")
    for split in ["train", "valid", "test"]:
        for path in sorted((root / split / "images").glob("*.jpg")):
            name = re.sub(r"^(train|valid|test)_", "", path.name)
            if re.match(r"[A-Za-z]+", name).group() in HERIDAL_TEST_PLACES:
                image = cv2.imread(str(path))
                yield image, read_boxes(root / split / "labels" / f"{path.stem}.txt", image.shape)


def detect(model, image):
    scale = IMGSZ / max(image.shape[:2])
    shrink = scale < 1
    if shrink:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    result = model.predict(image, imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, classes=[0, 1],
                           agnostic_nms=True, device=0, verbose=False)[0]
    boxes, scores = result.boxes.xyxy.cpu(), result.boxes.conf.cpu()
    keep = nms(boxes, scores, SITE_NMS_IOU)  # sorted by score
    return boxes[keep] / (scale if shrink else 1), scores[keep]


def average_precision(hits, total):
    recall = np.cumsum(hits) / max(total, 1)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.interp(np.linspace(0, 1, 101), recall, envelope, right=0).mean())


def score(model, images):
    confidences, hits, total, count = [], [], 0, 0
    for image, truth in images:
        boxes, conf = detect(model, image)
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
        count += 1
    confidences = np.concatenate(confidences)
    hits = np.concatenate(hits)
    order = np.argsort(-confidences, kind="stable")
    confidences, hits = confidences[order], hits[order]
    per_threshold = [average_precision(hits[:, t], total) for t in range(len(IOU_THRESHOLDS))]
    kept = hits[confidences >= SITE_CONF, 0]
    return {"images": count, "people": total, "mAP50-95": float(np.mean(per_threshold)), "mAP50": per_threshold[0],
            "site_precision": float(kept.mean()) if kept.size else 0.0,
            "site_recall": float(kept.sum() / max(total, 1))}


model = YOLO(str(mounted("dronesight-experiment-3-train") / "runs/vh_1280/weights/best.pt"))
results = {"model": "vh_1280 best.pt", "imgsz": IMGSZ}
for name, images in {"visdrone": visdrone_test(), "heridal": heridal_test()}.items():
    results[name] = score(model, images)
    print(name, json.dumps(results[name]), flush=True)
    (OUT / "experiment3_test_results.json").write_text(json.dumps(results, indent=2))

shutil.rmtree(OUT / "datasets", ignore_errors=True)  # Ultralytics downloads here, not to a temp folder
print(json.dumps(results, indent=2))
