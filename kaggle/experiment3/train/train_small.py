"""Experiment 3, part 2: train for tiny people.

Part 1 showed a 1280 input helps and tiling does not, and that C2A causes false alarms at higher
detail, so C2A is dropped. HERIDAL is repeated three times because it has only 848 labelled people.
Fine-tunes the Experiment 2 VisDrone + HERIDAL model two ways:

- vh_1280: the same YOLO11s at 1280
- vh_960_p2: YOLO11s at 960 with an extra stride-4 detection layer for small objects

Each run keeps best.pt and last.pt, because VisDrone outweighs HERIDAL 29 to 1 in mixed validation
and so largely decides best.pt. All are scored through the site's shrinking and NMS on each
validation set. Test sets are not touched.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torchvision.ops import nms  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

SMOKE = False  # True runs a short end-to-end trial

INPUT = Path("/kaggle/input")
WEIGHTS = INPUT / "notebooks/hassanagddsbdxfg/dronesight-experiment-2/runs/train_vh/weights/best.pt"
HERIDAL_SPLITS = {"BLI": "val", "JAS": "val", "MED": "val", "BRA": "test", "BRK": "test", "TRS": "test",
                  "ZRI": "test"}  # the rest are train, as in experiment2_results.json
HERIDAL_LONG_SIDE = 1280
HERIDAL_REPEATS = 3
DS = Path("/tmp/ds")  # outside /kaggle/working so the data is not saved as notebook output
OUT = Path("/kaggle/working")
RESULTS = OUT / "experiment3c_results.json"
SEED = 0
EPOCHS = 1 if SMOKE else 15
FRACTION = 0.02 if SMOKE else 1.0
SITE_CONF = 0.25
SITE_NMS_IOU = 0.5
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
# name: (image size, batch, extra stride-4 layer)
RUNS = {"vh_960_p2": (960, 8, True), "vh_1280": (1280, 6, False)}
P2_MODEL = """
nc: 1
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[19, 22, 25, 28], 1, Detect, [nc]]
"""


def mounted(slug):
    """Kaggle mounts attached datasets at either of these paths, so look in both."""
    return next(p for pattern in (slug, f"datasets/*/{slug}") for p in INPUT.glob(pattern))


def label_lines(label, keep=None):
    """YOLO lines as class 0, keeping only source classes in `keep`."""
    lines = []
    if label.exists():
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and (keep is None or int(float(parts[0])) in keep):
                lines.append(" ".join(["0"] + parts[1:5]))
    return lines


def to_pixels(lines, shape):
    height, width = shape[:2]
    boxes = []
    for line in lines:
        cx, cy, w, h = map(float, line.split()[1:5])
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def add_image(domain, split, name, write, lines):
    img_dir, lbl_dir = DS / domain / "images" / split, DS / domain / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    write(img_dir / name)
    (lbl_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines))


def build_visdrone():
    """Link VisDrone into the shared layout and return its validation images for scoring."""
    data = check_det_dataset("VisDrone.yaml")
    images = []
    for split in ["train", "val"]:
        for path in sorted(Path(data[split]).glob("*.jpg")):
            head, tail = str(path).rsplit("/images/", 1)
            lines = label_lines(Path(head) / "labels" / Path(tail).with_suffix(".txt"), keep={0, 1})  # pedestrian, people
            add_image("visdrone", split, path.name, lambda dst: dst.symlink_to(path), lines)
            if split == "val":
                image = cv2.imread(str(path))
                images.append((image, to_pixels(lines, image.shape)))
    return images


def build_heridal():
    """Shrink HERIDAL as Experiment 2 did and return its full-size validation images for scoring."""
    root = mounted("heridal-4k")
    images = []
    for source in ["train", "valid", "test"]:
        for path in sorted((root / source / "images").glob("*.jpg")):
            place = re.match(r"[A-Za-z]+", re.sub(r"^(train|valid|test)_", "", path.name)).group()
            split = HERIDAL_SPLITS.get(place, "train")
            if split == "test":
                continue
            lines = label_lines(root / source / "labels" / f"{path.stem}.txt")
            image = cv2.imread(str(path))
            if split == "val":
                images.append((image, to_pixels(lines, image.shape)))
            scale = HERIDAL_LONG_SIDE / max(image.shape[:2])
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
            jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            for repeat in range(HERIDAL_REPEATS if split == "train" else 1):
                add_image("heridal", split, f"{path.stem}_{repeat}.jpg", lambda dst: dst.write_bytes(jpeg), lines)
    return images


def detect(model, image, imgsz):
    scale = imgsz / max(image.shape[:2])
    shrink = scale < 1
    if shrink:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    result = model.predict(image, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, classes=[0, 1],
                           agnostic_nms=True, device=0, verbose=False)[0]
    boxes, scores = result.boxes.xyxy.cpu(), result.boxes.conf.cpu()
    keep = nms(boxes, scores, SITE_NMS_IOU)  # sorted by score
    return boxes[keep] / (scale if shrink else 1), scores[keep]


def average_precision(hits, total):
    recall = np.cumsum(hits) / max(total, 1)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.interp(np.linspace(0, 1, 101), recall, envelope, right=0).mean())


def score(model, images, imgsz):
    confidences, hits, total = [], [], 0
    for image, truth in images:
        boxes, conf = detect(model, image, imgsz)
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
    return {"mAP50-95": float(np.mean(per_threshold)), "mAP50": per_threshold[0],
            "site_precision": float(kept.mean()) if kept.size else 0.0,
            "site_recall": float(kept.sum() / max(total, 1)),
            "best_f1": float(f1[best]), "best_f1_conf": float(confidences[best]),
            "best_f1_precision": float(precision[best]), "best_f1_recall": float(recall[best])}


domains = {"visdrone": build_visdrone(), "heridal": build_heridal()}
assert all(domains.values()), {name: len(images) for name, images in domains.items()}
counts = defaultdict(dict)
for domain in domains:
    for split in ["train", "val"]:
        counts[domain][split] = len(list((DS / domain / "images" / split).iterdir()))
print(json.dumps(counts))
data_yaml = DS / "vh.yaml"
data_yaml.write_text(yaml.safe_dump({"path": str(DS), "names": {0: "person"},
                                     "train": ["visdrone/images/train", "heridal/images/train"],
                                     "val": ["visdrone/images/val", "heridal/images/val"]}))
p2_yaml = DS / "yolo11s-p2.yaml"  # the file name tells Ultralytics to use the `s` scale
p2_yaml.write_text(P2_MODEL)

results = {"smoke": SMOKE, "epochs": EPOCHS, "counts": counts, "runs": {}}
for run, (imgsz, batch, p2) in RUNS.items():
    model = YOLO(str(p2_yaml)).load(str(WEIGHTS)) if p2 else YOLO(str(WEIGHTS))
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=imgsz, batch=batch, device=0, workers=2, seed=SEED,
                close_mosaic=min(5, EPOCHS), fraction=FRACTION, project=str(OUT / "runs"), name=run,
                exist_ok=True, plots=False)
    results["runs"][run] = {"imgsz": imgsz}
    for weights in ["best", "last"]:
        trained = YOLO(str(OUT / "runs" / run / "weights" / f"{weights}.pt"))
        entry = {domain: score(trained, images, imgsz) for domain, images in domains.items()}
        entry["mean_mAP50-95"] = float(np.mean([entry[d]["mAP50-95"] for d in domains]))
        results["runs"][run][weights] = entry
        print(run, weights, json.dumps(entry), flush=True)
    # Saved after every run so a Kaggle timeout cannot lose finished runs.
    RESULTS.write_text(json.dumps(results, indent=2))

shutil.rmtree(OUT / "datasets", ignore_errors=True)  # Ultralytics downloads here, not to a temp folder
print(json.dumps(results, indent=2))
