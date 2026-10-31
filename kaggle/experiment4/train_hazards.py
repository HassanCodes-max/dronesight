"""Experiment 4: see the disaster, not just the people.

Risk needs to know where the danger is, so two models are trained from COCO weights:

- Flood: YOLO11s-seg on FloodNet. Flooded buildings, flooded roads and water become one `flood`
  class. Only 398 images are labelled, and neighbouring frames can look alike, so whole blocks of
  consecutive image ids go to one split.
- Fire: YOLO11s on D-Fire, which labels `smoke` and `fire`. Many images are not aerial, so its
  scores say little about a drone's view until it is checked on aerial scenes.

Each model is scored once on its own test split, and flood also reports pixel IoU.
"""
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

SMOKE = False  # True runs a short end-to-end trial

INPUT = Path("/kaggle/input")
DS = Path("/tmp/ds")  # outside /kaggle/working so the data is not saved as notebook output
OUT = Path("/kaggle/working")
RESULTS = OUT / "experiment4_results.json"
SEED = 0
FLOOD_CLASSES = [1, 3, 5]  # building-flooded, road-flooded, water
FLOOD_BLOCK = 20
FLOOD_LONG_SIDE = 1280
FLOOD_IMGSZ = 960
FLOOD_EPOCHS = 1 if SMOKE else 60
FIRE_IMGSZ = 640
FIRE_EPOCHS = 1 if SMOKE else 20
FIRE_FRACTION = 0.02 if SMOKE else 1.0
MIN_REGION = 0.0005  # share of the image; smaller regions are label noise
CONF = 0.25


def mounted(slug):
    """Kaggle mounts attached datasets at either of these paths, so look in both."""
    return next(p for pattern in (slug, f"datasets/*/{slug}") for p in INPUT.glob(pattern))


def polygons(mask):
    """YOLO segmentation lines for each sizeable region of a binary mask."""
    height, width = mask.shape
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_REGION * height * width:
            continue
        points = cv2.approxPolyDP(contour, 2.0, True).reshape(-1, 2) / [width, height]
        if len(points) >= 3:
            lines.append("0 " + " ".join(f"{v:.5f}" for v in points.flatten()))
    return lines


def build_flood():
    """Write the YOLO layout and return test images with their true masks for pixel IoU."""
    labeled = mounted("aerial-imagery-dataset-floodnet-challenge") / "FloodNet Challenge - Track 1/Train/Labeled"
    pairs = sorted((image, group / "mask" / f"{image.stem}_lab.png")
                   for group in [labeled / "Flooded", labeled / "Non-Flooded"]
                   for image in (group / "image").glob("*.jpg"))
    blocks = sorted({int(image.stem) // FLOOD_BLOCK for image, _ in pairs})
    random.Random(SEED).shuffle(blocks)
    cut_test, cut_val = round(0.15 * len(blocks)), round(0.30 * len(blocks))
    split_of = {block: "test" if i < cut_test else "val" if i < cut_val else "train" for i, block in enumerate(blocks)}
    counts = {}
    test = []
    for image_path, mask_path in pairs:
        split = split_of[int(image_path.stem) // FLOOD_BLOCK]
        image = cv2.imread(str(image_path))
        mask = np.isin(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), FLOOD_CLASSES)
        scale = FLOOD_LONG_SIDE / max(image.shape[:2])
        size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        for kind in ["images", "labels"]:
            (DS / "flood" / kind / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DS / "flood/images" / split / image_path.name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (DS / "flood/labels" / split / f"{image_path.stem}.txt").write_text("\n".join(polygons(mask)))
        counts[split] = counts.get(split, 0) + 1
        if split == "test":
            test.append((image, mask))
    path = DS / "flood.yaml"
    path.write_text(yaml.safe_dump({"path": str(DS / "flood"), "names": {0: "flood"}, "train": "images/train",
                                    "val": "images/val", "test": "images/test"}))
    return path, counts, test


def pixel_iou(model, images):
    """Pixels a person standing there would be told are flood, compared with the truth."""
    overlap = union = 0
    for image, truth in images:
        result = model.predict(image, imgsz=FLOOD_IMGSZ, conf=CONF, device=0, verbose=False, retina_masks=True)[0]
        predicted = np.zeros(truth.shape, dtype=bool)
        if result.masks is not None:
            predicted = result.masks.data.cpu().numpy().any(axis=0)
            if predicted.shape != truth.shape:
                predicted = cv2.resize(predicted.astype(np.uint8), truth.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        overlap += int((predicted & truth).sum())
        union += int((predicted | truth).sum())
    return overlap / max(union, 1)


def train_flood(results):
    data, counts, test = build_flood()
    print(json.dumps(counts), flush=True)
    model = YOLO("yolo11s-seg.pt")
    model.train(data=str(data), epochs=FLOOD_EPOCHS, imgsz=FLOOD_IMGSZ, batch=8, device=0, workers=2, seed=SEED,
                patience=20, project=str(OUT / "runs"), name="flood", exist_ok=True, plots=False)
    best = YOLO(str(OUT / "runs/flood/weights/best.pt"))
    metrics = best.val(data=str(data), split="test", imgsz=FLOOD_IMGSZ, device=0, plots=False, project="/tmp/val", name="flood")
    results["flood"] = {"images": counts, "mask_mAP50-95": float(metrics.seg.map), "mask_mAP50": float(metrics.seg.map50),
                        "pixel_iou": pixel_iou(best, test)}


def train_fire(results):
    root = mounted("smoke-fire-detection-yolo")
    data = DS / "fire.yaml"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(yaml.safe_dump({"path": str(root / "data"), "names": {0: "smoke", 1: "fire"}, "train": "train/images",
                                    "val": "val/images", "test": "test/images"}))
    model = YOLO("yolo11s.pt")
    model.train(data=str(data), epochs=FIRE_EPOCHS, imgsz=FIRE_IMGSZ, batch=32, device=0, workers=2, seed=SEED,
                fraction=FIRE_FRACTION, project=str(OUT / "runs"), name="fire", exist_ok=True, plots=False)
    best = YOLO(str(OUT / "runs/fire/weights/best.pt"))
    metrics = best.val(data=str(data), split="test", imgsz=FIRE_IMGSZ, device=0, plots=False, project="/tmp/val", name="fire")
    results["fire"] = {"mAP50-95": float(metrics.box.map), "mAP50": float(metrics.box.map50),
                       "per_class_mAP50-95": {best.names[int(c)]: float(v) for c, v in zip(metrics.box.ap_class_index, metrics.box.maps[metrics.box.ap_class_index])}}


results = {"smoke": SMOKE}
for step in [train_flood, train_fire]:
    step(results)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)

shutil.rmtree(OUT / "datasets", ignore_errors=True)
for name in ["yolo11s.pt", "yolo11s-seg.pt", "yolo26n.pt", "yolo11n.pt"]:
    (Path(name)).unlink(missing_ok=True)
