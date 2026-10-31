"""Run the real models over disaster scenes for the mission simulator.

Scenes come from C2A's test group as Experiment 2 split it, so the person model never trained on
them. C2A pastes people onto real aerial disaster photos, so the mission is a simulation. Every
box and hazard region in the output is model output, not a hand-placed marker.
"""
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
SCENES = OUT / "scenes"
SEED = 0
TEST_FRACTION = VAL_FRACTION = 0.15
SCENES_PER_MISSION = 9
CANDIDATES = 120
MIN_PEOPLE, MAX_PEOPLE = 5, 30
HAZARD_SHARE = {"flood": 0.4, "fire": 0.1}  # partly affected scenes give the ranking real differences to show
PERSON_CONF, HAZARD_CONF = 0.25, 0.25
PERSON_IMGSZ, FLOOD_IMGSZ, FIRE_IMGSZ = 960, 960, 640
IMAGE_WIDTH = 960
JPEG_QUALITY = 72
OUTLINE_EPSILON = 0.004  # polygon simplification, as a share of the outline length
MIN_REGION = 0.0005  # share of the image; smaller regions are noise
BOX_PAD = 0.15  # share of a fire box added on each side, so flames at its edge are not cut straight


def mounted(slug):
    """Kaggle mounts attached data at any of these paths, so look in all of them."""
    return next(p for pattern in (slug, f"datasets/*/{slug}", f"notebooks/*/{slug}") for p in INPUT.glob(pattern))


def group_split(groups):
    """Experiment 2's split, so its test group is reproduced exactly."""
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    total = sum(len(groups[k]) for k in keys)
    taken = {"test": 0, "val": 0}
    out = {}
    for key in keys:
        if taken["test"] < TEST_FRACTION * total:
            split = "test"
        elif taken["val"] < VAL_FRACTION * total:
            split = "val"
        else:
            split = "train"
        out[key] = split
        if split in taken:
            taken[split] += len(groups[key])
    return out


def unseen_scenes():
    """Images per disaster type that Experiment 2 kept out of training."""
    root = mounted("c2a-dataset") / "C2A_Dataset/new_dataset3"
    by_scene = defaultdict(lambda: defaultdict(list))
    for split in ["train", "val", "test"]:
        for image in sorted((root / split / "images").glob("*.png")):
            base = image.stem.rsplit("_", 1)[0]
            by_scene[re.sub(r"\d+$", "", base)][base].append(image)
    unseen = {}
    for scene, bases in by_scene.items():
        assigned = group_split(bases)
        unseen[scene] = [image for base, images in sorted(bases.items()) if assigned[base] == "test" for image in images]
    return unseen


def simplified(points):
    """A closed outline of 0 to 1 points with fewer corners, or None if nothing is left of it."""
    points = points.astype(np.float32).reshape(-1, 1, 2)
    simple = cv2.approxPolyDP(points, OUTLINE_EPSILON * cv2.arcLength(points, True), True).reshape(-1, 2)
    return np.round(simple, 4).tolist() if len(simple) >= 3 else None


def flood_outlines(model, image):
    result = model.predict(image, imgsz=FLOOD_IMGSZ, conf=HAZARD_CONF, device="cpu", verbose=False)[0]
    if result.masks is None:
        return []
    return [outline for outline in map(simplified, result.masks.xyn) if outline]


def flame_outlines(image, boxes):
    """Cut fire boxes down to flame-coloured pixels, so standing beside a flame is not being in it."""
    height, width = image.shape[:2]
    inside = np.zeros((height, width), np.uint8)
    for x1, y1, x2, y2 in boxes:
        pad_x, pad_y = (x2 - x1) * BOX_PAD, (y2 - y1) * BOX_PAD
        cv2.rectangle(inside, (int((x1 - pad_x) * width), int((y1 - pad_y) * height)),
                      (int((x2 + pad_x) * width), int((y2 + pad_y) * height)), 1, -1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    flame = cv2.inRange(hsv, (0, 90, 150), (35, 255, 255)) | cv2.inRange(hsv, (0, 0, 240), (40, 255, 255))
    size = max(3, width // 80) | 1
    mask = cv2.morphologyEx(((flame > 0) & (inside > 0)).astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = [c for c in contours if cv2.contourArea(c) >= MIN_REGION * height * width]
    return [outline for outline in (simplified(c.reshape(-1, 2) / [width, height]) for c in large) if outline]


def fire_regions(model, image):
    result = model.predict(image, imgsz=FIRE_IMGSZ, conf=HAZARD_CONF, device="cpu", verbose=False)[0]
    found = {"fire": [], "smoke": []}
    for box, cls in zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        found[model.names[int(cls)]].append(np.round(box, 4).tolist())
    found["fire"] = flame_outlines(image, found["fire"])
    return found


def people_in(model, image):
    result = model.predict(image, imgsz=PERSON_IMGSZ, conf=PERSON_CONF, iou=0.5, classes=[0], device="cpu", verbose=False)[0]
    return [{"box": np.round(box, 4).tolist(), "confidence": round(float(conf), 3)}
            for box, conf in zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy())]


def label_count(path):
    label = path.parent.parent / "labels" / f"{path.stem}.txt"
    return len(label.read_text().splitlines()) if label.exists() else 0


person_model = YOLO(str(mounted("dronesight-experiment-2") / "runs/train_vhc/weights/best.pt"))
flood_model = YOLO(str(mounted("dronesight-flood") / "runs/flood/weights/best.pt"))
fire_model = YOLO(str(mounted("dronesight-experiment-4") / "runs/fire/weights/best.pt"))
unseen = unseen_scenes()
print({name: len(images) for name, images in unseen.items()}, flush=True)
SCENES.mkdir(parents=True, exist_ok=True)
missions = {}
for mission, group in [("flood", "flood_image"), ("fire", "fire_image")]:
    pool = [p for p in unseen[group] if MIN_PEOPLE <= label_count(p) <= MAX_PEOPLE]
    random.Random(SEED).shuffle(pool)
    # C2A pastes different people onto one photo, so one of each keeps every sector a different place.
    photos = {}
    for path in pool:
        photos.setdefault(path.stem.rsplit("_", 1)[0], path)
    print(mission, len(photos), "photos", flush=True)
    scored = []
    for path in list(photos.values())[:CANDIDATES]:
        image = cv2.imread(str(path))
        # Only the mission's own hazard, so a false fire box cannot raise an alarm in a flood.
        scene = {"source": path.stem, "people": people_in(person_model, image), "true_people": label_count(path),
                 "flood": flood_outlines(flood_model, image) if mission == "flood" else [],
                 **(fire_regions(fire_model, image) if mission == "fire" else {"fire": [], "smoke": []})}
        share = sum(cv2.contourArea(np.array(o, np.float32)) for o in scene[mission])
        scored.append((abs(share - HAZARD_SHARE[mission]), scene, image))
    scored.sort(key=lambda item: item[0])
    chosen = []
    for _, scene, image in scored[:SCENES_PER_MISSION]:
        scale = IMAGE_WIDTH / image.shape[1]
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        scene["image"] = f"scenes/{scene['source']}.jpg"
        cv2.imwrite(str(OUT / scene["image"]), small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        scene["aspect"] = image.shape[1] / image.shape[0]
        chosen.append(scene)
    missions[mission] = chosen
    print(mission, [(s["source"], len(s["people"]), s["true_people"], len(s["flood"]), len(s["fire"]), len(s["smoke"])) for s in chosen], flush=True)

(OUT / "mission_scenes.json").write_text(json.dumps(missions))
