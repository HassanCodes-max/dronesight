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
MIN_PEOPLE, MAX_PEOPLE = 4, 30
PERSON_CONF, HAZARD_CONF = 0.25, 0.25
PERSON_IMGSZ, FLOOD_IMGSZ, FIRE_IMGSZ = 960, 960, 640
IMAGE_WIDTH = 960
JPEG_QUALITY = 72
OUTLINE_EPSILON = 0.004  # polygon simplification, as a share of the outline length


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


def flood_outlines(model, image):
    result = model.predict(image, imgsz=FLOOD_IMGSZ, conf=HAZARD_CONF, device=0, verbose=False)[0]
    if result.masks is None:
        return []
    outlines = []
    for polygon in result.masks.xyn:
        points = polygon.astype(np.float32).reshape(-1, 1, 2)
        simple = cv2.approxPolyDP(points, OUTLINE_EPSILON * cv2.arcLength(points, True), True).reshape(-1, 2)
        if len(simple) >= 3:
            outlines.append(np.round(simple, 4).tolist())
    return outlines


def fire_boxes(model, image):
    result = model.predict(image, imgsz=FIRE_IMGSZ, conf=HAZARD_CONF, device=0, verbose=False)[0]
    found = {"fire": [], "smoke": []}
    for box, cls in zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        found[model.names[int(cls)]].append(np.round(box, 4).tolist())
    return found


def people_in(model, image):
    result = model.predict(image, imgsz=PERSON_IMGSZ, conf=PERSON_CONF, iou=0.5, classes=[0], device=0, verbose=False)[0]
    return [{"box": np.round(box, 4).tolist(), "confidence": round(float(conf), 3)}
            for box, conf in zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy())]


def label_count(path):
    label = path.parent.parent / "labels" / f"{path.stem}.txt"
    return len(label.read_text().splitlines()) if label.exists() else 0


person_model = YOLO(str(mounted("dronesight-experiment-2") / "runs/train_vhc/weights/best.pt"))
flood_model = YOLO(str(mounted("dronesight-flood") / "runs/flood/weights/best.pt"))
# The fire mission runs once this job is attached to the kernel.
fire_weights = next((p / "runs/fire/weights/best.pt" for p in INPUT.glob("**/dronesight-experiment-4")), None)
fire_model = YOLO(str(fire_weights)) if fire_weights else None
unseen = unseen_scenes()
print({name: len(images) for name, images in unseen.items()}, flush=True)
SCENES.mkdir(parents=True, exist_ok=True)
missions = {}
for mission, group in [("flood", "flood_image"), ("fire", "fire_image")][:2 if fire_model else 1]:
    pool = [p for p in unseen[group] if MIN_PEOPLE <= label_count(p) <= MAX_PEOPLE]
    random.Random(SEED).shuffle(pool)
    scored = []
    for path in pool[:CANDIDATES]:
        image = cv2.imread(str(path))
        scene = {"source": path.stem, "people": people_in(person_model, image),
                 "true_people": label_count(path), "flood": flood_outlines(flood_model, image),
                 **(fire_boxes(fire_model, image) if fire_model else {"fire": [], "smoke": []})}
        hazard = sum(cv2.contourArea(np.array(o, np.float32)) for o in scene["flood"]) if mission == "flood" else len(scene["fire"]) + len(scene["smoke"])
        scored.append((hazard, len(scene["people"]), scene, image))
    # Scenes where the hazard model found something, so the mission has dangers to respond to.
    scored.sort(key=lambda item: (item[0] > 0, item[1]), reverse=True)
    chosen = []
    for hazard, _, scene, image in scored[:SCENES_PER_MISSION]:
        scale = IMAGE_WIDTH / image.shape[1]
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        scene["image"] = f"scenes/{scene['source']}.jpg"
        cv2.imwrite(str(OUT / scene["image"]), small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        scene["aspect"] = image.shape[1] / image.shape[0]
        chosen.append(scene)
    missions[mission] = chosen
    print(mission, [(s["source"], len(s["people"]), s["true_people"], len(s["flood"]), len(s["fire"]), len(s["smoke"])) for s in chosen], flush=True)

(OUT / "mission_scenes.json").write_text(json.dumps(missions))
