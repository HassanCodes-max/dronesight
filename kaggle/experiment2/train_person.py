"""Experiment 2: does mixing aerial datasets make person detection work on unseen footage?

Fine-tunes the Experiment 1 winner (YOLO11s at 960) to a single `person` class on VisDrone,
HERIDAL and C2A. Each leave-one-out run is scored on the dataset it never saw.

HERIDAL is re-split by location and C2A by base scene, because the published splits place
near-identical images on both sides.
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
import yaml  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

SMOKE = True  # short end-to-end trial; set False for the real run

INPUT = Path("/kaggle/input")
WEIGHTS = INPUT / "notebooks/hassanagddsbdxfg/dronesight-experiment-1/runs/s_960/weights/best.pt"
HERIDAL = INPUT / "datasets/mariannaesposito/heridal-4k"
C2A = INPUT / "datasets/rgbnihal/c2a-dataset/C2A_Dataset/new_dataset3"
# Outside /kaggle/working so the data is not saved as notebook output.
DS = Path("/tmp/ds")
OUT = Path("/kaggle/working")
RESULTS = OUT / "experiment2_results.json"

SEED = 0
IMGSZ = 960
EPOCHS = 1 if SMOKE else 15
FRACTION = 0.01 if SMOKE else 1.0
VAL_FRACTION = TEST_FRACTION = 0.15
HERIDAL_LONG_SIDE = 1280  # training shrinks images to IMGSZ anyway, so this only saves load time
DOMAINS = ["visdrone", "heridal", "c2a"]
RUNS = {
    "train_v": ["visdrone"],
    "train_vc": ["visdrone", "c2a"],
    "train_vh": ["visdrone", "heridal"],
    "train_hc": ["heridal", "c2a"],
    "train_vhc": ["visdrone", "heridal", "c2a"],
}
if SMOKE:
    RUNS = {k: RUNS[k] for k in ["train_v", "train_hc"]}

stats = defaultdict(lambda: defaultdict(lambda: {"images": 0, "positive_images": 0, "instances": 0}))


def read_person_boxes(label, keep=None):
    """Return YOLO lines as class 0, keeping only source classes in `keep`."""
    lines = []
    if label.exists():
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and (keep is None or int(float(parts[0])) in keep):
                lines.append(" ".join(["0"] + parts[1:5]))
    return lines


def add_image(domain, split, image, boxes, long_side=None):
    img_dir = DS / domain / "images" / split
    lbl_dir = DS / domain / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    dst = img_dir / image.name
    if long_side is None:
        if not dst.exists():
            dst.symlink_to(image)
    else:
        im = cv2.imread(str(image))
        scale = long_side / max(im.shape[:2])
        if scale < 1:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (lbl_dir / f"{image.stem}.txt").write_text("\n".join(boxes))
    s = stats[domain][split]
    s["images"] += 1
    s["positive_images"] += bool(boxes)
    s["instances"] += len(boxes)


def group_split(groups, weight):
    """Assign whole groups to test, then val, then train, seeded and by share of `weight`."""
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    total = sum(weight(groups[k]) for k in keys)
    out, taken = {}, {"test": 0, "val": 0}
    for key in keys:
        if taken["test"] < TEST_FRACTION * total:
            split = "test"
        elif taken["val"] < VAL_FRACTION * total:
            split = "val"
        else:
            split = "train"
        out[key] = split
        if split in taken:
            taken[split] += weight(groups[key])
    return out


def build_visdrone():
    settings.update({"datasets_dir": "/tmp/datasets"})
    data = check_det_dataset("VisDrone.yaml")
    for split in ["train", "val", "test"]:
        for image in sorted(Path(data[split]).glob("*.jpg")):
            head, tail = str(image).rsplit("/images/", 1)
            label = Path(head) / "labels" / Path(tail).with_suffix(".txt")
            add_image("visdrone", split, image, read_person_boxes(label, keep={0, 1}))  # pedestrian, people


def build_heridal():
    by_place = defaultdict(list)
    for split in ["train", "valid", "test"]:
        for image in sorted((HERIDAL / split / "images").glob("*.jpg")):
            boxes = read_person_boxes(HERIDAL / split / "labels" / f"{image.stem}.txt")
            name = re.sub(r"^(train|valid|test)_", "", image.name)
            by_place[re.match(r"[A-Za-z]+", name).group()].append((image, boxes))
    # Split on images that contain people: many HERIDAL images are empty.
    assigned = group_split(by_place, lambda items: sum(bool(b) for _, b in items))
    for place, items in by_place.items():
        for image, boxes in items:
            add_image("heridal", assigned[place], image, boxes, long_side=HERIDAL_LONG_SIDE)
    return {k: assigned[k] for k in sorted(assigned)}


def build_c2a():
    by_scene = defaultdict(lambda: defaultdict(list))
    for split in ["train", "val", "test"]:
        for image in sorted((C2A / split / "images").glob("*.png")):
            boxes = read_person_boxes(C2A / split / "labels" / f"{image.stem}.txt")
            base = image.stem.rsplit("_", 1)[0]  # flood_image0001_3 -> flood_image0001
            scene = re.sub(r"\d+$", "", base)
            by_scene[scene][base].append((image, boxes))
    for scene, bases in by_scene.items():  # split inside each disaster type to keep them balanced
        assigned = group_split(bases, len)
        for base, items in bases.items():
            for image, boxes in items:
                add_image("c2a", assigned[base], image, boxes)


def write_yaml(name, train_domains, val_domains, test_domain=None):
    cfg = {"path": str(DS), "names": {0: "person"},
           "train": [f"{d}/images/train" for d in train_domains],
           "val": [f"{d}/images/val" for d in val_domains]}
    if test_domain:
        cfg["test"] = f"{test_domain}/images/test"
    path = DS / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def box_metrics(metrics):
    return {"mAP50-95": float(metrics.box.map), "mAP50": float(metrics.box.map50),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr)}


build_visdrone()
heridal_places = build_heridal()
build_c2a()
for domain in DOMAINS:
    for split in ["train", "val", "test"]:
        assert stats[domain][split]["instances"] > 0, (domain, split, dict(stats[domain]))

results = {"smoke": SMOKE, "epochs": EPOCHS, "imgsz": IMGSZ, "stats": stats,
           "heridal_places": heridal_places, "runs": {}}
RESULTS.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

test_yamls = {d: write_yaml(f"test_{d}", [d], [d], d) for d in DOMAINS}
for run, domains in RUNS.items():
    model = YOLO(str(WEIGHTS))
    model.train(data=write_yaml(run, domains, domains), epochs=EPOCHS, imgsz=IMGSZ, batch=12,
                device=0, workers=2, seed=SEED, close_mosaic=min(5, EPOCHS), fraction=FRACTION,
                project=str(OUT / "runs"), name=run, exist_ok=True, plots=False)
    best = OUT / "runs" / run / "weights" / "best.pt"
    entry = {"train_domains": domains, "test": {}}
    for domain, data in test_yamls.items():
        metrics = YOLO(str(best)).val(data=data, split="test", imgsz=IMGSZ, batch=8, device=0,
                                      workers=2, plots=False, project="/tmp/val", name=f"{run}_{domain}")
        entry["test"][domain] = {**box_metrics(metrics), "unseen": domain not in domains}
    results["runs"][run] = entry
    # Saved after every run so a Kaggle timeout cannot lose finished runs.
    RESULTS.write_text(json.dumps(results, indent=2))
    (OUT / "runs" / run / "weights" / "last.pt").unlink(missing_ok=True)

print(json.dumps(results, indent=2))
