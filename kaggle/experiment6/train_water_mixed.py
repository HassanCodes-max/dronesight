"""Experiment 6: teach the water model what a real flood looks like.

Experiment 5 traced water well on grey and green scenes but missed brown flood water and shaded grey
roofs, because FloodNet is shot straight down and its water is rarely muddy. Colour shifting could
not close that gap, and its wide hue range made any colour look like water. This adds Flood Area
Segmentation, photos of real floods at many angles and in many muddy colours, and narrows the hue
range. FloodNet keeps Experiment 4's blocks. The best epoch is picked on the mean validation IoU of
the two datasets, and Experiment 5's model is scored on the same test images as the baseline.
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

SMOKE = False  # True runs a short end-to-end trial

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
SEED = 0
FLOOD_CLASSES = [1, 3, 5]  # building-flooded, road-flooded, water
FLOOD_BLOCK = 20
LONG_SIDE = 768
CROP = 512
EPOCHS = 1 if SMOKE else 60
BATCH = 8
RATE = 3e-4
HUE_SHIFT = 15
TEST_FRACTION = VAL_FRACTION = 0.15
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def mounted(slug):
    """Kaggle mounts attached data at any of these paths, so look in all of them."""
    return next(p for pattern in (slug, f"datasets/*/{slug}", f"notebooks/*/{slug}") for p in INPUT.glob(pattern))


def shrink(image, mask):
    """To the training size, keeping both sides at least as large as a crop."""
    height, width = image.shape[:2]
    scale = max(LONG_SIDE / max(height, width), CROP / min(height, width))
    size = (round(width * scale), round(height * scale))
    shrunk = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return shrunk, cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def load_floodnet():
    labeled = mounted("aerial-imagery-dataset-floodnet-challenge") / "FloodNet Challenge - Track 1/Train/Labeled"
    pairs = sorted((image, group / "mask" / f"{image.stem}_lab.png")
                   for group in [labeled / "Flooded", labeled / "Non-Flooded"]
                   for image in (group / "image").glob("*.jpg"))
    blocks = sorted({int(image.stem) // FLOOD_BLOCK for image, _ in pairs})
    random.Random(SEED).shuffle(blocks)
    cut_test, cut_val = round(0.15 * len(blocks)), round(0.30 * len(blocks))
    split_of = {block: "test" if i < cut_test else "val" if i < cut_val else "train" for i, block in enumerate(blocks)}
    splits = {"train": [], "val": [], "test": []}
    for image_path, mask_path in pairs:
        mask = np.isin(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), FLOOD_CLASSES).astype(np.uint8)
        splits[split_of[int(image_path.stem) // FLOOD_BLOCK]].append(shrink(cv2.imread(str(image_path)), mask))
    return splits


def load_flood_area():
    root = mounted("flood-area-segmentation")
    pairs = []
    for image_path in sorted((root / "Image").iterdir()):
        mask_path = next((root / "Mask").glob(f"{image_path.stem}.*"), None)
        image = cv2.imread(str(image_path))
        if mask_path is not None and image is not None:
            pairs.append((image, (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)))
    random.Random(SEED).shuffle(pairs)
    cut_test, cut_val = round(TEST_FRACTION * len(pairs)), round((TEST_FRACTION + VAL_FRACTION) * len(pairs))
    resized = [shrink(image, mask) for image, mask in pairs]
    return {"test": resized[:cut_test], "val": resized[cut_test:cut_val], "train": resized[cut_val:]}


def jitter_colour(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-HUE_SHIFT, HUE_SHIFT)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.6, 1.5), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def to_tensor(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)).float()


class Crops(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image, mask = self.items[index]
        top = random.randint(0, image.shape[0] - CROP)
        left = random.randint(0, image.shape[1] - CROP)
        image, mask = image[top:top + CROP, left:left + CROP], mask[top:top + CROP, left:left + CROP]
        if random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            image, mask = image[::-1], mask[::-1]
        return to_tensor(jitter_colour(image.copy())), torch.from_numpy(mask.copy()).float()


def predict(model, image):
    """Water probability at the image's own size. Sides must divide by 32 for the U-Net."""
    height, width = image.shape[:2]
    resized = cv2.resize(image, (round(width / 32) * 32, round(height / 32) * 32), interpolation=cv2.INTER_AREA)
    with torch.inference_mode():
        logits = model(to_tensor(resized)[None].to("cuda"))
    return cv2.resize(torch.sigmoid(logits)[0, 0].cpu().numpy(), (width, height))


def iou(model, items, threshold=0.5):
    overlap = union = 0
    for image, mask in items:
        predicted = predict(model, image) >= threshold
        overlap += int((predicted & (mask > 0)).sum())
        union += int((predicted | (mask > 0)).sum())
    return overlap / max(union, 1)


def scores(model, floodnet, flood_area, split):
    return {"floodnet": iou(model, floodnet[split]), "flood_area": iou(model, flood_area[split])}


def new_model(weights):
    return smp.Unet("resnet34", encoder_weights=weights, classes=1).to("cuda")


floodnet, flood_area = load_floodnet(), load_flood_area()
counts = {"floodnet": {k: len(v) for k, v in floodnet.items()}, "flood_area": {k: len(v) for k, v in flood_area.items()}}
print(counts, flush=True)

baseline = new_model(None)
baseline.load_state_dict(torch.load(mounted("dronesight-water") / "water_unet.pt"))
baseline.eval()
results = {"smoke": SMOKE, "epochs": EPOCHS, "images": counts, "experiment5_test": scores(baseline, floodnet, flood_area, "test")}
print(results, flush=True)
del baseline

# The smaller dataset repeats so each one gives about half of every epoch.
repeats = max(1, round(len(floodnet["train"]) / len(flood_area["train"])))
model = new_model("imagenet")
optimiser = torch.optim.AdamW(model.parameters(), lr=RATE)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, EPOCHS)
dice = smp.losses.DiceLoss("binary", from_logits=True)
binary = torch.nn.BCEWithLogitsLoss()
loader = DataLoader(Crops((floodnet["train"] + flood_area["train"] * repeats) * 2), batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
best = -1.0
for epoch in range(EPOCHS):
    model.train()
    for images, masks in loader:
        images, masks = images.to("cuda"), masks[:, None].to("cuda")
        logits = model(images)
        loss = dice(logits, masks) + binary(logits, masks)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    schedule.step()
    model.eval()
    val = scores(model, floodnet, flood_area, "val")
    score = (val["floodnet"] + val["flood_area"]) / 2
    print(f"epoch {epoch + 1}/{EPOCHS} loss {loss.item():.3f} val {val} mean {score:.4f}", flush=True)
    if score > best:
        best = score
        results["val"] = val
        torch.save(model.state_dict(), OUT / "water_unet.pt")

model.load_state_dict(torch.load(OUT / "water_unet.pt"))
model.eval()
results["test"] = scores(model, floodnet, flood_area, "test")
(OUT / "experiment6_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results), flush=True)
