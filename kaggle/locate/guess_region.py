"""Can a picture alone say roughly where a drone video was filmed?

GeoCLIP guesses a place on Earth from one image. It learned from ground-level photos, so this scores it
on drone pictures whose region is known: HERIDAL from Croatia and VisDrone from cities in China. If the
guess lands near enough, satellite maps of that region can be searched for the exact spot.
"""
import json
import math
import random
import subprocess
import sys
import zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "geoclip"], check=True)

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from geoclip import GeoCLIP  # noqa: E402

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
SEED = 0
PER_SET = 150
TOP = 5
VISDRONE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip"
SPLIT_CROATIA = (43.5, 16.4)  # HERIDAL was filmed in this part of Croatia and Bosnia and Herzegovina
CHINA = (18, 54, 73, 135)  # south, north, west, east


def km(a, b):
    (lat1, lon1), (lat2, lon2) = [(math.radians(x), math.radians(y)) for x, y in (a, b)]
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def sample(paths):
    paths = sorted(paths)
    random.Random(SEED).shuffle(paths)
    return paths[:PER_SET]


def guesses(model, path):
    places, chances = model.predict(str(path), top_k=TOP)
    return [(float(a), float(b)) for a, b in places.tolist()], [float(p) for p in chances.tolist()]


def visdrone_images():
    archive = Path("/tmp/visdrone.zip")
    with httpx.stream("GET", VISDRONE, follow_redirects=True, timeout=600) as response, archive.open("wb") as file:
        for part in response.iter_bytes():
            file.write(part)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall("/tmp/visdrone")
    return list(Path("/tmp/visdrone").rglob("*.jpg"))


model = GeoCLIP()
results = {}

heridal = [p for p in INPUT.rglob("*.[jJ][pP][gG]") if "location-frames" not in str(p)]
print("heridal images", len(heridal), flush=True)
errors, top5 = [], []
for path in sample(heridal):
    places, _ = guesses(model, path)
    distances = [km(place, SPLIT_CROATIA) for place in places]
    errors.append(distances[0])
    top5.append(min(distances))
results["heridal"] = {"images": len(errors), "median_km": float(np.median(errors)), "top5_median_km": float(np.median(top5)),
                      "within_200km": float(np.mean(np.array(errors) <= 200)), "top5_within_200km": float(np.mean(np.array(top5) <= 200))}
print(results["heridal"], flush=True)

inside = []
any_inside = []
south, north, west, east = CHINA
for path in sample(visdrone_images()):
    places, _ = guesses(model, path)
    hits = [south <= lat <= north and west <= lon <= east for lat, lon in places]
    inside.append(hits[0])
    any_inside.append(any(hits))
results["visdrone"] = {"images": len(inside), "first_guess_in_china": float(np.mean(inside)), "any_of_five_in_china": float(np.mean(any_inside))}
print(results["visdrone"], flush=True)

results["own_videos"] = {}
for path in sorted(p for p in INPUT.rglob("*.jpg") if "location-frames" in str(p)):
    places, chances = guesses(model, path)
    results["own_videos"][path.stem] = [{"lat": round(a, 3), "lon": round(b, 3), "chance": round(c, 4)} for (a, b), c in zip(places, chances)]
    print(path.stem, results["own_videos"][path.stem][:3], flush=True)

(OUT / "guess_region.json").write_text(json.dumps(results, indent=2))
