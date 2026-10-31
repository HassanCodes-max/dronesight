"""Turn model output into the mission file the simulator plays.

    python deploy/build_mission_data.py outputs/mission dronesight/app/static/mission

Reads the Kaggle job's mission_scenes.json and scenes, lays the scenes out as a search grid over a
simulated flight, then decides each person's hazard, ranking and map position with rescue.py.
Locations and flight data are simulated. People and hazards are model output.
"""
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dronesight.app.rescue import ALONE_RADIUS_M, Person, Telemetry, locate, location_error_m, rank  # noqa: E402

ALTITUDE_M = 40
CAMERA_FOV_DEGREES = 70
GRID = 3
RASTER_WIDTH = 256
AREAS = {"flood": {"title": "Flood", "latitude": 34.015, "longitude": 71.975},
         "fire": {"title": "Fire", "latitude": 24.860, "longitude": 67.010}}


def rasterise(scene, height):
    """One mask per hazard, in a small grid. Fire and smoke boxes become filled rectangles."""
    masks = {}
    flood = np.zeros((height, RASTER_WIDTH), np.uint8)
    for outline in scene["flood"]:
        points = (np.array(outline) * [RASTER_WIDTH, height]).astype(np.int32)
        cv2.fillPoly(flood, [points], 1)
    masks["flood water"] = flood
    for name in ["fire", "smoke"]:
        mask = np.zeros((height, RASTER_WIDTH), np.uint8)
        for x1, y1, x2, y2 in scene[name]:
            cv2.rectangle(mask, (int(x1 * RASTER_WIDTH), int(y1 * height)), (int(x2 * RASTER_WIDTH), int(y2 * height)), 1, -1)
        masks[name] = mask
    return masks


def hazard_at(masks, x, y, metres_per_cell):
    """What the point stands in and how deep, otherwise the nearest hazard and its distance in metres."""
    height, width = next(iter(masks.values())).shape
    column, row = min(width - 1, int(x * width)), min(height - 1, int(y * height))
    nearest, distance = None, None
    for name, mask in masks.items():
        if mask[row, column]:
            depth = cv2.distanceTransform(mask, cv2.DIST_L2, 5)[row, column]
            return name, float(depth * metres_per_cell), None, None
        if mask.any():
            cells = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)[row, column]
            if distance is None or cells * metres_per_cell < distance:
                nearest, distance = name, float(cells * metres_per_cell)
    return None, 0, nearest, distance


def build_mission(name, scenes):
    area = AREAS[name]
    aspect = scenes[0]["aspect"]
    probe = Telemetry(area["latitude"], area["longitude"], ALTITUDE_M, 0, CAMERA_FOV_DEGREES, aspect)
    sectors, people = [], []
    positions = {}
    for number, scene in enumerate(scenes):
        row, column = number // GRID, number % GRID
        # Sector centres step one camera footprint apart, so the grid covers the area with no gaps.
        north = (GRID / 2 - row - 0.5) * probe.ground_height_m
        east = (column - GRID / 2 + 0.5) * probe.ground_width_m
        telemetry = Telemetry(area["latitude"] + north / 111_320,
                              area["longitude"] + east / (111_320 * math.cos(math.radians(area["latitude"]))),
                              ALTITUDE_M, 0, CAMERA_FOV_DEGREES, scene["aspect"])
        height = round(RASTER_WIDTH / scene["aspect"])
        masks = rasterise(scene, height)
        metres_per_cell = telemetry.ground_width_m / RASTER_WIDTH
        sector = {"id": f"{'ABC'[row]}{column + 1}", "row": row, "column": column, "image": f"mission/{Path(scene['image']).name}",
                  "aspect": scene["aspect"], "flood": scene["flood"], "fire": scene["fire"], "smoke": scene["smoke"], "people": []}
        sectors.append(sector)
        for found in scene["people"]:
            x1, y1, x2, y2 = found["box"]
            x, y = (x1 + x2) / 2, y2  # feet touch the ground, so they decide what the person stands in
            hazard, depth, nearest, distance = hazard_at(masks, x, y, metres_per_cell)
            person = Person(id=len(people), x=x, y=y, confidence=found["confidence"], hazard=hazard, depth_m=depth,
                            distance_m=distance, nearest_hazard=nearest)
            people.append(person)
            positions[person.id] = (sector, found["box"], *locate(x, y, telemetry),
                                    east + (x - 0.5) * telemetry.ground_width_m,
                                    north + (0.5 - y) * telemetry.ground_height_m)
    for person in people:
        here = positions[person.id]
        person.neighbours = sum(1 for other in people if other.id != person.id
                                and math.hypot(here[4] - positions[other.id][4], here[5] - positions[other.id][5]) <= ALONE_RADIUS_M)
    for order, person in enumerate(rank(people)):
        sector, box, latitude, longitude, *_ = positions[person.id]
        sector["people"].append({"id": person.id, "rank": order + 1, "box": box, "confidence": person.confidence, "level": person.level,
                                 "score": person.score, "reasons": person.reasons, "latitude": round(latitude, 6), "longitude": round(longitude, 6)})
    return {"title": area["title"], "altitude_m": ALTITUDE_M, "error_m": round(location_error_m(probe)),
            "sector_width_m": round(probe.ground_width_m), "sectors": sectors}


def main(source, target):
    source, target = Path(source), Path(target)
    missions = json.loads((source / "mission_scenes.json").read_text())
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source / "scenes", target)
    data = {name: build_mission(name, scenes) for name, scenes in missions.items()}
    (target / "mission.json").write_text(json.dumps(data, separators=(",", ":")))
    for name, mission in data.items():
        all_people = [p for s in mission["sectors"] for p in s["people"]]
        levels = {level: sum(p["level"] == level for p in all_people) for level in ["CRITICAL", "HIGH", "WATCH", "CLEAR"]}
        print(name, len(all_people), "people", levels, "error", mission["error_m"], "m, sector", mission["sector_width_m"], "m")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
