"""Build the list of world towns that turns a position into a place name.

    python deploy/build_places.py

Names come from GeoNames (CC BY 4.0). They are published with the page, so naming a place sends nothing
anywhere. Each line is latitude, longitude, town, region and country code.
"""
import csv
import io
import zipfile
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent.parent / "site/places.csv"
TOWNS = "https://download.geonames.org/export/dump/cities15000.zip"
REGIONS = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"
SMALLEST_TOWN = 30000  # smaller towns add size, and "near" a larger one is enough to send help to an area


def main():
    archive = httpx.get(TOWNS, follow_redirects=True, timeout=300).raise_for_status().content
    regions = {}
    for line in httpx.get(REGIONS, follow_redirects=True, timeout=300).raise_for_status().text.splitlines():
        code, name, *_ = line.split("\t")
        regions[code] = name
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        rows = zipped.read("cities15000.txt").decode("utf8").splitlines()
    kept = []
    for row in rows:
        field = row.split("\t")
        if int(field[14]) < SMALLEST_TOWN:
            continue
        region = regions.get(f"{field[8]}.{field[10]}", "")
        kept.append([round(float(field[4]), 2), round(float(field[5]), 2), field[2], region, field[8]])
    with OUT.open("w", encoding="utf8", newline="\n") as file:
        csv.writer(file, lineterminator="\n").writerows(kept)
    print(f"{len(kept)} towns, {OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
