"""Read Experiment 1 results and map the attached datasets, so Experiment 2 uses real paths."""
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
report = {"top": [], "exp1_files": [], "datasets": {}}

for r, d, f in os.walk(ROOT):
    depth = len(Path(r).relative_to(ROOT).parts)
    if depth <= 4:
        report["top"].append({"dir": r, "dirs": len(d), "files": len(f)})
    if depth >= 7:
        d[:] = []
    for n in f:
        if n == "experiment1_results.json":
            shutil.copy(Path(r) / n, OUT / n)
            report["exp1_files"].append(str(Path(r) / n))
        if n == "best.pt":
            report["exp1_files"].append(str(Path(r) / n))

for name in ["c2a", "heridal"]:
    base = next((p for p in ROOT.rglob("*") if p.is_dir() and name in p.name.lower()), None)
    if base is None:
        report["datasets"][name] = "not found"
        continue
    per_dir = Counter()
    ext = Counter()
    yamls = {}
    labels = []
    images = []
    for r, d, f in os.walk(base):
        rel = str(Path(r).relative_to(base))
        per_dir[rel] = len(f)
        for n in f:
            ext[Path(n).suffix.lower()] += 1
            p = Path(r) / n
            if n.endswith((".yaml", ".yml")):
                yamls[str(p)] = p.read_text(errors="replace")[:600]
            if n.endswith(".txt") and len(labels) < 4 and "README" not in n:
                labels.append({"file": str(p), "head": p.read_text(errors="replace")[:220]})
            if n.lower().endswith((".jpg", ".png")):
                images.append(n)
    prefixes = Counter(re.sub(r"\d.*$", "", re.sub(r"^(train|valid|test)_", "", n))[:24] for n in images)
    report["datasets"][name] = {"base": str(base), "per_dir": dict(per_dir.most_common(30)),
                                "ext": dict(ext), "yamls": yamls, "labels": labels,
                                "image_count": len(images), "prefixes": dict(prefixes.most_common(40)),
                                "samples": images[:6]}

(OUT / "probe_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1)[:12000])
