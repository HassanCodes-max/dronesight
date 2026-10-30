"""Detect and track objects in a drone video; write one row per object per frame.

Usage:
    python -m dronesight.detect.track data/raw/clip.mp4 --out outputs/clip_tracks.parquet
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


def track_video(video, model_name="yolo11n.pt", stride=2, imgsz=960, conf=0.25):
    """Run detection + ByteTrack on every `stride`-th frame and return a DataFrame."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = YOLO(model_name)
    rows = []
    for idx in tqdm(range(total), desc=Path(video).name):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            continue
        res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          imgsz=imgsz, conf=conf, verbose=False)[0]
        if res.boxes.id is None:
            continue
        for box, tid, cls, c in zip(res.boxes.xyxy.tolist(), res.boxes.id.int().tolist(),
                                    res.boxes.cls.int().tolist(), res.boxes.conf.tolist()):
            rows.append({"frame": idx, "t": idx / fps, "track_id": tid,
                         "label": res.names[cls], "conf": c,
                         "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
    cap.release()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="outputs/tracks.parquet")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    df = track_video(args.video, args.model, args.stride)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"{len(df)} detections, {df.track_id.nunique() if len(df) else 0} tracks -> {args.out}")
