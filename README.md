# DroneSight

Ask your drone footage questions in plain English.

Drone teams return with hours of video and scrub through it by hand to find what matters (flood response, search and rescue, site inspection). DroneSight detects and tracks what is in the footage, indexes it with GPS, and answers questions like *"where were people near flooded roads?"* with the exact clips, a map pin and a short report.

**Status:** week 1, detection and tracking pipeline. Nothing is benchmarked yet, so no results are claimed.

## Plan

| Week | Goal |
|---|---|
| 1 | Detection and tracking on VisDrone and FloodNet, accuracy measured |
| 2 | Searchable index, plain-English questions return the right clips |
| 3 | Web app with accounts, video, map, chat and report export |
| 4 | Security hardening, public deployment, results write-up, launch post |

## Security goals

No website is attack-proof, so the aim is layered defence that is tested and published, not a claim of being unhackable. Nothing below is built yet.

- **DDoS and abuse:** CDN and WAF in front of the site, bot checks, and rate limits per IP and per account.
- **Runaway costs:** per-user upload and question quotas, plus a hard daily cap on GPU and LLM spend.
- **Accounts:** secure login and strict access checks, so users can only reach their own footage.
- **Uploads:** size, duration and real file type checks, processing in an isolated worker, and automatic deletion.
- **Prompt injection:** text from footage is treated as data, and the LLM cannot call anything that changes state.
- **Proof:** security headers, dependency and secret scanning in CI, and an OWASP ZAP scan of our own site with published results.

Footage can show people's faces, homes and locations, so the app needs a clear privacy notice and short retention by default.

## Run

```
pip install -r requirements.txt
python -m dronesight.detect.track data/raw/clip.mp4 --out outputs/clip_tracks.parquet
```

## Experiments

Training runs on Kaggle GPUs. Models are chosen on the validation split, and only the winner is scored once on the test split.

| # | Dataset | Question | Status |
|---|---|---|---|
| 1 | VisDrone | Which of YOLO11n at 640, YOLO11n at 960 or YOLO11s at 960 is best? | Done: YOLO11s at 960 |
| 2 | VisDrone, HERIDAL, C2A | Does mixing datasets help find people in footage from a dataset the model never saw? | Short trial running |

### Experiment 1 results (10 classes, 30 epochs)

| Model | Image size | Val mAP50-95 | Val mAP50 | Inference on T4 |
|---|---|---|---|---|
| YOLO11n | 640 | 0.163 | 0.292 | 2.2 ms |
| YOLO11n | 960 | 0.232 | 0.394 | 4.9 ms |
| YOLO11s | 960 | 0.291 | 0.477 | 12.8 ms |

The winner scored mAP50-95 0.231 and mAP50 0.397 on test-dev, which was scored once. People are the weak spot: test mAP50-95 is 0.154 for pedestrian and 0.072 for people, against 0.506 for car.

## Update log

- Experiment 1 finished, YOLO11s at 960 won. Experiment 2 added: HERIDAL is re-split by location and C2A by base scene to stop near-identical images leaking across splits.
- Experiment 1 script and Kaggle config added. Training moved to Kaggle.
- Repo scaffold and first tracking script (YOLO + ByteTrack, one row per object per frame).
