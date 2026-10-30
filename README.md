# DroneSight

Ask your drone footage questions in plain English.

Drone teams return with hours of video and scrub through it by hand to find what matters, for example in flood response, search and rescue, or site inspection. DroneSight finds the people in a flight, indexes every detection by time, and takes you straight to the moment you asked about.

**Status:** working web app with a real trained detector, tested locally. It is not deployed yet, and Experiment 2 is still training. Search currently understands questions about people, by count and time. Map pins, LLM questions and before and after comparison are planned, not built.

## How it works

```
Browser (plain JS, no third-party scripts)
   |
FastAPI: accounts, uploads, search, rate limits
   |
SQLite: users, videos, every detection with its time and track
   |
Worker process: decodes video, runs YOLO11s through ONNX Runtime on CPU, tracks people
```

The model runs through ONNX, so the web server needs no PyTorch or GPU. It took about 285 ms per frame on a Kaggle CPU.

Search is deterministic: a question becomes a filter on stored detections. It costs nothing per query, and text can never become instructions, so there is no prompt injection risk until an LLM is added.

## Security

No website is attack-proof. These are layered defences, and `tests/test_app.py` attacks them directly.

**Built and tested**

- **Passwords and sessions:** Argon2 hashing, only hashes of session tokens are stored, and cookies are HttpOnly and SameSite=Strict.
- **Access control:** another user's video returns 404 on every route, so its existence is not revealed.
- **Brute force and abuse:** rate limits per IP on logins, searches and the API, which forged forwarded headers cannot dodge.
- **Runaway costs:** per-user upload quotas, plus a site-wide daily cap on uploads and processing time.
- **Uploads:** declared size is checked and bytes are counted while streaming, file type is checked from content, file names are server-generated, and videos are deleted after 24 hours.
- **Browser attacks:** strict Content Security Policy, frame blocking, cross-site write rejection, and server text is only ever rendered as text.
- **SQL injection:** every value is passed as a query parameter.

**Not done yet**

- **DDoS:** the app can only limit requests that reach it. Real volumetric protection needs a CDN such as Cloudflare in front, which needs a custom domain.
- **Worker isolation:** the worker is a separate process, but it shares a container with the web server.
- **Accounts:** no email verification or password reset yet.
- **Rate limits:** counters are per process and reset on restart.
- **Independent testing:** no external scan or penetration test has been run.

## Run locally

Put the exported model at `models/person.onnx` and a sample at `demo/flight.mp4`, then:

```
pip install -r requirements-web.txt
python -m dronesight.app.seed demo/flight.mp4 "Sample flight"
python -m dronesight.app.worker
python -m uvicorn dronesight.app.main:app --port 8000
```

Run the worker and the server in separate terminals, then open http://127.0.0.1:8000. Tests run with `python -m pytest tests`.

## Deploy

```
huggingface-cli login
python deploy/push_space.py <your-hf-username>/dronesight
```

The Dockerfile has not been built yet, because the development machine has no Docker. Sample footage comes from VisDrone, so check its terms before publishing.

## Experiments

Training runs on Kaggle GPUs. Models are chosen on the validation split, and only the winner is scored once on the test split.

| # | Dataset | Question | Status |
|---|---|---|---|
| 1 | VisDrone | Which of YOLO11n at 640, YOLO11n at 960 or YOLO11s at 960 is best? | Done: YOLO11s at 960 |
| 2 | VisDrone, HERIDAL, C2A | Does mixing datasets help find people in footage from a dataset the model never saw? | Full run on Kaggle |

### Experiment 1 results (10 classes, 30 epochs)

| Model | Image size | Val mAP50-95 | Val mAP50 | Inference on T4 |
|---|---|---|---|---|
| YOLO11n | 640 | 0.163 | 0.292 | 2.2 ms |
| YOLO11n | 960 | 0.232 | 0.394 | 4.9 ms |
| YOLO11s | 960 | 0.291 | 0.477 | 12.8 ms |

The winner scored mAP50-95 0.231 and mAP50 0.397 on test-dev, which was scored once. People are the weak spot: test mAP50-95 is 0.154 for pedestrian and 0.072 for people, against 0.506 for car.

## Update log

- Web app, attack tests, ONNX export and deployment script added. The first site model is the Experiment 1 winner, keeping only its two person classes.
- Experiment 1 finished, YOLO11s at 960 won. Experiment 2 added: HERIDAL is re-split by location and C2A by base scene to stop near-identical images leaking across splits.
- Experiment 1 script and Kaggle config added. Training moved to Kaggle.
- Repo scaffold and first tracking script (YOLO + ByteTrack, one row per object per frame).
