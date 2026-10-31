# DroneSight

Ask your drone footage questions in plain English.

Drone teams return with hours of video and scrub through it by hand to find what matters, for example in flood response, search and rescue, or site inspection. DroneSight finds the people in a flight, indexes every detection by time, and takes you straight to the moment you asked about.

**Status:** working web app with a real trained detector, tested locally. It is not deployed yet. Search currently understands questions about people, by count and time. Map pins, LLM questions and before and after comparison are planned, not built.

## How it works

```
Browser (plain JS, no third-party scripts)
   |
FastAPI: accounts, uploads, search, rate limits
   |
SQLite: users, videos, every detection with its time and track
   |
Worker process: decodes video, runs YOLO11s at 1280 through ONNX Runtime on CPU, tracks people
```

The model runs through ONNX, so the web server needs no PyTorch or GPU. At the 1280 input it takes about 580 ms per frame on a Kaggle CPU and about 650 ms on the development laptop, so a 3-minute flight sampled every 0.25 s indexes in about 8 minutes.

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

The site runs as two containers on an Ubuntu server: the app, and Caddy in front for HTTPS. The app is not exposed directly, so its rate limits see each visitor's real IP.

```
sh deploy/push_server.sh azureuser@<server-ip> <site-domain>
```

The Dockerfile has not been built yet, because the development machine has no Docker. Hugging Face Docker Spaces were the first plan, but they now need a paid plan. Sample footage comes from VisDrone. Mirrors of that dataset state different licences, so check the official terms before publishing.

## Experiments

Training runs on Kaggle GPUs. Models are chosen on validation data, never on test scores.

| # | Dataset | Question | Status |
|---|---|---|---|
| 1 | VisDrone | Which of YOLO11n at 640, YOLO11n at 960 or YOLO11s at 960 is best? | Done: YOLO11s at 960 |
| 2 | VisDrone, HERIDAL, C2A | Does mixing datasets help find people in footage from a dataset the model never saw? | Done: no measurable gain |
| 3 | VisDrone, HERIDAL | What finds tiny people best within the site's CPU budget? | Inference tests done, training running |

### Experiment 1 results (10 classes, 30 epochs)

| Model | Image size | Val mAP50-95 | Val mAP50 | Inference on T4 |
|---|---|---|---|---|
| YOLO11n | 640 | 0.163 | 0.292 | 2.2 ms |
| YOLO11n | 960 | 0.232 | 0.394 | 4.9 ms |
| YOLO11s | 960 | 0.291 | 0.477 | 12.8 ms |

The winner scored mAP50-95 0.231 and mAP50 0.397 on test-dev, which was scored once. People are the weak spot: test mAP50-95 is 0.154 for pedestrian and 0.072 for people, against 0.506 for car.

### Experiment 2 results (one `person` class, YOLO11s at 960, 15 epochs, one seed)

Test mAP50-95. Each dataset was split so near-identical images cannot land on both sides: HERIDAL by location and C2A by base scene.

| Trained on | VisDrone | HERIDAL | C2A |
|---|---|---|---|
| VisDrone | 0.167 | 0.232 (unseen) | 0.053 (unseen) |
| VisDrone + C2A | 0.146 | 0.234 (unseen) | 0.617 |
| VisDrone + HERIDAL | 0.163 | 0.312 | 0.052 (unseen) |
| HERIDAL + C2A | 0.059 (unseen) | 0.196 | 0.623 |
| All three | 0.146 | 0.323 | 0.613 |

Mixing did not help on unseen data: adding a second dataset moved HERIDAL from 0.232 to 0.234 and C2A from 0.053 to 0.052. A model that never saw VisDrone scores 0.059 on it, so scores on familiar data say little about a new customer's footage.

C2A is synthetic: [its authors](https://arxiv.org/abs/2408.04922) paste cut-out people onto disaster scenes. Its high scores are not evidence about real footage. It also dominated training, with 252,234 labelled people against 106,396 for VisDrone and 848 for HERIDAL, and both VisDrone + C2A models lost about 0.02 on VisDrone. It made up about 80% of the people in mixed validation sets, so it also largely decided each model's best epoch.

Every run was scored on all three test sets because that comparison was the question, so test results were seen before the site model was picked. To keep the pick independent, the rule was fixed first: the highest mean mAP50-95 on VisDrone and HERIDAL validation images, excluding C2A.

### First site model comparison

Precision and recall are at confidence 0.25, with a box counted correct at IoU 0.5. The scoring code matched Ultralytics on a known case: 0.164 against 0.163. These numbers overstate the site: they use HERIDAL images already shrunk with area averaging and NMS at 0.7, while the site then used linear shrinking and NMS at 0.5. Experiment 3 scores through the site's own settings.

| Model | VisDrone mAP50-95 | VisDrone recall | VisDrone precision | HERIDAL mAP50-95 | HERIDAL recall | HERIDAL precision | Mean mAP50-95 |
|---|---|---|---|---|---|---|---|
| Experiment 1 winner | 0.259 | 0.498 | 0.726 | 0.080 | 0.085 | 0.641 | 0.169 |
| VisDrone | 0.301 | 0.629 | 0.664 | 0.102 | 0.201 | 0.683 | 0.201 |
| VisDrone + HERIDAL | 0.296 | 0.601 | 0.688 | 0.142 | 0.301 | 0.592 | 0.219 |
| **All three (shipped)** | 0.266 | 0.542 | 0.685 | 0.206 | 0.585 | 0.536 | 0.236 |

The all-three model won this comparison and was replaced in Experiment 3. The HERIDAL validation set is small, 482 people in three locations, and VisDrone validation runs much easier than test-dev.

### Experiment 3 inference tests (no training, validation data, site NMS at 0.5)

Found and correct are recall and precision at confidence 0.25.

| Model | Input | Shrinking | Mean mAP50-95 | VisDrone found / correct | HERIDAL found / correct |
|---|---|---|---|---|---|
| All three | 960 | Linear | 0.201 | 53% / 75% | 52% / 47% |
| All three | 960 | Area | 0.228 | 54% / 75% | 57% / 54% |
| All three | 1280 | Area | 0.246 | 60% / 75% | 63% / 40% |
| VisDrone + HERIDAL | 960 | Area | 0.219 | 60% / 75% | 30% / 65% |
| **VisDrone + HERIDAL (shipped)** | 1280 | Area | **0.250** | **65% / 75%** | **34% / 66%** |

- **Shrinking:** area averaging keeps tiny people that linear shrinking loses on 4K frames, at no speed cost. The site now uses it.
- **Input size:** moving from 960 to 1280 adds about 40% CPU time per frame, 579 ms against 410 ms on Kaggle.
- **Tiling:** 2 x 2 tiles plus the whole frame scored 0.245 for five model passes per frame with linear shrinking, against 0.244 for one pass at 1280, so it was dropped.
- **C2A:** with 2 x 2 tiles, only 16% of the all-three model's HERIDAL boxes were correct, against 51% without C2A, so the training run drops it.

The shipped model won on the mean, but 0.250 against 0.246 is close to a tie and it finds far fewer people in HERIDAL. Part 2 repeats HERIDAL three times in training to address that.

## Update log

- Experiment 3 inference tests: the site now shrinks frames with area averaging and runs the VisDrone + HERIDAL model at 1280. Deployment moved from Hugging Face to an Ubuntu server with Caddy.
- Experiment 2 finished: mixing datasets gave no measurable gain on unseen footage. The site moved to the model trained on all three, chosen on validation data.
- Web app, attack tests, ONNX export and deployment script added. The first site model is the Experiment 1 winner, keeping only its two person classes.
- Experiment 1 finished, YOLO11s at 960 won. Experiment 2 added: HERIDAL is re-split by location and C2A by base scene to stop near-identical images leaking across splits.
- Experiment 1 script and Kaggle config added. Training moved to Kaggle.
- Repo scaffold and first tracking script (YOLO + ByteTrack, one row per object per frame).
