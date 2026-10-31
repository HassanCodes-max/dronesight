# DroneSight

Add a drone video of a flood or fire. Every person is followed, ranked by danger with the reasons shown, and placed on the map when the drone's position is known. All of it runs on your own device.

After a disaster, teams come back with hours of drone video and someone has to watch all of it and decide who needs help first. DroneSight does that first pass. The video is analysed in your browser and is never uploaded.

**Live demo:** https://hassanali0987654-dronesight.static.hf.space

**This is a research demo, not a rescue system.** The models miss people, the risk rules have not been validated by emergency responders, and locations are estimates. In a real emergency, call your local emergency services.

**Status:** live on a free static site with no server. An earlier simulated mission was retired from the site, and its code remains in the repository. Telegram alerts are written but switched off.

## How it works

```
Your video, in your browser
   |
YOLO11s at 1280: finds people, on the graphics card through WebGPU or the processor through WebAssembly
   |
tracker.js: removes the drone's motion and follows each person
   |
U-Net (flood) or YOLO11s plus flame colour (fire): finds the hazard every 2 seconds
   |
risk.js: ranks each person at their worst moment, with reasons
   |
Drone position or DJI .SRT log: turns picture positions into latitude and longitude
   |
Cloudflare Worker: explains one finding in plain language, from measurements only
```

**Risk ranking** uses fixed rules in `dronesight/app/static/risk.js`, so every decision can be explained. Being in fire or water counts most, then how far the person is from safety, then being near a hazard, being alone and not moving. Someone 3 m or more into water is critical, because they cannot just step out. Dry land beside water is a watch, while dry land beside flames is high, because fire spreads and burns at a distance. Smoke alone is only a watch, because its boxes are coarse. The rules are written by hand, not learned from responders' decisions.

**Distances** use each person's own height as the ruler, because an ordinary video has no scale. They are rough, and worse when the camera looks straight down, where people appear short.

**Location** needs the drone's position, height, heading and camera tilt, typed in or read from a DJI `.SRT` flight log. An ordinary video has no GPS, so without these the page says the location is unknown instead of guessing. The maths assumes flat ground, and each result shows an error range that grows with tilt. Typed positions assume the drone stayed there. The flight log parser was tested on written examples of the two DJI formats, not on a real log.

**The shortcut** is the straight line to the nearest safe ground, measured on the hazard mask, plus a Google Maps directions link from the team to the person. Neither knows whether a route is passable, and the assistant is told never to say a route is safe.

**The assistant** gets measurements, not text. The worker checks every field against fixed values and ranges and writes the findings itself, so nothing a visitor types reaches the model.

The recorded flight page has its own pipeline:

```
Browser (plain JS, no third-party scripts)
   |
FastAPI: accounts, uploads, search, rate limits
   |
SQLite: users, videos, every detection with its time and track
   |
Worker process: decodes video, runs YOLO11s at 1280 through ONNX Runtime on CPU, tracks people
```

**Your own video:** the free site has no server to send a file to, so the visitor's device does the work. ONNX Runtime Web runs the full 1280 people model on their graphics card through WebGPU, or on their processor through WebAssembly. `tracker.js` removes the drone's own motion by comparing frames, predicts where each person will be, matches sure detections first and lets weak ones keep a track alive, then fills gaps from both sides, because analysis runs after the flight. It samples up to five times a second, as many as fit about seven minutes on that device. `tests/upload_check.py` checks that the page makes no upload request. On a 29-second, 1080p Pexels video it followed about 186 people in just under 3 minutes on the development laptop, and the median person was followed for 16 seconds. That is a visual check on one video, not a tracking benchmark.

A smaller 960 model was tried first for speed. Its test recall looked close, at 37% against 44% on VisDrone, but on that real video it found only 9 to 24 people per frame against the full model's 55 to 118, so it was dropped.

The model runs through ONNX, so the web server needs no PyTorch or GPU. At the 1280 input it takes about 565 ms per frame on a Kaggle CPU and about 650 ms on the development laptop, so a 3-minute flight sampled every 0.25 s indexes in about 8 minutes.

Search is deterministic and runs in the browser: a question becomes a filter on the flight's detections, which also drive the live boxes and the activity timeline. It costs nothing per query, and text can never become instructions.

Free questions go to a Cloudflare Worker that calls Workers AI. The facts live in the worker, so a visitor cannot supply false ones. The model has no tools, its replies are filtered for leaked instructions and shown as plain text, and requests are rate limited by IP and origin.

## Security

No website is attack-proof. These are layered defences, and `tests/test_app.py` attacks them directly.

**Built and tested**

- **Passwords and sessions:** Argon2 hashing, only hashes of session tokens are stored, and cookies are HttpOnly and SameSite=Strict.
- **Access control:** another user's video returns 404 on every route, so its existence is not revealed.
- **Brute force and abuse:** rate limits per IP on logins and the API, which forged forwarded headers cannot dodge.
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

`tests/worker_check.py` runs the built worker in a real browser against refused origins, oversized questions, odd mission names, leaked instructions and rate limits. Visitors can only choose which critical person to alert about, and the worker writes every message itself, so a visitor's text never reaches Telegram. Each alert says it is a simulation, and nothing contacts real emergency services.

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

The live demo is a static build. It needs no server, so there is nothing to keep running and no backend to attack, but visitors can only explore the sample flight:

```
python -m dronesight.app.seed demo/flight.mp4 "Sample flight"
python deploy/build_static.py dist
python deploy/push_static.py <your-hf-username>/dronesight dist
```

Static hosts cannot send security headers, so its Content Security Policy is in a meta tag, which cannot stop framing.

`run_public.bat` serves the full site from a Windows machine through a temporary Cloudflare link, which needs no account or payment. It only works while that machine is on, and the link changes on each restart. Sign-up, search, upload and the attack checks have passed through that link.

For a permanent home, the site runs as two containers on an Ubuntu server: the app, and Caddy in front for HTTPS. The app is not exposed directly, so its rate limits see each visitor's real IP.

```
sh deploy/push_server.sh azureuser@<server-ip> <site-domain>
```

The Dockerfile has not been built yet, because the development machine has no Docker. Hugging Face Docker Spaces were the first plan, but they now need a paid plan.

## Experiments

Training runs on Kaggle GPUs. Models are chosen on validation data, never on test scores.

| # | Dataset | Question | Status |
|---|---|---|---|
| 1 | VisDrone | Which of YOLO11n at 640, YOLO11n at 960 or YOLO11s at 960 is best? | Done: YOLO11s at 960 |
| 2 | VisDrone, HERIDAL, C2A | Does mixing datasets help find people in footage from a dataset the model never saw? | Done: no measurable gain |
| 3 | VisDrone, HERIDAL | What finds tiny people best within the site's CPU budget? | Done: VisDrone + HERIDAL at 1280 |
| 4 | FloodNet, D-Fire | Can models see the hazard as well as the people? | Done: fire boxes kept, flood outlines too rough |
| 5 | FloodNet | Does a pixel model trace water better than instance masks? | Done: U-Net, but it misses brown water |
| 6 | FloodNet, Flood Area | Does training on real floods fix brown water? | Done: test IoU 0.50 to 0.85 on real floods |

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
- **Input size:** moving from 960 to 1280 costs 1.4 to 2.3 times the CPU time: 579 ms against 410 ms per frame on Kaggle, and 115 s against 50 s to index the sample flight on the development laptop.
- **Tiling:** 2 x 2 tiles plus the whole frame scored 0.245 for five model passes per frame with linear shrinking, against 0.244 for one pass at 1280, so it was dropped.
- **C2A:** with 2 x 2 tiles, only 16% of the all-three model's HERIDAL boxes were correct, against 51% without C2A, so the training run drops it.

That model won on the mean, but 0.250 against 0.246 is close to a tie and it finds far fewer people in HERIDAL. Part 2 repeats HERIDAL three times in training to address that.

### Experiment 3 training (15 epochs, HERIDAL repeated three times, validation data)

| Model | Input | Mean mAP50-95 | VisDrone found / correct | HERIDAL found / correct |
|---|---|---|---|---|
| YOLO11s with a stride-4 layer | 960 | 0.226 | 60% / 78% | 36% / 56% |
| **YOLO11s (shipped)** | 1280 | **0.274** | **66% / 77%** | **43% / 68%** |

The stride-4 layer added detail for small objects but lost on both datasets, so it was dropped.

### Final people model on test data, scored once

Through the site's shrinking and NMS, on images that played no part in any choice. This is the number to quote.

| Test set | Images | People | mAP50-95 | mAP50 | Found | Correct |
|---|---|---|---|---|---|---|
| VisDrone test-dev | 1,610 | 27,382 | 0.197 | 0.478 | 44% | 69% |
| HERIDAL, four places held out | 404 | 1,743 | 0.391 | 0.714 | 67% | 72% |

Found and correct are at confidence 0.25, with a box counted correct at IoU 0.5. The same model found 66% in VisDrone validation and 43% in HERIDAL validation, so a single location or split can move this figure a lot. It misses between a third and just over half of people, which is why the site says so and why a real system would need a person watching the feed.

### Experiment 4: seeing the hazard (each model scored once on its own test split)

| Hazard | Model | Dataset | Test score |
|---|---|---|---|
| Flood | YOLO11s-seg | FloodNet, 398 labelled images, split by blocks of neighbouring frames | Pixel IoU 0.543, mask mAP50 0.170 |
| Fire and smoke | YOLO11s | D-Fire | mAP50-95 0.415, mAP50 0.739: smoke 0.476, fire 0.354 |

D-Fire is largely not aerial, so its score says little about a drone's view. Fire boxes were checked by eye on C2A's aerial scenes: they land on real flames but miss some, for example in forest scenes. Boxes also marked bystanders as in the fire, so the mission cuts each box down to flame-coloured pixels.

### Experiments 5 and 6: tracing water by pixel

Test pixel IoU, higher is better, with both models scored on the same images. Both use a U-Net with a ResNet-34 encoder.

| Model | Trained on | FloodNet | Flood Area |
|---|---|---|---|
| Experiment 5 | FloodNet, wide colour shifting | 0.571 | 0.500 |
| **Experiment 6 (shipped)** | FloodNet + Flood Area, narrow colour shifting | **0.574** | **0.848** |

Experiment 5 traced grey and blue water but missed brown floodwater and shaded grey roofs, because FloodNet is shot straight down and its water is rarely muddy. Colour shifting could not close that gap. Experiment 6 adds Flood Area Segmentation, 290 photos of real floods, and picks the best epoch on the mean validation IoU of both datasets. Flood Area is split at random, so photos of one flood can land on both sides, and 0.848 may be a little high. Its Kaggle listing says CC0, but it also says its images were collected from the internet, so only the trained weights are used here and its images are not republished.

## Data and credits

The source code is under the MIT licence. It does not cover images, video or datasets, which belong to their own owners. This is a free, non-commercial student project, and anything here will be removed if an owner asks.

- **Mission scenes:** [C2A](https://github.com/Ragib-Amin-Nihal/C2A) by Nihal et al. (2024), which adds people from LSP/MPII-MPHB to disaster photos from [AIDER](https://github.com/ckyrkou/AIDER) by Kyrkou and Theocharides (2019), released for non-commercial and academic use. C2A's repository states no licence.
- **Sample flight:** [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) by Zhu et al. (2021), AISKYEYE team, Tianjin University. Its repository states no licence, so it is shown as a non-commercial demonstration.
- **Training only, no images republished:** [HERIDAL](https://www.kaggle.com/datasets/mariannaesposito/heridal-4k), FloodNet, D-Fire and [Flood Area Segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation).

## Update log

- Added danger ranking, map locations and plain-language explanations on visitors' own videos, all in the browser.
- The product moved from the simulated mission to visitors' own videos. Tracking was added in the browser, and the browser model moved back to the full 1280 version after a real video showed the 960 version missing most people.
- Added analysis of a visitor's own video in their browser, with no server. Missions moved to the largest source photos, up to 3600 pixels wide against as little as 230, because many scenes were blurry when shown large. The assistant now reads its facts from the site, so new data needs no worker update.
- Missions moved to the Experiment 6 water model, and the people model was scored once on test data. The site now quotes the test figures, which are lower for VisDrone and higher for HERIDAL than the validation ones.
- With accurate water, 168 of 215 flood people ranked high, because dry land beside water scored almost the same as standing in it. Fire and flood now score distance differently, and nothing scores points for being near a smoke box.
- Rebuilt around emergencies: a simulated drone flies flood and fire missions, ranks people by risk, reports critical locations to headquarters and moves on. Hazards were traced first with boxes, then outlines, then pixels.
- Page rebuilt around the video: boxes stay on during playback, a timeline shows people across the flight, ranked moments appear before any search, and uploads can be deleted. Search moved to the browser, and a free static edition went live.
- Experiment 3 inference tests: the site now shrinks frames with area averaging and runs the VisDrone + HERIDAL model at 1280. Deployment moved from Hugging Face to an Ubuntu server with Caddy.
- Experiment 2 finished: mixing datasets gave no measurable gain on unseen footage. The site moved to the model trained on all three, chosen on validation data.
- Web app, attack tests, ONNX export and deployment script added. The first site model is the Experiment 1 winner, keeping only its two person classes.
- Experiment 1 finished, YOLO11s at 960 won. Experiment 2 added: HERIDAL is re-split by location and C2A by base scene to stop near-identical images leaking across splits.
- Experiment 1 script and Kaggle config added. Training moved to Kaggle.
- Repo scaffold and first tracking script (YOLO + ByteTrack, one row per object per frame).
