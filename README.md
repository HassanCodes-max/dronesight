# DroneSight

Add a drone video. Anyone in water, in fire, beside flames or lying still and alone gets a red box and a number on the video, and each number is explained. Everyone else stays in a black box and needs nothing from you. All of it runs on your own device.

After a disaster, teams come back with hours of drone video and someone has to watch all of it and decide who needs help first. DroneSight does that first pass. The video is analysed in your browser and is never uploaded.

**Live demo:** https://hassanali0987654-dronesight.static.hf.space

**This is a research demo, not a rescue system.** The models miss people, the danger rules have not been checked by emergency responders, and locations are estimates. Nothing is sent to officials unless you send it yourself. In a real emergency, call your local emergency services.

**Status:** live on a free static site with no server. It was checked by eye on one real drone video of a flooded camp, one real phone clip of a flood and one real, calm park video, and on clips panned across disaster photos. One real fire clip was tried: a dark, low-angle fire show, where it marked most of the crowd red, because it cannot tell a performer from a victim or judge distance from that angle. It works best from a drone in daylight. On the low-angle phone clip it over-marked people along the bank and missed a man on a boat who was only in view for two seconds.

## How it works

```
Your video, in your browser
   |
YOLO11s at 1280 for small people seen from above, plus YOLO11n at 640 for close or poor-quality ones,
on the graphics card through WebGPU or the processor through WebAssembly
   |
tracker.js: removes the drone's motion and follows each person
   |
U-Net for water and YOLO11s plus flame colour for fire, every second, on every video
   |
risk.js: marks each person red or black at their worst moment, with reasons
   |
The video file's own location tag, a DJI .SRT log, or the sending device: turn people into coordinates
   |
A report you can copy or open in WhatsApp
```

**Water outlines** are cleaned after the model: only confident pixels count, gaps the size of a person are filled, and thin strips and small patches are removed. This was tuned by eye on real flood videos, where the raw outline spread onto roofs, walls and heads and put red boxes on people standing beside the water. It costs some foamy water at the edges.

**Who turns red** comes from fixed rules in `site/risk.js`, so every decision can be explained. A person is red if they are in water or fire, within 5 m of flames, or lying still and alone. Dry land beside water stays black, and so does smoke alone, because its boxes are coarse. The water model learned daylight floods and takes smoke and dark ground for water, so its result is ignored in any frame that is 10% or more smoke, shows flames, or is mostly dark, and in the whole video once a quarter of its frames are. The page says when a video was too dark to judge water. Flames only count if the check before or after saw flames close by, because a real fire stays put, while a fireball or a swung torch would otherwise turn a whole crowd red. Points only order the red people: fire first, then how far they are from safety, then being alone and not moving. The rules are written by hand, not learned from responders' decisions.

**Distances** use each person's own height as the ruler, because an ordinary video has no scale. They are rough, and worse when the camera looks straight down, where people appear short.

**Lying still** is flagged from the track: a box wider than tall that stays put for 4 seconds. It is the only sign of someone hurt away from water or fire, and it will miss people and raise false alarms. No model here recognises cliffs or falls.

**Location** is never typed in. A normal video has no GPS, so coordinates come from the first of these that exists: a DJI .SRT log, which gives each person their own spot with an error range; the location tag many phone videos carry; or, when someone presses Send on WhatsApp, their own device, with its permission. The last two are the area, not each person's spot, and the report says which it was. With none of them the report says location unknown, because a made-up coordinate could send a team to the wrong place.

**The shortcut** is the straight line to the nearest safe ground, measured on the hazard mask, and it does not know whether a route is passable.

**Reporting** stays with a person. The page shows a demo alert, writes the report and opens WhatsApp, but it never says officials were told, because someone in an emergency could stop trying if they believed help was coming.

**What it cannot see.** No model here recognises fights, attacks or harassment, so a video of one gets no red box unless someone also ends up lying still and alone.

**A language model was removed.** A small one wrote plain-language explanations, but it invented times, contradicted itself and promised a team would arrive quickly, so the page now writes every message from the measurements.

**Tracking** runs in the browser. ONNX Runtime Web runs the full 1280 people model on the visitor's graphics card through WebGPU, or on their processor through WebAssembly. `tracker.js` removes the drone's own motion by comparing frames, predicts where each person will be, matches sure detections first and lets weak ones keep a track alive, then fills gaps from both sides, because analysis runs after the flight. On a 29-second, 1080p Pexels video it followed about 186 people, and the median person was followed for 16 seconds. That is a visual check on one video, not a tracking benchmark.

**Two people models.** The main model learned only small figures from above, and on a real phone clip of a flood it missed a man on a boat and most of a crowd beside the water. YOLO11n's COCO weights, exported by `kaggle/close`, found them, and they are not retrained here. The two are complementary: on a camp video the main model found 17 small people where YOLO11n found none. People whose feet are out of view are not judged, because where they stand cannot be seen.

A smaller 960 model was tried first for speed. Its test recall looked close, at 37% against 44% on VisDrone, but on that real video it found only 9 to 24 people per frame against the full model's 55 to 118, so it was dropped.

**Speed.** Nothing is uploaded, so the video shows at once and the check walks through it on screen, drawing boxes as each moment is read. On the development laptop's Intel graphics the main people model took 1.35 s a frame, and the water and fire models about 0.3 s each, but profiling a 20-second video showed time going to three avoidable places:

| Where the time went | Fix |
|---|---|
| Grey padding: a widescreen frame in a square input is about 44% padding | The three YOLO models were re-exported to take any shape (`kaggle/fast`), so they read the frame's own shape |
| Water and fire clean-up in JavaScript: about 0.8 s a check, more than the models | Distance maps compare squared distances from lookup tables, and canvases are reused |
| Reading the slow models on every sample | The small model and camera motion run twice a second, which keeps tracking through a moving camera. The main model and hazards run once a second |

| Video | Before | After |
|---|---|---|
| 20 s drone video of a flooded camp | 75 s | 40 s |
| 34 s phone clip of a flood | 89 s | 56 to 68 s |
| 29 s park video | 93 s | 74 s |
| 16 s fire clip | 70 s | 44 s |

Reading the main model once a second first dropped a man who stands in plain view on a roof, because tracks broke. Keeping the cheap samples at twice a second brought him back, and that is why the design has two speeds. Videos that would take over a minute read the slow models every 2 seconds. Models saved by an earlier visit are made ready as the page opens, so the first boxes appear in about 1.5 seconds. On a first visit that takes about 15 seconds, plus the download of about 100 MB, which then stays in the browser. A first visit downloads nothing until a video is added, to protect phone data.

Two ways to go faster were tested and rejected. 8-bit weights would halve the download, but on real frames one went from 8 people found to 12 and water outlines agreed only 88%, so the models stay at 16-bit. `deploy/shrink_models.py` halved them that way; on real frames the same people were found, scores moved by under 0.001, and water outlines agreed 99.96%. The 960 model is faster but finds far fewer people, as above. Only the runtime is compressed, from 29 MB to 7 MB, because the models barely shrink.

```
pip install -r requirements.txt
python deploy/fetch_ort.py
python deploy/build_static.py dist
python -m http.server 8000 --directory dist
```

The models go in `models/`: `person.onnx`, `close.onnx`, `water.onnx` and `fire.onnx`, exported by `kaggle/browser`, `kaggle/close` and `kaggle/hazards`. `python deploy/shrink_models.py` then makes the smaller copies that get published. To check the whole page, give it any video:

```
python tests/danger_check.py http://127.0.0.1:8000 your-video.mp4 outputs/check
```

## Deploy

```
python deploy/build_static.py dist
python deploy/push_static.py <your-hf-username>/dronesight dist
```

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

D-Fire is largely not aerial, so its score says little about a drone's view. Fire boxes were checked by eye on C2A's aerial scenes: they land on real flames but miss some, for example in forest scenes. Boxes also marked bystanders as in the fire, so the page cuts each box down to flame-coloured pixels.

### Experiments 5 and 6: tracing water by pixel

Test pixel IoU, higher is better, with both models scored on the same images. Both use a U-Net with a ResNet-34 encoder.

| Model | Trained on | FloodNet | Flood Area |
|---|---|---|---|
| Experiment 5 | FloodNet, wide colour shifting | 0.571 | 0.500 |
| **Experiment 6 (shipped)** | FloodNet + Flood Area, narrow colour shifting | **0.574** | **0.848** |

Experiment 5 traced grey and blue water but missed brown floodwater and shaded grey roofs, because FloodNet is shot straight down and its water is rarely muddy. Colour shifting could not close that gap. Experiment 6 adds Flood Area Segmentation, 290 photos of real floods, and picks the best epoch on the mean validation IoU of both datasets. Flood Area is split at random, so photos of one flood can land on both sides, and 0.848 may be a little high. Its Kaggle listing says CC0, but it also says its images were collected from the internet, so only the trained weights are used here and its images are not republished.

## Data and credits

The source code is under the MIT licence. The people and fire models were made with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), which is AGPL-3.0. It does not cover images, video or datasets, which belong to their own owners. This is a free, non-commercial student project, and anything here will be removed if an owner asks.

- **Training only, no images republished:** [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) by Zhu et al. (2021), [HERIDAL](https://www.kaggle.com/datasets/mariannaesposito/heridal-4k), [C2A](https://github.com/Ragib-Amin-Nihal/C2A) by Nihal et al. (2024), FloodNet, D-Fire and [Flood Area Segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation).

## Update log

- Added a second people model, a cleaner water outline, a water check every second, half-size model files and location from the device, after tests on real flood videos showed the first versions failing.
- Made the video appear at once, with the check running while it plays, and removed the retired simulation, server and assistant code. They remain in the git history.
- Made the result binary: red numbered boxes for anyone in water, in fire, beside flames or lying still and alone, black boxes for everyone else, and a list that explains only the numbered people.
- Rebuilt the page around one job: add a video and see who needs help first. Water and fire are now checked on every video, so there is nothing to choose, and the language model was replaced by messages written from the measurements. On a real, calm park video with about 186 people it raised no false alarms.
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
