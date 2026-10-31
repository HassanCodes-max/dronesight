"""Build the facts a language model needs to answer free questions about the sample flight.

Counts road users four times a second with the 10-class Experiment 1 model, and has SmolVLM describe
one frame every few seconds. The website's person counts stay with the 1280 model. Captions come
from a small model and can be wrong, so the assistant must present them as a rough description.
"""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "accelerate", "-U", "transformers"], check=True)

import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402
from ultralytics import YOLO  # noqa: E402

NOTEBOOKS = Path("/kaggle/input/notebooks/hassanagddsbdxfg")
VIDEO = NOTEBOOKS / "dronesight-package/flight.mp4"
WEIGHTS = NOTEBOOKS / "dronesight-experiment-1/runs/s_960/weights/best.pt"
OUT = Path("/kaggle/working/scene.json")
VISION_MODEL = "HuggingFaceTB/SmolVLM-Instruct"
SAMPLE_SECONDS = 0.25
CAPTION_SECONDS = 3
CONF = 0.25
PROMPT = ("Describe this aerial drone photo in two sentences: the place, the time of day and lighting, "
          "the roads and buildings, and what is happening. Only state what you can see.")

capture = cv2.VideoCapture(str(VIDEO))
fps = capture.get(cv2.CAP_PROP_FPS)
total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total / fps
step = max(1, round(fps * SAMPLE_SECONDS))
caption_step = round(CAPTION_SECONDS / SAMPLE_SECONDS)
print(json.dumps({"fps": fps, "frames": total, "duration": duration}), flush=True)

detector = YOLO(str(WEIGHTS))
processor = AutoProcessor.from_pretrained(VISION_MODEL)
vision = AutoModelForImageTextToText.from_pretrained(VISION_MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()


def describe(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to("cuda", torch.bfloat16)
    with torch.inference_mode():
        generated = vision.generate(**inputs, max_new_tokens=90, do_sample=False)
    reply = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return reply.strip()


samples, captions = [], []
index = taken = 0
while True:
    ok, frame = capture.read()
    if not ok:
        break
    if index % step == 0:
        t = round(index / fps, 2)
        boxes = detector.predict(frame, imgsz=960, conf=CONF, device=0, verbose=False)[0].boxes
        counts = Counter(detector.names[int(c)] for c in boxes.cls)
        samples.append({"t": t, "counts": dict(counts)})
        if taken % caption_step == 0:
            captions.append({"t": t, "text": describe(frame)})
            print(captions[-1], flush=True)
            OUT.write_text(json.dumps({"duration": duration, "samples": samples, "captions": captions}))
        taken += 1
    index += 1
capture.release()

OUT.write_text(json.dumps({"duration": duration, "samples": samples, "captions": captions}))
print("samples", len(samples), "captions", len(captions))
