// Finds and follows people in a visitor's video on their own device, so the file never leaves it.
import { HAZARD_MODELS, dropFalseWater, hazardFrame, hazardSeen, hazardViews, judge, maskPicture, modelView, standsIn, steadyFlames, toTensor, tooDarkForWater } from "./hazards.js";
import { Tracker, cameraShift } from "./tracker.js";

const MODEL = "person.onnx";
// The main model only learned small figures from far above. This one learned everyday photos, so it
// finds people close to the camera or in poor video. Boxes come from both.
const CLOSE = { model: "close.onnx", size: 640 };
const SAVED_FILES = "dronesight-model-5";  // change this when the model or runtime changes
const SIZE = 1280;  // the frame's longest side
const CONFIDENCE = 0.1;  // low on purpose: weak boxes keep a track alive, and only strong ones start one
const MAX_OVERLAP = 0.5;
// The main model learned figures from far above, so up close it boxes parts of a body as well as the whole.
// A real person behind them is far smaller than the box around them, so they stay.
const PART_INSIDE = 0.8;
const PART_SIZE = 0.2;
const MAX_SECONDS = 180;
// The small people model and the camera's motion are cheap, and keep tracking through a moving camera.
// The main people model and the hazard models are slow, so they only run on some of those samples.
const SAMPLE_SECONDS = 0.5;
const HEAVY_SECONDS = 1;
const LONG_VIDEO_HEAVY_SECONDS = 2;
const TIME_BUDGET_SECONDS = 60;
const DOWNLOAD_MB = 100;  // all the models and the runtime together
const SHOWN_SCORE = 0.25;  // boxes shown during the check: weaker ones only help follow a person
const STANDARD_VIDEO = [1920, 1080];  // the shape the models are warmed up on
const FRAMES_BEFORE_CHOOSING = 4;  // an even number, so light and heavy samples are measured equally
const THUMBNAIL_WIDTH = 192;
const HAS_PICTURE = 2;  // the video element's readyState once it can show a frame

function iou(a, b) {
  const width = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const height = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (width <= 0 || height <= 0) return 0;
  const overlap = width * height;
  return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap);
}

const area = ([x1, y1, x2, y2]) => (x2 - x1) * (y2 - y1);

function isPartOf(part, whole) {
  if (whole.score < SHOWN_SCORE || area(part.box) >= area(whole.box) || area(part.box) < PART_SIZE * area(whole.box)) return false;
  const width = Math.min(part.box[2], whole.box[2]) - Math.max(part.box[0], whole.box[0]);
  const height = Math.min(part.box[3], whole.box[3]) - Math.max(part.box[1], whole.box[1]);
  return width > 0 && height > 0 && width * height >= PART_INSIDE * area(part.box);
}

function nms(found) {
  const kept = [];
  for (const candidate of found.sort((a, b) => b.score - a.score)) {
    if (kept.every((other) => iou(candidate.box, other.box) <= MAX_OVERLAP)) kept.push(candidate);
  }
  return kept.filter((part) => !kept.some((whole) => isPartOf(part, whole)));
}

// Raw output has shape (1, 5 or more, N): centre x, centre y, width, height, then a score for each
// class in model pixels. Person is the first class in both models.
function decode(output, view, videoWidth, videoHeight) {
  const count = output.dims[2];
  const { data } = output;
  const [scaleX, scaleY] = [videoWidth / view.width, videoHeight / view.height];
  const found = [];
  for (let i = 0; i < count; i += 1) {
    const score = data[4 * count + i];
    if (score < CONFIDENCE) continue;
    const [x, y, width, height] = [data[i], data[count + i], data[2 * count + i], data[3 * count + i]];
    found.push({ score, box: [(x - width / 2) * scaleX, (y - height / 2) * scaleY, (x + width / 2) * scaleX, (y + height / 2) * scaleY] });
  }
  return found;
}

async function people(ort, video, session, view) {
  const results = await session.run({ [session.inputNames[0]]: toTensor(ort, video, view) });
  return decode(results[session.outputNames[0]], view, video.videoWidth, video.videoHeight);
}

const pictures = (hazard) => ["flood", "fire"].map((kind) => maskPicture(hazard[kind], kind)).filter(Boolean);

function greyThumbnail(ctx, video, width, height) {
  ctx.drawImage(video, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  const grey = new Uint8Array(width * height);
  for (let i = 0; i < grey.length; i += 1) grey[i] = (data[4 * i] * 2 + data[4 * i + 1] * 5 + data[4 * i + 2]) >> 3;
  return grey;
}

// The host's download links change on every request, so the browser's own cache never matches them.
// Keeping the files here means only the first video downloads them.
async function saved(path, onBytes) {
  const files = await caches.open(SAVED_FILES);
  const known = await files.match(path);
  if (known) return known.arrayBuffer();
  onBytes(0);
  const response = await fetch(path);
  if (!response.ok) throw new Error("The model could not be downloaded. Please try again.");
  const parts = [];
  const reader = response.body.getReader();
  for (let part = await reader.read(); !part.done; part = await reader.read()) {
    parts.push(part.value);
    onBytes(part.value.length);
  }
  const whole = new Blob(parts);
  const data = await (path.endsWith(".gz") ? new Response(whole.stream().pipeThrough(new DecompressionStream("gzip"))) : whole).arrayBuffer();
  // A fresh response, because the store refuses some of the host's headers. A full disk only means downloading again.
  await files.put(path, new Response(data)).catch(() => {});
  return data;
}

async function forgetOldFiles() {
  for (const name of await caches.keys()) {
    if (name.startsWith("dronesight-model-") && name !== SAVED_FILES) await caches.delete(name);
  }
}

const runtimeFile = (graphics) => `ort/ort-wasm-simd-threaded${graphics ? ".asyncify" : ""}.wasm.gz`;

// The first run of each model is by far the slowest, so it happens on a blank frame.
async function warm(ort, session, view) {
  await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", new Float32Array(3 * view.width * view.height), [1, 3, view.height, view.width]) });
}

async function loadModels(onProgress) {
  await forgetOldFiles();
  let bytes = 0;
  const onBytes = (size) => {
    bytes += size;
    const shown = Math.min(DOWNLOAD_MB, Math.round(bytes / 1e6));
    onProgress(shown / DOWNLOAD_MB, `Downloading the models, only needed once: ${shown} of about ${DOWNLOAD_MB} MB.`);
  };
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
  const ort = await import(adapter ? "./ort/ort.webgpu.min.mjs" : "./ort/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
  if (adapter) ort.env.webgpu.powerPreference = "high-performance";
  const [binary, model, closeModel, ...hazardModels] = await Promise.all([
    saved(runtimeFile(adapter), onBytes), saved(MODEL, onBytes), saved(CLOSE.model, onBytes), ...HAZARD_MODELS.map((path) => saved(path, onBytes)),
  ]);
  onProgress(0, "Getting the models ready.");
  ort.env.wasm.wasmBinary = binary;
  const options = { executionProviders: adapter ? ["webgpu", "wasm"] : ["wasm"] };
  const session = await ort.InferenceSession.create(new Uint8Array(model), options);
  const closeSession = await ort.InferenceSession.create(new Uint8Array(closeModel), options);
  const hazardSessions = [];
  for (const hazardModel of hazardModels) hazardSessions.push(await ort.InferenceSession.create(new Uint8Array(hazardModel), options));
  const device = adapter ? "graphics card" : "processor";
  const { water, fire } = hazardViews(...STANDARD_VIDEO);
  await warm(ort, session, modelView(SIZE, ...STANDARD_VIDEO));
  await warm(ort, closeSession, modelView(CLOSE.size, ...STANDARD_VIDEO));
  await warm(ort, hazardSessions[0], water.canvas);
  await warm(ort, hazardSessions[1], fire);
  return { ort, session, closeSession, hazardSessions, device };
}

let models = null;
function ready(onProgress) {
  models ??= loadModels(onProgress).catch((error) => {
    models = null;
    throw error;
  });
  return models;
}

// Models saved by an earlier visit are made ready as the page opens, so a video can start at once.
// A first visit downloads nothing until a video is added, which keeps phone data safe.
export async function prepare() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
  const files = await caches.open(SAVED_FILES);
  const known = await Promise.all([runtimeFile(adapter), MODEL, CLOSE.model, ...HAZARD_MODELS].map((path) => files.match(path)));
  if (known.every(Boolean)) await ready(() => {});
}

function canPlay(video) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= HAS_PICTURE) resolve();
    video.addEventListener("loadeddata", resolve, { once: true });
    video.addEventListener("error", () => reject(new Error("This video could not be played in your browser. Try an MP4.")), { once: true });
  });
}

const seek = (video, time) => new Promise((resolve) => {
  video.onseeked = resolve;
  video.currentTime = time;
});

function timeLeft(seconds) {
  if (seconds === null) return "working out how long this takes.";
  return seconds < 90 ? `about ${Math.ceil(seconds / 5) * 5} seconds left.` : `about ${Math.ceil(seconds / 60)} min left.`;
}

// Turns tracks into fractions of the frame, and lists who is in view at every sample.
function toIndex(tracks, duration, width, height) {
  const scaled = tracks.map(({ id, times, boxes, world }) => ({
    id,
    times,
    world: world.map(([x, y]) => [x / height, y / height]),
    boxes: boxes.map(([x1, y1, x2, y2]) => [
      Math.round(Math.max(0, x1 / width) * 10000) / 10000, Math.round(Math.max(0, y1 / height) * 10000) / 10000,
      Math.round(Math.min(1, x2 / width) * 10000) / 10000, Math.round(Math.min(1, y2 / height) * 10000) / 10000,
    ]),
  }));
  const moments = new Map();
  for (const track of scaled) {
    track.times.forEach((t, i) => {
      if (!moments.has(t)) moments.set(t, []);
      moments.get(t).push(track.boxes[i]);
    });
  }
  const samples = [...moments].sort((a, b) => a[0] - b[0]).map(([t, boxes]) => ({ t, boxes }));
  return { duration, samples, tracks: scaled, reach: 1 };
}

// Walks the visitor's own player through the video, so they watch each moment being checked.
export async function analyse(video, onProgress, onLook) {
  await canPlay(video);
  const { duration, videoWidth, videoHeight } = video;
  if (duration > MAX_SECONDS) throw new Error("Please choose a video of up to 3 minutes.");
  onProgress(0, "Getting the models ready.");
  const { ort, session, closeSession, hazardSessions, device } = await ready(onProgress);
  const [far, near] = [modelView(SIZE, videoWidth, videoHeight), modelView(CLOSE.size, videoWidth, videoHeight)];
  const views = hazardViews(videoWidth, videoHeight);
  const small = document.createElement("canvas");
  small.width = THUMBNAIL_WIDTH;
  small.height = Math.round((THUMBNAIL_WIDTH * videoHeight) / videoWidth);
  const smallCtx = small.getContext("2d", { willReadFrequently: true });
  const toVideoPixels = videoWidth / THUMBNAIL_WIDTH;

  const tracker = new Tracker();
  const hazards = [];
  let nextHeavy = 0;
  let heavySeconds = HEAVY_SECONDS;
  let previous = null;
  let time = 0;
  let done = 0;
  let started = 0;
  while (time < duration) {
    await seek(video, time);
    const heavy = time >= nextHeavy;
    const found = nms([...(heavy ? await people(ort, video, session, far) : []), ...await people(ort, video, closeSession, near)]);
    const grey = greyThumbnail(smallCtx, video, small.width, small.height);
    const shift = previous ? cameraShift(previous, grey, small.width, small.height) : { x: 0, y: 0 };
    previous = grey;
    tracker.update(time, found, { x: shift.x * toVideoPixels, y: shift.y * toVideoPixels });
    if (heavy) {
      const hazard = await hazardFrame(ort, hazardSessions, video, time, views);
      hazards.push(hazard);
      nextHeavy = time + heavySeconds;
      dropFalseWater([hazard]);
      const boxes = found.filter(({ score }) => score >= SHOWN_SCORE).map(({ box: [x1, y1, x2, y2] }) => [x1 / videoWidth, y1 / videoHeight, x2 / videoWidth, y2 / videoHeight]);
      onLook({ t: time, masks: pictures(hazard), people: boxes.map((box) => ({ box, danger: standsIn(box, hazard) })) });
    }
    done += 1;
    if (done === 1) started = performance.now();  // the first frame sets up the model, so it says little about the rest
    const secondsPerFrame = done > 1 ? (performance.now() - started) / (done - 1) / 1000 : null;
    if (done === FRAMES_BEFORE_CHOOSING + 1 && (duration / SAMPLE_SECONDS) * secondsPerFrame > TIME_BUDGET_SECONDS) heavySeconds = LONG_VIDEO_HEAVY_SECONDS;
    time += SAMPLE_SECONDS;
    const left = secondsPerFrame === null ? null : (Math.max(0, duration - time) / SAMPLE_SECONDS) * secondsPerFrame;
    onProgress(Math.min(1, time / duration), `Checking on your ${device}, ${timeLeft(left)} Please keep this page open.`);
    await new Promise((resolve) => { setTimeout(resolve); });  // lets the page repaint between frames
  }
  dropFalseWater(hazards);
  steadyFlames(hazards);
  const index = toIndex(tracker.finish(), duration, videoWidth, videoHeight);
  Object.assign(index, {
    aspect: videoWidth / videoHeight,
    people: judge(index.tracks, hazards),
    hazardFound: { flood: hazardSeen(hazards, "flood"), fire: hazardSeen(hazards, "fire"), tooDark: tooDarkForWater(hazards) },
    masks: hazards.flatMap((hazard) => pictures(hazard).map((picture) => ({ t: hazard.t, picture }))),
  });
  return index;
}
