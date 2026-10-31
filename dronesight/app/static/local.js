// Finds and follows people in a visitor's video on their own device, so the file never leaves it. The
// result has the same shape as the server's index, so the page's timeline and search work unchanged.
import { Tracker, cameraShift } from "./tracker.js";

const MODEL = "person.onnx";
const SAVED_FILES = "dronesight-model-2";  // change this when the model or runtime changes
const SIZE = 1280;
const CONFIDENCE = 0.1;  // low on purpose: weak boxes keep a track alive, and only strong ones start one
const MAX_OVERLAP = 0.5;
const MAX_SECONDS = 180;
const TIME_BUDGET_SECONDS = 420;
const RATES = [5, 4, 3, 2, 1];  // samples a second, best first
const FIRST_RATE = 3;
const FRAMES_BEFORE_CHOOSING = 3;
const THUMBNAIL_WIDTH = 192;

function iou(a, b) {
  const width = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const height = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (width <= 0 || height <= 0) return 0;
  const overlap = width * height;
  return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap);
}

function nms(found) {
  const kept = [];
  for (const candidate of found.sort((a, b) => b.score - a.score)) {
    if (kept.every((other) => iou(candidate.box, other.box) <= MAX_OVERLAP)) kept.push(candidate);
  }
  return kept;
}

// Raw output has shape (1, 5, N): centre x, centre y, width, height and score, in model pixels.
function decode(output, frame) {
  const count = output.dims[2];
  const { data } = output;
  const found = [];
  for (let i = 0; i < count; i += 1) {
    const score = data[4 * count + i];
    if (score < CONFIDENCE) continue;
    const [x, y, width, height] = [data[i], data[count + i], data[2 * count + i], data[3 * count + i]];
    found.push({ score, box: [x - width / 2, y - height / 2, x + width / 2, y + height / 2] });
  }
  return nms(found).map(({ score, box: [x1, y1, x2, y2] }) => ({
    score,
    box: [(x1 - frame.left) / frame.scale, (y1 - frame.top) / frame.scale, (x2 - frame.left) / frame.scale, (y2 - frame.top) / frame.scale],
  }));
}

// Same grey padding as the training images, so the frame keeps its shape.
function toTensor(ort, ctx, video, frame) {
  ctx.fillStyle = "rgb(114, 114, 114)";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(video, frame.left, frame.top, frame.width, frame.height);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  const pixels = SIZE * SIZE;
  const blob = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i += 1) {
    blob[i] = data[4 * i] / 255;
    blob[pixels + i] = data[4 * i + 1] / 255;
    blob[2 * pixels + i] = data[4 * i + 2] / 255;
  }
  return new ort.Tensor("float32", blob, [1, 3, SIZE, SIZE]);
}

function greyThumbnail(ctx, video, width, height) {
  ctx.drawImage(video, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  const grey = new Uint8Array(width * height);
  for (let i = 0; i < grey.length; i += 1) grey[i] = (data[4 * i] * 2 + data[4 * i + 1] * 5 + data[4 * i + 2]) >> 3;
  return grey;
}

// The host's download links change on every request, so the browser's own cache never matches them.
// Keeping the files here means only the first video downloads them.
async function saved(path) {
  const files = await caches.open(SAVED_FILES);
  const known = await files.match(path);
  if (known) return known.arrayBuffer();
  const response = await fetch(path);
  if (!response.ok) throw new Error("The model could not be downloaded. Please try again.");
  const data = await response.arrayBuffer();
  // A fresh response, because the store refuses some of the host's headers. A full disk only means downloading again.
  await files.put(path, new Response(data)).catch(() => {});
  return data;
}

async function forgetOldFiles() {
  for (const name of await caches.keys()) {
    if (name.startsWith("dronesight-model-") && name !== SAVED_FILES) await caches.delete(name);
  }
}

async function loadModel() {
  await forgetOldFiles();
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  const ort = await import(adapter ? "./ort/ort.webgpu.min.mjs" : "./ort/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
  const [binary, model] = await Promise.all([saved(`ort/ort-wasm-simd-threaded${adapter ? ".asyncify" : ""}.wasm`), saved(MODEL)]);
  ort.env.wasm.wasmBinary = binary;
  const session = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: adapter ? ["webgpu", "wasm"] : ["wasm"] });
  return { ort, session, device: adapter ? "graphics card" : "processor" };
}

function openVideo(url) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";
    video.onloadeddata = () => resolve(video);
    video.onerror = () => reject(new Error("This video could not be played in your browser. Try an MP4."));
    video.src = url;
  });
}

const seek = (video, time) => new Promise((resolve) => {
  video.onseeked = resolve;
  video.currentTime = time;
});

function timeLeft(seconds) {
  if (seconds === null) return "working out how long this takes.";
  return seconds < 60 ? "less than a minute left." : `about ${Math.ceil(seconds / 60)} min left.`;
}

// As many samples a second as fit the time budget on this device, because more samples track better.
const bestRate = (secondsPerFrame, duration) => RATES.find((rate) => rate * duration * secondsPerFrame <= TIME_BUDGET_SECONDS) ?? RATES.at(-1);

// Turns tracks into the server's index: every sample lists who is in view, in fractions of the frame.
function toIndex(tracks, duration, width, height) {
  const scaled = tracks.map(({ id, times, boxes }) => ({
    id,
    times,
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

export async function analyse(file, onProgress) {
  const url = URL.createObjectURL(file);
  try {
    const video = await openVideo(url);
    const { duration, videoWidth, videoHeight } = video;
    if (duration > MAX_SECONDS) throw new Error("Please choose a video of up to 3 minutes.");
    onProgress(0, "Getting the model ready. The first time downloads about 70 MB.");
    const { ort, session, device } = await loadModel();
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = SIZE;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingQuality = "high";
    const small = document.createElement("canvas");
    small.width = THUMBNAIL_WIDTH;
    small.height = Math.round((THUMBNAIL_WIDTH * videoHeight) / videoWidth);
    const smallCtx = small.getContext("2d", { willReadFrequently: true });
    const scale = SIZE / Math.max(videoWidth, videoHeight);
    const width = Math.round(videoWidth * scale);
    const height = Math.round(videoHeight * scale);
    const frame = { scale, width, height, left: Math.floor((SIZE - width) / 2), top: Math.floor((SIZE - height) / 2) };
    const toVideoPixels = videoWidth / THUMBNAIL_WIDTH;

    const tracker = new Tracker();
    let rate = FIRST_RATE;
    let previous = null;
    let time = 0;
    let done = 0;
    let started = 0;
    while (time < duration) {
      await seek(video, time);
      const results = await session.run({ [session.inputNames[0]]: toTensor(ort, ctx, video, frame) });
      const grey = greyThumbnail(smallCtx, video, small.width, small.height);
      const shift = previous ? cameraShift(previous, grey, small.width, small.height) : { x: 0, y: 0 };
      previous = grey;
      tracker.update(time, decode(results[session.outputNames[0]], frame), { x: shift.x * toVideoPixels, y: shift.y * toVideoPixels });
      done += 1;
      if (done === 1) started = performance.now();  // the first frame sets up the model, so it says little about the rest
      const secondsPerFrame = done > 1 ? (performance.now() - started) / (done - 1) / 1000 : null;
      if (done === FRAMES_BEFORE_CHOOSING + 1) rate = bestRate(secondsPerFrame, duration);
      time += 1 / rate;
      const left = secondsPerFrame === null ? null : Math.max(0, duration - time) * rate * secondsPerFrame;
      onProgress(Math.min(1, time / duration), `Following people on your ${device}, ${timeLeft(left)} Please keep this page open.`);
      await new Promise((resolve) => { setTimeout(resolve); });  // lets the page repaint between frames
    }
    return { url, index: toIndex(tracker.finish(), duration, videoWidth, videoHeight) };
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  }
}
