// Finds people in a visitor's video on their own device, so the file never leaves it. The result has
// the same shape as the server's index, so the page's boxes, timeline and search work unchanged.
const MODEL = "person_960.onnx";
const SAVED_FILES = "dronesight-model-1";  // change this when the model or runtime changes
const SIZE = 960;
const CONFIDENCE = 0.25;
const MAX_OVERLAP = 0.5;
const MAX_SECONDS = 180;
const SAMPLE_SECONDS = 1;  // the server samples four times a second, which is too slow here

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
  const clamp = (value) => Math.round(Math.min(1, Math.max(0, value)) * 10000) / 10000;
  return nms(found).map(({ box: [x1, y1, x2, y2] }) => [
    clamp((x1 - frame.left) / frame.width), clamp((y1 - frame.top) / frame.height),
    clamp((x2 - frame.left) / frame.width), clamp((y2 - frame.top) / frame.height),
  ]);
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

async function loadModel() {
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

export async function analyse(file, onProgress) {
  const url = URL.createObjectURL(file);
  try {
    const video = await openVideo(url);
    if (video.duration > MAX_SECONDS) throw new Error("Please choose a video of up to 3 minutes.");
    onProgress(0, "Getting the model ready. The first time downloads about 70 MB.");
    const { ort, session, device } = await loadModel();
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = SIZE;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingQuality = "high";
    const scale = SIZE / Math.max(video.videoWidth, video.videoHeight);
    const width = Math.round(video.videoWidth * scale);
    const height = Math.round(video.videoHeight * scale);
    const frame = { width, height, left: Math.floor((SIZE - width) / 2), top: Math.floor((SIZE - height) / 2) };
    const steps = Math.max(1, Math.floor(video.duration / SAMPLE_SECONDS));
    const samples = [];
    let started = 0;
    for (let step = 0; step < steps; step += 1) {
      const t = step * SAMPLE_SECONDS;
      await seek(video, t);
      const results = await session.run({ [session.inputNames[0]]: toTensor(ort, ctx, video, frame) });
      const boxes = decode(results[session.outputNames[0]], frame);
      if (boxes.length) samples.push({ t, boxes });
      if (!step) started = performance.now();  // the first frame sets up the model, so it says little about the rest
      const seconds = step ? ((performance.now() - started) / step) * (steps - step - 1) / 1000 : null;
      onProgress((step + 1) / steps, `Finding people on your ${device}, ${timeLeft(seconds)} Please keep this page open.`);
      await new Promise((resolve) => { setTimeout(resolve); });  // lets the page repaint between frames
    }
    return { url, index: { duration: video.duration, samples, reach: SAMPLE_SECONDS } };
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  }
}
