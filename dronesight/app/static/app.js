// All text from the server or the user goes through textContent, never innerHTML.
import { findMoments, parse } from "./search.js";

const $ = (id) => document.getElementById(id);
const STATIC_EDITION = document.documentElement.dataset.edition === "static";
const SAMPLE_FLIGHT = { id: "sample", title: "Sample flight", status: "ready", progress: 1, duration: 41.84, is_demo: 1 };
// Text shown on each chip, and the question it asks.
const EXAMPLES = [["Busiest moment", "busiest moment with people"], ["At least 20 people", "at least 20 people"],
  ["Groups in the first 15 seconds", "groups in the first 15 seconds"], ["Anyone in the last 10 seconds", "anyone in the last 10 seconds"]];
const POLL_MS = 2000;
const SAMPLE_REACH_SECONDS = 0.6;  // a sample stands for the video this long after it
const BAR_WIDTH = 4;
const BOX_COLOUR = "#ffd84d";
const BOX_EDGE = "rgba(0, 0, 0, 0.75)";
const BAR_COLOUR = "#5f7082";
const PLAYHEAD_COLOUR = "#f2f5f8";

const player = $("player");
let videos = [];
let selected = null;
let index = null;
let matches = new Set();
let moments = [];
let currentMoment = null;
let signedIn = false;
let signingUp = false;
let pollTimer = null;

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "That didn't work. Please check what you entered.";
    throw new Error(detail);
  }
  return body;
}

const postJson = (path, payload) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});

const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const people = (n) => `${n} ${n === 1 ? "person" : "people"}`;

function describe(video) {
  if (video.status === "ready") return `${clock(video.duration || 0)} long`;
  if (video.status === "failed") return video.message || "Could not process this video.";
  if (video.status === "queued") return "Waiting to start";
  return `Finding people, ${Math.round(video.progress * 100)}% done`;
}

function renderVideos() {
  $("videos").replaceChildren(...videos.map((video) => {
    const row = document.createElement("li");
    row.className = "video-row";
    const button = document.createElement("button");
    button.className = "video-item";
    button.type = "button";
    button.setAttribute("aria-current", String(selected?.id === video.id));
    const title = document.createElement("span");
    title.textContent = video.title;
    const info = document.createElement("small");
    info.textContent = describe(video);
    button.append(title, info);
    if (video.status === "processing") {
      const bar = document.createElement("progress");
      bar.value = video.progress;
      bar.setAttribute("aria-label", "Processing progress");
      button.append(bar);
    }
    button.addEventListener("click", () => selectVideo(video));
    row.append(button);
    if (!video.is_demo) {
      const remove = document.createElement("button");
      remove.className = "remove";
      remove.type = "button";
      remove.textContent = "Delete";
      remove.setAttribute("aria-label", `Delete ${video.title}`);
      remove.addEventListener("click", () => deleteVideo(video));
      row.append(remove);
    }
    return row;
  }));
}

async function refreshVideos() {
  videos = STATIC_EDITION ? [SAMPLE_FLIGHT] : await api("api/videos");
  clearTimeout(pollTimer);
  if (videos.some((v) => v.status === "queued" || v.status === "processing")) {
    pollTimer = setTimeout(refreshVideos, POLL_MS);
  }
  const current = videos.find((v) => v.id === selected?.id);
  if (current?.status === "ready" && selected.status !== "ready") {
    await selectVideo(current);
  } else if (current) {
    selected = current;
    if (current.status !== "ready") $("placeholder").textContent = describe(current);
  } else {
    await selectVideo(videos.find((v) => v.status === "ready") ?? null);
  }
  renderVideos();
}

async function deleteVideo(video) {
  if (!window.confirm(`Delete “${video.title}”? This cannot be undone.`)) return;
  try {
    await api(`api/videos/${encodeURIComponent(video.id)}`, { method: "DELETE" });
    if (selected?.id === video.id) selected = null;
    await refreshVideos();
  } catch (error) {
    $("upload-status").textContent = error.message;
  }
}

function setResults(title, answer, found, matching) {
  $("results-title").textContent = title;
  $("answer").textContent = answer;
  moments = found;
  matches = new Set(matching.map((sample) => sample.t));
  currentMoment = null;
  const strongest = Math.max(1, ...found.map((moment) => moment.boxes.length));
  $("clips").replaceChildren(...found.map((moment) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.className = "clip";
    button.type = "button";
    const time = document.createElement("time");
    time.textContent = clock(moment.t);
    const what = document.createElement("span");
    what.textContent = people(moment.boxes.length);
    const meter = document.createElement("span");
    meter.className = "meter";
    const fill = document.createElement("span");
    fill.style.width = `${(moment.boxes.length / strongest) * 100}%`;
    meter.append(fill);
    button.append(time, what, meter);
    button.addEventListener("click", () => jumpTo(moment));
    item.append(button);
    return item;
  }));
  drawActivity();
}

function showBusiest() {
  const query = { minCount: 1, start: null, end: null, busiest: false };
  const { moments: found } = findMoments(index, query);
  const empty = "No people were detected in this flight. The detector can miss very small people.";
  setResults("Busiest moments", found.length ? "Select one to jump to it." : empty, found, []);
}

function ask(question) {
  if (!index) {
    $("answer").textContent = selected?.status === "ready" ? "This flight is still loading. Try again in a moment." : "Choose a finished flight first.";
    return;
  }
  const query = parse(question, index.duration);
  if (!query) {
    setResults("Your search", "I can only search for people, by how many and when. Try “at least 20 people” or “groups after 30 seconds”.", [], []);
    return;
  }
  const { matching, moments: found } = findMoments(index, query);
  if (!found.length) {
    setResults("Your search", "No matching moments. The detector can miss very small people, so this does not prove nobody is there.", [], []);
    return;
  }
  const best = found[0];
  const answer = query.busiest
    ? `The busiest moment shows ${people(best.boxes.length)} at ${clock(best.t)}.`
    : `Found ${found.length} ${found.length === 1 ? "moment" : "moments"}. The strongest shows ${people(best.boxes.length)} at ${clock(best.t)}.`;
  setResults("Your search", answer, found, matching);
  jumpTo(best);
}

function jumpTo(moment) {
  currentMoment = moment;
  document.querySelectorAll(".clip").forEach((clip, i) => clip.setAttribute("aria-current", String(moments[i] === moment)));
  player.pause();
  player.currentTime = moment.t;
  $("seek").value = moment.t;
  draw();
}

function sampleAt(time) {
  if (!index?.samples.length) return null;
  let low = 0;
  let high = index.samples.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (index.samples[middle].t <= time) low = middle;
    else high = middle - 1;
  }
  const sample = index.samples[low];
  return sample.t <= time + 0.01 && time - sample.t <= SAMPLE_REACH_SECONDS ? sample : null;
}

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * scale);
  canvas.height = Math.round(rect.height * scale);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  return [ctx, rect.width, rect.height];
}

function drawBoxes() {
  const [ctx, stageWidth, stageHeight] = fitCanvas($("overlay"));
  const sample = selected?.status === "ready" ? sampleAt(player.currentTime) : null;
  const live = $("live");
  live.hidden = !sample;
  if (sample) live.textContent = `${people(sample.boxes.length)} in view`;
  if (!sample || !$("show-boxes").checked || !player.videoWidth) return;
  const fit = Math.min(stageWidth / player.videoWidth, stageHeight / player.videoHeight);
  const width = player.videoWidth * fit;
  const height = player.videoHeight * fit;
  const left = (stageWidth - width) / 2;
  const top = (stageHeight - height) / 2;
  for (const [colour, lineWidth] of [[BOX_EDGE, 3], [BOX_COLOUR, 1.5]]) {
    ctx.strokeStyle = colour;
    ctx.lineWidth = lineWidth;
    for (const [x1, y1, x2, y2] of sample.boxes) {
      ctx.strokeRect(left + x1 * width, top + y1 * height, (x2 - x1) * width, (y2 - y1) * height);
    }
  }
}

function drawActivity() {
  const [ctx, width, height] = fitCanvas($("activity"));
  if (!index?.duration) return;
  const count = Math.max(1, Math.floor(width / BAR_WIDTH));
  const peaks = new Array(count).fill(0);
  const matched = new Array(count).fill(false);
  for (const sample of index.samples) {
    const bar = Math.min(count - 1, Math.floor((sample.t / index.duration) * count));
    peaks[bar] = Math.max(peaks[bar], sample.boxes.length);
    matched[bar] ||= matches.has(sample.t);
  }
  const tallest = Math.max(1, ...peaks);
  peaks.forEach((peak, bar) => {
    const barHeight = peak ? Math.max(3, (peak / tallest) * (height - 8)) : 0;
    ctx.fillStyle = matched[bar] ? BOX_COLOUR : BAR_COLOUR;
    ctx.fillRect(bar * (width / count), height - barHeight, width / count - 1, barHeight);
  });
  ctx.fillStyle = PLAYHEAD_COLOUR;
  ctx.fillRect((player.currentTime / index.duration) * width - 1, 0, 2, height);
}

function draw() {
  drawBoxes();
  drawActivity();
  $("time").textContent = `${clock(player.currentTime)} / ${clock(index?.duration ?? 0)}`;
  if (document.activeElement !== $("seek")) $("seek").value = player.currentTime;
}

function playbackLoop() {
  if (player.paused || player.ended) return;
  draw();
  requestAnimationFrame(playbackLoop);
}

async function selectVideo(video) {
  selected = video;
  index = null;
  renderVideos();
  setResults("Busiest moments", "", [], []);
  player.pause();
  const ready = video?.status === "ready";
  $("placeholder").hidden = false;
  $("placeholder").textContent = video ? (ready ? "Loading the flight…" : describe(video)) : "Add a flight to begin.";
  $("live").hidden = true;
  if (!ready) {
    player.removeAttribute("src");
    player.load();
    return;
  }
  try {
    const base = `api/videos/${encodeURIComponent(video.id)}`;
    const loaded = await (STATIC_EDITION ? fetch("index.json").then((r) => r.json()) : api(`${base}/index`));
    if (selected !== video) return;
    index = loaded;
    player.src = STATIC_EDITION ? "flight.mp4" : `${base}/stream`;
    $("seek").max = index.duration;
    showBusiest();
    draw();
  } catch (error) {
    $("placeholder").textContent = error.message;
  }
}

async function upload(file) {
  if (!signedIn) {
    openAuth("Sign in to add your own flight. It stays private and is deleted after 24 hours.");
    return;
  }
  const status = $("upload-status");
  status.textContent = `Uploading ${file.name}…`;
  try {
    const title = encodeURIComponent(file.name.replace(/\.[^.]+$/, ""));
    const { id } = await api(`api/videos?title=${title}`, { method: "PUT", body: file });
    status.textContent = "Uploaded. Finding people now, which takes a few minutes.";
    selected = null;
    videos = await api("api/videos");
    await selectVideo(videos.find((v) => v.id === id) ?? null);
    await refreshVideos();
  } catch (error) {
    status.textContent = error.message;
  }
}

function setUser(email) {
  signedIn = Boolean(email);
  $("who").textContent = email || "";
  $("signin").hidden = signedIn;
  $("signout").hidden = !signedIn;
}

function setAuthMode(up) {
  signingUp = up;
  $("auth-title").textContent = up ? "Create your account" : "Sign in";
  $("auth-submit").textContent = up ? "Create account" : "Sign in";
  $("auth-switch").textContent = up ? "Already have an account? Sign in" : "New here? Create an account";
  $("password").autocomplete = up ? "new-password" : "current-password";
  $("password-hint").hidden = !up;
  $("auth-error").textContent = "";
}

function openAuth(reason = "") {
  setAuthMode(false);
  $("auth-why").textContent = reason;
  $("auth").showModal();
}

$("signin").addEventListener("click", () => openAuth());
$("auth-switch").addEventListener("click", () => setAuthMode(!signingUp));
$("auth-close").addEventListener("click", () => $("auth").close());
$("signout").addEventListener("click", async () => {
  await api("api/logout", { method: "POST" });
  setUser(null);
  selected = null;
  await refreshVideos();
});

$("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const body = { email: $("email").value, password: $("password").value };
    const user = await postJson(signingUp ? "api/register" : "api/login", body);
    setUser(user.email);
    $("password").value = "";
    $("auth").close();
    await refreshVideos();
  } catch (error) {
    $("auth-error").textContent = error.message;
  }
});

$("ask").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (question) ask(question);
  else if (index) showBusiest();
});

$("examples").replaceChildren(...EXAMPLES.map(([label, question]) => {
  const chip = document.createElement("button");
  chip.className = "chip";
  chip.type = "button";
  chip.textContent = label;
  chip.addEventListener("click", () => {
    $("question").value = question;
    ask(question);
  });
  return chip;
}));

function togglePlay() {
  if (!index) return;
  if (player.paused) player.play().catch(() => {});  // a load failure already shows in the placeholder
  else player.pause();
}

$("play").addEventListener("click", togglePlay);
$("stage").addEventListener("click", togglePlay);
$("show-boxes").addEventListener("change", drawBoxes);
$("fullscreen").addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else $("stage").requestFullscreen();
});
$("seek").addEventListener("input", (event) => {
  player.currentTime = Number(event.target.value);
  draw();
});

player.addEventListener("play", () => {
  $("play").textContent = "Pause";
  playbackLoop();
});
player.addEventListener("pause", () => {
  $("play").textContent = "Play";
  draw();
});
player.addEventListener("seeked", draw);
player.addEventListener("loadeddata", () => {
  $("placeholder").hidden = true;
  draw();
});
player.addEventListener("error", () => {
  if (player.getAttribute("src")) $("placeholder").textContent = "This video could not be played in your browser.";
});
window.addEventListener("resize", draw);
document.addEventListener("fullscreenchange", draw);

$("file").addEventListener("click", (event) => {
  if (!signedIn) {
    event.preventDefault();
    openAuth("Sign in to add your own flight. It stays private and is deleted after 24 hours.");
  }
});
$("file").addEventListener("change", (event) => {
  if (event.target.files[0]) upload(event.target.files[0]);
  event.target.value = "";
});

const drop = $("drop");
drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("over");
  if (event.dataTransfer.files[0]) upload(event.dataTransfer.files[0]);
});

(async () => {
  if (STATIC_EDITION) {
    $("account").hidden = true;
    $("upload-area").hidden = true;
    $("static-note").hidden = false;
  } else {
    const me = await api("api/me");
    setUser(me.email);
  }
  await refreshVideos();
})();
