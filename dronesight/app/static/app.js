// All text from the server or the user goes through textContent, never innerHTML.
const $ = (id) => document.getElementById(id);
const EXAMPLES = ["people", "groups of people", "busiest moment with people", "at least 2 people after 30 seconds"];
const POLL_MS = 2000;

let videos = [];
let selected = null;
let signedIn = false;
let signingUp = false;
let pollTimer = null;
let activeBoxes = [];

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

function describe(video) {
  if (video.status === "ready") return video.is_demo ? "Sample flight" : `${clock(video.duration || 0)} long`;
  if (video.status === "failed") return video.message || "Failed";
  if (video.status === "queued") return "Waiting to start";
  return `${video.message || "Working"}, ${Math.round(video.progress * 100)}% done`;
}

function renderVideos() {
  const list = $("videos");
  list.replaceChildren(...videos.map((video) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.className = "video-item";
    button.setAttribute("aria-current", String(selected?.id === video.id));
    const title = document.createElement("span");
    title.textContent = video.title;
    const info = document.createElement("small");
    info.textContent = describe(video);
    button.append(title, info);
    button.addEventListener("click", () => selectVideo(video));
    item.append(button);
    return item;
  }));
  $("empty").hidden = videos.length > 0;
}

async function refreshVideos() {
  videos = await api("/api/videos");
  renderVideos();
  clearTimeout(pollTimer);
  if (videos.some((v) => v.status === "queued" || v.status === "processing")) {
    pollTimer = setTimeout(refreshVideos, POLL_MS);
  }
  const current = videos.find((v) => v.id === selected?.id);
  if (current && current.status === "ready" && selected.status !== "ready") selectVideo(current);
  else if (current) selected = current;
  if (!selected) {
    const first = videos.find((v) => v.status === "ready");
    if (first) selectVideo(first);
  }
}

function selectVideo(video) {
  selected = video;
  renderVideos();
  clearResults();
  const ready = video.status === "ready";
  $("placeholder").hidden = ready;
  if (ready) {
    $("player").src = `/api/videos/${encodeURIComponent(video.id)}/stream`;
  } else {
    $("player").removeAttribute("src");
    $("player").load();
    $("placeholder").textContent = describe(video);
  }
}

function clearResults() {
  $("answer").textContent = "";
  $("clips").replaceChildren();
  activeBoxes = [];
  drawBoxes();
}

function drawBoxes() {
  const canvas = $("overlay");
  const player = $("player");
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!activeBoxes.length || !player.videoWidth) return;
  const fit = Math.min(rect.width / player.videoWidth, rect.height / player.videoHeight);
  const width = player.videoWidth * fit;
  const height = player.videoHeight * fit;
  const left = (rect.width - width) / 2;
  const top = (rect.height - height) / 2;
  ctx.strokeStyle = "#3ddc97";
  ctx.lineWidth = 2;
  for (const [x1, y1, x2, y2] of activeBoxes) {
    ctx.strokeRect(left + x1 * width, top + y1 * height, (x2 - x1) * width, (y2 - y1) * height);
  }
}

function jumpTo(clip) {
  const player = $("player");
  player.pause();
  player.currentTime = clip.peak_t;
  activeBoxes = clip.boxes;
  drawBoxes();
}

function renderClip(clip) {
  const item = document.createElement("li");
  const button = document.createElement("button");
  button.className = "clip";
  const time = document.createElement("time");
  time.textContent = clock(clip.peak_t);
  const what = document.createElement("span");
  what.textContent = `${clip.count} ${clip.count === 1 ? "person" : "people"}`;
  const conf = document.createElement("small");
  conf.textContent = `${Math.round(clip.confidence * 100)}% confidence`;
  button.append(time, what, conf);
  button.addEventListener("click", () => jumpTo(clip));
  item.append(button);
  return item;
}

async function ask(question) {
  if (!selected || selected.status !== "ready") {
    $("answer").textContent = "Choose a finished flight first.";
    return;
  }
  clearResults();
  $("answer").textContent = "Searching…";
  try {
    const result = await postJson(`/api/videos/${encodeURIComponent(selected.id)}/search`, { question });
    $("answer").textContent = result.answer;
    $("clips").replaceChildren(...result.clips.map(renderClip));
    if (result.clips.length) jumpTo(result.clips[0]);
  } catch (error) {
    $("answer").textContent = error.message;
  }
}

async function upload(file) {
  if (!signedIn) {
    openAuth();
    return;
  }
  const status = $("upload-status");
  status.textContent = `Uploading ${file.name}…`;
  try {
    const title = encodeURIComponent(file.name.replace(/\.[^.]+$/, ""));
    const { id } = await api(`/api/videos?title=${title}`, { method: "PUT", body: file });
    status.textContent = "Uploaded. Processing has started.";
    await refreshVideos();
    const video = videos.find((v) => v.id === id);
    if (video) selectVideo(video);
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
  $("auth-error").textContent = "";
}

function openAuth() {
  setAuthMode(false);
  $("auth").showModal();
}

$("signin").addEventListener("click", openAuth);
$("auth-switch").addEventListener("click", () => setAuthMode(!signingUp));
$("auth-close").addEventListener("click", () => $("auth").close());
$("signout").addEventListener("click", async () => {
  await api("/api/logout", { method: "POST" });
  setUser(null);
  selected = null;
  await refreshVideos();
});

$("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const body = { email: $("email").value, password: $("password").value };
    const user = await postJson(signingUp ? "/api/register" : "/api/login", body);
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
});

$("examples").replaceChildren(...EXAMPLES.map((text) => {
  const chip = document.createElement("button");
  chip.className = "chip";
  chip.type = "button";
  chip.textContent = text;
  chip.addEventListener("click", () => {
    $("question").value = text;
    ask(text);
  });
  return chip;
}));

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

$("player").addEventListener("play", () => { activeBoxes = []; drawBoxes(); });
window.addEventListener("resize", drawBoxes);

(async () => {
  const me = await api("/api/me");
  setUser(me.email);
  await refreshVideos();
})();
