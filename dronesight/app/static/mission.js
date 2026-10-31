// The mission simulator. The flight decisions here are real logic: the drone surveys sector by
// sector, breaks off to report anyone critical, queues everyone else by rank, then returns to them.
// Detections and rankings come from mission/mission.json, computed in advance by the real models.
const $ = (id) => document.getElementById(id);

const FLIGHT_SECONDS = 1.1;   // between neighbouring sectors
const SCAN_SECONDS = 3.2;
const LOCK_SECONDS = 2.0;
const ZOOM = 3;
const SPEEDS = [1, 2, 4];
const REPORTED_LEVELS = new Set(["CRITICAL", "HIGH"]);
const COLOURS = { CRITICAL: "#ff5a4f", HIGH: "#ff9f40", WATCH: "#ffd84d", CLEAR: "#b3bec9" };
const FLOOD_FILL = "rgba(64, 156, 255, 0.32)";
const FIRE_FILL = "rgba(255, 106, 40, 0.42)";
const SMOKE_FILL = "rgba(190, 190, 190, 0.32)";
const HOME = { row: 2.75, column: -0.75 };

let missions = null;
let name = "flood";
let mission = null;
let images = new Map();
let state = null;
let paused = false;
let speedIndex = 0;
let lastFrame = 0;

const sectorAt = (row, column) => mission.sectors.find((s) => s.row === row && s.column === column);
const everyone = () => mission.sectors.flatMap((sector) => sector.people.map((person) => ({ person, sector })));
const mapsLink = (person) => `https://www.google.com/maps?q=${person.latitude},${person.longitude}`;

// Back and forth along each row, the standard way to cover an area without gaps.
function surveyOrder() {
  const order = [];
  for (let row = 0; row < 3; row += 1) {
    const columns = row % 2 ? [2, 1, 0] : [0, 1, 2];
    for (const column of columns) {
      const sector = sectorAt(row, column);
      if (sector) order.push(sector);
    }
  }
  return order;
}

function newState() {
  const tasks = surveyOrder().flatMap((sector) => [{ type: "fly", sector }, { type: "scan", sector }]);
  return { drone: { ...HOME }, tasks, task: null, sector: null, reveal: {}, seen: new Set(), reported: new Set(),
    queue: [], clock: 0, lock: null, phase: "Taking off", done: false };
}

function start() {
  mission = missions[name];
  state = newState();
  paused = false;
  $("pause").textContent = "Pause";
  $("alerts").replaceChildren();
  $("hq-empty").hidden = false;
  $("lock").hidden = true;
  updateTally();
}

function discover(person, sector) {
  if (state.seen.has(person.id)) return;
  state.seen.add(person.id);
  if (person.level === "CRITICAL") {
    // Critical people interrupt the survey at once.
    state.tasks.unshift({ type: "lock", sector, person }, { type: "scan", sector, resume: true });
    state.task = null;
  } else if (REPORTED_LEVELS.has(person.level)) {
    state.queue.push({ person, sector });
    state.queue.sort((a, b) => a.person.rank - b.person.rank);
  }
  updateTally();
}

function startTask(task) {
  task.elapsed = task.resume ? state.reveal[task.sector.id] * SCAN_SECONDS : 0;
  if (task.type === "fly") {
    task.from = { ...state.drone };
    const distance = Math.hypot(task.sector.row - task.from.row, task.sector.column - task.from.column);
    task.duration = Math.max(0.3, distance * FLIGHT_SECONDS);
    state.phase = `Flying to sector ${task.sector.id}`;
  } else if (task.type === "scan") {
    state.sector = task.sector;
    state.reveal[task.sector.id] ??= 0;
    task.duration = SCAN_SECONDS;
    state.phase = `Searching sector ${task.sector.id}`;
  } else if (task.type === "lock") {
    state.sector = task.sector;
    state.lock = task.person;
    task.duration = LOCK_SECONDS;
    state.phase = `${task.person.level === "CRITICAL" ? "Critical person" : "Returning to person"} in sector ${task.sector.id}`;
  } else if (task.type === "home") {
    task.from = { ...state.drone };
    task.duration = Math.hypot(HOME.row - task.from.row, HOME.column - task.from.column) * FLIGHT_SECONDS;
    state.phase = "Returning to base";
  }
}

function finishTask(task) {
  if (task.type === "lock") {
    report(task.person, task.sector);
    state.lock = null;
  }
  if (task.type === "scan" && !state.tasks.some((t) => t.type === "scan")) {
    // The survey is finished, so visit everyone queued, most urgent first.
    state.tasks.push(...state.queue.flatMap(({ person, sector }) => [{ type: "fly", sector }, { type: "lock", sector, person }]));
    state.queue = [];
    state.tasks.push({ type: "home" });
    updateTally();
  }
  if (task.type === "home") {
    state.done = true;
    state.phase = `Mission complete: ${state.reported.size} people reported to headquarters`;
  }
}

function advance(seconds) {
  state.clock += seconds;
  if (state.done) return;
  if (!state.task) {
    state.task = state.tasks.shift() ?? null;
    if (!state.task) return;
    startTask(state.task);
  }
  const task = state.task;
  task.elapsed += seconds;
  const progress = Math.min(1, task.elapsed / task.duration);
  if (task.type === "fly" || task.type === "home") {
    const target = task.type === "home" ? HOME : task.sector;
    state.drone.row = task.from.row + (target.row - task.from.row) * progress;
    state.drone.column = task.from.column + (target.column - task.from.column) * progress;
  } else if (task.type === "scan") {
    state.reveal[task.sector.id] = progress;
    const next = [...task.sector.people].sort((a, b) => a.box[3] - b.box[3]).find((p) => !state.seen.has(p.id) && p.box[3] <= progress);
    if (next) {
      discover(next, task.sector);
      if (!state.task) return;
    }
  }
  if (progress >= 1) {
    finishTask(task);
    state.task = null;
  }
}

function report(person, sector) {
  state.reported.add(person.id);
  $("hq-empty").hidden = true;
  const item = document.createElement("li");
  item.className = `alert ${person.level.toLowerCase()}`;
  const title = document.createElement("strong");
  title.textContent = `${person.level}: sector ${sector.id}`;
  const why = document.createElement("p");
  why.textContent = person.reasons.join(", ") || "needs checking";
  const coords = document.createElement("p");
  coords.className = "coords";
  coords.textContent = `${person.latitude.toFixed(5)}, ${person.longitude.toFixed(5)}`;
  const meta = document.createElement("p");
  meta.className = "meta";
  meta.textContent = `Within about ${mission.error_m} m, simulated location. Priority ${person.rank}.`;
  const link = document.createElement("a");
  link.href = mapsLink(person);
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "Open in Google Maps";
  item.append(title, why, coords, meta, link);
  $("alerts").prepend(item);
  updateTally();
}

function updateTally() {
  const people = everyone().map(({ person }) => person);
  const seen = people.filter((p) => state.seen.has(p.id));
  $("found").textContent = seen.length;
  $("critical").textContent = seen.filter((p) => p.level === "CRITICAL").length;
  $("reported").textContent = state.reported.size;
  $("waiting").textContent = seen.filter((p) => REPORTED_LEVELS.has(p.level) && !state.reported.has(p.id)).length;
}

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = Math.round(rect.width * scale);
  const height = Math.round(rect.height * scale);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  return [ctx, rect.width, rect.height];
}

function drawOverlays(ctx, sector, x, y, width, height, reveal) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, width, height * reveal);
  ctx.clip();
  ctx.fillStyle = FLOOD_FILL;
  for (const outline of sector.flood) {
    ctx.beginPath();
    outline.forEach(([px, py], i) => (i ? ctx.lineTo(x + px * width, y + py * height) : ctx.moveTo(x + px * width, y + py * height)));
    ctx.closePath();
    ctx.fill();
  }
  for (const [boxes, fill] of [[sector.smoke, SMOKE_FILL], [sector.fire, FIRE_FILL]]) {
    ctx.fillStyle = fill;
    for (const [x1, y1, x2, y2] of boxes) ctx.fillRect(x + x1 * width, y + y1 * height, (x2 - x1) * width, (y2 - y1) * height);
  }
  ctx.restore();
  for (const person of sector.people) {
    if (!state.seen.has(person.id)) continue;
    const [x1, y1, x2, y2] = person.box;
    ctx.strokeStyle = "rgba(0, 0, 0, 0.75)";
    ctx.lineWidth = 3.5;
    ctx.strokeRect(x + x1 * width, y + y1 * height, (x2 - x1) * width, (y2 - y1) * height);
    ctx.strokeStyle = COLOURS[person.level];
    ctx.lineWidth = person.level === "CRITICAL" ? 2.5 : 1.5;
    ctx.strokeRect(x + x1 * width, y + y1 * height, (x2 - x1) * width, (y2 - y1) * height);
  }
}

function drawCamera() {
  const [ctx, width, height] = fitCanvas($("camera"));
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
  const sector = state.sector;
  const image = sector && images.get(sector.image);
  $("lock").hidden = !state.lock;
  if (!image) return;
  const task = state.task;
  let zoom = 1;
  let focusX = 0.5;
  let focusY = 0.5;
  if (state.lock) {
    const ease = Math.min(1, (task?.elapsed ?? 0) / (LOCK_SECONDS * 0.4));
    zoom = 1 + (ZOOM - 1) * ease;
    focusX = 0.5 + ((state.lock.box[0] + state.lock.box[2]) / 2 - 0.5) * ease;
    focusY = 0.5 + ((state.lock.box[1] + state.lock.box[3]) / 2 - 0.5) * ease;
    $("lock").textContent = `${state.lock.level}: ${state.lock.reasons.join(", ")}`;
  }
  const scaled = Math.max(width / image.width, height / image.height) * zoom;
  const drawWidth = image.width * scaled;
  const drawHeight = image.height * scaled;
  const x = Math.min(0, Math.max(width - drawWidth, width / 2 - focusX * drawWidth));
  const y = Math.min(0, Math.max(height - drawHeight, height / 2 - focusY * drawHeight));
  ctx.drawImage(image, x, y, drawWidth, drawHeight);
  const reveal = state.reveal[sector.id] ?? 0;
  drawOverlays(ctx, sector, x, y, drawWidth, drawHeight, reveal);
  if (task?.type === "scan") {
    ctx.fillStyle = "rgba(255, 216, 77, 0.8)";
    ctx.fillRect(0, y + reveal * drawHeight, width, 2);
  }
}

function drawMap() {
  const [ctx, width, height] = fitCanvas($("map"));
  ctx.fillStyle = "#10151b";
  ctx.fillRect(0, 0, width, height);
  const cellWidth = width / 3.6;
  const cellHeight = height / 3.6;
  const left = cellWidth * 0.5;
  const top = cellHeight * 0.1;
  const position = (row, column) => [left + (column + 0.5) * cellWidth, top + (row + 0.5) * cellHeight];
  for (const sector of mission.sectors) {
    const x = left + sector.column * cellWidth;
    const y = top + sector.row * cellHeight;
    const image = images.get(sector.image);
    ctx.globalAlpha = state.reveal[sector.id] === undefined ? 0.28 : 1;
    if (image) ctx.drawImage(image, x, y, cellWidth - 1, cellHeight - 1);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#fff";
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.fillText(sector.id, x + 4, y + 13);
    for (const person of sector.people) {
      if (!state.seen.has(person.id) || !REPORTED_LEVELS.has(person.level)) continue;
      ctx.fillStyle = state.reported.has(person.id) ? COLOURS[person.level] : "#fff";
      ctx.beginPath();
      ctx.arc(x + ((person.box[0] + person.box[2]) / 2) * cellWidth, y + person.box[3] * cellHeight, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  const [homeX, homeY] = position(HOME.row, HOME.column);
  ctx.fillStyle = "#b3bec9";
  ctx.font = "600 11px system-ui, sans-serif";
  ctx.fillText("BASE", homeX - 14, homeY + 4);
  const [droneX, droneY] = position(state.drone.row, state.drone.column);
  ctx.strokeStyle = "#ffd84d";
  ctx.fillStyle = "#ffd84d";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(droneX, droneY, 9, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(droneX, droneY, 3, 0, Math.PI * 2);
  ctx.fill();
}

function render() {
  drawCamera();
  drawMap();
  $("status").textContent = state.phase;
  const seconds = Math.floor(state.clock);
  $("hud").textContent = `${mission.title} mission  ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}  Height ${mission.altitude_m} m`;
}

function frame(time) {
  const seconds = Math.min(0.1, (time - lastFrame) / 1000);
  lastFrame = time;
  if (mission && state) {
    if (!paused) advance(seconds * SPEEDS[speedIndex]);
    render();
  }
  requestAnimationFrame(frame);
}

function choose(next) {
  name = next;
  $("choose-flood").setAttribute("aria-pressed", String(name === "flood"));
  $("choose-fire").setAttribute("aria-pressed", String(name === "fire"));
  start();
}

async function loadImages() {
  images = new Map();
  await Promise.all(Object.values(missions).flatMap((m) => m.sectors.map((sector) => new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      images.set(sector.image, image);
      resolve();
    };
    image.onerror = resolve;
    image.src = sector.image;
  }))));
}

$("choose-flood").addEventListener("click", () => choose("flood"));
$("choose-fire").addEventListener("click", () => choose("fire"));
$("restart").addEventListener("click", start);
$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
});
$("speed").addEventListener("click", () => {
  speedIndex = (speedIndex + 1) % SPEEDS.length;
  $("speed").textContent = `Speed ${SPEEDS[speedIndex]}×`;
});

(async () => {
  try {
    missions = await fetch("mission/mission.json").then((response) => response.json());
    await loadImages();
    $("choose-fire").hidden = !missions.fire;
    start();
    requestAnimationFrame((time) => {
      lastFrame = time;
      frame(time);
    });
  } catch {
    $("status").textContent = "The mission could not be loaded. Please refresh the page.";
  }
})();
