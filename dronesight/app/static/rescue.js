// The whole page: add a video, see who needs help and where. Red boxes are numbered, the rest are black.
// All text from the file or the visitor goes through textContent, never innerHTML.
import { analyse } from "./local.js";
import { exitDirection, locate, parseFlightLog, readVideoLocation } from "./risk.js";
import { boxAt } from "./tracker.js";

const $ = (id) => document.getElementById(id);
const DEFAULT_FOV = 84;  // most consumer drone cameras
const MAX_IN_REPORT = 10;  // a WhatsApp link has to stay short
const WAY_SECONDS = 1.5;  // the arrow stays near the moment it was measured, because the drone moves on
const RED = "#ff2d20";
const WAY_COLOUR = "#4dffa6";
const NUMBER_HEIGHT = 18;
const LOCATION_TIMEOUT_MS = 15000;
// Someone in a real emergency could stop trying if a page said help was coming, so this never says so.
const NOT_TOLD = "A live system would share each numbered person's location with officials and send help. This demo is not connected to officials, so nobody has been told and no help is on the way. Call your local emergency services.";

const player = $("player");
let videoUrl = null;
let index = null;
let listed = [];  // only the people who need help, numbered
let chosen = null;
let drone = null;
let team = null;
let flightLog = [];
let area = null;  // a rough place for the whole video: from its file or the visitor's device

const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const count = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const coordinates = ({ latitude, longitude }) => `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`;
const capital = (text) => text[0].toUpperCase() + text.slice(1);

function show(section) {
  for (const id of ["start", "working", "results"]) $(id).hidden = id !== section;
}

function droneAt(time) {
  if (!flightLog.length) return drone;
  const entry = flightLog.reduce((best, item) => (Math.abs(item.t - time) < Math.abs(best.t - time) ? item : best));
  return { fov: DEFAULT_FOV, heading: entry.heading ?? 0, tilt: entry.tilt ?? 0, ...drone, ...entry, height: entry.height ?? drone?.height };
}

// A person's own spot needs the drone's height and direction. Without them there is only an area.
function whereIs(person) {
  const position = droneAt(person.t);
  return position?.height ? locate(person.feet[0], person.feet[1], position, index.aspect) : null;
}

function placeText(person) {
  const place = whereIs(person);
  if (place) return `Location: ${coordinates(place)}, an estimate, within about ${place.errorM} m.`;
  return area ? `Area only: ${coordinates(area)}, ${area.source}.` : null;
}

const mapsLink = ({ latitude, longitude }) => (team
  ? `https://www.google.com/maps/dir/?api=1&origin=${team.latitude},${team.longitude}&destination=${latitude},${longitude}&travelmode=walking`
  : `https://www.google.com/maps?q=${latitude},${longitude}`);

function wayText(person) {
  if (!person.exit || person.depthM < 1) return null;
  return `Quickest way out: about ${Math.round(person.depthM)} m ${exitDirection(person.exit, droneAt(person.t)?.heading ?? null)}, in a straight line. Check the route first.`;
}

function reportText() {
  const lines = [`DroneSight report from a drone video. Found by software and not checked by a person.`,
    `${count(listed.length, "person needs", "people need")} help.`];
  for (const person of listed.slice(0, MAX_IN_REPORT)) {
    const place = whereIs(person) ?? area;
    lines.push(`${person.place}. ${capital(person.reasons.join(", "))}. ${place ? `Location ${coordinates(place)}` : "Location unknown"}. Seen at ${clock(person.t)} in the video.`);
  }
  if (listed.length > MAX_IN_REPORT) lines.push(`And ${listed.length - MAX_IN_REPORT} more.`);
  return lines.join("\n");
}

function renderReport() {
  $("report").hidden = !listed.length;
  if (listed.length) $("report-send").href = `https://wa.me/?text=${encodeURIComponent(reportText())}`;
}

function renderLocationNote() {
  $("location-note").hidden = Boolean(drone || flightLog.length);
  $("location-text").textContent = area
    ? `Only the general area is known: ${coordinates(area)}, ${area.source}. Each person's own spot needs the drone's height and direction.`
    : "No location is known yet, because a normal video has no GPS in it.";
  $("use-my-location").hidden = area !== null;
}

function card(person) {
  const item = document.createElement("li");
  item.className = "person";
  item.setAttribute("aria-current", String(person === chosen));
  const pick = document.createElement("button");
  pick.className = "person-pick";
  pick.type = "button";
  const number = document.createElement("span");
  number.className = "place";
  number.textContent = person.place;
  const [main, ...more] = person.reasons;
  const title = document.createElement("strong");
  title.textContent = capital(main);
  const why = document.createElement("span");
  why.textContent = `${more.length ? `${capital(more.join(", "))}. ` : ""}Seen at ${clock(person.t)}.`;
  pick.append(number, title, why);
  pick.addEventListener("click", () => choose(person));
  const details = document.createElement("div");
  details.className = "details";
  const way = wayText(person);
  if (way) {
    const line = document.createElement("p");
    line.textContent = way;
    details.append(line);
  }
  const location = placeText(person);
  if (location) {
    const line = document.createElement("p");
    line.textContent = location;
    details.append(line);
  }
  const place = whereIs(person) ?? area;
  if (place) {
    const link = document.createElement("a");
    link.href = mapsLink(place);
    link.rel = "noopener noreferrer";
    link.target = "_blank";
    link.textContent = team ? "Directions in Google Maps" : "Open in Google Maps";
    const line = document.createElement("p");
    line.append(link);
    details.append(line);
  }
  item.append(pick, details);
  return item;
}

function renderPeople() {
  $("people").replaceChildren(...listed.map(card));
  $("panel").hidden = !listed.length;
  $("result-grid").classList.toggle("calm", !listed.length);
}

function refresh() {
  renderLocationNote();
  renderReport();
  renderPeople();
}

function choose(person) {
  chosen = person;
  player.pause();
  player.currentTime = person.t;
  renderPeople();
  draw();
}

function fitted() {
  const canvas = $("overlay");
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * scale);
  canvas.height = Math.round(rect.height * scale);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  const fit = Math.min(rect.width / (player.videoWidth || 1), rect.height / (player.videoHeight || 1));
  const width = player.videoWidth * fit;
  const height = player.videoHeight * fit;
  return { ctx, width, height, left: (rect.width - width) / 2, top: (rect.height - height) / 2 };
}

function drawWay(ctx, person, { left, top, width, height }) {
  const [fromX, fromY] = [left + person.feet[0] * width, top + person.feet[1] * height];
  const [toX, toY] = [left + person.way[0] * width, top + person.way[1] * height];
  const angle = Math.atan2(toY - fromY, toX - fromX);
  ctx.strokeStyle = WAY_COLOUR;
  ctx.fillStyle = WAY_COLOUR;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - 11 * Math.cos(angle - 0.5), toY - 11 * Math.sin(angle - 0.5));
  ctx.lineTo(toX - 11 * Math.cos(angle + 0.5), toY - 11 * Math.sin(angle + 0.5));
  ctx.closePath();
  ctx.fill();
}

// Black has a pale edge, or it would vanish on dark ground.
function drawOrdinary(ctx, x, y, w, h) {
  ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(x, y, w, h);
}

function drawNumbered(ctx, x, y, w, h, person, { left, top }) {
  ctx.strokeStyle = RED;
  ctx.lineWidth = person === chosen ? 4 : 2.5;
  ctx.strokeRect(x, y, w, h);
  const text = String(person.place);
  const width = ctx.measureText(text).width + 10;
  const [labelX, labelY] = [Math.max(left, x), Math.max(top, y - NUMBER_HEIGHT)];
  ctx.fillStyle = RED;
  ctx.fillRect(labelX, labelY, width, NUMBER_HEIGHT);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, labelX + 5, labelY + 14);
}

function draw() {
  $("time").textContent = `${clock(player.currentTime)} / ${clock(index?.duration ?? 0)}`;
  if (document.activeElement !== $("seek")) $("seek").value = player.currentTime;
  const view = fitted();
  const { ctx, left, top, width, height } = view;
  if (!index || !player.videoWidth) return;
  for (const mask of index.masks) {
    if (Math.abs(mask.t - player.currentTime) <= 1) ctx.drawImage(mask.picture, left, top, width, height);
  }
  if (chosen?.way && Math.abs(player.currentTime - chosen.t) <= WAY_SECONDS) drawWay(ctx, chosen, view);
  const numbered = new Map(listed.map((person) => [person.id, person]));
  const seen = index.tracks.map((track) => ({ person: numbered.get(track.id), box: boxAt(track, player.currentTime) })).filter(({ box }) => box);
  seen.sort((a, b) => Boolean(a.person) - Boolean(b.person));  // red stays on top
  ctx.font = "700 14px system-ui, sans-serif";
  for (const { person, box } of seen) {
    const [x, y, w, h] = [left + box[0] * width, top + box[1] * height, (box[2] - box[0]) * width, (box[3] - box[1]) * height];
    if (person) drawNumbered(ctx, x, y, w, h, person, view);
    else drawOrdinary(ctx, x, y, w, h);
  }
}

function playbackLoop() {
  if (player.paused || player.ended) return;
  draw();
  requestAnimationFrame(playbackLoop);
}

function hazardsSeen() {
  const { flood, fire } = index.hazardFound;
  if (flood && fire) return "Water and fire were seen.";
  if (flood) return "Water was seen.";
  return fire ? "Fire was seen." : "No water or fire was seen.";
}

function showResults(result) {
  videoUrl = result.url;
  ({ index } = result);
  listed = index.people.filter((person) => person.critical).map((person, i) => Object.assign(person, { place: i + 1 }));
  chosen = null;
  const seen = count(index.tracks.length, "person", "people");
  $("headline").textContent = listed.length ? `${count(listed.length, "person needs", "people need")} help` : "Nobody in danger was found";
  $("subhead").textContent = listed.length
    ? `Out of about ${seen} in the video.`
    : `About ${seen} in the video. ${hazardsSeen()} The software can miss people, and it cannot recognise fights or attacks.`;
  $("report-note").textContent = NOT_TOLD;
  refresh();
  show("results");
  player.src = videoUrl;
  $("seek").max = index.duration;
  player.addEventListener("loadeddata", () => {
    if (listed.length) choose(listed[0]);
    else draw();
  }, { once: true });
}

async function check(file) {
  $("start-error").textContent = "";
  flightLog = $("log").files[0] ? parseFlightLog(await $("log").files[0].text()) : [];
  show("working");
  $("bar").removeAttribute("value");
  try {
    const found = await readVideoLocation(file);
    area = found ? { ...found, source: "from the video file" } : null;
    const result = await analyse(file, (progress, message) => {
      if (progress) $("bar").value = progress;
      $("working-note").textContent = message;
    });
    showResults(result);
  } catch (error) {
    $("start-error").textContent = error.message;
    show("start");
  }
}

function startAgain() {
  player.pause();
  player.removeAttribute("src");
  player.load();
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = null;
  index = null;
  chosen = null;
  drone = null;
  area = null;
  $("working-note").textContent = "";
  show("start");
}

function numberIn(id, low, high) {
  const text = $(id).value.trim();
  const value = Number(text);
  return text && value >= low && value <= high ? value : null;
}

$("again").addEventListener("click", startAgain);
$("file").addEventListener("change", (event) => {
  if (event.target.files[0]) check(event.target.files[0]);
  event.target.value = "";
});
const drop = $("drop");
drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("over");
  if (event.dataTransfer.files[0]) check(event.dataTransfer.files[0]);
});

$("report-send").addEventListener("click", () => {
  $("report-note").textContent = "WhatsApp opened. Nothing is sent until you choose who gets it and press send there.";
});
$("report-copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText(reportText()).catch(() => {});
  $("report-note").textContent = "Report copied. Nobody has been told until you send it.";
});

// Only ever asked for after a click, and the browser asks the visitor's permission too.
$("use-my-location").addEventListener("click", () => {
  $("location-text").textContent = "Finding your location…";
  navigator.geolocation.getCurrentPosition(({ coords }) => {
    area = { latitude: coords.latitude, longitude: coords.longitude, source: "where you are now, not where the video was filmed" };
    refresh();
  }, () => {
    $("location-text").textContent = "Your location could not be read. You can enter the drone's details instead.";
  }, { timeout: LOCATION_TIMEOUT_MS });
});

$("where-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const [latitude, longitude] = [numberIn("drone-lat", -90, 90), numberIn("drone-lon", -180, 180)];
  const height = numberIn("drone-height", 1, 1000);
  const [heading, tilt] = [numberIn("drone-heading", -360, 360), numberIn("drone-tilt", 0, 89)];
  if (latitude === null || longitude === null || height === null || heading === null || tilt === null) {
    $("where-error").textContent = "Please check the numbers. Height is 1 to 1000 m, and tilt is 0 to 89 degrees.";
    return;
  }
  $("where-error").textContent = "";
  drone = { latitude, longitude, height, heading, tilt, fov: DEFAULT_FOV };
  const [teamLatitude, teamLongitude] = [numberIn("team-lat", -90, 90), numberIn("team-lon", -180, 180)];
  team = teamLatitude === null || teamLongitude === null ? null : { latitude: teamLatitude, longitude: teamLongitude };
  $("where").open = false;
  refresh();
});

const togglePlay = () => (player.paused ? player.play().catch(() => {}) : player.pause());
$("play").addEventListener("click", togglePlay);
$("stage").addEventListener("click", togglePlay);
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
window.addEventListener("resize", draw);
