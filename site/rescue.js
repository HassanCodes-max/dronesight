// The whole page: add a video, see who needs help and where. Red boxes are numbered, the rest are black.
// All text from the file or the visitor goes through textContent, never innerHTML.
import { analyse, prepare } from "./local.js";
import { exitDirection, nameOfPlace, parseFlightLog, readVideoLocation } from "./risk.js";
import { boxAt } from "./tracker.js";

const $ = (id) => document.getElementById(id);
const WAY_SECONDS = 3;  // the arrow stays near the moment it was measured, because the drone moves on
const MASK_SECONDS = 1;
const SHORT_ARROW = 45;
const RED = "#ff2d20";
const WAY_COLOUR = "#4dffa6";
const NUMBER_HEIGHT = 18;

const player = $("player");
let videoUrl = null;
let index = null;
let live = null;  // what the check has found at the moment it is looking at
let listed = [];  // only the people who need help, numbered
let chosen = null;
let flightLog = [];
let area = null;  // the main area the video was filmed in, by name

const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const count = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const capital = (text) => text[0].toUpperCase() + text.slice(1);

function show(section) {
  for (const id of ["start", "results"]) $(id).hidden = id !== section;
}

const heading = () => flightLog[0]?.heading ?? null;

function wayText(person) {
  if (!person.exit || person.exitM < 1) return null;
  return `Quickest way out: about ${Math.round(person.exitM)} m ${exitDirection(person.exit, heading())}, in a straight line. Check the route first.`;
}

function renderLocationNote() {
  $("location-text").textContent = area ? `Main area: ${area.name}.` : "Main area: not known. This video has no location in it.";
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
  why.className = "why";
  why.textContent = `${more.length ? `${capital(more.join(", "))}. ` : ""}Seen at ${clock(person.t)}.`;
  pick.append(number, title, why);
  const way = wayText(person);
  if (way) {
    const line = document.createElement("span");
    line.className = "way";
    line.textContent = way;
    pick.append(line);
  }
  pick.addEventListener("click", () => choose(person));
  item.append(pick);
  return item;
}

function renderPeople() {
  $("people").replaceChildren(...listed.map(card));
  $("panel").hidden = !listed.length;
  $("result-grid").classList.toggle("calm", !listed.length);
}

function refresh() {
  renderLocationNote();
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

// From where the person is now, in the direction measured at their worst moment. Only the chosen person
// gets the whole route, because a full line for everyone would hide the picture.
function drawWay(ctx, person, box, { left, top, width, height }) {
  const [feetX, feetY] = [(box[0] + box[2]) / 2, box[3]];
  const [fromX, fromY] = [left + feetX * width, top + feetY * height];
  const [routeX, routeY] = [(person.way[0] - person.feet[0]) * width, (person.way[1] - person.feet[1]) * height];
  const shown = person === chosen ? 1 : Math.min(1, SHORT_ARROW / Math.max(1, Math.hypot(routeX, routeY)));
  const [toX, toY] = [fromX + routeX * shown, fromY + routeY * shown];
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

// Plain red, because numbers are only given once the whole video has been judged.
function drawLive(ctx, { left, top, width, height }) {
  for (const picture of live.masks) ctx.drawImage(picture, left, top, width, height);
  for (const { box, danger } of live.people) {
    const [x, y, w, h] = [left + box[0] * width, top + box[1] * height, (box[2] - box[0]) * width, (box[3] - box[1]) * height];
    if (danger) {
      ctx.strokeStyle = RED;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);
    } else {
      drawOrdinary(ctx, x, y, w, h);
    }
  }
}

function draw() {
  $("time").textContent = `${clock(player.currentTime)} / ${clock(player.duration || 0)}`;
  if (document.activeElement !== $("seek")) $("seek").value = player.currentTime;
  const view = fitted();
  const { ctx, left, top, width, height } = view;
  if (live && player.videoWidth) drawLive(ctx, view);
  if (!index || !player.videoWidth) return;
  const distance = (mask) => Math.abs(mask.t - player.currentTime);
  const nearest = Math.min(MASK_SECONDS, ...index.masks.map(distance));  // flames move, so only one moment is shown
  for (const mask of index.masks) {
    if (distance(mask) === nearest) ctx.drawImage(mask.picture, left, top, width, height);
  }
  const numbered = new Map(listed.map((person) => [person.id, person]));
  const seen = index.tracks.map((track) => ({ person: numbered.get(track.id), box: boxAt(track, player.currentTime) })).filter(({ box }) => box);
  seen.sort((a, b) => Boolean(a.person) - Boolean(b.person));  // red stays on top
  for (const { person, box } of seen) {
    if (person?.way && Math.abs(player.currentTime - person.t) <= WAY_SECONDS) drawWay(ctx, person, box, view);
  }
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
  if (fire) return "Fire was seen.";
  return index.hazardFound.tooDark ? "No fire was seen, and the video is too dark to judge water." : "No water or fire was seen.";
}

function setChecking(checking) {
  $("play").disabled = checking;
  $("seek").disabled = checking;
  $("working").hidden = !checking;
  $("again").hidden = checking;
}

function showResults(found) {
  live = null;
  index = found;
  listed = index.people.filter((person) => person.critical).map((person, i) => Object.assign(person, { place: i + 1 }));
  chosen = null;
  const seen = count(index.tracks.length, "person", "people");
  $("report").hidden = !listed.length;
  $("headline").textContent = listed.length ? `${count(listed.length, "person needs", "people need")} help` : "Nobody in danger was found";
  $("subhead").textContent = listed.length
    ? `Out of about ${seen} in the video.${index.hazardFound.tooDark ? " The video is too dark to judge water, so only fire was checked." : ""}`
    : `About ${seen} in the video. ${hazardsSeen()} The software can miss people, and it cannot recognise fights or attacks.`;
  setChecking(false);
  refresh();
  if (listed.length) {
    choose(listed[0]);
  } else {
    player.currentTime = 0;
    draw();
  }
}

// Nothing is uploaded, so the video shows at once, and the check walks through it in front of the visitor.
async function check(file) {
  $("start-error").textContent = "";
  flightLog = $("log").files[0] ? parseFlightLog(await $("log").files[0].text()) : [];
  videoUrl = URL.createObjectURL(file);
  listed = [];
  $("headline").textContent = "Checking your video";
  $("subhead").textContent = "Boxes appear as each moment is checked. The numbered list comes at the end.";
  $("bar").removeAttribute("value");
  $("report").hidden = true;
  setChecking(true);
  renderPeople();
  show("results");
  player.src = videoUrl;
  try {
    const place = flightLog[0] ?? await readVideoLocation(file);
    area = place ? { name: await nameOfPlace(place) } : null;
    showResults(await analyse(player, (progress, message) => {
      if (progress) $("bar").value = progress;
      $("working-note").textContent = message;
    }, (look) => {
      live = look;
      draw();
    }));
  } catch (error) {
    startAgain();
    $("start-error").textContent = error.message;
  }
}

function startAgain() {
  live = null;
  setChecking(false);
  player.pause();
  player.removeAttribute("src");
  player.load();
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = null;
  index = null;
  listed = [];
  chosen = null;
  area = null;
  $("working-note").textContent = "";
  show("start");
}

$("again").addEventListener("click", startAgain);
$("file").addEventListener("change", (event) => {
  if (event.target.files[0]) check(event.target.files[0]);
  event.target.value = "";
});
if ($("file").files[0]) check($("file").files[0]);  // chosen before this script had loaded
const drop = $("drop");
drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("over");
  if (event.dataTransfer.files[0]) check(event.dataTransfer.files[0]);
});

const togglePlay = () => {
  if (live || !$("working").hidden) return;
  if (player.paused) player.play().catch(() => {});
  else player.pause();
};
$("play").addEventListener("click", togglePlay);
$("stage").addEventListener("click", togglePlay);
$("seek").addEventListener("input", (event) => {
  player.currentTime = Number(event.target.value);
  draw();
});
player.addEventListener("loadedmetadata", () => {
  $("seek").max = player.duration;
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
prepare().catch(() => {});  // only a speed-up: a failure here just means the models load with the first video
