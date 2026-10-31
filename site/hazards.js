// Finds flood water or fire in a video's frames, then judges everyone the tracker followed. Hazards are
// checked less often than people, because they change slowly and their models are heavier.
import { boxAt } from "./tracker.js";
import { rank } from "./risk.js";

// Tuned by eye on real flood videos. A stricter score and these clean-ups take the outline off roofs,
// walls and heads, at the cost of some foamy water at the edges.
const WATER = { model: "water.onnx", width: 640, height: 384, score: 2, closeCells: 3, openCells: 8, smallestShare: 0.01 };
const FIRE = { model: "fire.onnx", size: 640, confidence: 0.25 };
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const BOX_PAD = 0.15;
const MASK_WIDTH = 640;
const PERSON_METRES = 1.7;  // a person's height is the only ruler in an ordinary video
const NEIGHBOUR_HEIGHTS = 6;
const STILL_SECONDS = 4;
const STILL_HEIGHTS = 0.6;
const LYING_RATIO = 1.2;  // wider than tall, which a standing person's box never is from the side
const FEET_OUT_OF_VIEW = 0.98;  // a box reaching the bottom edge hides where the person is standing
const MIN_HAZARD_SHARE = 0.005;  // specks are usually false alarms, so they are ignored
const MAX_SMOKE_SHARE = 0.1;  // with this much of the frame in smoke, its water result is not trusted
const SMOKY_VIDEO_SHARE = 0.25;  // with this many smoky frames, no water result in the video is trusted

export const HAZARD_SECONDS = 1;
export const HAZARD_MODELS = [WATER.model, FIRE.model];

function sized(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  return ctx;
}

function floodFrame(ort, session, video) {
  const ctx = sized(WATER.width, WATER.height);
  ctx.drawImage(video, 0, 0, WATER.width, WATER.height);
  const { data } = ctx.getImageData(0, 0, WATER.width, WATER.height);
  const pixels = WATER.width * WATER.height;
  const blob = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i += 1) {
    for (let channel = 0; channel < 3; channel += 1) blob[channel * pixels + i] = (data[4 * i + channel] / 255 - MEAN[channel]) / STD[channel];
  }
  const input = new ort.Tensor("float32", blob, [1, 3, WATER.height, WATER.width]);
  return session.run({ [session.inputNames[0]]: input }).then((results) => {
    const logits = results[session.outputNames[0]].data;
    const mask = cleanWater(Uint8Array.from(logits, (value) => (value >= WATER.score ? 1 : 0)), WATER.width, WATER.height);
    return { width: WATER.width, height: WATER.height, mask, smoke: [] };
  });
}

// Every cell within a distance of the mask, or only those further than it from the mask's edge.
const grown = (mask, width, height, cells) => Uint8Array.from(nearestSeed((i) => mask[i], width, height).distance, (distance) => (distance <= cells ? 1 : 0));
const shrunk = (mask, width, height, cells) => Uint8Array.from(nearestSeed((i) => !mask[i], width, height).distance, (distance) => (distance > cells ? 1 : 0));

function withoutSmallPatches(mask, width, height, smallest) {
  const seen = new Uint8Array(mask.length);
  const kept = new Uint8Array(mask.length);
  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || seen[start]) continue;
    const patch = [start];
    seen[start] = 1;
    for (let next = 0; next < patch.length; next += 1) {
      const cell = patch[next];
      const sides = [cell % width ? cell - 1 : -1, (cell + 1) % width ? cell + 1 : -1, cell - width, cell + width];
      for (const side of sides) {
        if (side >= 0 && side < mask.length && mask[side] && !seen[side]) {
          seen[side] = 1;
          patch.push(side);
        }
      }
    }
    if (patch.length >= smallest) for (const cell of patch) kept[cell] = 1;
  }
  return kept;
}

// Fills gaps such as a person standing in the water, then removes thin strips and small patches.
export function cleanWater(mask, width, height) {
  const closed = shrunk(grown(mask, width, height, WATER.closeCells), width, height, WATER.closeCells);
  const opened = grown(shrunk(closed, width, height, WATER.openCells), width, height, WATER.openCells);
  return withoutSmallPatches(opened, width, height, WATER.smallestShare * mask.length);
}

function flamePixel(red, green, blue) {
  const high = Math.max(red, green, blue);
  const low = Math.min(red, green, blue);
  if (high === low) return false;
  const saturation = ((high - low) / high) * 255;
  let hue = high === red ? (green - blue) / (high - low) : high === green ? 2 + (blue - red) / (high - low) : 4 + (red - green) / (high - low);
  hue = ((hue * 30) + 180) % 180;
  return (hue <= 35 && saturation >= 110 && high >= 190) || (hue <= 40 && high >= 240);  // glowing orange, then white-hot
}

async function fireFrame(ort, session, video) {
  const scale = FIRE.size / Math.max(video.videoWidth, video.videoHeight);
  const width = Math.round(video.videoWidth * scale);
  const height = Math.round(video.videoHeight * scale);
  const left = Math.floor((FIRE.size - width) / 2);
  const top = Math.floor((FIRE.size - height) / 2);
  const ctx = sized(FIRE.size, FIRE.size);
  ctx.fillStyle = "rgb(114, 114, 114)";
  ctx.fillRect(0, 0, FIRE.size, FIRE.size);
  ctx.drawImage(video, left, top, width, height);
  const { data } = ctx.getImageData(0, 0, FIRE.size, FIRE.size);
  const pixels = FIRE.size * FIRE.size;
  const blob = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i += 1) {
    for (let channel = 0; channel < 3; channel += 1) blob[channel * pixels + i] = data[4 * i + channel] / 255;
  }
  const results = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", blob, [1, 3, FIRE.size, FIRE.size]) });
  const output = results[session.outputNames[0]];
  const count = output.dims[2];
  const boxes = { smoke: [], fire: [] };
  for (let i = 0; i < count; i += 1) {
    const smoke = output.data[4 * count + i];
    const flame = output.data[5 * count + i];
    if (Math.max(smoke, flame) < FIRE.confidence) continue;
    const [x, y, w, h] = [output.data[i], output.data[count + i], output.data[2 * count + i], output.data[3 * count + i]];
    (flame >= smoke ? boxes.fire : boxes.smoke).push([(x - w / 2 - left) / width, (y - h / 2 - top) / height, (x + w / 2 - left) / width, (y + h / 2 - top) / height]);
  }
  // A box holds bystanders too, so only flame-coloured pixels inside it count as fire.
  const maskWidth = MASK_WIDTH;
  const maskHeight = Math.round((MASK_WIDTH * video.videoHeight) / video.videoWidth);
  const small = sized(maskWidth, maskHeight);
  small.drawImage(video, 0, 0, maskWidth, maskHeight);
  const colours = small.getImageData(0, 0, maskWidth, maskHeight).data;
  const mask = new Uint8Array(maskWidth * maskHeight);
  for (const [x1, y1, x2, y2] of boxes.fire) {
    const padX = (x2 - x1) * BOX_PAD;
    const padY = (y2 - y1) * BOX_PAD;
    const fromX = Math.max(0, Math.floor((x1 - padX) * maskWidth));
    const toX = Math.min(maskWidth, Math.ceil((x2 + padX) * maskWidth));
    const fromY = Math.max(0, Math.floor((y1 - padY) * maskHeight));
    const toY = Math.min(maskHeight, Math.ceil((y2 + padY) * maskHeight));
    for (let row = fromY; row < toY; row += 1) {
      for (let column = fromX; column < toX; column += 1) {
        const cell = row * maskWidth + column;
        if (flamePixel(colours[4 * cell], colours[4 * cell + 1], colours[4 * cell + 2])) mask[cell] = 1;
      }
    }
  }
  return { width: maskWidth, height: maskHeight, mask, smoke: boxes.smoke };
}

export const smokeShare = (boxes) => boxes.reduce((sum, [x1, y1, x2, y2]) => sum + Math.max(0, Math.min(x2, 1) - Math.max(x1, 0)) * Math.max(0, Math.min(y2, 1) - Math.max(y1, 0)), 0);

export async function hazardFrame(ort, sessions, video, t) {
  const flood = await floodFrame(ort, sessions[0], video);
  const fire = await fireFrame(ort, sessions[1], video);
  return { t, flood: { ...flood, t }, fire: { ...fire, t } };
}

// The water model takes smoke for water, and smoke drifts in and out of view between frames.
export function dropSmokyWater(frames) {
  const smoky = frames.filter((frame) => smokeShare(frame.fire.smoke) >= MAX_SMOKE_SHARE);
  for (const frame of smoky.length / frames.length >= SMOKY_VIDEO_SHARE ? frames : smoky) frame.flood.mask.fill(0);
}

// For every cell, how far the nearest seed is and which one it is, in two sweeps.
function nearestSeed(seeded, width, height) {
  const size = width * height;
  const distance = new Float32Array(size).fill(Infinity);
  const from = new Int32Array(size).fill(-1);
  for (let i = 0; i < size; i += 1) {
    if (seeded(i)) {
      distance[i] = 0;
      from[i] = i;
    }
  }
  const relax = (cell, other) => {
    if (from[other] < 0) return;
    const dx = (cell % width) - (from[other] % width);
    const dy = Math.floor(cell / width) - Math.floor(from[other] / width);
    const length = Math.hypot(dx, dy);
    if (length < distance[cell]) {
      distance[cell] = length;
      from[cell] = from[other];
    }
  };
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const cell = row * width + column;
      if (column) relax(cell, cell - 1);
      if (row) {
        relax(cell, cell - width);
        if (column) relax(cell, cell - width - 1);
        if (column < width - 1) relax(cell, cell - width + 1);
      }
    }
  }
  for (let row = height - 1; row >= 0; row -= 1) {
    for (let column = width - 1; column >= 0; column -= 1) {
      const cell = row * width + column;
      if (column < width - 1) relax(cell, cell + 1);
      if (row < height - 1) {
        relax(cell, cell + width);
        if (column < width - 1) relax(cell, cell + width + 1);
        if (column) relax(cell, cell + width - 1);
      }
    }
  }
  return { distance, from };
}

function isLying(track) {
  const shapes = track.boxes.map(([x1, y1, x2, y2]) => (x2 - x1) / Math.max(y2 - y1, 1e-6)).sort((a, b) => a - b);
  return shapes[Math.floor(shapes.length / 2)] >= LYING_RATIO;
}

function isStill(track) {
  const span = track.times.at(-1) - track.times[0];
  if (span < STILL_SECONDS) return false;
  const heights = track.boxes.map((box) => box[3] - box[1]).sort((a, b) => a - b);
  const body = heights[Math.floor(heights.length / 2)];
  const [startX, startY] = track.world[0];
  return track.world.every(([x, y]) => Math.hypot(x - startX, y - startY) / body < STILL_HEIGHTS);
}

// One person at one moment. Their own height is the ruler, because an ordinary video has no scale.
function personAt(track, box, frame, fields, name, everyone) {
  const { width, height, mask, smoke } = frame;
  const [x1, y1, x2, y2] = box;
  const feet = [(x1 + x2) / 2, y2];
  const cell = Math.min(height - 1, Math.floor(feet[1] * height)) * width + Math.min(width - 1, Math.floor(feet[0] * width));
  const metres = PERSON_METRES / Math.max(2, (y2 - y1) * height);
  const body = y2 - y1;
  const company = everyone.some((other) => other !== feet && Math.hypot((other[0] - feet[0]) * (width / height), other[1] - feet[1]) < NEIGHBOUR_HEIGHTS * body);
  const inSmoke = smoke.some(([left, top, right, bottom]) => feet[0] >= left && feet[0] <= right && feet[1] >= top && feet[1] <= bottom);
  const inside = Boolean(mask[cell]) && y2 < FEET_OUT_OF_VIEW;
  const way = inside ? fields.safe.from[cell] : -1;
  return {
    id: track.id,
    t: 0,
    box,
    feet,
    hazard: inside ? name : inSmoke ? "smoke" : null,
    depthM: inside ? fields.safe.distance[cell] * metres : 0,
    distanceM: inside || !Number.isFinite(fields.danger.distance[cell]) ? null : fields.danger.distance[cell] * metres,
    nearestHazard: inside ? null : name,
    exit: way >= 0 ? { dx: (way % width) - (cell % width), dy: Math.floor(way / width) - Math.floor(cell / width) } : null,
    way: way >= 0 ? [((way % width) + 0.5) / width, (Math.floor(way / width) + 0.5) / height] : null,
    alone: !company,
    still: isStill(track),
    lying: isLying(track),
  };
}

const hasHazard = ({ mask }) => mask.reduce((sum, cell) => sum + cell, 0) / mask.length >= MIN_HAZARD_SHARE;

export const hazardSeen = (frames, kind) => frames.some((frame) => hasHazard(frame[kind]));

// Each person's worst moment in the video decides their place in the ranking.
export function judge(tracks, frames) {
  const worst = new Map();
  for (const [kind, name] of [["flood", "flood water"], ["fire", "fire"]]) judgeHazard(tracks, frames.map((frame) => frame[kind]), name, worst);
  return [...worst.values()].sort((a, b) => b.score - a.score || b.depthM - a.depthM || a.id - b.id);
}

function judgeHazard(tracks, frames, name, worst) {
  for (const frame of frames) {
    const { width, height, t } = frame;
    const mask = hasHazard(frame) ? frame.mask : new Uint8Array(frame.mask.length);
    const usable = { ...frame, mask };
    const fields = {
      safe: nearestSeed((i) => !mask[i], width, height),
      danger: nearestSeed((i) => mask[i], width, height),
    };
    const seen = tracks.map((track) => ({ track, box: boxAt(track, t) })).filter((item) => item.box);
    const everyone = seen.map(({ box }) => [(box[0] + box[2]) / 2, box[3]]);
    const judged = rank(seen.map(({ track, box }, i) => {
      everyone[i] = [(box[0] + box[2]) / 2, box[3]];
      return { ...personAt(track, box, usable, fields, name, everyone.filter((_, other) => other !== i)), t };
    }));
    for (const person of judged) {
      const known = worst.get(person.id);
      if (!known || person.score > known.score) worst.set(person.id, person);
    }
  }
}

// Hazards as a see-through picture for the video overlay.
export function maskPicture(frame, kind) {
  const { width, height, mask } = frame;
  if (!hasHazard(frame)) return null;
  const ctx = sized(width, height);
  const image = ctx.createImageData(width, height);
  const colour = kind === "flood" ? [64, 156, 255, 90] : [255, 106, 40, 120];
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i]) image.data.set(colour, 4 * i);
  }
  ctx.putImageData(image, 0, 0);
  return ctx.canvas;
}
