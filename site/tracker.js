// Follows each person through a video. The drone moves too, so its own motion is removed before
// boxes are compared. Analysis runs after the flight, so gaps can be filled from both sides.
const STRONG = 0.25;  // only these can start a track; weaker boxes can only continue one
const REACH = 1.6;  // how far a person can move between samples, in their own body heights
const MIN_SIZE_RATIO = 0.5;
const MAX_MISSED_SECONDS = 2;
const MIN_SECONDS_SEEN = 0.8;  // anything shorter is usually a false detection
const MIN_HITS = 2;  // long videos are read once a second, so someone in view briefly is only seen a few times
const SIZE_SMOOTHING = 2;  // samples either side
const VELOCITY_KEPT = 0.7;

const SEARCH = 24;  // thumbnail pixels
const COARSE_STEP = 3;

const centre = (box) => [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];

// How far the picture moved between two grey thumbnails, in thumbnail pixels.
export function cameraShift(before, after, width, height) {
  const difference = (dx, dy, limit) => {
    let total = 0;
    for (let y = SEARCH; y < height - SEARCH; y += 2) {
      for (let x = SEARCH; x < width - SEARCH; x += 2) {
        total += Math.abs(after[y * width + x] - before[(y - dy) * width + (x - dx)]);
      }
      if (total >= limit) return total;
    }
    return total;
  };
  const search = (range, step, startX, startY) => {
    let best = { x: startX, y: startY, score: Infinity };
    for (let dy = -range; dy <= range; dy += step) {
      for (let dx = -range; dx <= range; dx += step) {
        const x = startX + dx;
        const y = startY + dy;
        if (Math.abs(x) > SEARCH || Math.abs(y) > SEARCH) continue;
        const score = difference(x, y, best.score);
        if (score < best.score) best = { x, y, score };
      }
    }
    return best;
  };
  const rough = search(SEARCH, COARSE_STEP, 0, 0);
  return search(COARSE_STEP - 1, 1, rough.x, rough.y);
}

function closest(tracks, detections) {
  const pairs = [];
  tracks.forEach((track, i) => {
    const height = track.last[3] - track.last[1];
    detections.forEach((detection, j) => {
      const size = (detection.box[3] - detection.box[1]) / height;
      if (size < MIN_SIZE_RATIO || size > 1 / MIN_SIZE_RATIO) return;
      const [x, y] = centre(detection.box);
      const distance = Math.hypot(x - track.expected[0], y - track.expected[1]) / height;
      if (distance < REACH) pairs.push({ distance, i, j });
    });
  });
  const usedTracks = new Set();
  const usedDetections = new Set();
  const matched = [];
  for (const { distance, i, j } of pairs.sort((a, b) => a.distance - b.distance)) {
    if (usedTracks.has(i) || usedDetections.has(j)) continue;
    usedTracks.add(i);
    usedDetections.add(j);
    matched.push({ track: tracks[i], detection: detections[j], distance });
  }
  return {
    matched,
    tracks: tracks.filter((_, i) => !usedTracks.has(i)),
    detections: detections.filter((_, j) => !usedDetections.has(j)),
  };
}

export class Tracker {
  constructor() {
    this.open = [];
    this.closed = [];
    this.nextId = 1;
    this.time = 0;
    this.drift = [0, 0];  // how far the picture has moved in total, to tell walking from camera motion
  }

  update(time, detections, shift) {
    const elapsed = time - this.time;
    this.time = time;
    this.drift = [this.drift[0] + shift.x, this.drift[1] + shift.y];
    for (const track of this.open) {
      // The picture moved, so everyone in it moved with it.
      track.last = [track.last[0] + shift.x, track.last[1] + shift.y, track.last[2] + shift.x, track.last[3] + shift.y];
      const [x, y] = centre(track.last);
      track.expected = [x + track.velocity[0] * elapsed, y + track.velocity[1] * elapsed];
    }
    // Sure detections choose first, then weak ones may continue whoever is left.
    const first = closest(this.open, detections.filter((d) => d.score >= STRONG));
    const second = closest(first.tracks, detections.filter((d) => d.score < STRONG));
    for (const { track, detection } of [...first.matched, ...second.matched]) this.extend(track, detection, time);
    for (const detection of first.detections) this.start(detection, time);
    this.open = this.open.filter((track) => {
      if (time - track.times.at(-1) <= MAX_MISSED_SECONDS) return true;
      this.closed.push(track);
      return false;
    });
  }

  start(detection, time) {
    const box = detection.box;
    this.open.push({ id: this.nextId, last: box, expected: centre(box), velocity: [0, 0], times: [time], boxes: [box], world: [this.onGround(box)], hits: 1 });
    this.nextId += 1;
  }

  extend(track, detection, time) {
    const elapsed = time - track.times.at(-1);
    const [x, y] = centre(detection.box);
    const [oldX, oldY] = centre(track.last);
    track.velocity = track.velocity.map((v, axis) => VELOCITY_KEPT * v + (1 - VELOCITY_KEPT) * ([x, y][axis] - [oldX, oldY][axis]) / elapsed);
    track.last = detection.box;
    track.times.push(time);
    track.boxes.push(detection.box);
    track.world.push(this.onGround(detection.box));
    track.hits += 1;
  }

  onGround(box) {
    const [x, y] = centre(box);
    return [x - this.drift[0], y - this.drift[1]];
  }

  finish() {
    const tracks = [...this.closed, ...this.open].filter(
      (track) => track.hits >= MIN_HITS && track.times.at(-1) - track.times[0] >= MIN_SECONDS_SEEN,
    );
    return tracks.map((track, order) => ({ id: order + 1, times: track.times, boxes: smooth(track.boxes), world: track.world }));
  }
}

// Detectors change a box's size from frame to frame, so size is averaged. Its centre stays as seen.
function smooth(boxes) {
  return boxes.map((box, i) => {
    const near = boxes.slice(Math.max(0, i - SIZE_SMOOTHING), i + SIZE_SMOOTHING + 1);
    const width = near.reduce((sum, b) => sum + b[2] - b[0], 0) / near.length;
    const height = near.reduce((sum, b) => sum + b[3] - b[1], 0) / near.length;
    const [x, y] = centre(box);
    return [x - width / 2, y - height / 2, x + width / 2, y + height / 2];
  });
}

// A track's box at any moment, between the samples it was seen in.
export function boxAt(track, time) {
  const { times, boxes } = track;
  if (time < times[0] || time > times.at(-1)) return null;
  let low = 0;
  let high = times.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (times[middle] <= time) low = middle;
    else high = middle - 1;
  }
  if (low === times.length - 1) return boxes[low];
  const part = (time - times[low]) / (times[low + 1] - times[low]);
  return boxes[low].map((value, i) => value + (boxes[low + 1][i] - value) * part);
}
