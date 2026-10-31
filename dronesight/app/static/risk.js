// Ranks people by danger and turns picture positions into map positions.
//
// The ranking is rules, not a trained model, because no dataset labels who is really at risk, and a
// rescue team must be able to see why someone ranked first. It is a research prototype: the weights
// are reasoned choices that have not been validated by emergency responders.
const METRES_PER_DEGREE = 111320;
const GPS_ERROR_M = 5;  // typical consumer drone GPS
const ANGLE_ERROR_DEGREES = 3;  // how far the entered tilt and heading may be from the truth
const LAST_USABLE_DEGREES = 80;  // nearer the horizon, one pixel covers too much ground to trust

const HAZARD_POINTS = { fire: 65, "flood water": 45, smoke: 10 };  // smoke boxes are coarse, so smoke alone is only a watch
const DEEP_POINTS = 20;  // someone far from safe ground cannot simply walk out
const DEEP_METRES = 3;
// Metres from the hazard and points. Fire spreads and burns at a distance. Dry land beside water is
// safer, so it stays a watch unless other reasons add up. Smoke boxes are too coarse to measure from.
const NEAR_POINTS = { fire: [[5, 40], [20, 25], [50, 10]], "flood water": [[5, 25], [20, 10]] };
const ALONE_POINTS = 15;
const STILL_POINTS = 15;
const LEVELS = [[65, "CRITICAL"], [30, "HIGH"], [10, "WATCH"], [0, "CLEAR"]];
const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];

// Nothing here can see the ground, so a promise about speed or safety would be false reassurance.
const PROMISE = /\b(quick|quickly|fast|easy|easily|safely|soon|should be able|can reach|will reach|is safe|are safe|route is|clear path|no obstacles?)\b/i;

const radians = (degrees) => (degrees * Math.PI) / 180;

// Score one person and record each reason, so the result can be explained.
export function assess(person) {
  let score = 0;
  const reasons = [];
  if (person.hazard) {
    score += HAZARD_POINTS[person.hazard];
    reasons.push(`in ${person.hazard}`);
    if (person.hazard !== "smoke" && person.depthM >= DEEP_METRES) {
      score += DEEP_POINTS;
      reasons.push(`about ${Math.round(person.depthM)} m from safe ground`);
    }
  }
  if ((!person.hazard || person.hazard === "smoke") && person.distanceM !== null && person.nearestHazard) {
    const near = NEAR_POINTS[person.nearestHazard].find(([metres]) => person.distanceM <= metres);
    if (near) {
      score += near[1];
      reasons.push(`about ${Math.round(person.distanceM)} m from ${person.nearestHazard}`);
    }
  }
  if (score && person.alone) {
    score += ALONE_POINTS;
    reasons.push("alone");
  }
  if (score && person.still) {
    score += STILL_POINTS;
    reasons.push("not moving");
  }
  return { ...person, score, reasons, level: LEVELS.find(([minimum]) => score >= minimum)[1] };
}

// Most urgent first. Ties go to whoever is furthest from safety.
export const rank = (people) => people.map(assess).sort((a, b) => b.score - a.score || b.depthM - a.depthM || a.id - b.id);

// Where a picture point is on the ground, assuming flat ground. Tilt is 0 for straight down.
export function locate(x, y, drone, aspect) {
  const across = Math.tan(radians(drone.fov) / 2);
  const up = (0.5 - y) * 2 * (across / aspect);
  const fromCentre = Math.atan(up);
  const angle = radians(drone.tilt) + fromCentre;
  if (angle >= radians(LAST_USABLE_DEGREES)) return null;
  const forward = drone.height * Math.tan(angle);
  const right = (x - 0.5) * 2 * across * (drone.height / Math.cos(angle)) * Math.cos(fromCentre);
  const heading = radians(drone.heading);
  const north = forward * Math.cos(heading) - right * Math.sin(heading);
  const east = forward * Math.sin(heading) + right * Math.cos(heading);
  const slip = drone.height * (Math.tan(angle + radians(ANGLE_ERROR_DEGREES)) - Math.tan(angle));
  const turn = Math.hypot(forward, right) * radians(ANGLE_ERROR_DEGREES);
  return {
    latitude: drone.latitude + north / METRES_PER_DEGREE,
    longitude: drone.longitude + east / (METRES_PER_DEGREE * Math.cos(radians(drone.latitude))),
    errorM: Math.round(Math.hypot(GPS_ERROR_M, slip, turn)),
  };
}

export const withoutPromises = (text) => text.split(/(?<=[.!?])\s+/).filter((sentence) => !PROMISE.test(sentence)).join(" ");

export function between(from, to) {
  const north = (to.latitude - from.latitude) * METRES_PER_DEGREE;
  const east = (to.longitude - from.longitude) * METRES_PER_DEGREE * Math.cos(radians(from.latitude));
  return { metres: Math.round(Math.hypot(north, east)), direction: compass(Math.atan2(east, north)) };
}

export const compass = (angle) => COMPASS[Math.round((((angle * 180) / Math.PI + 360) % 360) / 45) % 8];

// The way out, as a compass direction when the heading is known, or as a direction in the picture.
export function exitDirection({ dx, dy }, heading) {
  if (heading !== null) return compass(Math.atan2(dx, -dy) + radians(heading));
  const across = Math.abs(dx) * 2 < Math.abs(dy) ? "" : dx > 0 ? "right" : "left";
  const down = Math.abs(dy) * 2 < Math.abs(dx) ? "" : dy > 0 ? "bottom" : "top";
  return `towards the ${[down, across].filter(Boolean).join(" ")} of the picture`;
}

// DJI flight logs differ by model, so each field is matched on its own. Times come from the subtitle's start.
export function parseFlightLog(text) {
  const entries = [];
  for (const block of text.replace(/\r/g, "").split(/\n\s*\n/)) {
    const time = block.match(/(\d+):(\d+):(\d+)[,.](\d+)\s*-->/);
    const latitude = block.match(/latitude\s*:\s*(-?\d+\.\d+)/i) ?? block.match(/GPS\s*\(\s*(-?\d+\.\d+)/i);
    const longitude = block.match(/longitude\s*:\s*(-?\d+\.\d+)/i) ?? block.match(/GPS\s*\(\s*-?\d+\.\d+\s*,\s*(-?\d+\.\d+)/i);
    const height = block.match(/rel_alt\s*:\s*(-?\d+(?:\.\d+)?)/i) ?? block.match(/\bH\s*(-?\d+(?:\.\d+)?)\s*M/i);
    if (!time || !latitude || !longitude) continue;
    const [, hours, minutes, seconds, parts] = time;
    const yaw = block.match(/gb_yaw\s*:\s*(-?\d+(?:\.\d+)?)/i);
    const pitch = block.match(/gb_pitch\s*:\s*(-?\d+(?:\.\d+)?)/i);
    entries.push({
      t: Number(hours) * 3600 + Number(minutes) * 60 + Number(seconds) + Number(`0.${parts}`),
      latitude: Number(latitude[1]),
      longitude: Number(longitude[1]),
      height: height ? Number(height[1]) : null,
      heading: yaw ? Number(yaw[1]) : null,
      tilt: pitch ? 90 + Number(pitch[1]) : null,  // a gimbal pitch of -90 looks straight down
    });
  }
  return entries;
}
