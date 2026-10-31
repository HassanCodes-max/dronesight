// Ranks people by danger and turns picture positions into map positions.
//
// The ranking is rules, not a trained model, because no dataset labels who is really at risk, and a
// rescue team must be able to see why someone ranked first. It is a research prototype: the weights
// are reasoned choices that have not been validated by emergency responders.
const METRES_PER_DEGREE = 111320;
const GPS_ERROR_M = 5;  // typical consumer drone GPS
const ANGLE_ERROR_DEGREES = 3;  // how far the entered tilt and heading may be from the truth
const LAST_USABLE_DEGREES = 80;  // nearer the horizon, one pixel covers too much ground to trust

const HAZARD_POINTS = { fire: 65, "flood water": 45, smoke: 10 };  // smoke boxes are coarse, so smoke alone never counts
const DEEP_POINTS = 20;  // someone far from safe ground cannot simply walk out
const DEEP_METRES = 3;
const NEAR_FIRE_METRES = 5;  // flames burn at a distance, but dry land beside water is safe
const NEAR_FIRE_POINTS = 40;
const ALONE_POINTS = 15;
const STILL_POINTS = 15;
const LYING_POINTS = 50;  // lying down, alone and not moving, wherever they are, may mean someone is hurt
const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];

const radians = (degrees) => (degrees * Math.PI) / 180;

// Someone is critical if they are in water or fire, beside flames, or lying still alone. Points only
// decide the order among them. Each reason is recorded, so the result can be explained.
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
  const besideFire = person.nearestHazard === "fire" && person.distanceM !== null && person.distanceM <= NEAR_FIRE_METRES;
  if (besideFire) {
    score += NEAR_FIRE_POINTS;
    reasons.push(`about ${Math.max(1, Math.round(person.distanceM))} m from fire`);
  }
  const hurt = person.lying && person.still && person.alone;
  if (hurt) {
    score += LYING_POINTS;
    reasons.push("lying down and not moving");
  }
  const critical = hurt || besideFire || person.hazard === "fire" || person.hazard === "flood water";
  if (critical && person.alone) {
    score += ALONE_POINTS;
    reasons.push("alone");
  }
  if (critical && person.still && !person.lying) {
    score += STILL_POINTS;
    reasons.push("not moving");
  }
  return { ...person, score, reasons, critical };
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

const LOCATION_MARKERS = ["©xyz", "location.ISO6709"];
const EDGE_BYTES = 8 * 1024 * 1024;  // the tag sits near one end of the file
const ISO_6709 = /([+-]\d{1,2}\.\d{2,})([+-]\d{1,3}\.\d{2,})/;

// Phones and some drones store where a video was filmed in its file, as text after one of these tags.
export async function readVideoLocation(file) {
  const parts = [file.slice(0, EDGE_BYTES), file.slice(Math.max(0, file.size - EDGE_BYTES))];
  for (const part of parts) {
    const text = new TextDecoder("latin1").decode(await part.arrayBuffer());
    for (const marker of LOCATION_MARKERS) {
      const start = text.indexOf(marker);
      const found = start < 0 ? null : text.slice(start, start + 200).match(ISO_6709);
      if (found && Math.abs(Number(found[1])) <= 90 && Math.abs(Number(found[2])) <= 180) {
        return { latitude: Number(found[1]), longitude: Number(found[2]) };
      }
    }
  }
  return null;
}

// Coordinates from typed text or a Google Maps link. The pin's !3d!4d comes before the map's centre.
export function parsePlace(text) {
  const found = text.match(/!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)/) ?? text.match(/@(-?\d+\.\d+),(-?\d+\.\d+)/)
    ?? text.match(/[?&](?:q|query|ll|destination)=(-?\d+\.\d+)(?:,|%2C)(-?\d+\.\d+)/i) ?? text.match(/^\s*(-?\d{1,2}(?:\.\d+)?)\s*[,\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*$/);
  if (!found) return null;
  const [latitude, longitude] = [Number(found[1]), Number(found[2])];
  return Math.abs(latitude) <= 90 && Math.abs(longitude) <= 180 ? { latitude, longitude } : null;
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
