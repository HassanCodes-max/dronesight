// Ranks people by danger and names the area a video was filmed in.
//
// The ranking is rules, not a trained model, because no dataset labels who is really at risk, and a
// rescue team must be able to see why someone ranked first. It is a research prototype: the weights
// are reasoned choices that have not been validated by emergency responders.

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

export const compass = (angle) => COMPASS[Math.round((((angle * 180) / Math.PI + 360) % 360) / 45) % 8];

// The way out, as a compass direction when the heading is known, or as a direction in the picture.
export function exitDirection({ dx, dy }, heading) {
  if (heading !== null) return compass(Math.atan2(dx, -dy) + radians(heading));
  const across = Math.abs(dx) * 2 < Math.abs(dy) ? "" : dx > 0 ? "right" : "left";
  const down = Math.abs(dy) * 2 < Math.abs(dx) ? "" : dy > 0 ? "bottom" : "top";
  return `towards the ${[down, across].filter(Boolean).join(" ")} of the picture`;
}

const PLACES = "places.csv";

// The nearest listed town, found on this device, so naming a place sends nothing anywhere. Commas in a
// town's name are quoted in the file, and the first two fields are always numbers.
export async function nameOfPlace({ latitude, longitude }) {
  const list = await (await fetch(PLACES)).text();
  const squash = Math.cos(radians(latitude));
  let best = null;
  let shortest = Infinity;
  for (const line of list.split("\n")) {
    const [lat, lon] = line.split(",", 2).map(Number);
    const away = (lat - latitude) ** 2 + ((lon - longitude) * squash) ** 2;
    if (away < shortest) {
      shortest = away;
      best = line;
    }
  }
  const [, , town, region, country] = best.match(/(?:^|,)("(?:[^"]|"")*"|[^,]*)/g).map((field) => field.replace(/^,/, "").replace(/^"|"$/g, ""));
  const nation = new Intl.DisplayNames(["en"], { type: "region" }).of(country);
  return `near ${[town, region, nation].filter(Boolean).filter((part, i, parts) => part !== parts[i - 1]).join(", ")}`;
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
