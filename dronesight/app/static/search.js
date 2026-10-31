// Turns a plain-English question into a filter over a flight's detections, in the browser.
// Deterministic on purpose: no LLM key, no cost per query, and text can never become instructions.
const PERSON_WORDS = new Set(["person", "people", "human", "humans", "pedestrian", "pedestrians", "someone",
  "somebody", "anybody", "anyone", "survivor", "survivors", "victim", "victims", "group", "groups", "crowd",
  "crowds", "busiest", "busy"]);
const NUMBERS = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10 };
const GROUP_SIZE = 3;
const MOMENT_SPACING_SECONDS = 4;
const MAX_MOMENTS = 6;
const TIME = String.raw`(\d+:\d{2}|\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|m|min|mins|minute|minutes)\b)`;

const number = (word) => NUMBERS[word] ?? (/^\d+$/.test(word) ? Number(word) : null);

function seconds(value) {
  if (value.includes(":")) {
    const [minutes, rest] = value.split(":");
    return Number(minutes) * 60 + Number(rest);
  }
  return parseFloat(value) * (/m/.test(value) ? 60 : 1);
}

// Returns a query, or null when the question is outside what the index can answer.
export function parse(text, duration = 0) {
  const q = text.toLowerCase().trim();
  const tokens = q.match(/[a-z]+|\d+/g) ?? [];
  if (!tokens.some((token) => PERSON_WORDS.has(token))) return null;
  const query = { minCount: 1, start: null, end: null, busiest: /\b(most|busiest|largest|biggest)\b/.test(q) };

  let m;
  if ((m = q.match(/(at least|minimum of|minimum|more than|over)\s+(\d+|[a-z]+)/)) && number(m[2]) !== null) {
    query.minCount = number(m[2]) + (/^(more|over)/.test(m[1]) ? 1 : 0);
  } else if ((m = q.match(/\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\+?\s+(?:or more\s+)?(?:people|persons|humans)/))) {
    query.minCount = number(m[1]);
  } else if (/\b(group|groups|crowd|crowds|several|many)\b/.test(q)) {
    query.minCount = GROUP_SIZE;
  }

  if ((m = q.match(new RegExp(`(?:after|from)\\s+${TIME}`)))) query.start = seconds(m[1]);
  if ((m = q.match(new RegExp(`(?:before|until|to)\\s+${TIME}`)))) query.end = seconds(m[1]);
  if ((m = q.match(new RegExp(`first\\s+${TIME}`)))) query.end = seconds(m[1]);
  if ((m = q.match(new RegExp(`last\\s+${TIME}`)))) query.start = Math.max(0, duration - seconds(m[1]));
  return query;
}

// Picks the strongest samples, spaced apart so one crowded stretch does not fill the whole list.
export function findMoments(index, query) {
  const matching = index.samples.filter((sample) => sample.boxes.length >= query.minCount
    && (query.start === null || sample.t >= query.start)
    && (query.end === null || sample.t <= query.end));
  const strongest = [...matching].sort((a, b) => b.boxes.length - a.boxes.length || a.t - b.t);
  const limit = query.busiest ? 1 : MAX_MOMENTS;
  const moments = [];
  for (const sample of strongest) {
    if (moments.length === limit) break;
    if (moments.every((moment) => Math.abs(moment.t - sample.t) >= MOMENT_SPACING_SECONDS)) moments.push(sample);
  }
  return { matching: query.busiest ? moments : matching, moments };
}
