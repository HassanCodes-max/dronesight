"""Check the whole page as a visitor uses it: rules, map math, flight logs, then a whole video.

    python tests/danger_check.py https://your-site video outputs/danger_check

Mistakes here would send a team to the wrong person or the wrong place, so the rules and the map math
are checked on cases worked out by hand. Not part of pytest because it needs a browser and a video.
"""
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).parent))

from browser_check import BROWSERS, FLAGS, HAS_BOXES, PORT, Page, report  # noqa: E402

LIMIT_SECONDS = 420
RULES = """(async () => {
  const { assess, rank } = await import('./risk.js');
  const { dropFalseWater, smokeShare } = await import('./hazards.js');
  const person = (values) => ({ id: 1, hazard: null, depthM: 0, distanceM: null, nearestHazard: null, alone: false, still: false, ...values });
  const critical = (values) => assess(person(values)).critical;
  const order = (...people) => rank(people.map((values, i) => person({ id: i + 1, ...values }))).map((p) => p.id);
  const out = {};
  out['anyone in water or fire is red'] = critical({ hazard: 'flood water', depthM: 1 }) && critical({ hazard: 'flood water', depthM: 12 }) && critical({ hazard: 'fire' });
  out['beside flames is red, further away is black'] = critical({ distanceM: 4, nearestHazard: 'fire' }) && !critical({ distanceM: 20, nearestHazard: 'fire' });
  out['beside water is black'] = !critical({ distanceM: 1, nearestHazard: 'flood water', alone: true, still: true });
  out['smoke alone is black'] = !critical({ hazard: 'smoke', depthM: 30 });
  out['smoke beside flames counts both'] = assess(person({ hazard: 'smoke', distanceM: 4, nearestHazard: 'fire' })).reasons.join() === 'in smoke,about 4 m from fire';
  out['lying still and alone is red'] = critical({ lying: true, still: true, alone: true });
  out['lying still in company or moving is black'] = !critical({ lying: true, still: true }) && !critical({ lying: true, alone: true });
  out['far from danger is black even if alone'] = !critical({ distanceM: 400, nearestHazard: 'fire', alone: true });
  out['deep beats the edge'] = order({ hazard: 'flood water', depthM: 1 }, { hazard: 'flood water', depthM: 12 })[0] === 2;
  out['fire beats shallow water'] = order({ hazard: 'flood water', depthM: 1 }, { hazard: 'fire' })[0] === 2;
  out['red always comes before black'] = order({ hazard: 'smoke', alone: true }, { distanceM: 4, nearestHazard: 'fire' })[0] === 2;
  const { cleanWater } = await import('./hazards.js');
  const blank = (width, height) => new Uint8Array(width * height);
  const pond = blank(100, 100);
  for (let y = 20; y < 80; y += 1) for (let x = 20; x < 80; x += 1) pond[y * 100 + x] = 1;
  const middle = 50 * 100 + 50;
  const withGap = pond.slice();
  withGap[middle] = 0;
  out['a person-sized gap in water is filled'] = cleanWater(withGap, 100, 100)[middle] === 1;
  const strip = pond.slice();
  for (let x = 0; x < 100; x += 1) strip[5 * 100 + x] = 1;
  out['a thin strip of false water goes'] = cleanWater(strip, 100, 100)[5 * 100 + 50] === 0;
  out['smoke is measured inside the frame only'] = smokeShare([[0, 0, 0.5, 1]]) === 0.5 && smokeShare([[-1, -1, 0.5, 0.5]]) === 0.25 && smokeShare([]) === 0;
  const frame = (smoke, { dark = false, flames = 0 } = {}) => ({ flood: { mask: new Uint8Array([1, 1]), dark }, fire: { smoke, mask: new Uint8Array([flames, flames]) } });
  const heavy = [[0, 0, 0.5, 1]];
  const mixed = [frame(heavy), frame([]), frame([]), frame([])];
  dropFalseWater(mixed);
  out['one smoky frame in four rules out water in the whole video'] = mixed.every((f) => f.flood.mask[0] === 0);
  const calm = [frame(heavy), frame([]), frame([]), frame([]), frame([])];
  dropFalseWater(calm);
  out['a rarer smoky frame only rules out its own water'] = calm[0].flood.mask[0] === 0 && calm[1].flood.mask[0] === 1;
  const night = [frame([], { dark: true }), frame([]), frame([]), frame([])];
  dropFalseWater(night);
  out['dark frames rule out water like smoke does'] = night.every((f) => f.flood.mask[0] === 0);
  const flames = [frame([], { flames: 1 }), frame([]), frame([]), frame([]), frame([])];
  dropFalseWater(flames);
  out['a frame with flames does not trust its water'] = flames[0].flood.mask[0] === 0 && flames[1].flood.mask[0] === 1;
  const { steadyFlames } = await import('./hazards.js');
  const flameAt = (cell) => {
    const mask = new Uint8Array(100 * 100);
    for (let row = 0; row < 10; row += 1) mask.fill(1, cell + row * 100, cell + row * 100 + 10);
    return { flood: {}, fire: { width: 100, height: 100, mask } };
  };
  const lasting = [flameAt(4040), flameAt(4040), flameAt(0)];
  steadyFlames(lasting);
  out['a lasting fire stays and a passing flame goes'] = lasting[0].fire.mask.some(Boolean) && lasting[1].fire.mask.some(Boolean) && !lasting[2].fire.mask.some(Boolean);
  out['reasons are explained'] = assess(person({ hazard: 'flood water', depthM: 1, alone: true, still: true })).reasons.join() === 'in flood water,alone,not moving';
  return JSON.stringify(out);
})()"""
MAP = """(async () => {
  const { between, exitDirection, locate, parseFlightLog, readVideoLocation } = await import('./risk.js');
  const drone = { latitude: 33.7, longitude: 73, height: 100, heading: 0, tilt: 0, fov: 90 };
  const north = (place) => (place.latitude - 33.7) * 111320;
  const east = (place) => (place.longitude - 73) * 111320 * Math.cos(33.7 * Math.PI / 180);
  const near = (a, b, tolerance = 0.5) => Math.abs(a - b) < tolerance;
  const out = {};
  const middle = locate(0.5, 0.5, drone, 1);
  out['the picture centre is below the drone'] = near(north(middle), 0) && near(east(middle), 0);
  out['the top of the picture is ahead'] = near(north(locate(0.5, 0, drone, 1)), 100);
  out['the right of the picture is east'] = near(east(locate(1, 0.5, drone, 1)), 100);
  const facingEast = locate(0.5, 0, { ...drone, heading: 90 }, 1);
  out['heading turns the ground'] = near(east(facingEast), 100) && near(north(facingEast), 0);
  const tilted = locate(0.5, 0.5, { ...drone, tilt: 45 }, 1);
  out['a tilted camera looks ahead'] = near(north(tilted), 100);
  out['above the horizon has no location'] = locate(0.5, 0, { ...drone, tilt: 80 }, 1) === null;
  out['error grows with tilt'] = locate(0.5, 0.5, { ...drone, tilt: 60 }, 1).errorM > locate(0.5, 0.5, drone, 1).errorM;
  const trip = between({ latitude: 33.7, longitude: 73 }, { latitude: 33.7009, longitude: 73 });
  out['distance and direction'] = near(trip.metres, 100, 1) && trip.direction === 'north';
  out['the way out uses the heading'] = exitDirection({ dx: 0, dy: -5 }, 90) === 'east' && exitDirection({ dx: 0, dy: -5 }, null) === 'towards the top of the picture';
  const log = parseFlightLog('1\\n00:00:01,000 --> 00:00:02,000\\n[latitude: 33.123456] [longitude: 73.654321] [rel_alt: 42.5 abs_alt: 600.1] [gb_yaw: 12.5 gb_pitch: -60]\\n\\n2\\n00:00:02,000 --> 00:00:03,000\\nGPS (33.5, 73.5, 18) H 30.0M\\n');
  out['newer flight logs are read'] = log[0].latitude === 33.123456 && log[0].height === 42.5 && log[0].heading === 12.5 && log[0].tilt === 30 && log[0].t === 1;
  out['older flight logs are read'] = log[1].latitude === 33.5 && log[1].longitude === 73.5 && log[1].height === 30;
  const tag = new TextEncoder().encode('©xyz');
  const latin = (text) => Uint8Array.from([...text].map((c) => c.charCodeAt(0)));
  const withTag = new File([new Uint8Array(500), latin('©xyz\u0000\u0012\u0015Ç+33.6844+073.0479/'), new Uint8Array(500)], 'a.mp4');
  const apple = new File([new Uint8Array(100), latin('com.apple.quicktime.location.ISO6709  data  -33.8688+151.2093+058.000/')], 'b.mov');
  const plain = new File([new Uint8Array(2000)], 'c.mp4');
  const read = await readVideoLocation(withTag);
  out['a location in the file is read'] = read?.latitude === 33.6844 && read?.longitude === 73.0479;
  out['a phone video location is read'] = (await readVideoLocation(apple))?.longitude === 151.2093;
  out['no location in the file gives none'] = (await readVideoLocation(plain)) === null;
  return JSON.stringify(out);
})()"""


async def run(url, video, out):
    target = next(t for t in httpx.get(f"http://127.0.0.1:{PORT}/json").json() if t["type"] == "page" and t["url"] == "about:blank")
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None, ping_interval=None) as socket:
        page = Page(socket)
        for domain in ["Page", "Runtime", "Log", "DOM", "Network"]:
            await page.call(f"{domain}.enable")
        await page.call("Emulation.setDeviceMetricsOverride", width=1360, height=1000, deviceScaleFactor=1, mobile=False)
        await page.call("Page.navigate", url=f"{url}/")
        await page.wait("document.getElementById('start').offsetHeight > 0", 30)
        checks = {"there is nothing to choose first": await page.js("document.querySelectorAll('#start button, #start input[type=radio]').length === 0")}
        await page.shot(Path(out) / "start.png")
        checks.update(json.loads(await page.js(RULES)))
        checks.update(json.loads(await page.js(MAP)))

        async def add_video():
            root = (await page.call("DOM.getDocument"))["root"]["nodeId"]
            node = (await page.call("DOM.querySelector", nodeId=root, selector="#file"))["nodeId"]
            await page.call("DOM.setFileInputFiles", nodeId=node, files=[str(Path(video).resolve())])

        started = time.perf_counter()
        await add_video()
        checks["progress shows"] = await page.wait("document.getElementById('working').offsetHeight > 0", 10)
        checks["the video shows within 5 seconds"] = await page.wait("document.getElementById('player').readyState >= 2", 5)
        print("video showing after", round(time.perf_counter() - started, 1), "s", flush=True)
        await asyncio.sleep(1)
        await page.shot(Path(out) / "working.png")
        checks["analysis finishes"] = await page.wait("document.getElementById('working').hidden", LIMIT_SECONDS)
        print("finished after", round(time.perf_counter() - started), "s:", await page.js("document.getElementById('headline').textContent"), "|",
              await page.js("document.getElementById('subhead').textContent"), "| error:",
              await page.js("document.getElementById('start-error').textContent"), flush=True)
        checks["the answer comes first"] = await page.js("document.getElementById('headline').textContent.length > 5")
        await asyncio.sleep(1.5)
        found = await page.js("document.querySelectorAll('.person').length")
        if not found:  # a calm video should say so, not list anyone
            checks["a calm video says nobody is in danger"] = await page.js("document.getElementById('headline').textContent === 'Nobody in danger was found'")
            checks["everyone else needs nothing"] = await page.js("document.getElementById('panel').hidden")
            checks["no report is offered"] = await page.js("document.getElementById('report').hidden")
            await page.js("document.getElementById('player').currentTime = 1")
            await asyncio.sleep(1.5)
            checks["boxes are drawn on the video"] = await page.js(HAS_BOXES)
            await page.shot(Path(out) / "results.png")
        else:
            print("reasons:", await page.js("[...document.querySelectorAll('.person-pick strong')].map((e) => e.textContent).join(' | ')"))
            checks["every number is explained"] = await page.js("[...document.querySelectorAll('.person')].every((e, i) => e.querySelector('.place').textContent === String(i + 1) && e.querySelector('strong').textContent.length > 3)")
            checks["there is no list of normal people"] = await page.js("document.getElementById('others') === null")
            checks["the worst person is shown by itself"] = await page.wait("document.querySelector('.person[aria-current=true]') !== null", 15)
            await asyncio.sleep(1.5)
            checks["boxes are drawn on the video"] = await page.js(HAS_BOXES)
            print("first person:", await page.js("document.querySelector('.person')?.innerText"))
            await page.shot(Path(out) / "results.png")
            wording = "(document.getElementById('report').innerText)"
            checks["an alert shows without a click"] = await page.js("!document.getElementById('report').hidden")
            checks["it never claims officials were told"] = await page.js(f"!/has been sent to|officials have|help is coming|help is on its way/i.test({wording})")
            checks["it says nobody has been told"] = await page.js(f"/nobody has been told/.test({wording})")
            checks["the WhatsApp link holds the report"] = await page.js("document.getElementById('report-send').href.startsWith('https://wa.me/?text=DroneSight')")
            checks["a missing location says so once"] = await page.js("/no location in it/.test(document.getElementById('location-text').textContent) && !document.getElementById('people').innerText.includes('Coordinates')")
            checks["the page never asks where it was filmed"] = await page.js("!document.querySelector('#results input:not(#seek), #results form')")
            await page.call("Browser.grantPermissions", permissions=["geolocation"], origin=url)
            await page.call("Emulation.setGeolocationOverride", latitude=33.6844, longitude=73.0479, accuracy=20)
            await page.js("document.getElementById('report-send').click()")
            checks["sharing adds the device's location by itself"] = await page.wait("document.getElementById('report-send').href.includes(encodeURIComponent('33.684400, 73.047900'))", 20)
            checks["every red person gets coordinates"] = await page.js("[...document.querySelectorAll('.person')].every((e) => e.innerText.includes('33.684400, 73.047900'))")
            checks["the report says where the coordinates came from"] = await page.js("decodeURIComponent(document.getElementById('report-send').href).includes('the device that sent this report')")
            checks["each person links to the map"] = await page.js("document.querySelector('.person a')?.href.startsWith('https://www.google.com/maps?q=')")
            await page.shot(Path(out) / "located.png")

        sent = [e["params"]["request"] for e in page.events if e.get("method") == "Network.requestWillBeSent"]
        posts = [r for r in sent if r["method"] not in ("GET", "HEAD", "OPTIONS")]
        print("sent from the device:", [(r["method"], r["url"][:70]) for r in posts])
        checks["nothing was sent from the device"] = not posts

        await page.js("document.getElementById('again').click()")
        checks["another video can be checked"] = await page.wait("document.getElementById('start').offsetHeight > 0", 5)
        before = len(page.events)
        await add_video()
        await page.wait("/Checking on your/.test(document.getElementById('working-note').textContent)", 60)
        again = [e["params"]["request"]["url"] for e in page.events[before:] if e.get("method") == "Network.requestWillBeSent"]
        large = [u for u in again if u.endswith((".onnx", ".wasm")) or "xet" in u]
        checks["a second video downloads nothing big"] = not large
        await page.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True)
        await asyncio.sleep(1)
        checks["mobile has no sideways scroll"] = await page.js("document.documentElement.scrollWidth <= window.innerWidth")
        return report(page, checks)


def main(url, video, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    binary = next(b for b in BROWSERS if Path(b).exists())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        browser = subprocess.Popen([binary, *FLAGS, f"--user-data-dir={profile}", "about:blank"])
        try:
            for _ in range(40):
                try:
                    httpx.get(f"http://127.0.0.1:{PORT}/json/version")
                    break
                except httpx.HTTPError:
                    asyncio.run(asyncio.sleep(0.25))
            ok = asyncio.run(run(url.rstrip("/"), video, out))
        finally:
            browser.terminate()
            browser.wait(timeout=10)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
