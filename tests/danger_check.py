"""Check the danger analysis on a real page: rules, map math, flight logs, then a whole video.

    python tests/danger_check.py https://your-site video flood outputs/danger_check

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

from browser_check import BROWSERS, PORT, Page, report  # noqa: E402

LIMIT_SECONDS = 420
RULES = """(async () => {
  const { assess, rank } = await import('./risk.js');
  const person = (values) => ({ id: 1, hazard: null, depthM: 0, distanceM: null, nearestHazard: null, alone: false, still: false, ...values });
  const level = (values) => assess(person(values)).level;
  const order = (...people) => rank(people.map((values, i) => person({ id: i + 1, ...values }))).map((p) => p.id);
  const out = {};
  out['deep water is critical, the edge is high'] = level({ hazard: 'flood water', depthM: 12 }) === 'CRITICAL' && level({ hazard: 'flood water', depthM: 1 }) === 'HIGH';
  out['deep beats the edge'] = order({ hazard: 'flood water', depthM: 1 }, { hazard: 'flood water', depthM: 12 })[0] === 2;
  out['fire is critical and beats shallow water'] = level({ hazard: 'fire' }) === 'CRITICAL' && order({ hazard: 'flood water', depthM: 1 }, { hazard: 'fire' })[0] === 2;
  out['in water beats beside it'] = order({ distanceM: 4, nearestHazard: 'flood water' }, { hazard: 'flood water', depthM: 1 })[0] === 2;
  out['beside water is a watch, beside flames is high'] = level({ distanceM: 4, nearestHazard: 'flood water' }) === 'WATCH' && level({ distanceM: 4, nearestHazard: 'fire' }) === 'HIGH';
  out['smoke alone is only a watch'] = level({ hazard: 'smoke', depthM: 30 }) === 'WATCH';
  out['smoke beside flames counts both'] = assess(person({ hazard: 'smoke', distanceM: 4, nearestHazard: 'fire' })).reasons.join() === 'in smoke,about 4 m from fire';
  out['alone and still are explained'] = assess(person({ hazard: 'flood water', depthM: 1, alone: true, still: true })).reasons.join() === 'in flood water,alone,not moving';
  out['far from danger is clear even if alone'] = level({ distanceM: 400, nearestHazard: 'fire', alone: true }) === 'CLEAR';
  return JSON.stringify(out);
})()"""
MAP = """(async () => {
  const { between, exitDirection, locate, parseFlightLog, withoutPromises } = await import('./risk.js');
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
  const said = 'They are in water. The shortest way out is north. The team should be able to reach them quickly. Check the route yourself.';
  out['promises of speed or safety are removed'] = withoutPromises(said) === 'They are in water. The shortest way out is north. Check the route yourself.';
  out['safe ground can still be mentioned'] = withoutPromises('About 3 m from safe ground.') === 'About 3 m from safe ground.';
  return JSON.stringify(out);
})()"""


async def run(url, video, kind, out):
    target = next(t for t in httpx.get(f"http://127.0.0.1:{PORT}/json").json() if t["type"] == "page")
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None, ping_interval=None) as socket:
        page = Page(socket)
        for domain in ["Page", "Runtime", "Log", "DOM"]:
            await page.call(f"{domain}.enable")
        await page.call("Emulation.setDeviceMetricsOverride", width=1360, height=1200, deviceScaleFactor=1, mobile=False)
        await page.call("Page.navigate", url=f"{url}/")
        await page.wait("document.querySelectorAll('.clip').length > 1", 30)
        checks = {**json.loads(await page.js(RULES)), **json.loads(await page.js(MAP))}
        await page.js(f"document.querySelector('input[name=kind][value={kind}]').click()")
        root = (await page.call("DOM.getDocument"))["root"]["nodeId"]
        node = (await page.call("DOM.querySelector", nodeId=root, selector="#file"))["nodeId"]
        started = time.perf_counter()
        await page.call("DOM.setFileInputFiles", nodeId=node, files=[str(Path(video).resolve())])
        checks["analysis finishes"] = await page.wait("document.getElementById('upload-status').textContent.startsWith('Done.')", LIMIT_SECONDS)
        print("finished after", round(time.perf_counter() - started), "s:", await page.js("document.getElementById('risk-summary').textContent"), flush=True)
        await asyncio.sleep(1)
        checks["people in danger are listed"] = await page.wait("document.querySelectorAll('.risk').length > 0", 10)
        print("first card:", await page.js("document.querySelector('.risk')?.innerText"))
        checks["an unknown location says so"] = await page.js("document.querySelector('.risk').innerText.includes('Location unknown')")
        checks["no map link without a location"] = await page.js("!document.querySelector('.risk a')")
        await page.js("document.querySelector('.risk .actions button').click()")
        await asyncio.sleep(1.5)
        await page.shot(Path(out) / "danger_video.png")
        await page.js("""document.getElementById('where').open = true;
            document.getElementById('drone-lat').value = 'nonsense';
            document.getElementById('drone-lon').value = '73';
            document.getElementById('drone-height').value = '60';
            document.getElementById('where-form').requestSubmit()""")
        checks["bad numbers are refused"] = await page.js("document.getElementById('where-error').textContent !== ''")
        await page.js("""document.getElementById('drone-lat').value = '33.6844';
            document.getElementById('drone-lon').value = '73.0479';
            document.getElementById('drone-height').value = '60';
            document.getElementById('drone-heading').value = '45';
            document.getElementById('team-lat').value = '33.6900';
            document.getElementById('team-lon').value = '73.0500';
            document.getElementById('where-form').requestSubmit()""")
        await asyncio.sleep(0.5)
        print("with location:", await page.js("document.querySelector('.risk')?.innerText"))
        checks["coordinates appear"] = await page.js("/33\\.68\\d+, 73\\.04?\\d+/.test(document.querySelector('.risk').innerText) || /33\\.\\d+, 73\\.\\d+/.test(document.querySelector('.risk').innerText)")
        checks["the error range is shown"] = await page.js("/within about \\d+ m/.test(document.querySelector('.risk').innerText)")
        checks["directions link appears"] = await page.js("document.querySelector('.risk a')?.href.startsWith('https://www.google.com/maps/dir/')")
        if "127.0.0.1" not in url:  # the worker refuses other sites
            await page.js("[...document.querySelector('.risk .actions').querySelectorAll('button')].at(-1).click()")
            checks["the assistant explains"] = await page.wait(
                "(() => { const t = document.querySelector('.risk .explained').textContent; return t.length > 30 && !t.includes('Thinking'); })()", 40)
            print("explained:", await page.js("document.querySelector('.risk .explained').textContent"))
            checks["the explanation always gives the emergency line"] = await page.js(
                "document.querySelector('.risk .explained').textContent.includes('call your local emergency services')")
        await page.shot(Path(out) / "danger_located.png")
        return report(page, checks)


def main(url, video, kind, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    binary = next(b for b in BROWSERS if Path(b).exists())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        browser = subprocess.Popen([binary, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                                    "--no-first-run", "about:blank"])
        try:
            for _ in range(40):
                try:
                    httpx.get(f"http://127.0.0.1:{PORT}/json/version")
                    break
                except httpx.HTTPError:
                    asyncio.run(asyncio.sleep(0.25))
            ok = asyncio.run(run(url.rstrip("/"), video, kind, out))
        finally:
            browser.terminate()
            browser.wait(timeout=10)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
