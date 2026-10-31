"""Watch the mission simulator run in a real Edge or Chrome and check that it behaves.

    python tests/mission_check.py http://127.0.0.1:8130 outputs/mission_check
"""
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).parent))

from browser_check import BROWSERS, PORT, Page, report  # noqa: E402

READ_ALERTS = """[...document.querySelectorAll('.alert')].reverse().map(a => ({
    level: a.querySelector('strong').textContent.split(':')[0],
    why: a.querySelector('p').textContent, coords: a.querySelector('.coords').textContent}))"""
TALLY = """Object.fromEntries(['found', 'critical', 'reported', 'waiting'].map(
    id => [id, Number(document.getElementById(id).textContent)]))"""
HAS_PIXELS = """(() => { const c = document.getElementById('CANVAS');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    for (let i = 0; i < d.length; i += 4) if (d[i] > 40) return true; return false; })()"""


async def run(url, out):
    targets = httpx.get(f"http://127.0.0.1:{PORT}/json").json()
    target = next(t for t in targets if t["type"] == "page")
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None) as socket:
        page = Page(socket)
        for domain in ["Page", "Runtime", "Log"]:
            await page.call(f"{domain}.enable")
        await page.call("Emulation.setDeviceMetricsOverride", width=1360, height=1000, deviceScaleFactor=1, mobile=False)
        await page.call("Page.navigate", url=url)
        checks = {}
        checks["mission starts by itself"] = await page.wait("/Flying|Searching/.test(document.getElementById('status').textContent)", 20)
        await asyncio.sleep(3)
        checks["camera shows the scene"] = await page.js(HAS_PIXELS.replace("CANVAS", "camera"))
        checks["map shows the area"] = await page.js(HAS_PIXELS.replace("CANVAS", "map"))
        await page.js("document.getElementById('speed').click(); document.getElementById('speed').click()")
        checks["first alert arrives"] = await page.wait("document.querySelectorAll('.alert').length > 0", 40)
        await page.shot(Path(out) / "mission_running.png")
        checks["mission finishes"] = await page.wait("/Mission complete/.test(document.getElementById('status').textContent)", 150)
        await asyncio.sleep(0.5)
        print("status:", await page.js("document.getElementById('status').textContent"))
        alerts = await page.js(READ_ALERTS)
        tally = await page.js(TALLY)
        print("tally:", tally)
        print("first alerts:", json.dumps(alerts[:4]))
        levels = [a["level"] for a in alerts]
        print("order:", levels)
        checks["every alert has coordinates"] = bool(alerts) and all("," in a["coords"] for a in alerts)
        checks["every alert has a reason"] = bool(alerts) and all(a["why"] for a in alerts)
        checks["critical people reported"] = tally["critical"] == 0 or levels.count("CRITICAL") == tally["critical"]
        checks["nobody left waiting"] = tally["waiting"] == 0
        checks["everyone found"] = tally["found"] == await page.js(
            "fetch('mission/mission.json').then(r => r.json()).then(m => m.flood.sectors.reduce((n, s) => n + s.people.length, 0))")
        await page.shot(Path(out) / "mission_done.png")
        await page.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True)
        await asyncio.sleep(1)
        checks["mobile has no sideways scroll"] = await page.js("document.documentElement.scrollWidth <= window.innerWidth")
        await page.shot(Path(out) / "mission_mobile.png")
        return report(page, checks)


def main(url, out):
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
            ok = asyncio.run(run(url, out))
        finally:
            browser.terminate()
            browser.wait(timeout=10)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
