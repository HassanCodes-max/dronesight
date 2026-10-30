"""Drive a real Edge or Chrome over the DevTools protocol against a running site.

    python tests/browser_check.py http://127.0.0.1:8124 outputs/check

Not part of pytest because it needs a browser and a server with a seeded sample flight.
"""
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import websockets

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
PORT = 9333


class Page:
    def __init__(self, socket):
        self.socket = socket
        self.next_id = 0
        self.events = []

    async def call(self, method, **params):
        self.next_id += 1
        await self.socket.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            message = json.loads(await self.socket.recv())
            if message.get("id") == self.next_id:
                return message.get("result", message)
            self.events.append(message)

    async def js(self, expression):
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        return result["result"].get("value")

    async def wait(self, expression, seconds=15):
        for _ in range(seconds * 4):
            if await self.js(expression):
                return True
            await asyncio.sleep(0.25)
        return False

    async def shot(self, path):
        data = await self.call("Page.captureScreenshot", format="png")
        Path(path).write_bytes(base64.b64decode(data["data"]))


async def run(url, out):
    targets = httpx.get(f"http://127.0.0.1:{PORT}/json").json()
    target = next(t for t in targets if t["type"] == "page")
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None) as socket:
        page = Page(socket)
        for domain in ["Page", "Runtime", "Log"]:
            await page.call(f"{domain}.enable")
        await page.call("Emulation.setDeviceMetricsOverride", width=1280, height=1000,
                        deviceScaleFactor=1, mobile=False)
        await page.call("Page.navigate", url=url)
        checks = {}
        checks["flight listed"] = await page.wait("document.querySelectorAll('.video-item').length > 0")
        checks["video loaded"] = await page.wait("document.getElementById('player').readyState >= 2")
        print("player:", await page.js("""(() => { const p = document.getElementById('player');
            return {src: p.currentSrc, ready: p.readyState, duration: p.duration, width: p.videoWidth,
                    error: p.error && p.error.message}; })()"""))
        checks["placeholder hidden"] = await page.js("getComputedStyle(document.getElementById('placeholder')).display === 'none'")
        await page.js("document.querySelectorAll('.chip')[0].click()")
        checks["search answered"] = await page.wait("document.querySelectorAll('.clip').length > 0")
        print("answer:", await page.js("document.getElementById('answer').textContent"))
        await page.wait("document.getElementById('player').seeking === false")
        await asyncio.sleep(1)
        checks["jumped to moment"] = await page.js("document.getElementById('player').currentTime > 0")
        checks["boxes drawn"] = await page.js("""(() => { const c = document.getElementById('overlay');
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            for (let i = 3; i < d.length; i += 4) if (d[i]) return true; return false; })()""")
        await page.shot(Path(out) / "site_search.png")

        await page.js("document.getElementById('signin').click()")
        await page.js("document.getElementById('auth-switch').click()")
        await page.js("""document.getElementById('email').value = 'browser@example.com';
            document.getElementById('password').value = 'correct horse battery';
            document.getElementById('auth-form').requestSubmit()""")
        checks["sign up works"] = await page.wait("document.getElementById('who').textContent !== ''")
        checks["password field cleared"] = await page.js("document.getElementById('password').value === ''")
        root = (await page.call("DOM.getDocument"))["root"]["nodeId"]
        node = (await page.call("DOM.querySelector", nodeId=root, selector="#file"))["nodeId"]
        await page.call("DOM.setFileInputFiles", nodeId=node, files=[str(Path("demo/flight.mp4").resolve())])
        checks["upload appears in the list"] = await page.wait(
            "document.querySelectorAll('.video-item').length === 2", 30)
        checks["upload finishes processing"] = await page.wait(
            "[...document.querySelectorAll('.video-item small')].filter(e => e.textContent.endsWith('long')).length === 1", 120)
        await page.js("document.getElementById('question').value = 'at least 40 people'")
        await page.js("document.getElementById('ask').requestSubmit()")
        checks["search own upload"] = await page.wait("document.querySelectorAll('.clip').length > 1", 20)
        print("own upload:", await page.js("document.getElementById('answer').textContent"))

        await page.call("Emulation.setDeviceMetricsOverride", width=390, height=844,
                        deviceScaleFactor=2, mobile=True)
        await asyncio.sleep(1)
        checks["mobile has no sideways scroll"] = await page.js(
            "document.documentElement.scrollWidth <= window.innerWidth")
        await page.shot(Path(out) / "site_mobile.png")

        problems = [e["params"] for e in page.events
                    if e.get("method") == "Runtime.exceptionThrown"
                    or (e.get("method") == "Log.entryAdded" and e["params"]["entry"]["level"] == "error")]
        checks["no page errors"] = not problems
        for problem in problems[:5]:
            print("problem:", json.dumps(problem)[:300])
    for name, ok in checks.items():
        print("PASS" if ok else "FAIL", name)
    return all(checks.values())


def main(url, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    binary = next(b for b in BROWSERS if Path(b).exists())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        browser = subprocess.Popen([binary, "--headless=new", f"--remote-debugging-port={PORT}",
                                    f"--user-data-dir={profile}", "--autoplay-policy=no-user-gesture-required",
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
