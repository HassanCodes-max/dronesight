"""Analyse a video in a real browser and prove it never left the device.

    python tests/upload_check.py https://your-site demo/flight.mp4 outputs/upload_check [no-graphics]

no-graphics hides the graphics card from the page, as on many phones, so the slower path runs.
Not part of pytest because it needs a browser, a published model and several minutes.
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

from browser_check import BROWSERS, HAS_BOXES, HAS_TIMELINE, PORT, Page, report  # noqa: E402

LIMIT_SECONDS = 540


async def run(url, video, out, graphics):
    target = next(t for t in httpx.get(f"http://127.0.0.1:{PORT}/json").json() if t["type"] == "page")
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None) as socket:
        page = Page(socket)
        for domain in ["Page", "Runtime", "Log", "DOM", "Network"]:
            await page.call(f"{domain}.enable")
        await page.call("Emulation.setDeviceMetricsOverride", width=1280, height=1000, deviceScaleFactor=1, mobile=False)
        if not graphics:
            await page.call("Page.addScriptToEvaluateOnNewDocument", source="delete Navigator.prototype.gpu")
        await page.call("Page.navigate", url=f"{url}/footage.html")
        checks = {"sample flight loads": await page.wait("document.querySelectorAll('.clip').length > 1", 30)}
        checks["upload box shows"] = await page.js("document.getElementById('drop').offsetHeight > 0")
        root = (await page.call("DOM.getDocument"))["root"]["nodeId"]
        node = (await page.call("DOM.querySelector", nodeId=root, selector="#file"))["nodeId"]
        started = time.perf_counter()
        await page.call("DOM.setFileInputFiles", nodeId=node, files=[str(Path(video).resolve())])
        checks["progress shows"] = await page.wait("/Getting the model|Finding people/.test(document.getElementById('upload-status').textContent)", 20)
        ready = await page.wait("/Done\\.|Finding people/.test(document.getElementById('upload-status').textContent)", 180)
        print("model ready after", round(time.perf_counter() - started), "s:", await page.js("document.getElementById('upload-status').textContent"), flush=True)
        checks["analysis finishes"] = ready and await page.wait("document.getElementById('upload-status').textContent.startsWith('Done.')", LIMIT_SECONDS)
        print("finished after", round(time.perf_counter() - started), "s:", await page.js("document.getElementById('upload-status').textContent"), flush=True)
        await asyncio.sleep(2)
        print("moments:", await page.js("document.querySelectorAll('.clip').length"), await page.js("document.getElementById('answer').textContent"))
        checks["moments found"] = await page.wait("document.querySelectorAll('.clip').length > 0", 10)
        await page.js("document.querySelector('.clip').click()")
        await asyncio.sleep(1.5)
        checks["jumped to moment"] = await page.js("document.getElementById('player').currentTime > 0 || document.getElementById('live').textContent !== ''")
        checks["boxes drawn"] = await page.js(HAS_BOXES)
        checks["timeline drawn"] = await page.js(HAS_TIMELINE)
        print("in view:", await page.js("document.getElementById('live').textContent"))
        await page.shot(Path(out) / "upload_done.png")
        await page.js("document.getElementById('question').value = 'at least 2 people'; document.getElementById('ask').requestSubmit()")
        await asyncio.sleep(1)
        print("search:", await page.js("document.getElementById('answer').textContent"))
        checks["search works on the video"] = await page.js("/moment|No matching/.test(document.getElementById('answer').textContent)")
        sent = [e["params"]["request"] for e in page.events if e.get("method") == "Network.requestWillBeSent"]
        posts = [(r["method"], r["url"][:80]) for r in sent if r["method"] not in ("GET", "HEAD", "OPTIONS")]
        print("non-GET requests:", json.dumps(posts))
        checks["nothing was sent from the device"] = not posts
        before = len(page.events)
        await page.call("Page.navigate", url=f"{url}/footage.html")
        await page.wait("document.querySelectorAll('.clip').length > 1", 30)
        root = (await page.call("DOM.getDocument"))["root"]["nodeId"]
        node = (await page.call("DOM.querySelector", nodeId=root, selector="#file"))["nodeId"]
        started = time.perf_counter()
        await page.call("DOM.setFileInputFiles", nodeId=node, files=[str(Path(video).resolve())])
        await page.wait("/Finding people/.test(document.getElementById('upload-status').textContent)", 60)
        print("second video ready after", round(time.perf_counter() - started), "s", flush=True)
        again = [e["params"]["request"]["url"] for e in page.events[before:] if e.get("method") == "Network.requestWillBeSent"]
        big = [u for u in again if ".onnx" in u or ".wasm" in u or "xet" in u]
        print("large requests:", [u[:90] for u in big])
        checks["a second video downloads nothing big"] = not [u for u in big if "flight.mp4" not in u and "xet-bridge" not in u or u.endswith((".onnx", ".wasm"))]
        return report(page, checks)


def main(url, video, out, mode=""):
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
            ok = asyncio.run(run(url.rstrip("/"), video, out, mode != "no-graphics"))
        finally:
            browser.terminate()
            browser.wait(timeout=10)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
