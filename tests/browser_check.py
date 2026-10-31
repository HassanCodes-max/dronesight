"""Drive a real Edge or Chrome over the DevTools protocol."""
import asyncio
import base64
import json
from pathlib import Path

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
PORT = 9333
HAS_BOXES = """(() => { const c = document.getElementById('overlay');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    for (let i = 3; i < d.length; i += 4) if (d[i]) return true; return false; })()"""
# Edge's sync prompt freezes the page under test, so sync stays off.
FLAGS = ["--headless=new", f"--remote-debugging-port={PORT}", "--no-first-run", "--disable-sync", "--disable-features=msEdgeSync,EdgeSignin"]


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


def report(page, checks):
    problems = [e["params"] for e in page.events
                if e.get("method") == "Runtime.exceptionThrown"
                or (e.get("method") == "Log.entryAdded" and e["params"]["entry"]["level"] == "error")]
    checks["no page errors"] = not problems
    for problem in problems[:5]:
        print("problem:", json.dumps(problem)[:300])
    for name, ok in checks.items():
        print("PASS" if ok else "FAIL", name)
    return all(checks.values())
