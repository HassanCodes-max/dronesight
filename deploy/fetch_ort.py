"""Download the pinned ONNX Runtime Web files the upload page runs, checking each one's hash.

    python deploy/fetch_ort.py

They are served from this site, not a CDN, so no outside server can change the code visitors run.
"""
import base64
import hashlib
from pathlib import Path

import httpx

VERSION = "1.31.1"
TARGET = Path(__file__).resolve().parent.parent / "vendor/ort"
HASHES = {
    "ort.wasm.min.mjs": "wh2cMm1FOkphOYJwyATVl54RhUBZfLDAf3s9ZOYJgjQ=",
    "ort-wasm-simd-threaded.mjs": "MO0LXNEYLt6TH+pEivtWfU3HBeyVevCfEIor5fdw2sM=",
    "ort-wasm-simd-threaded.wasm": "S4hMIDsPpI0Fe+MzLGVRP/hao8W93WlGrMD6t+zaAB0=",
    "ort.webgpu.min.mjs": "MpzZVncjl6Gu3WaOkRGV05r++xgvb8FKr6TwisjY83k=",
    "ort-wasm-simd-threaded.asyncify.mjs": "Wc94FTDhoW3yopHXh5jj7BFSRn7a7EAtb5WZ2FBHwwI=",
    "ort-wasm-simd-threaded.asyncify.wasm": "yjGelTnNDx4pHPVgdHYBIerzllD12/hLblLAf0spROw=",
}


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        path = TARGET / name
        if not path.exists() or hash_of(path.read_bytes()) != expected:
            url = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist/{name}"
            data = httpx.get(url, follow_redirects=True, timeout=300).raise_for_status().content
            if hash_of(data) != expected:
                raise SystemExit(f"{name} does not match its pinned hash")
            path.write_bytes(data)
        print(name, f"{path.stat().st_size / 1e6:.1f} MB")


def hash_of(data):
    return base64.b64encode(hashlib.sha256(data).digest()).decode()


if __name__ == "__main__":
    main()
