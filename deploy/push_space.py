"""Publish the site to a Hugging Face Docker Space.

    hf auth login
    python deploy/push_space.py <your-hf-username>/dronesight

Only an explicit allowlist is uploaded, so local data and secrets cannot be published by accident.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
ALLOW = ["Dockerfile", ".dockerignore", "requirements-web.txt", "start.sh",
         "dronesight/**/*.py", "dronesight/app/static/*", "models/person.onnx", "demo/flight.mp4"]
SPACE_CARD = """---
title: DroneSight
emoji: 🛰️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Search drone footage in plain English
---

Ask a question about drone footage and jump to the moment that matters.
"""


def main(repo_id):
    for required in ["models/person.onnx", "demo/flight.mp4"]:
        if not (ROOT / required).exists():
            sys.exit(f"Missing {required}")
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=ROOT, allow_patterns=ALLOW,
                      commit_message="Deploy DroneSight")
    api.upload_file(repo_id=repo_id, repo_type="space", path_or_fileobj=SPACE_CARD.encode(),
                    path_in_repo="README.md", commit_message="Add Space card")
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
