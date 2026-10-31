"""Publish the static edition to a free Hugging Face static Space.

    hf auth login
    python deploy/build_static.py dist
    python deploy/push_static.py <your-hf-username>/dronesight dist

Only the build folder is uploaded, so local data and secrets cannot be published by accident.
"""
import sys

from huggingface_hub import HfApi


def main(repo_id, folder):
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=folder, commit_message="Publish DroneSight")
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
