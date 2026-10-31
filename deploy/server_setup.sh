#!/bin/sh
# Runs on the server from ~/dronesight. Installs Docker if needed, then builds and starts the site.
set -e
if ! command -v docker >/dev/null; then
    sudo apt-get update -q
    sudo apt-get install -yq docker.io docker-compose-v2
fi
printf 'SITE_ADDRESS=%s\n' "$1" > deploy/.env
sudo docker compose -f deploy/docker-compose.yml up -d --build
sudo docker compose -f deploy/docker-compose.yml ps
