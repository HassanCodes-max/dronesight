#!/bin/sh
# Copy the site to an Ubuntu server and start it:
#     sh deploy/push_server.sh azureuser@<server-ip> <site-domain>
# Only the files listed here leave this machine, so local data and secrets cannot be sent by accident.
set -e
host=$1
site=$2
key=${DRONESIGHT_SSH_KEY:-$HOME/.ssh/dronesight_oracle}
[ -n "$host" ] && [ -n "$site" ] || { sed -n 2,4p "$0"; exit 1; }
tar czf - --exclude=__pycache__ Dockerfile .dockerignore requirements-web.txt start.sh dronesight \
    models/person.onnx demo/flight.mp4 deploy/docker-compose.yml deploy/Caddyfile deploy/server_setup.sh |
    ssh -i "$key" "$host" 'mkdir -p ~/dronesight && tar xzf - -C ~/dronesight'
ssh -i "$key" "$host" "cd ~/dronesight && sh deploy/server_setup.sh '$site'"
echo "https://$site"
