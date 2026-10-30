#!/bin/sh
# Index the sample flight, then run the worker beside the web server.
set -e
for video in demo/*.mp4; do
    [ -e "$video" ] && python -m dronesight.app.seed "$video" "Sample flight"
done
(while true; do python -m dronesight.app.worker; sleep 5; done) &
exec uvicorn dronesight.app.main:app --host 0.0.0.0 --port "${PORT:-7860}" \
    --no-server-header --limit-concurrency 64 --timeout-keep-alive 5
