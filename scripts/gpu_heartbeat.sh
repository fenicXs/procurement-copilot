#!/bin/bash
# Keeps the Gemma model warm on the GPU during idle gaps between real work,
# per advisor guidance on SOL fairshare accounting. Light-touch: one short
# generation every 15 minutes with a long keep_alive, not continuous load.
set -euo pipefail

MODEL="gemma4:31b-it-bf16"
INTERVAL_SECONDS=900

while true; do
    curl -s http://localhost:11434/api/generate -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"ok\",
        \"stream\": false,
        \"keep_alive\": \"20m\",
        \"options\": {\"num_predict\": 1}
    }" > /dev/null
    echo "$(date -Iseconds) heartbeat sent"
    sleep "${INTERVAL_SECONDS}"
done
