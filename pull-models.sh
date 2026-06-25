#!/usr/bin/env bash
# Pulls the Ollama fusion model into the Docker volume (or a local path) before
# the stack starts. Safe to re-run — Ollama skips already-present models.
#
# Usage (from stream-lens/ or sgai-demo root):
#   ./pull-models.sh
#   FUSION_MODEL=smollm2:1.7b ./pull-models.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_IMAGE="ollama/ollama:0.30.5"

# Load .env — script dir first (stream-lens standalone), then parent (sgai-demo root)
for envfile in "$SCRIPT_DIR/.env" "$SCRIPT_DIR/../.env"; do
    if [[ -f "$envfile" ]]; then
        set -a; source "$envfile"; set +a
        break
    fi
done

FUSION_MODEL="${FUSION_MODEL:-gemma4:e4b}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-}"

# Nothing to pull when fusion uses the Google API (model ID has no colon)
if [[ "$FUSION_MODEL" != *:* ]]; then
    echo "FUSION_MODEL='$FUSION_MODEL' uses the Google API — no Ollama pull needed."
    exit 0
fi

# Determine the volume/bind-mount spec
if [[ -n "$OLLAMA_MODELS_DIR" ]]; then
    VOLUME="$OLLAMA_MODELS_DIR:/root/.ollama"
    echo "Using local Ollama path: $OLLAMA_MODELS_DIR"
else
    # Derive Docker Compose project name from the parent directory
    if [[ -f "$SCRIPT_DIR/../docker-compose.yml" ]]; then
        PARENT="$(basename "$(cd "$SCRIPT_DIR/.." && pwd)")"
        VOLUME="${PARENT}_ollama_models:/root/.ollama"
    else
        VOLUME="ollama_models:/root/.ollama"
    fi
    echo "Using Docker named volume: ${VOLUME%%:*}"
fi

CONTAINER="ollama-pull-$$"
echo "Pulling '$FUSION_MODEL' via a temporary Ollama container..."

docker run -d --name "$CONTAINER" -v "$VOLUME" "$OLLAMA_IMAGE" > /dev/null
trap 'docker stop "$CONTAINER" > /dev/null 2>&1; docker rm "$CONTAINER" > /dev/null 2>&1' EXIT

echo -n "Waiting for Ollama to start"
until docker exec "$CONTAINER" ollama list > /dev/null 2>&1; do
    echo -n "."; sleep 1
done
echo " ready."

docker exec "$CONTAINER" ollama pull "$FUSION_MODEL"
echo "Done. '$FUSION_MODEL' is available in ${VOLUME%%:*}."
