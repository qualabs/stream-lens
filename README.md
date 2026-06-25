# stream-lens

AI context-analysis module that watches a DASH live stream to extract scene context. Used as part of the [sgai-demo](https://github.com/qualabs/sgai-demo) project.

Sits between live-sim and morpheus in the streaming pipeline. Receives DASH segments from all renditions, buffers the selected rendition for analysis, runs parallel video+audio analysis tracks, fuses the results into a key-value context string, and forwards segments + MPD to morpheus.

## Architecture

```
live-sim ──(MPD + segments, all renditions)──▶ stream-lens ──(MPD + segments)──▶ morpheus ──▶ player
```

On each analysis trigger:

- **Video track**: ffmpeg extracts frames from the selected rendition → sent to Gemma 4 26B via Google AI API
- **Audio track**: ffmpeg extracts WAV → faster-whisper (transcript) + librosa (tempo, energy, tone)
- **Fusion**: synthesizes both tracks into a KV context string (e.g. `ctx_activity=surfing&ctx_mood=energetic`)

Tracks run in parallel. The fusion step waits for both.

## Rendition selection

live-sim pushes segments from **all** renditions. stream-lens only buffers the rendition matching `ANALYSIS_VIDEO_RENDITION` (a stream index). All other renditions are forwarded to morpheus without buffering.

Stream indices are derived from the filename (`chunk-stream{N}-XXXXX.m4s`) — no extra headers needed.

```
LIVE_SIM_RENDITIONS=1920x1080:4000k,1280x720:2000k,854x480:1000k
                    → stream0         → stream1       → stream2
ANALYSIS_VIDEO_RENDITION=2  →  analyse 854×480 (lowest res = fastest extraction)
```

## Output format

The fusion model outputs a URL-safe key=value string:

```
ctx_activity=surfing&ctx_mood=energetic&ctx_setting=ocean&ctx_sport=surfing&ctx_audience=sports+fans
```

All keys are prefixed with `FUSION_CONTEXT_PREFIX` (default `ctx_`). Values are URL-encoded. Morpheus appends this string directly to ad request URLs as query params.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `PUT` | `/segment` | Ingest a raw DASH segment (init or media). Headers: `X-Segment-Type: init\|media`, `X-Stream-Type: video\|audio`, `X-Segment-Number: N`, `X-Segment-Name: <filename>` |
| `PUT` | `/live.mpd` | Receive MPD from live-sim; forward to morpheus (fire-and-forget) |
| `POST` | `/processing` | Toggle analysis: `{"enabled": true\|false}`. Segment forwarding is unaffected. |
| `GET` | `/context` | Latest context result |
| `POST` | `/config` | Update config at runtime and reset buffers |
| `GET` | `/health` | Liveness check |

### `GET /context` response

```json
{
  "status": "waiting | processing | ready | partial | error",
  "context": "ctx_activity=surfing&ctx_mood=energetic&ctx_sport=surfing",
  "clip_start": "2026-06-01T14:32:10.123Z",
  "clip_end":   "2026-06-01T14:32:20.456Z",
  "processed_at": "2026-06-01T14:32:23.789Z",
  "timings": { "total_ms": 1240, "video_ms": 980, "audio_ms": 420, "fusion_ms": 310 }
}
```

Returns `503` if `ANALYSIS_VIDEO_RENDITION` does not match any received video stream index, with an error body listing the stream IDs that were seen.

Returns `409` on `PUT /segment` if a media segment arrives before the corresponding init segment.

### `POST /config`

Accepts any subset of the configurable variables at runtime. Resets buffers and counters on change.

```json
{ "BUFFER_SIZE": 5, "ANALYSIS_TRIGGER_SEGMENTS": 10, "ANALYSIS_VIDEO_RENDITION": 1 }
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Google AI API key — video analysis disabled if not set |
| `VIDEO_MODEL` | `gemma-4-26b-a4b-it` | Gemma model for video frame analysis (Google API) |
| `VIDEO_ANALYSIS_INSTRUCTIONS` | — | Override video model output instructions (uses built-in JSON schema if unset) |
| `OLLAMA_URL` | `http://ollama:11434/api/generate` | Ollama endpoint for fusion model |
| `FUSION_MODEL` | `gemma4:e4b` | Fusion model ID. Contains `:` → Ollama; no `:` → Google API |
| `FUSION_MODEL_TIMEOUT` | `300` | Fusion request timeout in seconds (Ollama path only) |
| `FUSION_CONTEXT_PREFIX` | `ctx_` | Prefix for all KV tags in the fusion output |
| `FUSION_INSTRUCTIONS` | — | Override fusion model instructions (uses built-in KV format if unset) |
| `BUFFER_SIZE` | `7` | Number of segments to feed into each analysis run |
| `ANALYSIS_TRIGGER_SEGMENTS` | `BUFFER_SIZE` | Segments from the selected rendition to receive before triggering |
| `SEG_DURATION_S` | `2` | Expected segment duration (must match live-sim `seg_duration`) |
| `ANALYSIS_VIDEO_RENDITION` | `2` | Stream index of the rendition to buffer (matches `LIVE_SIM_RENDITIONS` order) |
| `FRAME_SAMPLE_MODE` | `iframes` | `fps` = fixed rate; `iframes` = keyframes only |
| `FRAME_SAMPLE_FPS` | `1.0` | Frames per second (fps mode) |
| `MAX_FRAMES` | `15` | Maximum frames sent to the video model per analysis |
| `FRAME_MAX_WIDTH` | `640` | Max frame width in pixels (downscaled before sending) |
| `WHISPER_MODEL` | `medium` | faster-whisper model name |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MORPHEUS_BASE_URL` | `http://morpheus` | Base URL for morpheus segment + MPD forwarding |
| `SERVER_PORT` | `8001` | Port the server listens on |

## Model setup

Two models are required:

**Whisper** — pre-downloaded at image build time. No manual step needed; `docker build` (or `docker compose up --build`) handles it automatically.

**Ollama fusion model** — must be pulled before (or after) the stack starts. Two options:

- **Script** (recommended): run `./pull-models.sh` from within `stream-lens/` or from the sgai-demo repo root. It reads `FUSION_MODEL` and `OLLAMA_MODELS_DIR` from your `.env` automatically, then starts a temporary Ollama container to pull the model:

  ```bash
  ./stream-lens/pull-models.sh
  ```

- **Manual** (after stack is running): `docker compose exec ollama ollama pull <model>`

If `FUSION_MODEL` contains no `:` (e.g. `gemma-4-26b-a4b-it`), the Google API path is used and no Ollama pull is needed.

## Running

```bash
# From sgai-demo root via Docker Compose
docker compose up stream-lens

# Standalone
docker run -p 8001:8001 \
  -e GOOGLE_API_KEY=... \
  -e LIVE_SIM_RENDITIONS="1920x1080:4000k,1280x720:2000k,854x480:1000k" \
  -e ANALYSIS_VIDEO_RENDITION=2 \
  -v "$HOME/.ollama:/root/.ollama" \
  stream-lens
```

## Docker notes

- The Whisper `medium` model (~1.5 GB) is pre-downloaded at image build time
- Ollama runs as a separate service (`ollama` container) in the sgai-demo stack
- Mount a local `~/.ollama` path to reuse host-downloaded Ollama models and avoid re-pulling (~9.6 GB for E4B)
