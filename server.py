#!/usr/bin/env python3
"""
Context Analyzer: receives DASH segments from live-sim, runs parallel video+audio
analysis tracks, fuses the results via Gemma 4 E4B, and produces a structured
context object for ad personalization.

Pipeline:
  live-sim ──PUT /segment──▶ forward to Morpheus (fire-and-forget, always)
                          └─▶ buffer for analysis (when processing enabled)

  video segs ──▶ frame sampler ──▶ Gemma 4 26B (Google API) ──▶ video context ──┐
                                                                                   ├──▶ Gemma 4 E4B (Ollama) ──▶ JSON
  audio segs ──▶ WAV extract ──▶ faster-whisper + librosa ──▶ audio context ────┘

  live-sim ──PUT /live.mpd──▶ forward to Morpheus (fire-and-forget, always)

Endpoints:
  PUT /segment     — ingest a raw init or media segment; forwarded to Morpheus
  PUT /live.mpd    — receive MPD, forward to Morpheus (fire-and-forget)
  POST /processing — toggle {"enabled": true|false} for video analysis
  GET /context     — retrieve the latest analysis result
  GET /health      — liveness check
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import requests as _requests
import soundfile  # noqa: F401 — ensure libsndfile is importable at startup
from faster_whisper import WhisperModel
from google import genai
from google.genai import types as genai_types
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

SEG_DURATION_S: int = int(os.environ.get("SEG_DURATION_S", "2"))
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
VIDEO_MODEL: str = os.environ.get("VIDEO_MODEL", "gemma-4-26b-a4b-it")
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://ollama:11434/api/generate")
FUSION_MODEL_TIMEOUT: int = int(os.environ.get("FUSION_MODEL_TIMEOUT", "300"))
FUSION_MODEL: str = os.environ.get("FUSION_MODEL", "gemma4:e4b")
FRAME_SAMPLE_MODE: str = os.environ.get("FRAME_SAMPLE_MODE", "fps")  # "fps" | "iframes"
FRAME_SAMPLE_FPS: float = float(os.environ.get("FRAME_SAMPLE_FPS", "1.0"))
MAX_FRAMES: int = int(os.environ.get("MAX_FRAMES", "10"))
FRAME_MAX_WIDTH: int = int(os.environ.get("FRAME_MAX_WIDTH", "640"))
WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "medium")
WHISPER_DEVICE: str = os.environ.get("WHISPER_DEVICE", "cpu")
AUDIO_SR: int = 16000
SERVER_PORT: int = int(os.environ.get("SERVER_PORT", "8001"))
MORPHEUS_BASE_URL: str = os.environ.get("MORPHEUS_BASE_URL", "http://morpheus")
MORPHEUS_MPD_URL: str = f"{MORPHEUS_BASE_URL}/live.mpd"

_DASH_NS        = "urn:mpeg:dash:schema:mpd:2011"
_UP_NS          = "urn:mpeg:dash:schema:urlparam:2016"
_UP_SCHEME      = "urn:mpeg:dash:urlparam:2016"
_OVERLAY_SCHEME = "urn:scte:dash:scte214-events"

ET.register_namespace("",       _DASH_NS)
ET.register_namespace("up",     _UP_NS)
ET.register_namespace("scte35", "http://www.scte.org/schemas/35/2016")
ET.register_namespace("xsi",    "http://www.w3.org/2001/XMLSchema-instance")

BUFFER_SIZE: int = int(os.environ.get("BUFFER_SIZE", "7"))
ANALYSIS_TRIGGER_SEGMENTS: int = int(
    os.environ.get("ANALYSIS_TRIGGER_SEGMENTS", str(BUFFER_SIZE))
)
MAX_BUFFER: int = max(BUFFER_SIZE, ANALYSIS_TRIGGER_SEGMENTS) + 2
ANALYSIS_VIDEO_RENDITION: int = int(os.environ.get("ANALYSIS_VIDEO_RENDITION", "0"))
FUSION_CONTEXT_PREFIX: str = os.environ.get("FUSION_CONTEXT_PREFIX", "ctx_")
FUSION_INSTRUCTIONS: str = os.environ.get("FUSION_INSTRUCTIONS", "")
VIDEO_ANALYSIS_INSTRUCTIONS: str = os.environ.get("VIDEO_ANALYSIS_INSTRUCTIONS", "")


def _fusion_uses_google(model: str) -> bool:
    return ":" not in model


def _ms(start: float) -> int:
    """Elapsed milliseconds since a time.perf_counter() start."""
    return int((time.perf_counter() - start) * 1000)

# ── State ─────────────────────────────────────────────────────────────────────

video_init: Optional[bytes] = None
audio_init: Optional[bytes] = None
# Each entry: (seg_num, raw_bytes, received_at)
video_buffer: list[tuple[int, bytes, datetime]] = []
audio_buffer: list[tuple[int, bytes, datetime]] = []
analysis_in_progress: bool = False
processing_enabled: bool = True
latest_context: dict = {
    "status": "waiting",
    "context": None,
    "clip_start": None,
    "clip_end": None,
    "processed_at": None,
    "timings": None,
}
_lock: asyncio.Lock          # initialised in lifespan
_gemini_client: genai.Client | None = None
_whisper_model: Optional[WhisperModel] = None
_trigger_counter: int = 0
_total_video_segs_received: int = 0
_seen_video_stream_ids: set[int] = set()
_invalid_rendition: bool = False
_last_injected_mpd: Optional[bytes] = None
_last_ready_context: Optional[str] = None


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _lock, _gemini_client, processing_enabled
    _lock = asyncio.Lock()

    if GOOGLE_API_KEY:
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("[stream-lens] Gemini client configured (video_model=%s)", VIDEO_MODEL)
    else:
        logger.warning("[stream-lens] GOOGLE_API_KEY not set — video analysis disabled")

    # Verify fusion model is available. For Ollama models, probe /api/show.
    # For Google API models, skip the check (API key validity is caught at first call).
    if not _fusion_uses_google(FUSION_MODEL):
        _ollama_show_url = OLLAMA_URL.replace("/api/generate", "/api/show")
        try:
            resp = _requests.post(_ollama_show_url, json={"name": FUSION_MODEL}, timeout=10)
            if resp.status_code != 200:
                raise ValueError(resp.text[:200])
            logger.info("[stream-lens] fusion model OK: %s (ollama)", FUSION_MODEL)
        except Exception as exc:
            logger.error(
                "[stream-lens] fusion model '%s' not available in Ollama (%s): %s — "
                "processing disabled; Morpheus forwarding continues",
                FUSION_MODEL, _ollama_show_url, exc,
            )
            processing_enabled = False
            latest_context.update({
                "status": "error",
                "context": {"error": f"fusion model '{FUSION_MODEL}' not pulled in Ollama"},
                "processed_at": datetime.now(timezone.utc).isoformat(),
            })
    else:
        if _gemini_client is None:
            logger.error(
                "[stream-lens] FUSION_MODEL=%s targets Google API but GOOGLE_API_KEY is not set — "
                "processing disabled",
                FUSION_MODEL,
            )
            processing_enabled = False
            latest_context.update({
                "status": "error",
                "context": {"error": "GOOGLE_API_KEY not set — fusion via Google API unavailable"},
                "processed_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            logger.info("[stream-lens] fusion model OK: %s (google-api)", FUSION_MODEL)

    fusion_backend = "google-api" if _fusion_uses_google(FUSION_MODEL) else "ollama"
    logger.info(
        "[stream-lens] buffer_size=%d trigger_segs=%d seg=%ds "
        "frame_mode=%s frame_fps=%.1f fusion_model=%s(%s)",
        BUFFER_SIZE, ANALYSIS_TRIGGER_SEGMENTS, SEG_DURATION_S,
        FRAME_SAMPLE_MODE, FRAME_SAMPLE_FPS, FUSION_MODEL, fusion_backend,
    )
    yield


app = FastAPI(title="stream-lens", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Sync helpers (run in executor) ────────────────────────────────────────────

def _get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        logger.info("[audio] loading Whisper model %s...", WHISPER_MODEL)
        _whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="int8")
    return _whisper_model


def _prepare_raw_files(
    v_init: bytes,
    video_segs: list[tuple[int, bytes]],
    a_init: bytes,
    audio_segs: list[tuple[int, bytes]],
    tmpdir: str,
) -> tuple[Path, Path]:
    """Byte-concatenate init+fragments for each stream into temp fMP4 files."""
    base = Path(tmpdir)

    sorted_video = sorted(video_segs, key=lambda x: x[0])
    sorted_audio = sorted(audio_segs, key=lambda x: x[0])

    video_raw = base / "video_raw.mp4"
    video_raw.write_bytes(v_init + b"".join(d for _, d in sorted_video))

    audio_raw = base / "audio_raw.mp4"
    audio_raw.write_bytes(a_init + b"".join(d for _, d in sorted_audio))

    return video_raw, audio_raw


def _extract_video_frames(
    video_raw: Path,
    clip_start_sec: float,
) -> list[tuple[float, bytes]]:
    """Extract JPEG frames from the video fMP4 using ffmpeg."""
    base = video_raw.parent

    if FRAME_SAMPLE_MODE == "iframes":
        vf = f"select=eq(pict_type\\,I),scale={FRAME_MAX_WIDTH}:-1"
        vsync = ["-vsync", "vfr"]
    else:
        vf = f"fps={FRAME_SAMPLE_FPS},scale={FRAME_MAX_WIDTH}:-1"
        vsync = []

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_raw), "-vf", vf]
        + vsync
        + ["-q:v", "3", str(base / "frame_%04d.jpg")],
        capture_output=True,
    )
    if result.returncode != 0:
        logger.error("[video] ffmpeg frame extract failed: %s",
                     result.stderr.decode(errors="replace")[-500:])
        return []

    frame_files = sorted(base.glob("frame_*.jpg"))
    if not frame_files:
        return []

    # Subsample to MAX_FRAMES evenly spread
    if len(frame_files) > MAX_FRAMES:
        indices = [int(i * (len(frame_files) - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
        frame_files = [frame_files[i] for i in indices]

    frames = []
    for i, path in enumerate(frame_files):
        ts = clip_start_sec + i / max(FRAME_SAMPLE_FPS, 1.0)
        frames.append((ts, path.read_bytes()))

    return frames


def _extract_audio_wav(audio_raw: Path, tmpdir: str) -> Path:
    """Convert audio fMP4 to 16kHz mono WAV."""
    wav_path = Path(tmpdir) / "audio.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_raw), "-vn", "-ar", str(AUDIO_SR), "-ac", "1",
         str(wav_path)],
        capture_output=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args,
            output=result.stdout, stderr=result.stderr,
        )
    return wav_path


def _run_whisper(wav_path: Path) -> dict:
    model = _get_whisper()
    segs, info = model.transcribe(str(wav_path), beam_size=3, vad_filter=True)
    text = " ".join(s.text.strip() for s in segs).strip()
    lang_prob = round(float(info.language_probability), 3)
    has_speech = lang_prob >= 0.5
    logger.info("[audio] whisper: lang=%s (p=%.3f) has_speech=%s transcript=%s",
                info.language, lang_prob, has_speech, text[:80])
    return {
        "transcript": text,
        "language": info.language,
        "language_probability": lang_prob,
        "has_speech": has_speech,
    }


def _run_librosa(wav_path: Path) -> dict:
    """Extract audio features: tempo, energy, spectral centroid, tone label."""
    y, sr = librosa.load(str(wav_path), sr=AUDIO_SR, mono=True)
    if len(y) == 0:
        logger.warning("[audio] librosa: empty audio buffer — skipping feature extraction")
        return {"tempo": 0.0, "energy": 0.0, "spectral_centroid": 0.0, "zcr": 0.0, "tone": "silent"}
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = float(np.mean(rms))
    rms_var = float(np.var(rms))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    tempo_v = float(np.asarray(tempo).flat[0])

    if avg_rms > 0.08 and tempo_v > 120:
        label = "intense / high-energy"
    elif avg_rms > 0.05 and tempo_v > 90:
        label = "energetic / upbeat"
    elif avg_rms < 0.02:
        label = "quiet / calm"
    elif rms_var > 0.005:
        label = "dynamic / variable"
    else:
        label = "moderate / neutral"

    return {
        "tempo": round(tempo_v, 1),
        "energy": round(avg_rms, 4),
        "spectral_centroid": round(float(np.mean(centroid)), 1),
        "zcr": round(avg_zcr, 4),
        "tone": label,
    }


def _run_audio_track(audio_raw: Path, tmpdir: str, audio_segs_count: int) -> tuple[dict, dict, list[dict]]:
    errors: list[dict] = []
    whisper_result: dict = {}
    librosa_result: dict = {}
    t0 = time.perf_counter()
    timings = {"audio_wav_ms": 0, "whisper_ms": 0, "librosa_ms": 0, "audio_track_ms": 0}

    if audio_segs_count == 0:
        logger.warning("[audio] no audio segments in buffer — skipping audio track")
        errors.append({"step": "audio", "message": "no audio segments in buffer"})
        timings["audio_track_ms"] = _ms(t0)
        return {}, timings, errors

    try:
        wav_path = _extract_audio_wav(audio_raw, tmpdir)
        timings["audio_wav_ms"] = _ms(t0)
    except Exception as exc:
        logger.error("[audio] WAV extraction failed: %s", exc)
        errors.append({"step": "audio_wav", "message": str(exc)})
        timings["audio_track_ms"] = _ms(t0)
        return {}, timings, errors

    t1 = time.perf_counter()
    try:
        whisper_result = _run_whisper(wav_path)
    except Exception as exc:
        logger.error("[audio] Whisper failed: %s", exc)
        errors.append({"step": "whisper", "message": str(exc)})
    timings["whisper_ms"] = _ms(t1)

    t2 = time.perf_counter()
    try:
        librosa_result = _run_librosa(wav_path)
    except Exception as exc:
        logger.error("[audio] librosa failed: %s", exc)
        errors.append({"step": "librosa", "message": str(exc)})
    timings["librosa_ms"] = _ms(t2)

    timings["audio_track_ms"] = _ms(t0)
    return {**whisper_result, **librosa_result}, timings, errors


def _call_video_model(
    frames: list[tuple[float, bytes]],
    clip_start: datetime,
    clip_end: datetime,
) -> tuple[dict, dict]:
    """Send frames to Gemma 4 26B via Google API; return (video context, token counts)."""
    if not frames or _gemini_client is None:
        return {}, {}

    window_secs = (clip_end - clip_start).total_seconds()
    start_sec = clip_start.timestamp()
    end_sec = clip_end.timestamp()
    fps = FRAME_SAMPLE_FPS
    timestamps = [f"{ts:.1f}s" for ts, _ in frames]

    _default_video_instructions = (
        "Describe what is happening for ad personalization. "
        "Return ONLY valid JSON with these exact keys:\n"
        "- scene_summary: one sentence describing the overall scene\n"
        "- activity: what is happening (e.g. football match, news broadcast, film)\n"
        "- visual_mood: dominant atmosphere (tense / calm / energetic / dark / bright)\n"
        "- color_palette: list of 3 dominant color descriptors\n"
        "- content_tags: flat list of keywords covering genre, sport type, setting, mood, "
        "and detected entities (e.g. [\"surfing\", \"Olympic\", \"high-energy\", \"ocean\", \"competition\"])\n"
        "- ad_cues: list of up to 3 contextual signals for ad targeting\n"
        "- temporal_notes: any notable changes across the frames\n\n"
        "Return only JSON. No preamble, no markdown fences."
    )
    video_instructions = VIDEO_ANALYSIS_INSTRUCTIONS or _default_video_instructions

    prompt = (
        f"You are analyzing {len(frames)} video frames sampled at {fps}fps "
        f"from a {window_secs:.0f}s window of a live video stream "
        f"(t={start_sec:.0f}s to t={end_sec:.0f}s). "
        f"Timestamps: {', '.join(timestamps)}.\n\n"
        f"{video_instructions}"
    )

    contents = [
        genai_types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")
        for _, jpeg in frames
    ]
    contents.append(prompt)

    response = _gemini_client.models.generate_content(
        model=VIDEO_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            response_modalities=["TEXT"],
            temperature=0.1,
        ),
    )

    um = response.usage_metadata
    video_tokens = {
        "video_input_tokens":  (um.prompt_token_count or 0) if um else 0,
        "video_output_tokens": (um.candidates_token_count or 0) if um else 0,
    }
    logger.info("[video] tokens in=%d out=%d", video_tokens["video_input_tokens"], video_tokens["video_output_tokens"])

    raw = response.text.strip()
    logger.info("[video] Gemma raw: %s", raw[:120])

    # Strip optional markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw), video_tokens
    except json.JSONDecodeError:
        logger.error("[video] JSON parse failed: %s", raw[:300])
        return {"error": "video parse failed", "raw": raw[:500]}, video_tokens



def _default_fusion_instructions() -> str:
    p = FUSION_CONTEXT_PREFIX
    return (
        "Synthesize a set of ad targeting tags based on both the video and audio analyses above.\n"
        "Output format: key=value pairs separated by & (query string format).\n"
        "Rules:\n"
        f"- All keys must start with {p} (e.g. {p}activity, {p}mood, {p}sport)\n"
        "- Keys are chosen freely — use whatever is most relevant to ad targeting\n"
        "- Values are short (1-4 words)\n"
        "- Generate between 5 and 10 tags\n"
        "- Output only the key=value string. No preamble, no explanation, no JSON, no markdown.\n"
        f"Example: {p}activity=surfing&{p}mood=energetic&{p}setting=ocean&{p}sport=surfing"
    )


def _call_fusion_model(
    video_ctx: dict,
    audio_ctx: dict,
    clip_start: datetime,
    clip_end: datetime,
) -> tuple[str, dict]:
    """Fuse video+audio context via the configured fusion model; return (context_str, token counts)."""
    transcript = audio_ctx.get("transcript", "(no speech detected)")
    tone = {k: audio_ctx[k] for k in ("tempo", "energy", "tone") if k in audio_ctx}

    window_secs = (clip_end - clip_start).total_seconds()
    trigger_sec = clip_end.timestamp()

    instructions = FUSION_INSTRUCTIONS or _default_fusion_instructions()

    prompt = (
        f"You are a context synthesizer for a video ad personalization system.\n"
        f"The following analyses cover a {window_secs:.0f}s content buffer ending at "
        f"t={trigger_sec:.0f}s.\n\n"
        f"VIDEO ANALYSIS:\n{json.dumps(video_ctx, indent=2)}\n\n"
        f"AUDIO ANALYSIS:\n"
        f'Transcript: "{transcript}"\n'
        f"Tone: {json.dumps(tone, indent=2)}\n\n"
        f"{instructions}"
    )

    if _fusion_uses_google(FUSION_MODEL):
        if _gemini_client is None:
            raise RuntimeError("GOOGLE_API_KEY not set — cannot use Google API for fusion")
        response = _gemini_client.models.generate_content(
            model=FUSION_MODEL,
            contents=[prompt],
            config=genai_types.GenerateContentConfig(
                response_modalities=["TEXT"],
                temperature=0.1,
            ),
        )
        raw = response.text.strip()
        um = response.usage_metadata
        fusion_tokens = {
            "fusion_input_tokens":  (um.prompt_token_count or 0) if um else 0,
            "fusion_output_tokens": (um.candidates_token_count or 0) if um else 0,
        }
    else:
        payload = {
            "model": FUSION_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.1, "num_predict": 2000, "num_gpu": 0},
        }
        try:
            resp = _requests.post(OLLAMA_URL, json=payload, timeout=FUSION_MODEL_TIMEOUT)
            resp.raise_for_status()
        except _requests.exceptions.Timeout:
            raise TimeoutError(
                f"fusion model did not respond within {FUSION_MODEL_TIMEOUT}s"
                " — Ollama may be overloaded or model not loaded"
            )
        except _requests.exceptions.ConnectionError as exc:
            raise ConnectionError(f"cannot connect to Ollama at {OLLAMA_URL}: {exc}") from exc
        except _requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            body = exc.response.text[:200] if exc.response is not None else ""
            raise RuntimeError(f"Ollama HTTP {code}: {body}") from exc
        ollama_data = resp.json()
        raw = ollama_data.get("response", "").strip()
        fusion_tokens = {
            "fusion_input_tokens":  ollama_data.get("prompt_eval_count") or 0,
            "fusion_output_tokens": ollama_data.get("eval_count") or 0,
        }
    logger.info("[fusion] tokens in=%d out=%d", fusion_tokens["fusion_input_tokens"], fusion_tokens["fusion_output_tokens"])
    logger.info("[fusion] raw: %s", raw[:120])

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    return raw, fusion_tokens


def _run_video_track(
    video_raw: Path,
    clip_start: datetime,
    clip_end: datetime,
) -> tuple[dict, dict, int, dict, list[dict]]:
    errors: list[dict] = []
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    try:
        frames = _extract_video_frames(video_raw, clip_start.timestamp())
    except Exception as exc:
        logger.error("[video] frame extraction failed: %s", exc)
        errors.append({"step": "frame_extract", "message": str(exc)})
        frames = []
    t_frame_extract = _ms(t1)
    logger.info("[video] extracted %d frames", len(frames))

    t2 = time.perf_counter()
    video_ctx: dict = {}
    video_tokens: dict = {}
    try:
        video_ctx, video_tokens = _call_video_model(frames, clip_start, clip_end)
        if "error" in video_ctx:
            errors.append({"step": "gemini_parse", "message": video_ctx["error"]})
    except Exception as exc:
        logger.error("[video] Google API failed: %s", exc)
        errors.append({"step": "gemini", "message": str(exc), "service": "google_api"})
    t_gemma_video = _ms(t2)

    timings = {
        "frame_extract_ms": t_frame_extract,
        "gemma_video_ms":   t_gemma_video,
        "video_track_ms":   _ms(t0),
    }
    return video_ctx, timings, len(frames), video_tokens, errors


# ── Analysis task ─────────────────────────────────────────────────────────────

async def _run_analysis(
    v_init: bytes,
    a_init: bytes,
    video_segs: list[tuple[int, bytes]],
    audio_segs: list[tuple[int, bytes]],
    clip_start: datetime,
    clip_end: datetime,
) -> None:
    global analysis_in_progress, latest_context

    loop = asyncio.get_event_loop()
    t_total_start = time.perf_counter()
    all_errors: list[dict] = []

    try:
        # Phase A: prepare raw files — abort on failure
        try:
            with tempfile.TemporaryDirectory(prefix="ctx_") as tmpdir:
                t0 = time.perf_counter()
                video_raw, audio_raw = await loop.run_in_executor(
                    None, _prepare_raw_files, v_init, video_segs, a_init, audio_segs, tmpdir
                )
                t_prepare = _ms(t0)
                logger.info("[analysis] wrote video_raw (%s) audio_raw (%s)",
                            video_raw.stat().st_size, audio_raw.stat().st_size)

                # Phase B: parallel tracks — each returns errors, never raises
                t_par = time.perf_counter()
                video_result, audio_result = await asyncio.gather(
                    loop.run_in_executor(None, _run_video_track, video_raw, clip_start, clip_end),
                    loop.run_in_executor(None, _run_audio_track, audio_raw, tmpdir, len(audio_segs)),
                    return_exceptions=True,
                )
                t_parallel = _ms(t_par)

                if isinstance(video_result, Exception):
                    logger.error("[analysis] video track raised unexpectedly: %s", video_result)
                    video_ctx, vt, frames_extracted, video_tokens = {}, {"frame_extract_ms": 0, "gemma_video_ms": 0, "video_track_ms": 0}, 0, {}
                    all_errors.append({"step": "video_track", "message": str(video_result)})
                else:
                    video_ctx, vt, frames_extracted, video_tokens, video_errors = video_result
                    all_errors.extend(video_errors)

                if isinstance(audio_result, Exception):
                    logger.error("[analysis] audio track raised unexpectedly: %s", audio_result)
                    audio_ctx, at = {}, {"audio_wav_ms": 0, "whisper_ms": 0, "librosa_ms": 0, "audio_track_ms": 0}
                    all_errors.append({"step": "audio_track", "message": str(audio_result)})
                else:
                    audio_ctx, at, audio_errors = audio_result
                    all_errors.extend(audio_errors)

                logger.info("[analysis] video_ctx keys=%s audio_ctx keys=%s frames_extracted=%d",
                            list(video_ctx.keys()), list(audio_ctx.keys()), frames_extracted)

                # Phase C: fusion — specific error types
                t0 = time.perf_counter()
                context_str: str = ""
                fusion_tokens: dict = {}
                fusion_error: str = ""
                try:
                    context_str, fusion_tokens = await loop.run_in_executor(
                        None, _call_fusion_model, video_ctx, audio_ctx, clip_start, clip_end
                    )
                except TimeoutError as exc:
                    logger.error("[fusion] timeout: %s", exc)
                    all_errors.append({"step": "fusion", "message": str(exc), "service": "ollama"})
                    fusion_error = f"fusion error: {exc}"
                except ConnectionError as exc:
                    logger.error("[fusion] connection error: %s", exc)
                    all_errors.append({"step": "fusion", "message": str(exc), "service": "ollama"})
                    fusion_error = f"fusion error: {exc}"
                except Exception as exc:
                    logger.error("[fusion] failed: %s", exc)
                    all_errors.append({"step": "fusion", "message": str(exc)})
                    fusion_error = f"fusion error: {exc}"
                t_fusion = _ms(t0)
                logger.info("[analysis] fusion result: %s", context_str[:120])

                fusion_failed = bool(fusion_error)
                if fusion_failed:
                    status = "error"
                elif all_errors:
                    status = "partial"
                else:
                    status = "ready"

                timings = {
                    "prepare_ms":       t_prepare,
                    **vt,
                    **at,
                    "parallel_ms":      t_parallel,
                    "fusion_ms":        t_fusion,
                    "total_ms":         _ms(t_total_start),
                    **video_tokens,
                    **fusion_tokens,
                }
                logger.info(
                    "[timing] prepare=%dms | video=%dms (frames=%dms gemma=%dms) | "
                    "audio=%dms (wav=%dms whisper=%dms librosa=%dms) | "
                    "parallel=%dms | fusion=%dms | total=%dms",
                    timings["prepare_ms"],
                    timings.get("video_track_ms", 0), timings.get("frame_extract_ms", 0), timings.get("gemma_video_ms", 0),
                    timings.get("audio_track_ms", 0), timings.get("audio_wav_ms", 0), timings.get("whisper_ms", 0), timings.get("librosa_ms", 0),
                    timings["parallel_ms"], timings["fusion_ms"], timings["total_ms"],
                )

                async with _lock:
                    latest_context = {
                        "status": status,
                        "context": fusion_error if fusion_failed else context_str,
                        "clip_start": clip_start.isoformat(),
                        "clip_end": clip_end.isoformat(),
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "timings": timings,
                        "frames_extracted": frames_extracted,
                        "has_speech": audio_ctx.get("has_speech"),
                        **({"errors": all_errors} if all_errors else {}),
                    }
                    global _last_ready_context
                    if not fusion_failed:
                        _last_ready_context = context_str

        except Exception as exc:
            logger.error("[analysis] prepare failed: %s", exc, exc_info=True)
            async with _lock:
                latest_context = {
                    "status": "error",
                    "context": str(exc),
                    "clip_start": clip_start.isoformat(),
                    "clip_end": clip_end.isoformat(),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "timings": {"total_ms": _ms(t_total_start)},
                    "errors": [{"step": "prepare", "message": str(exc)}],
                }

    finally:
        async with _lock:
            analysis_in_progress = False


def _inject_context_into_mpd(mpd_bytes: bytes) -> bytes:
    global _last_injected_mpd
    ctx = _last_ready_context
    if not ctx:
        _last_injected_mpd = mpd_bytes
        return mpd_bytes

    try:
        root = ET.fromstring(mpd_bytes)
    except ET.ParseError:
        logger.warning("[mpd-inject] XML parse failed, forwarding as-is")
        _last_injected_mpd = mpd_bytes
        return mpd_bytes

    scte_stream = None
    for period in root.findall(f"{{{_DASH_NS}}}Period"):
        for es in period.findall(f"{{{_DASH_NS}}}EventStream"):
            if "scte35" in (es.get("schemeIdUri") or ""):
                scte_stream = es
                break
        if scte_stream is not None:
            break

    if scte_stream is None:
        _last_injected_mpd = mpd_bytes
        return mpd_bytes

    for sp in scte_stream.findall(f"{{{_DASH_NS}}}SupplementalProperty"):
        if sp.get("schemeIdUri") == _UP_SCHEME:
            scte_stream.remove(sp)

    sp = ET.Element(f"{{{_DASH_NS}}}SupplementalProperty")
    sp.set("schemeIdUri", _UP_SCHEME)
    eqi = ET.SubElement(sp, f"{{{_UP_NS}}}ExtUrlQueryInfo")
    eqi.set("queryTemplate", "$querypart$")
    eqi.set("includeInRequests", _OVERLAY_SCHEME)
    eqi.set("queryString", ctx)  # ET XML-encodes & → &amp; automatically
    scte_stream.insert(0, sp)

    result = ET.tostring(root, encoding="unicode", xml_declaration=True).encode()
    _last_injected_mpd = result
    return result


async def _forward_mpd_to_morpheus(mpd_bytes: bytes) -> None:
    mpd_bytes = _inject_context_into_mpd(mpd_bytes)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.put(
                MORPHEUS_MPD_URL,
                content=mpd_bytes,
                headers={"Content-Type": "application/dash+xml"},
            )
            if resp.status_code not in (200, 201, 204):
                logger.warning("[mpd-forward] Morpheus PUT %s: %s", resp.status_code, resp.text[:200])
            else:
                logger.debug("[mpd-forward] PUT OK → %s", MORPHEUS_MPD_URL)
    except Exception as exc:
        logger.error("[mpd-forward] failed: %s", exc)


async def _forward_segment_to_morpheus(seg_name: str, data: bytes) -> None:
    url = f"{MORPHEUS_BASE_URL}/{seg_name}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.put(url, content=data, headers={"Content-Type": "application/octet-stream"})
            if resp.status_code not in (200, 201, 204):
                logger.warning("[seg-forward] Morpheus PUT %s → %s", seg_name, resp.status_code)
            else:
                logger.debug("[seg-forward] PUT OK → %s", url)
    except Exception as exc:
        logger.error("[seg-forward] failed %s: %s", seg_name, exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.put("/segment")
async def put_segment(request: Request) -> JSONResponse:
    """Receive a raw DASH segment (init or media) and buffer it."""
    global video_init, audio_init, analysis_in_progress, latest_context
    global _trigger_counter, _total_video_segs_received, _seen_video_stream_ids, _invalid_rendition

    seg_type = request.headers.get("X-Segment-Type", "")
    stream_type = request.headers.get("X-Stream-Type", "")
    seg_num_header = request.headers.get("X-Segment-Number")

    if seg_type not in ("init", "media"):
        return JSONResponse(
            {"ok": False, "msg": f"Invalid X-Segment-Type: {seg_type!r}"},
            status_code=400,
        )
    if stream_type not in ("video", "audio"):
        return JSONResponse(
            {"ok": False, "msg": f"Invalid X-Stream-Type: {stream_type!r}"},
            status_code=400,
        )
    if seg_type == "media" and seg_num_header is None:
        return JSONResponse(
            {"ok": False, "msg": "X-Segment-Number required for media segments"},
            status_code=400,
        )

    data = await request.body()

    # Forward segment to Morpheus regardless of processing_enabled state
    seg_name = request.headers.get("X-Segment-Name", "")
    if seg_name:
        asyncio.create_task(_forward_segment_to_morpheus(seg_name, data))

    async with _lock:
        if seg_type == "init":
            if stream_type == "video":
                video_init = data
                logger.info("[segment] received video init (%d bytes)", len(data))
            else:
                audio_init = data
                logger.info("[segment] received audio init (%d bytes)", len(data))
            return JSONResponse({"ok": True})

        # Rendition filter for video media segments
        if stream_type == "video":
            m_rid = re.search(r"stream(\d+)", seg_name)
            rendition_id = int(m_rid.group(1)) if m_rid else 0
            _seen_video_stream_ids.add(rendition_id)
            _total_video_segs_received += 1

            if rendition_id == ANALYSIS_VIDEO_RENDITION:
                _invalid_rendition = False
                # fall through to existing buffer logic
            else:
                if (
                    _total_video_segs_received >= ANALYSIS_TRIGGER_SEGMENTS
                    and ANALYSIS_VIDEO_RENDITION not in _seen_video_stream_ids
                ):
                    _invalid_rendition = True
                return JSONResponse({"ok": True})

        if stream_type == "video" and video_init is None:
            return JSONResponse(
                {"ok": False, "msg": "init not received yet"}, status_code=409
            )
        if stream_type == "audio" and audio_init is None:
            return JSONResponse(
                {"ok": False, "msg": "init not received yet"}, status_code=409
            )

        seg_num = int(seg_num_header)
        received_at = datetime.now(timezone.utc)

        if stream_type == "video":
            if len(video_buffer) < MAX_BUFFER:
                video_buffer.append((seg_num, data, received_at))
                _trigger_counter += 1
                logger.debug("[segment] buffered video seg=%d buf=%d trigger=%d", seg_num, len(video_buffer), _trigger_counter)
            else:
                logger.debug("[segment] video buffer full, dropping seg=%d", seg_num)
        else:
            if len(audio_buffer) < MAX_BUFFER:
                audio_buffer.append((seg_num, data, received_at))
                logger.debug("[segment] buffered audio seg=%d buf=%d", seg_num, len(audio_buffer))
            else:
                logger.debug("[segment] audio buffer full, dropping seg=%d", seg_num)

        should_analyze = (
            _trigger_counter >= ANALYSIS_TRIGGER_SEGMENTS
            and len(video_buffer) >= BUFFER_SIZE
            and not analysis_in_progress
            and bool(GOOGLE_API_KEY)
            and processing_enabled
        )

        if should_analyze:
            buffer_video = video_buffer[-BUFFER_SIZE:]
            buffer_audio = list(audio_buffer)
            del video_buffer[:]
            audio_buffer.clear()
            _trigger_counter = 0

            clip_start = buffer_video[0][2]
            clip_end = buffer_video[-1][2]

            analysis_in_progress = True
            latest_context["status"] = "processing"

            v_init_snap = video_init
            a_init_snap = audio_init

            video_segs = [(n, b) for n, b, _ in buffer_video]
            audio_segs = [(n, b) for n, b, _ in buffer_audio]

            asyncio.create_task(
                _run_analysis(
                    v_init_snap, a_init_snap,
                    video_segs, audio_segs,
                    clip_start, clip_end,
                )
            )
            logger.info(
                "[segment] triggered analysis: %d video segs, %d audio segs",
                len(video_segs), len(audio_segs),
            )

    return JSONResponse({"ok": True})


@app.put("/live.mpd")
async def put_live_mpd(request: Request) -> JSONResponse:
    """Receive MPD from live-sim and forward to Morpheus (fire-and-forget)."""
    mpd_bytes = await request.body()
    if not mpd_bytes:
        return JSONResponse({"ok": False, "msg": "empty body"}, status_code=400)
    asyncio.create_task(_forward_mpd_to_morpheus(mpd_bytes))
    return JSONResponse({"ok": True})


@app.post("/processing")
async def post_processing(request: Request) -> JSONResponse:
    """Toggle video analysis on or off without affecting MPD forwarding."""
    global processing_enabled
    body = await request.json()
    if "enabled" not in body or not isinstance(body["enabled"], bool):
        return JSONResponse(
            {"ok": False, "msg": 'body must be {"enabled": true|false}'},
            status_code=400,
        )
    async with _lock:
        processing_enabled = body["enabled"]
    logger.info("[processing] analysis enabled=%s", processing_enabled)
    return JSONResponse({"ok": True, "processing_enabled": processing_enabled})


@app.get("/config")
async def get_config() -> JSONResponse:
    return JSONResponse({"config": {
        "BUFFER_SIZE": BUFFER_SIZE,
        "ANALYSIS_TRIGGER_SEGMENTS": ANALYSIS_TRIGGER_SEGMENTS,
        "ANALYSIS_VIDEO_RENDITION": ANALYSIS_VIDEO_RENDITION,
        "FRAME_SAMPLE_MODE": FRAME_SAMPLE_MODE,
        "FRAME_SAMPLE_FPS": FRAME_SAMPLE_FPS,
        "MAX_FRAMES": MAX_FRAMES,
        "FRAME_MAX_WIDTH": FRAME_MAX_WIDTH,
        "SEG_DURATION_S": SEG_DURATION_S,
        "FUSION_MODEL": FUSION_MODEL,
        "VIDEO_MODEL": VIDEO_MODEL,
        "FUSION_CONTEXT_PREFIX": FUSION_CONTEXT_PREFIX,
    }})


@app.post("/config")
async def post_config(request: Request) -> JSONResponse:
    """Update analysis config at runtime and reset buffers. Used by the benchmark."""
    global BUFFER_SIZE, ANALYSIS_TRIGGER_SEGMENTS, MAX_BUFFER, _trigger_counter, _total_video_segs_received
    global ANALYSIS_VIDEO_RENDITION, _seen_video_stream_ids, _invalid_rendition
    global FRAME_SAMPLE_MODE, FRAME_SAMPLE_FPS, MAX_FRAMES

    body = await request.json()
    async with _lock:
        if "BUFFER_SIZE" in body:
            BUFFER_SIZE = max(1, int(body["BUFFER_SIZE"]))
            MAX_BUFFER = max(BUFFER_SIZE, ANALYSIS_TRIGGER_SEGMENTS) + 2
        if "ANALYSIS_TRIGGER_SEGMENTS" in body:
            ANALYSIS_TRIGGER_SEGMENTS = max(1, int(body["ANALYSIS_TRIGGER_SEGMENTS"]))
            MAX_BUFFER = max(BUFFER_SIZE, ANALYSIS_TRIGGER_SEGMENTS) + 2
        if "ANALYSIS_VIDEO_RENDITION" in body:
            ANALYSIS_VIDEO_RENDITION = int(body["ANALYSIS_VIDEO_RENDITION"])
            video_init = None
            _seen_video_stream_ids = set()
            _total_video_segs_received = 0
            _invalid_rendition = False
        if "FRAME_SAMPLE_MODE" in body:
            FRAME_SAMPLE_MODE = str(body["FRAME_SAMPLE_MODE"])
        if "FRAME_SAMPLE_FPS" in body:
            FRAME_SAMPLE_FPS = float(body["FRAME_SAMPLE_FPS"])
        if "MAX_FRAMES" in body:
            MAX_FRAMES = int(body["MAX_FRAMES"])

        video_buffer.clear()
        audio_buffer.clear()
        _trigger_counter = 0
        _total_video_segs_received = 0
        latest_context.update({
            "status": "waiting",
            "context": None,
            "clip_start": None,
            "clip_end": None,
            "processed_at": None,
            "timings": None,
        })

    logger.info(
        "[config] buffer_size=%d trigger_segs=%d rendition=%d mode=%s fps=%.2f max_frames=%d",
        BUFFER_SIZE, ANALYSIS_TRIGGER_SEGMENTS, ANALYSIS_VIDEO_RENDITION,
        FRAME_SAMPLE_MODE, FRAME_SAMPLE_FPS, MAX_FRAMES,
    )
    return JSONResponse({"ok": True, "config": {
        "BUFFER_SIZE": BUFFER_SIZE,
        "ANALYSIS_TRIGGER_SEGMENTS": ANALYSIS_TRIGGER_SEGMENTS,
        "ANALYSIS_VIDEO_RENDITION": ANALYSIS_VIDEO_RENDITION,
        "FRAME_SAMPLE_MODE": FRAME_SAMPLE_MODE,
        "FRAME_SAMPLE_FPS": FRAME_SAMPLE_FPS,
        "MAX_FRAMES": MAX_FRAMES,
    }})


@app.get("/mpd")
async def get_mpd():
    """Return the last MPD forwarded to Morpheus (after context injection)."""
    from fastapi.responses import Response
    if _last_injected_mpd is None:
        return JSONResponse({"error": "no MPD received yet"}, status_code=404)
    return Response(content=_last_injected_mpd, media_type="application/dash+xml")


@app.get("/context")
async def get_context() -> JSONResponse:
    """Return the latest analysis result."""
    async with _lock:
        if _invalid_rendition:
            return JSONResponse(
                {
                    "error": (
                        f"ANALYSIS_VIDEO_RENDITION={ANALYSIS_VIDEO_RENDITION} does not match "
                        f"any received video stream. Received streams: {sorted(_seen_video_stream_ids)}"
                    )
                },
                status_code=503,
            )
        return JSONResponse(dict(latest_context))


@app.get("/health")
async def get_health() -> JSONResponse:
    return JSONResponse({"ok": True})
