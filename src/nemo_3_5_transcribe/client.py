"""Send audio to a vLLM server running Nemotron 3.5 Transcribe and cache the raw responses.

Requests follow the model card's "Send a Request" recipe: greedy decoding, reasoning off,
special tokens kept (speaker/event/style tags), and CTC word timestamps captured. The
server must be started with the CTC timestamp middleware (see ``scripts/serve.sh``).

Each (recording, pass) response is written to ``<raw_dir>/<conversation_id>.<pass>.json``
and skipped on later runs, so an interrupted job can be resumed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"

_BASE_PROMPT = (
    "Transcribe this audio with punctuation and capitalization. "
    "Include all spoken words and verbal fillers. "
    "Write numbers in their standard written form. "
)
# Prompts from the model card's prompt recommendations.
PROMPTS = {
    "speaker": _BASE_PROMPT + "Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription.",
    "rich": _BASE_PROMPT
    + "Use <ev:*> and <sty:*> special tokens to recognize all non-speech verbatims and speaking style.",
}

AUDIO_MIME = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".m4a": "audio/mp4",
}


@dataclass
class ClientConfig:
    server_url: str = "http://127.0.0.1:8000"
    model: str = MODEL
    concurrency: int = 16
    # The model card notes a dense one-hour conversation can need ~20k output tokens.
    max_tokens: int = 24576
    timeout_s: float = 1800.0
    retries: int = 2


def conversation_id(audio: Path, root: Path) -> str:
    relative = audio.relative_to(root) if audio.is_relative_to(root) else Path(audio.name)
    return "__".join(relative.with_suffix("").parts)


def find_audio(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_MIME and p.is_file())


def build_request(audio: Path, prompt: str, config: ClientConfig) -> dict:
    data = base64.b64encode(audio.read_bytes()).decode()
    mime = AUDIO_MIME.get(audio.suffix.lower(), "audio/wav")
    return {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{data}"}},
                ],
            }
        ],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": config.max_tokens,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "mm_processor_kwargs": {"capture_ctc_timestamps": True},
    }


async def _transcribe_one(
    http: httpx.AsyncClient, audio: Path, pass_name: str, out_path: Path, config: ClientConfig
) -> None:
    prompt = PROMPTS[pass_name]
    body = build_request(audio, prompt, config)
    for attempt in range(config.retries + 1):
        try:
            response = await http.post("/v1/chat/completions", json=body)
            response.raise_for_status()
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            retryable = isinstance(error, httpx.TransportError) or error.response.status_code >= 500
            if attempt == config.retries or not retryable:
                raise
            await asyncio.sleep(2**attempt)
    payload = response.json()
    choice = payload["choices"][0]
    record = {
        "audio": str(audio),
        "pass": pass_name,
        "prompt": prompt,
        "text": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "ctc_timestamps": payload.get("ctc_timestamps"),
        "usage": payload.get("usage"),
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(out_path)


async def transcribe(
    audio_files: list[Path],
    audio_root: Path,
    raw_dir: Path,
    passes: list[str],
    config: ClientConfig,
    overwrite: bool = False,
) -> dict[str, str]:
    """Transcribe every file with every pass; returns ``{"<file>:<pass>": error}`` for failures."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for audio in audio_files:
        cid = conversation_id(audio, audio_root)
        for pass_name in passes:
            out_path = raw_dir / f"{cid}.{pass_name}.json"
            if overwrite or not out_path.exists():
                jobs.append((audio, pass_name, out_path))
    log.info("%d requests to send (%d files, passes=%s)", len(jobs), len(audio_files), ",".join(passes))

    failures: dict[str, str] = {}
    semaphore = asyncio.Semaphore(config.concurrency)
    timeout = httpx.Timeout(config.timeout_s, connect=30.0)
    async with httpx.AsyncClient(base_url=config.server_url, timeout=timeout) as http:

        async def run(audio: Path, pass_name: str, out_path: Path) -> None:
            async with semaphore:
                try:
                    await _transcribe_one(http, audio, pass_name, out_path, config)
                    log.info("done %s [%s]", audio, pass_name)
                except Exception as error:  # one bad file must not stop the batch
                    failures[f"{audio}:{pass_name}"] = repr(error)
                    log.error("failed %s [%s]: %r", audio, pass_name, error)

        await asyncio.gather(*(run(*job) for job in jobs))
    return failures
