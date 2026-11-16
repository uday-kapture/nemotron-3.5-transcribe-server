"""Load-test a running Nemotron 3.5 Transcribe server with real recordings.

For each concurrency level, keeps that many requests in flight for --duration seconds, cycling
through the audio files. Stats cover requests that finish in the second half of each level, once
the first wave is through and the server is in steady state. Requests are the ones `transcribe`
sends. Every request (with its transcript) is logged to --log.

  uvx --from huggingface_hub hf download kapturecx/audited-transcripts-multispeaker \\
      --repo-type dataset --include 'audio/raw/*.wav' --local-dir data/audited
  uv run python -m nemo_3_5_transcribe.loadtest data/audited/audio/raw --server http://gpu-host:8000
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import TextIO

import httpx
import soundfile

from .client import PROMPTS, ClientConfig, build_request, find_audio


def summarize(rows: list[dict], duration_s: float) -> dict:
    """Throughput and latency of the requests that finished in the second half of a level."""
    half = duration_s / 2
    ok = [r for r in rows if r["ok"] and half <= r["finished_at"] <= duration_s]
    latencies = sorted(r["latency_s"] for r in ok)

    def pct(q: float) -> float:
        return latencies[math.ceil(q * len(latencies)) - 1] if latencies else math.nan

    return {
        "ok": len(ok),
        "errors": sum(not r["ok"] for r in rows),
        "truncated": sum(r["finish_reason"] == "length" for r in ok),
        "ctc_errors": sum(r["ctc_error"] is not None for r in ok),
        "calls_per_hour": len(ok) / half * 3600,
        "rtfx": sum(r["audio_s"] for r in ok) / half,
        "out_tok_s": sum(r["completion_tokens"] for r in ok) / half,
        "p50_s": pct(0.50),
        "p95_s": pct(0.95),
        "p99_s": pct(0.99),
    }


async def send(http: httpx.AsyncClient, body: bytes) -> dict:
    response = await http.post("/v1/chat/completions", content=body, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    return {
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": (payload.get("usage") or {}).get("completion_tokens", 0),
        # A server started without the CTC middleware returns no timestamps; count that as an error.
        "ctc_error": (payload.get("ctc_timestamps") or {"error": "no ctc_timestamps in response"}).get("error"),
        "text": choice["message"].get("content") or "",
    }


async def run_level(
    http: httpx.AsyncClient, calls: list[tuple[str, float, bytes]], concurrency: int, duration_s: float, log: TextIO
) -> list[dict]:
    queue = itertools.cycle(calls)
    rows: list[dict] = []
    start = time.perf_counter()

    async def worker() -> None:
        while time.perf_counter() - start < duration_s:
            name, audio_s, body = next(queue)
            row = {"concurrency": concurrency, "call": name, "audio_s": audio_s}
            sent = time.perf_counter()
            try:
                result = await send(http, body)
                row |= {"ok": True, **result}
            except Exception as error:  # timeouts and 5xx under overload are results, not crashes
                row |= {"ok": False, "error": repr(error)}
            done = time.perf_counter()
            row |= {"latency_s": done - sent, "finished_at": done - start}
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not row["ok"]:
                await asyncio.sleep(1)  # don't spin on a server that fails fast

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return rows


def print_summary(s: dict) -> None:
    print(
        f"c={s['concurrency']:>4}  ok={s['ok']:>5}  err={s['errors']}  trunc={s['truncated']}  "
        f"ctc_err={s['ctc_errors']}  calls/h={s['calls_per_hour']:>7.0f}  RTFx={s['rtfx']:>6.1f}  "
        f"tok/s={s['out_tok_s']:>6.0f}  p50={s['p50_s']:.1f}s  p95={s['p95_s']:.1f}s  p99={s['p99_s']:.1f}s",
        flush=True,
    )


async def _run(args: argparse.Namespace, config: ClientConfig, calls: list[tuple[str, float, bytes]]) -> list[dict]:
    timeout = httpx.Timeout(config.timeout_s, connect=30.0)
    limits = httpx.Limits(max_connections=None)  # httpx's default of 100 would silently cap concurrency
    async with httpx.AsyncClient(base_url=config.server_url, timeout=timeout, limits=limits) as http:
        start = time.perf_counter()
        warm = await send(http, calls[0][2])  # fails fast on a wrong URL or model name
        elapsed = time.perf_counter() - start
        print(f"warm-up: {elapsed:.1f}s for {calls[0][1]:.0f}s of audio, ctc_error={warm['ctc_error']}")
        summaries = []
        with args.log.open("w") as log:
            for concurrency in args.concurrency:
                rows = await run_level(http, calls, concurrency, args.duration, log)
                summaries.append({"concurrency": concurrency, **summarize(rows, args.duration)})
                print_summary(summaries[-1])
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nemo_3_5_transcribe.loadtest",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="audio file or directory (searched recursively)")
    parser.add_argument("--server", default=ClientConfig.server_url, help="vLLM server base URL")
    parser.add_argument("--model", default=ClientConfig.model)
    parser.add_argument("--pass", dest="pass_name", default="speaker", choices=PROMPTS, help="prompt to send")
    parser.add_argument(
        "--concurrency",
        default="1,8,32,64,128,192,256",
        type=lambda s: [int(c) for c in s.split(",")],
        help="comma-separated in-flight request counts to sweep",
    )
    parser.add_argument("--duration", type=float, default=240.0, help="seconds per level; stats use the second half")
    parser.add_argument("--max-p95", type=float, help="latency target (s); reports the busiest level that meets it")
    parser.add_argument(
        "--log", type=Path, default=Path(f"out/loadtest-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"), help="per-request log"
    )
    args = parser.parse_args(argv)

    files = find_audio(args.audio.resolve())
    if not files:
        parser.error(f"no audio files found under {args.audio}")
    config = ClientConfig(server_url=args.server, model=args.model)
    # Encode every request once up front so client-side base64/JSON work can't throttle the test.
    # shortcut: soundfile can't read .m4a durations; convert first if you load-test m4a.
    calls = [
        (f.stem, soundfile.info(f).duration, json.dumps(build_request(f, PROMPTS[args.pass_name], config)).encode())
        for f in files
    ]
    total_s = sum(c[1] for c in calls)
    print(f"{len(calls)} calls, {total_s / 60:.0f} min of audio, mean {total_s / len(calls):.0f}s; log: {args.log}")
    args.log.parent.mkdir(parents=True, exist_ok=True)

    summaries = asyncio.run(_run(args, config, calls))
    if args.max_p95 is not None:
        good = [s for s in summaries if s["errors"] == 0 and s["p95_s"] <= args.max_p95]
        if not good:
            print(f"no level kept p95 <= {args.max_p95:.0f}s with zero errors")
            return 1
        best = max(good, key=lambda s: s["calls_per_hour"])
        print(
            f"capacity at p95 <= {args.max_p95:.0f}s: {best['concurrency']} in flight, "
            f"{best['calls_per_hour']:.0f} calls/hour, {best['rtfx']:.0f}x real time"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
