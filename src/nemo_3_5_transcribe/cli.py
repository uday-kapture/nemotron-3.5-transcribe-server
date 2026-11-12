"""Command line entry point: ``transcribe`` (needs the GPU server), ``annotate`` (offline), ``run`` (both)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from .annotate import build_conversation
from .client import PROMPTS, ClientConfig, find_audio, transcribe
from .export import write_outputs
from .interactions import InteractionConfig

log = logging.getLogger("nemo_3_5_transcribe")


def _transcribe(args: argparse.Namespace) -> int:
    audio_root = args.audio.resolve()
    files = find_audio(audio_root)
    if not files:
        log.error("no audio files found under %s", audio_root)
        return 1
    config = ClientConfig(
        server_url=args.server, concurrency=args.concurrency, max_tokens=args.max_tokens, model=args.model
    )
    failures = asyncio.run(
        transcribe(
            files,
            audio_root if audio_root.is_dir() else audio_root.parent,
            args.out / "raw",
            args.passes,
            config,
            overwrite=args.overwrite,
        )
    )
    if failures:
        (args.out / "transcribe_failures.json").write_text(json.dumps(failures, indent=1))
        log.warning("%d requests failed; see %s", len(failures), args.out / "transcribe_failures.json")
    return 0


def _annotate(args: argparse.Namespace) -> int:
    config = InteractionConfig()
    if args.config:
        config = InteractionConfig.from_dict(tomllib.loads(args.config.read_text()).get("interactions", {}))

    records: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted((args.out / "raw").glob("*.json")):
        cid, pass_name, _ = path.name.rsplit(".", 2)
        records[cid][pass_name] = json.loads(path.read_text())
    if not records:
        log.error("no raw responses in %s; run `transcribe` first", args.out / "raw")
        return 1

    conversations = [build_conversation(cid, passes, config) for cid, passes in records.items()]
    write_outputs(conversations, args.out, max_segment_s=args.max_segment_s, pad_s=args.segment_pad_s)

    statuses = Counter(c["status"].split(":")[0] for c in conversations)
    onsets = Counter()
    for conversation in conversations:
        onsets.update(conversation.get("stats", {}).get("onsets", {}))
    log.info("conversations: %s", dict(statuses))
    log.info("IPU onsets: %s", dict(onsets.most_common()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nemo-3-5-transcribe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_transcribe_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("audio", type=Path, help="audio file or directory (searched recursively)")
        p.add_argument("--server", default=ClientConfig.server_url, help="vLLM server base URL")
        p.add_argument("--model", default=ClientConfig.model)
        p.add_argument(
            "--passes",
            default="speaker,rich",
            type=lambda s: s.split(","),
            help=f"comma-separated prompts to run per file, from: {', '.join(PROMPTS)}",
        )
        p.add_argument("--concurrency", type=int, default=ClientConfig.concurrency)
        p.add_argument("--max-tokens", type=int, default=ClientConfig.max_tokens)
        p.add_argument("--overwrite", action="store_true", help="re-request files that already have a response")

    def add_annotate_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path, help="TOML file with an [interactions] table of thresholds")
        p.add_argument("--max-segment-s", type=float, default=20.0, help="longest segment in segments.jsonl")
        p.add_argument("--segment-pad-s", type=float, default=0.1, help="audio padding around each segment")

    for name, help_text in [
        ("transcribe", "send audio to the server and cache raw responses"),
        ("annotate", "build labelled transcripts and manifests from cached responses"),
        ("run", "transcribe, then annotate"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--out", type=Path, required=True, help="output directory")
        if name in ("transcribe", "run"):
            add_transcribe_args(p)
        if name in ("annotate", "run"):
            add_annotate_args(p)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    unknown = set(getattr(args, "passes", [])) - set(PROMPTS)
    if unknown:
        parser.error(f"unknown passes: {', '.join(sorted(unknown))}")

    if args.command == "transcribe":
        return _transcribe(args)
    if args.command == "annotate":
        return _annotate(args)
    return _transcribe(args) or _annotate(args)


if __name__ == "__main__":
    sys.exit(main())
