"""Look for wrong-script output (e.g. Chinese for Hindi calls) by transcribing every call under several
prompt and sampling settings, then counting letters that are neither Devanagari nor Latin.

  uv run python -m nemo_3_5_transcribe.scriptcheck data/audited/audio/raw --server http://127.0.0.1:8000

Prompts: the repo's ("speaker", "speaker_hi", "rich") plus "plain", the model card's quick-start prompt with no
language or tags (what most other users send), and "verbatim". Temperature 0 is what the card requires; 1 is
roughly what a client that sets no temperature may get (see the generation_config.json in the model folder).
Every response is logged to --log so the offending transcripts can be read.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from .client import PROMPTS, ClientConfig, build_request, find_audio
from .loadtest import send

ALL_PROMPTS = PROMPTS | {
    "plain": "Transcribe the following audio.",
    # The card's Monsoon Hindi prompt: stronger script control than speaker_hi, but no speaker tags.
    "hindi_strict": (
        "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
        "Formatting: punctuation and capitalization. Output: transcript only. "
        "Use Devanagari for Hindi words, never romanized Hindi. Keep English words in their original spelling."
    ),
    # The card's experimental near-field prompt: ignores background voices, so no tags for them either.
    "nearfield": (
        "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. "
        "No punctuation or capitalization. Transcribe only the speaker nearest to the microphone. "
        "Do not transcribe any other voices."
    ),
    "verbatim": (
        "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
        "Formatting: punctuation and capitalization. Output: transcript only."
    ),
}
EXPECTED_SCRIPTS = {"DEVANAGARI", "LATIN"}
LOW_DEVANAGARI = 0.3  # below this share of Devanagari letters, a Hindi call's transcript is in the wrong language


def _script(char: str) -> str:
    return unicodedata.name(char, "UNKNOWN").split()[0]


def foreign_scripts(text: str) -> Counter:
    """Letters outside Devanagari/Latin, counted by script name (``CJK``, ``ARABIC``, ...)."""
    return Counter(s for c in text if c.isalpha() and (s := _script(c)) not in EXPECTED_SCRIPTS)


def devanagari_share(text: str) -> float | None:
    """Fraction of letters (tags like <spk:0> excluded) that are Devanagari; None if there are no letters.

    Catches wrong *language* in an expected script (English or romanized text for a Hindi call): the
    reviewed transcripts of all 71 calls are at least 63% Devanagari, the English failures seen are 0%.
    """
    letters = [c for c in re.sub(r"<[^>]*>", "", text) if c.isalpha()]
    return sum(_script(c) == "DEVANAGARI" for c in letters) / len(letters) if letters else None


def is_wrong_language(row: dict) -> bool:
    share = row.get("deva")
    return bool(row.get("foreign")) or (share is not None and share < LOW_DEVANAGARI)


def snippet(text: str) -> str:
    """The text around the first foreign letter, or the start if there is none."""
    for i, c in enumerate(text):
        if c.isalpha() and _script(c) not in EXPECTED_SCRIPTS:
            return text[max(0, i - 60) : i + 60].replace("\n", " ")
    return text[:120].replace("\n", " ")


async def run(args: argparse.Namespace) -> list[dict]:
    config = ClientConfig(server_url=args.server, model=args.model, max_tokens=args.max_tokens)
    files = find_audio(args.audio.resolve())
    jobs = [
        (f, prompt, temperature, seed)
        for prompt, temperature in itertools.product(args.prompts, args.temperature)
        for seed in range(args.repeats)
        for f in files
    ]
    print(
        f"{len(files)} calls x {len(args.prompts)} prompts x {len(args.temperature)} temperatures x "
        f"{args.repeats} repeats = {len(jobs)} requests; log: {args.log}",
        flush=True,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []
    start = time.perf_counter()
    timeout = httpx.Timeout(config.timeout_s, connect=30.0)
    limits = httpx.Limits(max_connections=None)  # httpx's default of 100 would silently cap --concurrency
    with args.log.open("w") as log:
        async with httpx.AsyncClient(base_url=config.server_url, timeout=timeout, limits=limits) as http:

            async def one(audio: Path, prompt: str, temperature: float, seed: int) -> None:
                body = build_request(audio, ALL_PROMPTS[prompt], config)
                body |= {"temperature": temperature, "seed": seed}  # distinct seeds so repeats differ when sampling
                row = {"call": audio.stem, "prompt": prompt, "temperature": temperature, "seed": seed}
                async with semaphore:
                    try:
                        row |= await send(http, json.dumps(body).encode())
                        row["foreign"] = dict(foreign_scripts(row["text"]))
                        row["deva"] = devanagari_share(row["text"])
                    except Exception as error:  # one failed request must not stop the sweep
                        row |= {"error": repr(error)}
                rows.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                if len(rows) % 50 == 0:
                    print(f"  {len(rows)}/{len(jobs)} done, {time.perf_counter() - start:.0f}s", flush=True)

            await asyncio.gather(*(one(*job) for job in jobs))
    return rows


def report(rows: list[dict], examples: int) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["prompt"], r["temperature"])].append(r)
    print(
        f"\n{'prompt':<10} {'temp':>5} {'runs':>5} {'foreign':>8} {'CJK':>5} {'lowdeva':>8} {'trunc':>6} {'errors':>7}"
    )
    for (prompt, temperature), rs in sorted(groups.items()):
        ok = [r for r in rs if "error" not in r]
        print(
            f"{prompt:<10} {temperature:>5} {len(rs):>5} {sum(bool(r['foreign']) for r in ok):>8} "
            f"{sum('CJK' in r['foreign'] for r in ok):>5} "
            f"{sum(r['deva'] is not None and r['deva'] < LOW_DEVANAGARI for r in ok):>8} "
            f"{sum(r['finish_reason'] == 'length' for r in ok):>6} {len(rs) - len(ok):>7}"
        )
    print("foreign = letters outside Devanagari/Latin; lowdeva = under 30% Devanagari (English/romanized for Hindi)")
    bad = [r for r in rows if is_wrong_language(r)]
    for r in bad[:examples]:
        deva = "n/a" if r["deva"] is None else f"{r['deva']:.0%}"
        print(f"\n{r['call']} [{r['prompt']} t={r['temperature']} seed={r['seed']}] devanagari={deva} {r['foreign']}")
        print(f"  ...{snippet(r['text'])}...")
    if len(bad) > examples:
        print(f"\n(+{len(bad) - examples} more in the log)")
    if not bad:
        print("\nNo wrong-language responses with these settings: not reproduced.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nemo_3_5_transcribe.scriptcheck",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="audio file or directory (searched recursively)")
    parser.add_argument("--server", default=ClientConfig.server_url)
    parser.add_argument("--model", default=ClientConfig.model)
    parser.add_argument(
        "--prompts",
        default="plain,verbatim,speaker,speaker_hi",
        type=lambda s: s.split(","),
        help=f"comma-separated, from: {', '.join(ALL_PROMPTS)}",
    )
    parser.add_argument("--temperature", default="0,1", type=lambda s: [float(t) for t in s.split(",")])
    parser.add_argument("--repeats", type=int, default=1, help="runs per call; only differs when temperature > 0")
    parser.add_argument("--concurrency", type=int, default=64, help="in-flight requests (1 rules out batching effects)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--examples", type=int, default=8, help="offending transcripts to print")
    parser.add_argument("--log", type=Path, default=Path(f"out/scriptcheck-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"))
    args = parser.parse_args(argv)

    unknown = set(args.prompts) - set(ALL_PROMPTS)
    if unknown:
        parser.error(f"unknown prompts: {', '.join(sorted(unknown))}")
    if not find_audio(args.audio.resolve()):
        parser.error(f"no audio files found under {args.audio}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    report(asyncio.run(run(args)), args.examples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
