"""Merge the per-pass responses of one recording into an annotated conversation.

The ``speaker`` pass supplies speaker-attributed word timings, from which IPUs and
turn-taking labels are derived. The ``rich`` pass is aligned as a single speaker, so
its ``<ev:*>``/``<sty:*>`` tags are given to whichever speaker was talking at that time.
If the speaker pass itself carries rich tags, those keep their own speaker.
"""

from __future__ import annotations

import difflib
from collections import Counter, defaultdict
from dataclasses import asdict

from .interactions import (
    IPU,
    OVERLAP_LABELS,
    InteractionConfig,
    build_ipus,
    label_ipus,
    normalize_word,
)
from .parse import UNKNOWN_SPEAKER, Alignment, Event, StyleSpan, parse_response, split_by_speaker_tags

INLINE_ONSETS = OVERLAP_LABELS | {"backchannel"}
# Events further than this from any speech are left unattached.
ATTACH_TOLERANCE_S = 0.5
_EPS = 0.05


def _gap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(b0 - a1, a0 - b1, 0.0)


def _attach(start: float, end: float, ipus: list[IPU], speaker: int | None) -> IPU | None:
    """The IPU (of ``speaker``, if given) with a word nearest to [start, end]."""
    best, best_gap = None, None
    for ipu in ipus:
        if speaker is not None and ipu.speaker != speaker:
            continue
        if _gap(start, end, ipu.start, ipu.end) > ATTACH_TOLERANCE_S:
            continue
        gap = min(_gap(start, end, w.start, w.end) for w in ipu.words)
        if best_gap is None or gap < best_gap:
            best, best_gap = ipu, gap
    return best


def _attach_tags(alignment: Alignment, ipus: list[IPU], keep_speaker: bool) -> tuple[list[Event], list[StyleSpan]]:
    for item in [*alignment.events, *alignment.styles]:
        speaker = item.speaker if keep_speaker and item.speaker != UNKNOWN_SPEAKER else None
        ipu = _attach(item.start, item.end, ipus, speaker)
        item.ipu = None if ipu is None else ipu.id
        if ipu is not None:
            item.speaker = ipu.speaker
        elif not keep_speaker:
            item.speaker = UNKNOWN_SPEAKER
    return alignment.events, alignment.styles


def _word_groups(ipu: IPU, events: list[Event], styles: list[StyleSpan]) -> tuple[list[str], str]:
    """Render each word of ``ipu`` with its inline tags; returns (per-word strings, trailing tags)."""
    words = ipu.words
    before: dict[int, list[str]] = defaultdict(list)
    after: dict[int, list[str]] = defaultdict(list)
    for style in styles:
        if style.speaker != ipu.speaker or _gap(style.start, style.end, ipu.start, ipu.end) > 0:
            continue
        first = next((i for i, w in enumerate(words) if w.start >= style.start - _EPS), None)
        last = max((i for i, w in enumerate(words) if w.end <= style.end + _EPS), default=None)
        if first is None or last is None or last < first:
            continue
        before[first].append(f"<sty:stt:{style.label}>")
        after[last].append(f"<sty:end:{style.label}>")
    for event in events:
        if event.ipu != ipu.id:
            continue
        index = next((i for i, w in enumerate(words) if w.start >= event.start - _EPS), len(words))
        before[index].append(f"<ev:{event.label}>")
    groups = [" ".join([*before[i], w.text, *after[i]]) for i, w in enumerate(words)]
    return groups, " ".join(before[len(words)])


def _onset_tag(ipu: IPU) -> str:
    return f"<int:{ipu.onset}>" if ipu.onset in INLINE_ONSETS else ""


def _join(*parts: str) -> str:
    return " ".join(p for p in parts if p)


def pass_agreement(a: Alignment, b: Alignment) -> float:
    """Word-sequence similarity of two passes (1.0 = identical), a cheap quality signal."""
    a_words = [normalize_word(w.text) for w in a.words]
    b_words = [normalize_word(w.text) for w in b.words]
    return round(difflib.SequenceMatcher(None, a_words, b_words, autojunk=False).ratio(), 4)


def build_conversation(cid: str, records: dict[str, dict], config: InteractionConfig) -> dict:
    """Annotate one recording from its raw pass records (keyed by pass name)."""
    primary_pass = "speaker" if "speaker" in records else next(iter(records))
    primary = records[primary_pass]
    alignment = parse_response(primary)
    conversation = {
        "id": cid,
        "audio": primary["audio"],
        "passes": {
            name: {"finish_reason": r.get("finish_reason"), "text": r.get("text", "")} for name, r in records.items()
        },
        "truncated": any(r.get("finish_reason") == "length" for r in records.values()),
    }

    if alignment.error or not alignment.words:
        # Keep the untimed per-speaker text so the transcript is not lost.
        by_speaker: dict[str, list[str]] = defaultdict(list)
        for speaker, text in split_by_speaker_tags(alignment.text):
            by_speaker[str(speaker)].append(text)
        conversation["status"] = f"alignment_failed:{alignment.error or 'no_words'}"
        conversation["speakers"] = {s: {"text": " ".join(t)} for s, t in by_speaker.items()}
        return conversation

    ipus = build_ipus(alignment.words, config)
    label_ipus(ipus, config)

    events, styles = _attach_tags(alignment, ipus, keep_speaker=True)
    agreement = None
    rich = records.get("rich") if primary_pass != "rich" else None
    if rich is not None:
        rich_alignment = parse_response(rich)
        if not rich_alignment.error and rich_alignment.words:
            rich_events, rich_styles = _attach_tags(rich_alignment, ipus, keep_speaker=False)
            events, styles = events + rich_events, styles + rich_styles
            agreement = pass_agreement(alignment, rich_alignment)
    events.sort(key=lambda e: e.start)
    styles.sort(key=lambda s: s.start)

    ipu_dicts, tagged_turns, plain_turns = [], [], []
    for ipu in ipus:
        groups, trailing = _word_groups(ipu, events, styles)
        tagged = _join(_onset_tag(ipu), *groups, trailing)
        tagged_turns.append(f"<spk:{ipu.speaker}> {tagged}")
        plain_turns.append(f"<spk:{ipu.speaker}> {ipu.text}")
        ipu_dicts.append(
            {
                "id": ipu.id,
                "speaker": ipu.speaker,
                "start": ipu.start,
                "end": ipu.end,
                "text": ipu.text,
                "text_tagged": tagged,
                "onset": ipu.onset,
                "target": ipu.target,
                "gap_s": ipu.gap_s,
                "overlap_s": ipu.overlap_s,
                "yield_s": ipu.yield_s,
                "lexical_backchannel": ipu.lexical_backchannel,
                "cut_off_by": ipu.cut_off_by,
                "words": [[w.text, w.start, w.end] for w in ipu.words],
                "word_groups": groups,
                "trailing_tags": trailing,
            }
        )

    speakers = {}
    for speaker in sorted({ipu.speaker for ipu in ipus}):
        own = [ipu for ipu in ipus if ipu.speaker == speaker]
        speakers[str(speaker)] = {
            "diarization_speaker": alignment.tag_to_diarization.get(speaker),
            "text": " ".join(ipu.text for ipu in own),
            "n_words": sum(len(ipu.words) for ipu in own),
            "speech_s": round(sum(ipu.duration for ipu in own), 3),
            "ipus": [ipu.id for ipu in own],
        }

    conversation.update(
        {
            "status": "ok",
            "pass_agreement": agreement,
            "speakers": speakers,
            "ipus": ipu_dicts,
            "events": [asdict(e) for e in events],
            "styles": [asdict(s) for s in styles],
            "diarization": [asdict(d) for d in alignment.diarization],
            "transcript": " ".join(plain_turns),
            "transcript_tagged": " ".join(tagged_turns),
            "stats": {
                "onsets": dict(Counter(ipu.onset for ipu in ipus)),
                "events": dict(Counter(e.label for e in events)),
                "styles": dict(Counter(s.label for s in styles)),
                "unknown_speaker_words": sum(w.speaker == UNKNOWN_SPEAKER for w in alignment.words),
                "unmatched_style_tags": alignment.unmatched_style_tags,
            },
        }
    )
    return conversation
