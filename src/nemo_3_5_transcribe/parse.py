"""Parse a Nemotron 3.5 Transcribe response into timed words, events and style spans.

The vLLM CTC middleware returns ``ctc_timestamps`` with words shaped
``{"word", "start", "end", "speaker"}``, where ``speaker`` is the transcript's
``<spk:N>`` index as a string. ``<ev:*>`` and ``<sty:*>`` tags survive alignment
as acoustic targets, either as their own "word" or glued to a neighbouring word
(the request sets ``spaces_between_special_tokens=False``), so both cases are split here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

RICH_TAG_RE = re.compile(r"(<(?:ev|sty):[^<>]*>)")
SPEAKER_TAG_RE = re.compile(r"<spk:(\d+)>")

UNKNOWN_SPEAKER = -1


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: int


@dataclass
class Event:
    """A non-speech event such as ``laughter`` (from ``<ev:laughter>``)."""

    label: str
    start: float
    end: float
    speaker: int
    ipu: int | None = None


@dataclass
class StyleSpan:
    """A speaking-style span between ``<sty:stt:X>`` and ``<sty:end:X>``."""

    label: str
    start: float
    end: float
    speaker: int
    ipu: int | None = None


@dataclass
class DiarSegment:
    speaker: int
    start: float
    end: float


@dataclass
class Alignment:
    text: str
    words: list[Word] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    styles: list[StyleSpan] = field(default_factory=list)
    diarization: list[DiarSegment] = field(default_factory=list)
    # transcript <spk:N> index -> Sortformer diarization column (None when unmatched)
    tag_to_diarization: dict[int, int | None] = field(default_factory=dict)
    unmatched_style_tags: int = 0
    error: str | None = None


def _speaker(value: object) -> int:
    text = str(value)
    return int(text) if text.isdigit() else UNKNOWN_SPEAKER


def _split_record(record: dict) -> list[tuple[str, float, float]]:
    """Split one aligned word record into (piece, start, end) for its text and tags.

    A tag that is its own record keeps the record's span. A tag glued before the text
    takes the record's start, one glued after takes its end.
    """
    start, end = float(record["start"]), float(record["end"])
    parts = [p for p in RICH_TAG_RE.split(record["word"]) if p]
    if len(parts) == 1:
        return [(parts[0], start, end)]
    text_positions = [i for i, p in enumerate(parts) if not RICH_TAG_RE.fullmatch(p)]
    if not text_positions:
        return [(p, start, end) for p in parts]
    first, last = text_positions[0], text_positions[-1]
    pieces = []
    for i, part in enumerate(parts):
        if i < first:
            pieces.append((part, start, start))
        elif i > last:
            pieces.append((part, end, end))
        else:
            pieces.append((part, start, end))
    return pieces


def parse_response(record: dict) -> Alignment:
    """Build an :class:`Alignment` from a raw record saved by :mod:`client`."""
    ctc = record.get("ctc_timestamps") or {}
    alignment = Alignment(
        text=record.get("text") or "",
        diarization=[
            DiarSegment(int(s["speaker"]), float(s["start"]), float(s["end"])) for s in ctc.get("diarization") or []
        ],
        tag_to_diarization={
            int(tag): None if col is None else int(col)
            for tag, col in (ctc.get("speaker_tag_to_diarization_speaker") or {}).items()
            if str(tag).isdigit()
        },
        error=ctc.get("error") if ctc else "missing_ctc_timestamps",
    )

    open_styles: dict[tuple[int, str], float] = {}
    for word_record in ctc.get("words") or []:
        speaker = _speaker(word_record.get("speaker"))
        for piece, start, end in _split_record(word_record):
            if not RICH_TAG_RE.fullmatch(piece):
                alignment.words.append(Word(piece, start, end, speaker))
                continue
            kind, _, label = piece[1:-1].partition(":")
            if kind == "ev":
                alignment.events.append(Event(label, start, end, speaker))
                continue
            edge, _, style = label.partition(":")
            key = (speaker, style)
            if edge == "stt":
                if key in open_styles:
                    alignment.unmatched_style_tags += 1
                open_styles[key] = start
            elif edge == "end" and key in open_styles:
                alignment.styles.append(StyleSpan(style, open_styles.pop(key), end, speaker))
            else:
                alignment.unmatched_style_tags += 1
    alignment.unmatched_style_tags += len(open_styles)
    alignment.styles.sort(key=lambda s: s.start)
    return alignment


def split_by_speaker_tags(text: str) -> list[tuple[int, str]]:
    """Untimed fallback: split a ``<spk:N>``-tagged transcript into (speaker, text) turns."""
    parts = SPEAKER_TAG_RE.split(text)
    turns = []
    if parts[0].strip():
        turns.append((UNKNOWN_SPEAKER, parts[0].strip()))
    for speaker, chunk in zip(parts[1::2], parts[2::2], strict=True):
        if chunk.strip():
            turns.append((int(speaker), chunk.strip()))
    return turns
