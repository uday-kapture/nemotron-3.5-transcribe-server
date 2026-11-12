"""Derive turn-taking labels (interruptions, backchannels, overlaps) from timed words.

Nemotron 3.5 Transcribe has no interruption tokens, so labels come from timing alone.
Each speaker's words are grouped into inter-pausal units (IPUs), and every IPU gets
one ``onset`` label describing how it started relative to the other speakers:

==================== =====================================================================
start                first IPU of the conversation
continue             same speaker as the current floor holder, after a pause
switch               clean turn change: the previous speaker had stopped
backchannel          short feedback ("mm-hmm", "haan") while the floor holder keeps going;
                     overlapped, or in a pause if every word is in the backchannel lexicon
interruption         starts inside another speaker's IPU, and that speaker stops first
                     (successful barge-in; ``yield_s`` is how long they kept talking)
failed_interruption  starts inside another speaker's IPU, but that speaker outlasts it
overlap_turn_end     starts in the last ``terminal_overlap_s`` of another speaker's IPU
                     (anticipated turn end, not disruptive)
simultaneous_start   both speakers started within ``simultaneous_start_s``
==================== =====================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .parse import UNKNOWN_SPEAKER, Word

OVERLAP_LABELS = frozenset({"interruption", "failed_interruption", "overlap_turn_end", "simultaneous_start"})

DEFAULT_BACKCHANNEL_LEXICON = frozenset(
    {
        # English
        "yeah", "yes", "yep", "yup", "uh-huh", "mm-hmm", "mhm", "mm", "hmm", "hm", "huh",
        "okay", "ok", "right", "sure", "alright", "exactly", "true", "cool", "great",
        # Hindi / Hinglish
        "haan", "han", "ha", "haa", "ji", "haanji", "achha", "accha", "acha", "theek", "thik", "hai",
    }
)  # fmt: skip

_NORMALIZE_RE = re.compile(r"[^\w\-']+")


@dataclass
class InteractionConfig:
    # A silence longer than this splits a speaker's words into separate IPUs.
    ipu_gap_s: float = 0.5
    simultaneous_start_s: float = 0.2
    terminal_overlap_s: float = 0.6
    backchannel_max_s: float = 1.2
    backchannel_max_words: int = 3
    # A pause backchannel needs the floor holder to resume within this many seconds.
    backchannel_resume_s: float = 2.0
    backchannel_lexicon: frozenset[str] = DEFAULT_BACKCHANNEL_LEXICON

    @classmethod
    def from_dict(cls, values: dict) -> InteractionConfig:
        values = dict(values)
        if "backchannel_lexicon" in values:
            values["backchannel_lexicon"] = frozenset(w.lower() for w in values["backchannel_lexicon"])
        return cls(**values)


@dataclass
class IPU:
    id: int
    speaker: int
    start: float
    end: float
    words: list[Word]
    onset: str = "start"
    # The other speaker's IPU this one is labelled against (overlapped, or the one it follows).
    target: int | None = None
    # start minus the target's end: negative while overlapping, the response gap otherwise.
    gap_s: float | None = None
    overlap_s: float = 0.0
    # For interruptions: how long the interrupted speaker kept talking after this onset.
    yield_s: float | None = None
    lexical_backchannel: bool = False
    cut_off_by: int | None = None

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def duration(self) -> float:
        return self.end - self.start


def normalize_word(word: str) -> str:
    return _NORMALIZE_RE.sub("", word.lower()).strip("-'")


def build_ipus(words: list[Word], config: InteractionConfig) -> list[IPU]:
    """Group each speaker's words into IPUs, returned in onset order."""
    by_speaker: dict[int, list[Word]] = {}
    for word in words:
        if word.speaker != UNKNOWN_SPEAKER:
            by_speaker.setdefault(word.speaker, []).append(word)

    groups: list[list[Word]] = []
    for speaker_words in by_speaker.values():
        speaker_words.sort(key=lambda w: (w.start, w.end))
        current = [speaker_words[0]]
        for word in speaker_words[1:]:
            if word.start - max(w.end for w in current) > config.ipu_gap_s:
                groups.append(current)
                current = []
            current.append(word)
        groups.append(current)

    groups.sort(key=lambda g: (g[0].start, g[0].speaker))
    return [
        IPU(id=i, speaker=g[0].speaker, start=g[0].start, end=max(w.end for w in g), words=g)
        for i, g in enumerate(groups)
    ]


def label_ipus(ipus: list[IPU], config: InteractionConfig) -> None:
    """Set the onset label and timing features of every IPU in place."""

    def is_short(ipu: IPU) -> bool:
        return ipu.duration <= config.backchannel_max_s and len(ipu.words) <= config.backchannel_max_words

    for ipu in ipus:
        ipu.lexical_backchannel = bool(ipu.words) and all(
            normalize_word(w.text) in config.backchannel_lexicon for w in ipu.words
        )

    # IPUs that hold the floor; backchannels never take it.
    floor: list[IPU] = []
    for index, ipu in enumerate(ipus):
        active = [other for other in floor if other.speaker != ipu.speaker and other.start <= ipu.start < other.end]
        if active:
            target = max(active, key=lambda o: o.end)
            ipu.target = target.id
            ipu.gap_s = round(ipu.start - target.end, 3)
            ipu.overlap_s = round(min(target.end, ipu.end) - ipu.start, 3)
            if ipu.start - target.start < config.simultaneous_start_s:
                ipu.onset = "simultaneous_start"
            elif is_short(ipu) and target.end >= ipu.end:
                ipu.onset = "backchannel"
            elif target.end - ipu.start <= config.terminal_overlap_s:
                ipu.onset = "overlap_turn_end"
            elif target.end < ipu.end:
                ipu.onset = "interruption"
                ipu.yield_s = round(target.end - ipu.start, 3)
                target.cut_off_by = ipu.id
            else:
                ipu.onset = "failed_interruption"
            if ipu.onset != "backchannel":
                floor.append(ipu)
            continue

        previous = max(floor, key=lambda o: o.end, default=None)
        if previous is None:
            floor.append(ipu)
            continue
        ipu.gap_s = round(ipu.start - previous.end, 3)
        if previous.speaker == ipu.speaker:
            ipu.onset = "continue"
        else:
            ipu.target = previous.id
            resumes = next((o for o in ipus[index + 1 :] if o.start >= ipu.end), None)
            if (
                is_short(ipu)
                and ipu.lexical_backchannel
                and resumes is not None
                and resumes.speaker == previous.speaker
                and resumes.start - ipu.end <= config.backchannel_resume_s
            ):
                ipu.onset = "backchannel"
                continue
            ipu.onset = "switch"
        floor.append(ipu)
