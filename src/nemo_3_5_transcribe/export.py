"""Write annotated conversations as JSON plus NeMo-style JSONL manifests.

Layout under ``out_dir``::

    conversations/<id>.json        full annotation (IPUs, words, labels, events, diarization)
    manifests/segments.jsonl       one row per speaker segment, for training ASR models
    manifests/speakers.jsonl       one row per (recording, speaker): the per-person transcript
    manifests/conversations.jsonl  one row per recording with the speaker/event/interaction-tagged transcript

Segment rows use ``offset``/``duration`` into the original audio (NeMo manifest
convention), so no audio is cut. ``text`` is the plain transcript and ``text_tagged``
adds ``<ev:*>``, ``<sty:*>`` and ``<int:*>`` tags.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import soundfile

from .annotate import INLINE_ONSETS

log = logging.getLogger(__name__)


def audio_duration(path: str) -> float | None:
    try:
        return round(soundfile.info(path).duration, 3)
    except Exception:  # unreadable or unsupported container; durations are then left out
        return None


def _segments(conversation: dict, max_segment_s: float) -> list[tuple[dict, int, int]]:
    """Split each IPU into (ipu, first_word, end_word) runs no longer than ``max_segment_s``."""
    runs = []
    for ipu in conversation["ipus"]:
        words = ipu["words"]
        first = 0
        for index in range(1, len(words)):
            if words[index][2] - words[first][1] > max_segment_s:
                runs.append((ipu, first, index))
                first = index
        runs.append((ipu, first, len(words)))
    return runs


def segment_rows(conversation: dict, duration: float | None, max_segment_s: float, pad_s: float) -> list[dict]:
    ipus = conversation["ipus"]
    rows = []
    for ipu, first, end in _segments(conversation, max_segment_s):
        words = ipu["words"][first:end]
        start, stop = words[0][1], words[-1][2]
        offset = max(0.0, start - pad_s)
        stop = stop + pad_s if duration is None else min(duration, stop + pad_s)
        tagged = ipu["word_groups"][first:end]
        if first == 0 and ipu["onset"] in INLINE_ONSETS:
            tagged = [f"<int:{ipu['onset']}>", *tagged]
        if end == len(ipu["words"]) and ipu["trailing_tags"]:
            tagged = [*tagged, ipu["trailing_tags"]]
        rows.append(
            {
                "audio_filepath": conversation["audio"],
                "offset": round(offset, 3),
                "duration": round(stop - offset, 3),
                "text": " ".join(w[0] for w in words),
                "text_tagged": " ".join(tagged),
                "speaker": ipu["speaker"],
                "conversation_id": conversation["id"],
                "ipu": ipu["id"],
                "onset": ipu["onset"] if first == 0 else "continue",
                # Another speaker is audible in this clip; filter these out for single-speaker ASR.
                "overlap": any(
                    o["speaker"] != ipu["speaker"] and o["start"] < stop and offset < o["end"] for o in ipus
                ),
            }
        )
    return rows


def speaker_rows(conversation: dict) -> list[dict]:
    ipus = {ipu["id"]: ipu for ipu in conversation.get("ipus", [])}
    rows = []
    for speaker, info in conversation["speakers"].items():
        rows.append(
            {
                "conversation_id": conversation["id"],
                "audio_filepath": conversation["audio"],
                "speaker": speaker,
                "diarization_speaker": info.get("diarization_speaker"),
                "text": info["text"],
                "n_words": info.get("n_words"),
                "speech_s": info.get("speech_s"),
                "segments": [[ipus[i]["start"], ipus[i]["end"], ipus[i]["text"]] for i in info.get("ipus", [])],
            }
        )
    return rows


def conversation_row(conversation: dict, duration: float | None) -> dict:
    return {
        "audio_filepath": conversation["audio"],
        "duration": duration,
        "text": conversation.get("transcript_tagged", ""),
        "text_plain": conversation.get("transcript", ""),
        "conversation_id": conversation["id"],
        "status": conversation["status"],
        "truncated": conversation["truncated"],
        "pass_agreement": conversation.get("pass_agreement"),
        "num_speakers": len(conversation["speakers"]),
        "onsets": conversation.get("stats", {}).get("onsets", {}),
    }


def write_outputs(conversations: list[dict], out_dir: Path, max_segment_s: float = 20.0, pad_s: float = 0.1) -> None:
    conv_dir = out_dir / "conversations"
    manifest_dir = out_dir / "manifests"
    conv_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (
        open(manifest_dir / "segments.jsonl", "w") as segments_file,
        open(manifest_dir / "speakers.jsonl", "w") as speakers_file,
        open(manifest_dir / "conversations.jsonl", "w") as conversations_file,
    ):
        for conversation in conversations:
            duration = audio_duration(conversation["audio"])
            conversation["duration"] = duration
            (conv_dir / f"{conversation['id']}.json").write_text(json.dumps(conversation, ensure_ascii=False, indent=1))
            conversations_file.write(json.dumps(conversation_row(conversation, duration), ensure_ascii=False) + "\n")
            for row in speaker_rows(conversation):
                speakers_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if conversation["status"] == "ok":
                for row in segment_rows(conversation, duration, max_segment_s, pad_s):
                    segments_file.write(json.dumps(row, ensure_ascii=False) + "\n")
