"""Builders for synthetic responses in the vLLM ``ctc_timestamps`` format."""


def talk(speaker: int | None, start: float, end: float, text: str) -> list[dict]:
    """Aligned word records spreading ``text`` evenly over [start, end]."""
    words = text.split()
    step = (end - start) / len(words)
    return [
        {
            "word": word,
            "start": round(start + i * step, 3),
            "end": round(start + (i + 0.8) * step, 3),
            "speaker": str(speaker),
        }
        for i, word in enumerate(words)
    ]


def record(words: list[dict], text: str = "", pass_name: str = "speaker", error: str | None = None) -> dict:
    words = sorted(words, key=lambda w: (w["start"], w["end"]))
    return {
        "audio": "/data/call.wav",
        "pass": pass_name,
        "text": text,
        "finish_reason": "stop",
        "ctc_timestamps": {
            "words": [] if error else words,
            "diarization": [{"speaker": 0, "start": 0.0, "end": 1.0}],
            "speaker_tag_to_diarization_speaker": {"0": 1, "1": 0},
            "error": error,
        },
    }
