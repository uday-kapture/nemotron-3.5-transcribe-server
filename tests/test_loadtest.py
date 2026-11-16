import math

from nemo_3_5_transcribe.loadtest import summarize


def row(finished_at: float, latency_s: float = 2.0, ok: bool = True, **extra) -> dict:
    base = {"audio_s": 60.0, "completion_tokens": 100, "finish_reason": "stop", "ctc_error": None}
    return {"ok": ok, "finished_at": finished_at, "latency_s": latency_s, **base, **extra}


def test_summarize_counts_only_the_steady_second_half():
    rows = [
        row(10.0, latency_s=50.0),  # first wave: excluded
        row(60.0),
        row(80.0, latency_s=4.0, finish_reason="length", ctc_error="boom"),
        row(130.0, finish_reason="length", ctc_error="alignment_failed"),  # runaway drained after the level
        row(20.0, ok=False),  # errors count wherever they happen
    ]
    s = summarize(rows, duration_s=100.0)
    assert s["ok"] == 2  # throughput and latency: the steady window only
    assert s["errors"] == 1
    assert s["truncated"] == 2 and s["ctc_errors"] == 2  # but failures count wherever they finish
    assert s["calls_per_hour"] == 2 / 50 * 3600
    assert s["rtfx"] == 120 / 50
    assert s["p50_s"] == 2.0 and s["p95_s"] == 4.0


def test_summarize_with_no_successes_reports_nan_latency():
    s = summarize([row(70.0, ok=False)], duration_s=100.0)
    assert s["ok"] == 0 and s["errors"] == 1 and math.isnan(s["p95_s"])
