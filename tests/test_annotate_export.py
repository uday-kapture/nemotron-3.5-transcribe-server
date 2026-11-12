import json

from nemo_3_5_transcribe.annotate import build_conversation
from nemo_3_5_transcribe.export import write_outputs
from nemo_3_5_transcribe.interactions import InteractionConfig

from helpers import record, talk

AGENT = "thank you for calling how can I help you today"
CUSTOMER = "sorry but my order still has not arrived"


def two_pass_records() -> dict[str, dict]:
    speaker_words = talk(0, 0, 4, AGENT) + talk(1, 3, 6, CUSTOMER)
    # The rich pass is aligned as one speaker; laughter falls inside the customer's speech.
    rich_words = talk(0, 0, 4, AGENT) + talk(0, 3, 6, CUSTOMER)
    rich_words.append({"word": "<ev:laughter>", "start": 4.6, "end": 4.6, "speaker": "0"})
    return {"speaker": record(speaker_words), "rich": record(rich_words, pass_name="rich")}


def test_two_pass_merge():
    conversation = build_conversation("call", two_pass_records(), InteractionConfig())

    assert conversation["status"] == "ok"
    assert conversation["pass_agreement"] == 1.0
    agent, customer = conversation["ipus"]
    assert customer["onset"] == "interruption"
    assert agent["cut_off_by"] == customer["id"]

    [laughter] = conversation["events"]
    assert (laughter["label"], laughter["speaker"], laughter["ipu"]) == ("laughter", 1, customer["id"])
    assert customer["text_tagged"].startswith("<int:interruption> sorry but my order still")
    assert "<ev:laughter>" in customer["text_tagged"]
    assert conversation["transcript_tagged"].startswith(f"<spk:0> {AGENT} <spk:1> <int:interruption>")

    assert conversation["speakers"]["0"]["text"] == AGENT
    assert conversation["speakers"]["1"]["diarization_speaker"] == 0


def test_alignment_failure_keeps_untimed_speaker_text():
    failed = record([], text="<spk:0>Hello.<spk:1>Hi there.<spk:0>Bye.", error="alignment_failed")
    conversation = build_conversation("call", {"speaker": failed}, InteractionConfig())
    assert conversation["status"] == "alignment_failed:alignment_failed"
    assert conversation["speakers"] == {"0": {"text": "Hello. Bye."}, "1": {"text": "Hi there."}}


def test_write_outputs(tmp_path):
    conversation = build_conversation("call", two_pass_records(), InteractionConfig())
    write_outputs([conversation], tmp_path, max_segment_s=2.0, pad_s=0.1)

    saved = json.loads((tmp_path / "conversations" / "call.json").read_text())
    assert saved["id"] == "call" and saved["duration"] is None  # the test audio does not exist

    segments = [json.loads(line) for line in (tmp_path / "manifests" / "segments.jsonl").read_text().splitlines()]
    # Each IPU is ~4 s and max_segment_s is 2, so both are split.
    assert len(segments) > 2
    assert " ".join(s["text"] for s in segments if s["speaker"] == 1) == CUSTOMER
    first_customer = next(s for s in segments if s["speaker"] == 1)
    assert first_customer["onset"] == "interruption"
    assert first_customer["text_tagged"].startswith("<int:interruption>")
    assert first_customer["overlap"] is True
    assert segments[-1]["overlap"] is False
    assert all(s["duration"] <= 2.0 + 2 * 0.1 + 1e-6 for s in segments)

    [speaker_0, speaker_1] = [
        json.loads(line) for line in (tmp_path / "manifests" / "speakers.jsonl").read_text().splitlines()
    ]
    assert (speaker_0["speaker"], speaker_0["text"]) == ("0", AGENT)
    assert speaker_1["segments"][0][2] == CUSTOMER

    [row] = [json.loads(line) for line in (tmp_path / "manifests" / "conversations.jsonl").read_text().splitlines()]
    assert row["num_speakers"] == 2 and row["onsets"] == {"start": 1, "interruption": 1}
