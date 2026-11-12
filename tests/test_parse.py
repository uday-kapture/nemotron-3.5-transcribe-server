from nemo_3_5_transcribe.parse import UNKNOWN_SPEAKER, parse_response, split_by_speaker_tags

from helpers import record


def test_standalone_and_glued_tags():
    words = [
        {"word": "<ev:laughter>", "start": 0.0, "end": 0.4, "speaker": "0"},
        {"word": "<sty:stt:positive_affect>That's", "start": 0.5, "end": 0.7, "speaker": "0"},
        {"word": "great!<sty:end:positive_affect>", "start": 0.8, "end": 1.1, "speaker": "0"},
        {"word": "Okay<ev:breath>", "start": 1.5, "end": 1.7, "speaker": "1"},
    ]
    alignment = parse_response(record(words))

    assert [(w.text, w.speaker) for w in alignment.words] == [("That's", 0), ("great!", 0), ("Okay", 1)]
    assert [(e.label, e.start, e.end, e.speaker) for e in alignment.events] == [
        ("laughter", 0.0, 0.4, 0),
        ("breath", 1.7, 1.7, 1),
    ]
    assert [(s.label, s.start, s.end, s.speaker) for s in alignment.styles] == [("positive_affect", 0.5, 1.1, 0)]
    assert alignment.tag_to_diarization == {0: 1, 1: 0}
    assert alignment.error is None


def test_unmatched_style_tags_are_counted():
    words = [
        {"word": "<sty:end:flat>", "start": 0.0, "end": 0.1, "speaker": "0"},
        {"word": "<sty:stt:fear_urgency>Help", "start": 0.2, "end": 0.5, "speaker": "0"},
    ]
    alignment = parse_response(record(words))
    assert alignment.styles == []
    assert alignment.unmatched_style_tags == 2


def test_untagged_words_get_unknown_speaker():
    alignment = parse_response(record([{"word": "hi", "start": 0.0, "end": 0.2, "speaker": "None"}]))
    assert alignment.words[0].speaker == UNKNOWN_SPEAKER


def test_alignment_error_and_missing_timestamps():
    assert parse_response(record([], error="alignment_failed")).error == "alignment_failed"
    assert parse_response({"audio": "a.wav", "text": "hi"}).error == "missing_ctc_timestamps"


def test_split_by_speaker_tags():
    assert split_by_speaker_tags("hm <spk:0>Hello there.<spk:1> Hi!") == [
        (UNKNOWN_SPEAKER, "hm"),
        (0, "Hello there."),
        (1, "Hi!"),
    ]
