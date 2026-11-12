import pytest

from nemo_3_5_transcribe.interactions import InteractionConfig, build_ipus, label_ipus
from nemo_3_5_transcribe.parse import parse_response

from helpers import record, talk

LONG = "so what I was saying is that the plan really needs more time"


def onsets(*utterances: list[dict]) -> list[tuple[int, str]]:
    config = InteractionConfig()
    words = parse_response(record([w for u in utterances for w in u])).words
    ipus = build_ipus(words, config)
    label_ipus(ipus, config)
    return [(ipu.speaker, ipu.onset) for ipu in ipus]


@pytest.mark.parametrize(
    ("utterances", "expected"),
    [
        pytest.param(
            [talk(0, 0, 2, "how can I help"), talk(1, 2.5, 4, "my order is late")],
            [(0, "start"), (1, "switch")],
            id="clean-switch",
        ),
        pytest.param(
            [talk(0, 0, 3, LONG), talk(1, 2, 6, "sorry but that is not what I asked for")],
            [(0, "start"), (1, "interruption")],
            id="interruption-speaker-yields",
        ),
        pytest.param(
            [talk(0, 0, 8, LONG), talk(1, 2, 5, "no no wait let me explain this part")],
            [(0, "start"), (1, "failed_interruption")],
            id="interruption-attempt-ignored",
        ),
        pytest.param(
            [talk(0, 0, 6, LONG), talk(1, 2, 2.4, "mm-hmm")],
            [(0, "start"), (1, "backchannel")],
            id="overlapped-backchannel",
        ),
        pytest.param(
            [talk(0, 0, 3, LONG), talk(1, 2.7, 5, "yes that works for me")],
            [(0, "start"), (1, "overlap_turn_end")],
            id="anticipated-turn-end",
        ),
        pytest.param(
            [talk(0, 0, 3, LONG), talk(1, 0.1, 2, "hello is this support")],
            [(0, "start"), (1, "simultaneous_start")],
            id="simultaneous-start",
        ),
        pytest.param(
            [talk(0, 0, 2, "the plan is simple"), talk(1, 2.6, 2.9, "okay"), talk(0, 3.5, 5, "we ship on friday")],
            [(0, "start"), (1, "backchannel"), (0, "continue")],
            id="backchannel-in-pause",
        ),
        pytest.param(
            [talk(0, 0, 2, "what is your name"), talk(1, 2.6, 2.9, "Ravi"), talk(0, 3.5, 5, "thanks Ravi")],
            [(0, "start"), (1, "switch"), (0, "switch")],
            id="short-answer-is-a-turn",
        ),
    ],
)
def test_onset_labels(utterances, expected):
    assert onsets(*utterances) == expected


def test_interruption_features():
    config = InteractionConfig()
    words = parse_response(record(talk(0, 0, 3, LONG) + talk(1, 2, 6, "sorry but that is not what I asked"))).words
    ipus = build_ipus(words, config)
    label_ipus(ipus, config)
    speaker_a, speaker_b = ipus
    assert speaker_b.target == speaker_a.id
    assert speaker_a.cut_off_by == speaker_b.id
    assert speaker_b.yield_s == pytest.approx(speaker_a.end - 2.0)
    assert speaker_b.gap_s == pytest.approx(-speaker_b.yield_s)


def test_pause_splits_ipus():
    config = InteractionConfig(ipu_gap_s=0.5)
    words = parse_response(record(talk(0, 0, 1, "first part") + talk(0, 2, 3, "second part"))).words
    ipus = build_ipus(words, config)
    label_ipus(ipus, config)
    assert [(ipu.text, ipu.onset) for ipu in ipus] == [("first part", "start"), ("second part", "continue")]
    assert ipus[1].gap_s == pytest.approx(2 - ipus[0].end)


def test_config_from_dict_lowercases_lexicon():
    config = InteractionConfig.from_dict({"ipu_gap_s": 0.3, "backchannel_lexicon": ["Haan", "OK"]})
    assert config.ipu_gap_s == 0.3
    assert config.backchannel_lexicon == frozenset({"haan", "ok"})
