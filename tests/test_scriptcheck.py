from collections import Counter

from nemo_3_5_transcribe.scriptcheck import devanagari_share, foreign_scripts, is_wrong_language, snippet


def test_english_for_a_hindi_call_is_flagged_even_though_latin_is_an_expected_script():
    assert devanagari_share("<spk:0> Yeah. Yeah, we are. <spk:1> Huh, what") == 0
    # tags like <spk:0> don't count as Latin letters; vowel signs aren't letters, so shares run low
    assert devanagari_share("<spk:0> हाँ जी बताइए। नमस्ते। payment जमा कहाँ करनी है") > 0.6
    assert devanagari_share("<spk:0> ...") is None
    assert is_wrong_language({"foreign": {}, "deva": 0.0})
    assert is_wrong_language({"foreign": {"BENGALI": 5}, "deva": 0.9})
    assert not is_wrong_language({"foreign": {}, "deva": 0.62})


def test_hindi_english_tags_and_digits_are_expected():
    assert not foreign_scripts("<spk:0> जी मैम, hello। 500 रुपये <ev:laughter> mam")


def test_wrong_scripts_are_counted_by_script():
    assert foreign_scripts("जी मैम 你好吗 hello") == Counter({"CJK": 3})
    assert foreign_scripts("مرحبا") == Counter({"ARABIC": 5})


def test_snippet_centres_on_the_first_foreign_letter():
    text = "a" * 200 + "你" + "b" * 200
    assert "你" in snippet(text) and len(snippet(text)) <= 120
    assert snippet("hello") == "hello"
