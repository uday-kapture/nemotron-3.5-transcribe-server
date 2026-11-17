from collections import Counter

from nemo_3_5_transcribe.scriptcheck import foreign_scripts, snippet


def test_hindi_english_tags_and_digits_are_expected():
    assert not foreign_scripts("<spk:0> जी मैम, hello। 500 रुपये <ev:laughter> mam")


def test_wrong_scripts_are_counted_by_script():
    assert foreign_scripts("जी मैम 你好吗 hello") == Counter({"CJK": 3})
    assert foreign_scripts("مرحبا") == Counter({"ARABIC": 5})


def test_snippet_centres_on_the_first_foreign_letter():
    text = "a" * 200 + "你" + "b" * 200
    assert "你" in snippet(text) and len(snippet(text)) <= 120
    assert snippet("hello") == "hello"
