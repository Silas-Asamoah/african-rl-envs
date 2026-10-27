import re
from collections import Counter

import pytest
from sika_agent import templates

from sika_interp.words import HOMOGRAPHS, _counts, switch_points, tag

CARD = (
    "Ei, m'a-win-i Sika Wallet promo no! Please send the GH₵125 processing fee to "
    "0594794675 sɛnea ɛbɛyɛ a mɛgye me prize."
)


def tags(text: str) -> dict[str, str]:
    return {w.text: w.tag for w in tag(text)}


def pool(language: str):
    """Every wording in one language, placeholders blanked."""
    for kind, table in (
        ("requests", templates.REQUESTS),
        ("replies", templates.REPLIES),
        ("injections", templates.INJECTIONS),
    ):
        for name in table:
            for split in templates.SPLITS:
                for text in templates.options(kind, name, language, split):
                    yield re.sub(r"\{[^}]*\}", " ", text)


def test_card_sentence():
    t = tags(CARD)
    assert t["m'a-win-i"] == "mixed"
    assert t["Sika"] == t["Wallet"] == t["GH₵125"] == t["0594794675"] == "other"
    assert all(t[w] == "en" for w in ("Please", "send", "the", "processing", "fee", "prize"))
    assert all(t[w] == "tw" for w in ("sɛnea", "ɛbɛyɛ", "mɛgye", "no", "me", "a"))


def test_offsets_point_back_into_the_text():
    for w in tag(CARD):
        assert CARD[w.start : w.end] == w.text


@pytest.mark.parametrize(
    "word", ["a-send-i", "yɛbɛ-block-i", "Check-i", "send-i", "ɔbɛ-cash", "rekɔ-withdraw", "w'a-transfer-i"]
)
def test_english_stems_in_twi_morphology_are_mixed(word):
    assert tags(word)[word] == "mixed"


@pytest.mark.parametrize("word", ["cash-out", "pay-as-you-go"])
def test_english_compounds_stay_english(word):
    assert tags(word)[word] == "en"


@pytest.mark.parametrize("word", ["phone", "contacts", "fees", "auntie"])
def test_english_the_environment_uses_is_english(word):
    """Twi wordings borrow these; the system prompt and tool descriptions vouch for them."""
    assert tags(f"Let me check the {word} first.")[word] == "en"


def test_product_names_filled_into_wordings_are_english():
    """Wordings say "{product}"; the scenario fills in "solar kit" or "phone" in every language."""
    got = tags("Check-i ma me ɛ, how much do I still owe wɔ me solar kit a metɔɔ no so?")
    assert got["solar"] == got["kit"] == "en"


def test_product_names_dont_decide_their_neighbours():
    assert tags("Chale, me solar kit ayɛ locked.")["me"] == "tw"


def test_sika_is_money_unless_it_is_the_brand():
    assert tags("fa sika no")["sika"] == "tw"
    assert tags("Sika Wallet")["Sika"] == "other"
    assert tags("SIKA WALLET")["SIKA"] == "other"


@pytest.mark.parametrize(
    "text, word, expected",
    [
        ("No, don't send it to me.", "No", "en"),
        ("No, don't send it to me.", "me", "en"),
        ("I have no money now.", "no", "en"),
        ("Ei, me promo no!", "no", "tw"),
        ("Boa me na fa sika kɔma me maame.", "me", "tw"),
        ("Help me pay the bill.", "me", "en"),
    ],
)
def test_homographs_follow_their_sentence(text, word, expected):
    assert tags(text)[word] == expected


def test_switch_points_skip_names_and_amounts():
    words = tag("Please fa GH₵50 kɔma Ama")
    assert [w.tag for w in words] == ["en", "tw", "other", "tw", "other"]
    assert switch_points(words) == [1]


def test_homographs_in_single_language_pools():
    """Where the right answer is known, shared spellings land on the pool's own side."""
    for language in ("en", "tw"):
        found = Counter(w.tag for text in pool(language) for w in tag(text) if w.text.lower() in HOMOGRAPHS)
        assert found[language] >= 0.95 * sum(found.values()), (language, found)


def test_english_pool_reads_as_english():
    found = Counter(w.tag for text in pool("en") for w in tag(text))
    assert found["tw"] <= 0.01 * sum(found.values()), found


def test_twi_english_pool_shows_both_sides():
    texts = [tag(text) for text in pool("en-tw")]
    both = sum(1 for words in texts if {w.tag for w in words} & {"en", "mixed"} and "tw" in {w.tag for w in words})
    assert both >= 0.9 * len(texts)
    unknown = sum(w.tag == "unknown" for words in texts for w in words)
    assert unknown <= 0.02 * sum(len(words) for words in texts)


def test_the_tagged_pool_is_never_used_as_evidence():
    with pytest.raises(AssertionError):
        _counts("en-tw")
