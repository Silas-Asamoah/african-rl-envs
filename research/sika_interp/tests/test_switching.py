from sika_interp.switching import MASK, align, masked, token_tags

TEXT = "Mesrɛ wo, send GH₵30 kɔma Ama."


def prefixes(pieces: list[str]) -> list[str]:
    out, acc = [], ""
    for p in pieces:
        acc += p
        out.append(acc)
    return out


def test_tokens_land_on_their_words_even_when_split():
    pieces = ["Mes", "rɛ", " wo", ",", " send", " GH₵", "30", " k", "ɔma", " Ama", "."]
    owner = align(TEXT, prefixes(pieces))
    words = ["Mesrɛ", "wo", "send", "GH₵30", "kɔma", "Ama"]
    assert [words[i] if i is not None else None for i in owner] == [
        "Mesrɛ", "Mesrɛ", "wo", None, "send", "GH₵30", "GH₵30", "kɔma", "kɔma", "Ama", None,
    ]  # fmt: skip


def test_a_leading_space_from_the_template_is_allowed():
    pieces = [" Mes", "rɛ", " wo", ",", " send", " GH₵30", " kɔma", " Ama", "."]
    owner = align(TEXT, prefixes(pieces))
    assert owner[0] == owner[1] == 0 and owner[4] == 2


def test_switch_points_open_on_the_first_token_of_the_switching_word():
    pieces = ["Mes", "rɛ", " wo", ",", " send", " GH₵", "30", " k", "ɔma", " Ama", "."]
    tags, opens = token_tags(TEXT, prefixes(pieces))
    assert tags[:2] == ["tw", "tw"] and tags[4] == "en" and tags[5] == "other"
    assert [i for i, o in enumerate(opens) if o] == [4, 7]  # "send" (tw→en), "kɔma" (en→tw)


def test_masking_one_language_keeps_the_rest_of_the_message():
    assert masked(TEXT, {"en"}) == f"Mesrɛ wo, {MASK} GH₵30 kɔma Ama."
    assert masked(TEXT, {"tw"}) == f"{MASK} {MASK}, send GH₵30 {MASK} Ama."
