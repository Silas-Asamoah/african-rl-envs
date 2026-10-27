from sika_interp.spans import Span, message_spans, sentences, to_tokens

MESSAGES = [
    {"role": "system", "content": "You are Sika. Be careful."},
    {"role": "user", "content": "Send GH₵50 to Kwesi."},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should check contacts.",
        "tool_calls": [{"function": {"name": "list_contacts", "arguments": "{}"}}],
    },
    {"role": "tool", "content": "Kwesi Mensah 0241234567"},
]
TEXT = (
    "<s>[SYS]You are Sika. Be careful.[/SYS][U]Send GH₵50 to Kwesi.[/U]"
    "[A]<call>list_contacts</call>[/A][T]Kwesi Mensah 0241234567[/T]"
)


def test_spans_follow_the_messages_in_order():
    spans = message_spans(TEXT, MESSAGES)
    assert [(s.message, s.part) for s in spans] == [(0, "content"), (1, "content"), (2, "call"), (3, "content")]
    assert TEXT[spans[1].start : spans[1].end] == "Send GH₵50 to Kwesi."


def test_dropped_reasoning_is_skipped():
    """This template drops earlier turns' reasoning; the call is still found."""
    spans = message_spans(TEXT, MESSAGES)
    assert not [s for s in spans if s.part == "reasoning"]


def test_a_long_piece_found_by_its_opening():
    long = "Rule one. " * 10
    text = "[SYS]" + long[:80] + "&nbsp;" + long[80:] + "[/SYS][U]Hi[/U]"  # escaped by the template
    spans = message_spans(text, [{"role": "system", "content": long}, {"role": "user", "content": "Hi"}])
    assert spans[0].start == 5 and spans[0].end > spans[0].start + 60
    assert text[spans[1].start : spans[1].end] == "Hi"


def test_sentences_split_at_ends_and_newlines():
    text = "First one. Second?\nThird"
    got = sentences(text, Span(0, len(text), 0, "content"))
    assert [text[s.start : s.end] for s in got] == ["First one.", "Second?", "Third"]


def test_token_ranges_cover_each_span():
    # One token per four characters.
    assert to_tokens([Span(0, 4, 0, "content"), Span(5, 9, 0, "content")], lambda c: c // 4) == [(0, 1), (1, 3)]
