"""Where each message, and each sentence in it, sits in a rendered conversation.

Chat templates wrap messages in role markers and may drop parts (Qwen keeps only the latest
turn's reasoning). So spans are found by searching the rendered text for each message's own
pieces, in order: its content, reasoning and tool-call names. Character spans become token
spans through `to_token`, the first token whose decoded prefix passes a character offset.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

SENTENCE_END = re.compile(r"(?<=[^\W\d_][.!?])\s+|(?<=[)\"'”’][.!?])\s+|\n+")
"""A sentence ends after a word's (or a closing quote's) .!? and a space, or at a line break.
Not after a number's full stop: a numbered rule ("1. Only move money…") stays one sentence."""


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    """Character offsets in the rendered text (end exclusive)."""
    message: int
    """Index of the message it belongs to; len(messages) for the response being rendered."""
    part: str
    """content, reasoning or call."""


def pieces(message: dict) -> list[tuple[str, str]]:
    """(part, text) a message shows, in the order templates render them."""
    out = []
    if message.get("reasoning_content"):
        out.append(("reasoning", message["reasoning_content"].strip()))
    if isinstance(message.get("content"), str) and message["content"].strip():
        out.append(("content", message["content"].strip()))
    for call in message.get("tool_calls") or []:
        fn = call.get("function", call)
        out.append(("call", fn["name"]))
    return [(p, t) for p, t in out if t]


def message_spans(text: str, messages: list[dict]) -> list[Span]:
    """Each message's pieces found in `text` in order; a piece the template dropped (or
    escaped) is skipped. A long piece not found whole is looked for by its first 60
    characters, and then spans to the next piece."""
    spans, cursor = [], 0
    for i, message in enumerate(messages):
        for part, piece in pieces(message):
            at = text.find(piece, cursor)
            end = at + len(piece)
            if at < 0 and len(piece) > 60:
                at = text.find(piece[:60], cursor)
                end = at + 60
            if at < 0:
                continue
            spans.append(Span(at, end, i, part))
            cursor = end
    # A piece found only by its opening runs on to where the next piece starts.
    out = []
    for k, s in enumerate(spans):
        piece = dict(pieces(messages[s.message])).get(s.part, "")
        if s.end - s.start < len(piece):
            limit = spans[k + 1].start if k + 1 < len(spans) else len(text)
            s = Span(s.start, max(s.end, min(s.start + len(piece), limit)), s.message, s.part)
        out.append(s)
    return out


def sentences(text: str, span: Span) -> list[Span]:
    """A span split at sentence ends and line breaks; empty pieces dropped."""
    out, start = [], span.start
    chunk = text[span.start : span.end]
    for m in SENTENCE_END.finditer(chunk):
        end = span.start + m.start()
        if text[start:end].strip():
            out.append(Span(start, end, span.message, span.part))
        start = span.start + m.end()
    if text[start : span.end].strip():
        out.append(Span(start, span.end, span.message, span.part))
    return out


def to_tokens(spans: list[Span], to_token: Callable[[int], int]) -> list[tuple[int, int]]:
    """Token ranges [start, end) for character spans; `to_token(c)` is the index of the
    token that holds character c."""
    out = []
    for s in spans:
        a, b = to_token(s.start), to_token(max(s.start, s.end - 1)) + 1
        out.append((a, max(a + 1, b)))
    return out
