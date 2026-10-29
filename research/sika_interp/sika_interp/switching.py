"""Token-level view of a Twi–English request: which word each token belongs to, which
tokens open a switch between languages, and versions of the request with one language's
words masked.

Models read tokens; the tagger (`words.py`) reads words. `align` lines the two up from the
text each prefix of the message's tokens decodes to, so it works with any tokenizer.
Masking replaces words, not tokens, and never writes new Twi: it removes a language's words
to see what the model loses without them.
"""

from sika_interp.words import switch_points, tag

MASK = "…"
"""What a masked word becomes: punctuation the tagger and the models both treat as empty."""


def align(text: str, prefixes: list[str]) -> list[int | None]:
    """For each token of `text`, the index of the word (in `tag(text)`) it overlaps first,
    or None for tokens between words. `prefixes[k]` is what the first k+1 tokens of the
    message decode to; leading text before `text` (a space, say) is allowed."""
    words = tag(text)
    full = prefixes[-1] if prefixes else ""
    offset = max(full.find(text), 0)
    out: list[int | None] = []
    start = 0
    for prefix in prefixes:
        end = max(len(prefix) - offset, start)
        out.append(next((i for i, w in enumerate(words) if w.start < end and w.end > start), None))
        start = end
    return out


def token_tags(text: str, prefixes: list[str]) -> tuple[list[str], list[bool]]:
    """Each token's word tag ("" between words) and whether it opens a switch point (the
    first token of a word whose side differs from the previous word's)."""
    words = tag(text)
    switches = set(switch_points(words))
    owner = align(text, prefixes)
    tags, opens, seen = [], [], set()
    for i in owner:
        tags.append(words[i].tag if i is not None else "")
        opens.append(i is not None and i in switches and i not in seen)
        if i is not None:
            seen.add(i)
    return tags, opens


def masked(text: str, sides: set[str]) -> str:
    """`text` with every word tagged with one of `sides` replaced by MASK, keeping
    punctuation and spacing."""
    out, at = [], 0
    for w in tag(text):
        out.append(text[at : w.start])
        out.append(MASK if w.tag in sides else w.text)
        at = w.end
    out.append(text[at:])
    return "".join(out)

