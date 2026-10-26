"""Word-level language tags for Twi–English text, for following a model across switch points.

Each word gets one tag:
- `en` or `tw`;
- `mixed`: an English stem inside Twi morphology, such as "a-send-i" or "yɛbɛ-block-i";
- `other`: names, brands, amounts and phone numbers, which belong to neither side;
- `unknown`: a word the evidence below can't place.

The evidence, in order:
1. digits and the cedi sign, then known names and brands (case-sensitive, so "Sika" is the
   brand and "sika" is Twi for money): `other`;
2. Twi prefixes or the "-i" ending around a hyphenated English stem: `mixed`;
3. the Twi letters ɛ and ɔ: `tw`;
4. the wording pools. A word used in the English pool is English, even where Twi wordings
   borrow it ("fee", "PIN", "airtime"): those borrowings are the English islands this study
   follows. A word only the Twi pool uses is Twi. The Twi–English pool is never consulted,
   so the tags don't learn from the text they tag;
5. words spelled the same in both languages ("me", "a", "so", "no", "o") take the side of
   the nearest unambiguous words in their sentence when those agree, else the side the
   pools favour. "No" opening a sentence is English and before a pause is the Twi
   determiner ("promo no!").

These are heuristics: check a sample by ear before relying on them.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from functools import cache

from sika_agent import templates
from sika_agent.scenarios import DAY_NAMES, FEMALE, MALE, SURNAMES
from sika_agent.wallet import LENDER

TAGS = ("en", "tw", "mixed", "other", "unknown")
SIDES = ("en", "tw", "mixed")
"""Tags that take part in switching."""
TWI_LETTERS = frozenset("ɛɔƐƆ")
BRANDS = ("Sika Wallet", "ECG Prepaid", "Ghana Water", "DStv", LENDER)
PROPER = frozenset(MALE + FEMALE + SURNAMES).union(*DAY_NAMES, (w for b in BRANDS for w in b.split()))
"""Names and brand words, matched case-sensitively."""
TWI_PREFIXES = frozenset({"a", "n", "me", "men", "mɛ", "ɔ", "ɛ", "ɛn", "bɛ", "ɛbɛ", "ɔbɛ", "yɛbɛ", "wobɛ", "akɔ", "rekɔ", "m'a", "w'a", "y'a"})  # fmt: skip
"""Pieces before a hyphen that put Twi morphology on an English stem, as in "a-send-i"."""
HOMOGRAPHS = frozenset({"me", "a", "so", "no", "o"})
"""Common in both languages: "me" is Twi for my/me, "a" a Twi relative marker, "so" Twi for
on/also, "no" the Twi determiner, "o" a particle both use."""
TWI_WORDS = frozenset({"sika"})
"""Twi however often the English pool uses them (English wordings say "my sika" too)."""
EDGE = "\"'“”‘’.,!?;:()…"
"""Punctuation stripped from the ends of a word."""
SENTENCE_END = (".", "!", "?")
PLACEHOLDER = re.compile(r"\{[^}]*\}")
MIN_ENGLISH = 2
"""Uses in the English pool that make a word English (one stray use doesn't)."""


@dataclass(frozen=True)
class Word:
    text: str
    start: int
    """Character offsets into the tagged text, for lining words up with model tokens."""
    end: int
    tag: str
    sentence: int
    """Index of the sentence the word is in."""


def tag(text: str) -> list[Word]:
    """Split `text` on whitespace, strip edge punctuation and tag each word."""
    words, sentence, opens = [], 0, True
    for match in re.finditer(r"\S+", text):
        raw = match.group()
        core = raw.strip(EDGE)
        if core:
            start = match.start() + raw.index(core)
            label = word_tag(core)
            if core.lower() == "no":
                # English opening a sentence ("No, don't"); the Twi determiner before a pause
                # ("promo no!"); otherwise its neighbours decide.
                label = "en" if opens else "tw" if raw[-1] in EDGE else "?"
            words.append(Word(core, start, start + len(core), label, sentence))
            opens = False
        if raw.rstrip("\"'”’)").endswith(SENTENCE_END):
            sentence, opens = sentence + 1, True
    return [replace(w, tag=_in_context(words, i)) if w.tag == "?" else w for i, w in enumerate(words)]


def switch_points(words: list[Word]) -> list[int]:
    """Indices of words whose side (`en`, `tw` or `mixed`) differs from the previous such
    word. `other` and `unknown` words are skipped, so a name between two Twi words is not a
    switch."""
    points, side = [], None
    for i, word in enumerate(words):
        if word.tag in SIDES:
            if side is not None and word.tag != side:
                points.append(i)
            side = word.tag
    return points


def word_tag(word: str) -> str:
    """The tag for one word on its own, or "?" for a homograph that needs its sentence."""
    if any(ch.isdigit() for ch in word) or "₵" in word:
        return "other"
    if word in PROPER or (word.isupper() and word.title() in PROPER):
        return "other"
    lower = word.lower()
    pieces = lower.split("-")
    if len(pieces) > 1 and _is_mixed(pieces):
        return "mixed"
    if TWI_LETTERS & set(word) or lower in TWI_WORDS:
        return "tw"
    if lower in HOMOGRAPHS:
        return "?"
    side = _side(lower)
    if side is None and len(pieces) > 1:
        sides = {s for p in pieces if (s := _side(p))}
        side = sides.pop() if len(sides) == 1 else None
    if side is None:
        return "other" if word[:1].isupper() else "unknown"
    return side


def _in_context(words: list[Word], i: int) -> str:
    """A homograph's tag: the side of the nearest unambiguous words in its sentence when
    they agree, else the side the pools favour. Borrowed words like "fee", used in both
    pools, don't count as evidence."""
    same = [w for w in words if w.sentence == words[i].sentence]
    j = same.index(words[i])
    left = next((s for w in reversed(same[:j]) if (s := _clear_side(w.text))), None)
    right = next((s for w in same[j + 1 :] if (s := _clear_side(w.text))), None)
    sides = {s for s in (left, right) if s}
    if len(sides) == 1:
        return sides.pop()
    return "tw" if _lean(words[i].text.lower()) > 0 else "en"


def _clear_side(word: str) -> str | None:
    """`tw` or `en` for a word that only one language uses (Twi letters, Twi-only words,
    English words Twi wordings rarely borrow), else None."""
    lower = word.lower()
    if TWI_LETTERS & set(word) or lower in TWI_WORDS:
        return "tw"
    en, tw = _counts("en")[lower], _counts("tw")[lower]
    if tw and not en:
        return "tw"
    if en >= MIN_ENGLISH and tw * 4 <= en:
        return "en"
    return None


def _is_mixed(pieces: list[str]) -> bool:
    """Twi prefixes and/or the "-i" ending around a stem with no Twi letters."""
    stem = pieces[:-1] if pieces[-1] == "i" else pieces
    while len(stem) > 1 and (stem[0] in TWI_PREFIXES or TWI_LETTERS & set(stem[0])):
        stem = stem[1:]
    return len(stem) < len(pieces) and not TWI_LETTERS & set("".join(stem))


def _side(word: str) -> str | None:
    """English if the English pool uses the word, Twi if only the Twi pool does."""
    if _counts("en")[word] >= MIN_ENGLISH:
        return "en"
    if _counts("tw")[word]:
        return "tw"
    return "en" if _counts("en")[word] else None


def _lean(word: str) -> float:
    """Log-odds that a lower-case word is Twi rather than English, from the wording pools."""
    tw, en = _counts("tw"), _counts("en")
    return math.log((tw[word] + 0.5) / tw.total()) - math.log((en[word] + 0.5) / en.total())


@cache
def _counts(language: str) -> Counter[str]:
    """Word counts over every wording in one language, both splits, with placeholders and
    brand names removed."""
    assert language in ("en", "tw"), "the Twi–English pool is what gets tagged"
    counts: Counter[str] = Counter()
    for kind, table in (
        ("requests", templates.REQUESTS),
        ("replies", templates.REPLIES),
        ("injections", templates.INJECTIONS),
    ):
        for name in table:
            for split in templates.SPLITS:
                for text in templates.options(kind, name, language, split):
                    for brand in BRANDS:
                        text = text.replace(brand, " ")
                    words = (w.strip(EDGE).lower() for w in PLACEHOLDER.sub(" ", text).split())
                    counts.update(w for w in words if w)
    counts.update(rel[language].lower() for rel in templates.RELATIONS.values())
    return counts
