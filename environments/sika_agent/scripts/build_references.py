"""Collect Twi reference material for phrasing generation from Ghana NLP's datasets.

    uv run --no-project --with duckdb python scripts/build_references.py

Streams only the text columns of the Parquet files over HTTP (nothing large is stored) and
writes `scripts/.cache/references.json` (gitignored):

- `code_switching`: KasaSpeech transcripts (Apache-2.0), money-related ones first. Real
  Twi–English code-switching, e.g. "Confirm-i account name no ansa na w'a-send-i sika no."
- `glossary`: English -> Twi finance terms from the dictionary lookups in
  `twi-translation-tool-calling` (CC-BY-NC-4.0). Reference only: some senses are wrong
  ("bill" -> a legislative bill), so the generator is told to choose.
- `dialogue`: short money-related excerpts from `pristine-twi-english` dialogues
  (CC-BY-NC-4.0), as style reference only.

Only KasaSpeech text may be reused verbatim; the non-commercial material guides the
generator and is never copied into the environment.
"""

import collections
import json
import random
import re
from pathlib import Path

import duckdb

HF = "https://huggingface.co/datasets/ghananlpcommunity"
KASA = f"{HF}/Ghana_English-Twi_Code-switching_Speech/resolve/refs%2Fconvert%2Fparquet/default"
KASA_FILES = [f"train/{i:04d}.parquet" for i in range(7)] + ["validation/0000.parquet", "test/0000.parquet"]
TOOLCALL = f"{HF}/twi-translation-tool-calling/resolve/refs%2Fconvert%2Fparquet/default"
PRISTINE = f"{HF}/pristine-twi-english/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
OUT = Path(__file__).parent / ".cache" / "references.json"

MONEY = re.compile(
    r"\b(sika|cedis?|money|momo|send|sende|tua|pay|ka|balance|account|airtime|credit|alert|transfer|"
    r"withdraw|loan|bill|bank|lend|borrow|debt|phone|number|confirm|verify|otp|pin)\b",
    re.I,
)
GLOSSARY_TERMS = [
    "money", "pay", "payment", "debt", "loan", "owe", "send", "receive", "balance", "fee", "charge",
    "account", "number", "message", "mistake", "thief", "fraud", "cheat", "secret", "prize", "win",
    "refund", "return", "water", "electricity", "lock", "unlock", "officer", "urgent", "quickly",
    "enough", "remain", "remaining", "short", "exactly", "extra", "cash", "buy", "sister", "brother",
    "mother", "landlord", "friend", "aunt", "check", "confirm", "agent", "transfer", "today",
    "yesterday", "ignore", "instruction", "please", "help", "how much", "finish", "complete",
]  # fmt: skip


def main() -> None:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    transcripts: set[str] = set()
    for f in KASA_FILES:
        for (text,) in con.execute(f"SELECT DISTINCT transcript FROM read_parquet('{KASA}/{f}')").fetchall():
            transcripts.add(text.strip().strip('",').strip("“”\"").strip())
    money = sorted(t for t in transcripts if MONEY.search(t) and len(t) < 160)
    rest = sorted(t for t in transcripts if t not in money and len(t) < 120)
    random.Random(0).shuffle(rest)

    glossary: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for split in ("train", "validation", "test"):
        for (result,) in con.execute(f"SELECT tool_result FROM read_parquet('{TOOLCALL}/{split}/0000.parquet')").fetchall():
            for m in re.finditer(r"^- (.+?) -> (.+)$", result or "", re.M):
                en = m.group(1).strip().lower()
                if len(en.split()) <= 3 and any(re.search(rf"\b{re.escape(t)}\b", en) for t in GLOSSARY_TERMS):
                    for alt in m.group(2).split(" | "):
                        glossary[en][alt.strip()] += 1

    dialogue = []
    for (twi,) in con.execute(
        f"SELECT twi FROM read_parquet('{PRISTINE}') WHERE style = 'Dialogue' "
        "AND (english_translation ILIKE '%money%' OR english_translation ILIKE '%cedi%' "
        "OR english_translation ILIKE '%pay%') LIMIT 40"
    ).fetchall():
        lines = [line.strip() for line in twi.splitlines() if line.strip()]
        dialogue.append(" ".join(lines[:3])[:400])

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "code_switching": money + rest[:60],
                "glossary": {en: [tw for tw, _ in c.most_common(3)] for en, c in sorted(glossary.items())},
                "dialogue": dialogue[:12],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"{len(money)} money-related transcripts, {len(glossary)} glossary entries, {len(dialogue)} dialogue excerpts -> {OUT}")


if __name__ == "__main__":
    main()
