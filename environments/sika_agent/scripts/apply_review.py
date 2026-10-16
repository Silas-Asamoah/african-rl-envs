"""Fold native-speaker verdicts from the review page into `sika_agent/phrasings.json`.

    uv run python scripts/apply_review.py verdicts.json

`verdicts.json` is the review page's `verdicts` collection as a list of
`{"id": <phrasing key>, "verdict": "good" | "fix" | "reject", "text": <fix>}` (or a mapping
from key to that body). For every Twi and Twi–English item, the pool becomes: phrasings
the checker kept and the reviewer didn't reject, plus checker rejects the reviewer rescued,
with fixes applied; then it is re-split into train and held-out eval. English pools are
left as generated. The file's status records how much was reviewed.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_review_page import key  # noqa: E402
from generate_phrasings import OUTPUT, REVIEW, placeholders, seeds, split_pools  # noqa: E402


def load_verdicts(path: str) -> dict[str, dict]:
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items()}
    return {row.get("id") or row.get("key"): row.get("data", row) for row in raw}


def main(path: str) -> None:
    verdicts = load_verdicts(path)
    candidates = [c for c in json.loads(REVIEW.read_text()) if c["language"] != "en" and not c.get("problems")]
    phrasings = json.loads(OUTPUT.read_text())
    tally: Counter = Counter()
    pools: dict[tuple, list[str]] = {}
    for c in candidates:
        v = verdicts.get(key(c), {})
        verdict = v.get("verdict")
        tally[verdict or ("unreviewed" if c.get("kept") else "rejected by checker")] += 1
        if verdict == "reject" or (not c.get("kept") and verdict not in ("good", "fix")):
            continue
        text = c["text"]
        if verdict == "fix":
            fixed = (v.get("text") or "").strip()
            if fixed and placeholders(fixed) == placeholders(text):
                text = fixed
            else:
                print(f"skipping a fix with changed placeholders: {fixed!r}", file=sys.stderr)
        pools.setdefault((c["kind"], c["name"], c["language"]), [])
        if text not in pools[(c["kind"], c["name"], c["language"])]:
            pools[(c["kind"], c["name"], c["language"])].append(text)

    for kind in ("requests", "replies", "injections"):
        for name, by_language in phrasings[kind].items():
            for language in by_language:
                if language == "en":
                    continue
                texts = pools.get((kind, name, language)) or seeds(kind, name, language)
                by_language[language] = split_pools(texts)

    reviewed = tally["good"] + tally["fix"] + tally["reject"]
    kept_total = reviewed + tally["unreviewed"]
    phrasings["status"] = (
        "machine-generated from Ghana NLP references; meaning back-checked by a second model; "
        f"native review: {reviewed} of {kept_total} Twi and Twi–English phrasings reviewed "
        f"({tally['good']} good, {tally['fix']} fixed, {tally['reject']} rejected)"
    )
    OUTPUT.write_text(json.dumps(phrasings, ensure_ascii=False, indent=1) + "\n")
    print(phrasings["status"])
    print(dict(tally))


if __name__ == "__main__":
    main(sys.argv[1])
