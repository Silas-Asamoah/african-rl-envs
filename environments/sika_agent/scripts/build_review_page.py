"""Build the native-speaker review page from the last generation run.

    uv run python scripts/build_review_page.py

Reads `scripts/.cache/review.json` (written by `generate_phrasings.py`), keeps the Twi and
Twi–English candidates that passed the mechanical checks (placeholders, language, no
duplicates), both those the checker model kept and those it rejected, and writes
`scripts/.cache/sika-twi-review.html` with them embedded. The page saves one verdict per
phrasing (good / fix with corrected text / reject) to its `verdicts` collection;
`apply_review.py` folds those verdicts back into `sika_agent/phrasings.json`.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_phrasings import MEANING, REVIEW  # noqa: E402

TEMPLATE = Path(__file__).parent / "review" / "page.html"
OUT = REVIEW.parent / "sika-twi-review.html"
KIND_ORDER = {"requests": 0, "replies": 1, "injections": 2}


def key(c: dict) -> str:
    return hashlib.sha256(f"{c['kind']}|{c['name']}|{c['language']}|{c['text']}".encode()).hexdigest()[:20]


def main() -> None:
    candidates = json.loads(REVIEW.read_text())
    items = [
        {
            "key": key(c),
            "kind": c["kind"],
            "name": c["name"],
            "language": c["language"],
            "text": c["text"],
            "english": c.get("english", ""),
            "natural": c.get("natural", 0),
            "source": c["source"],
            "checker": c.get("checker", ""),
            "reason": "" if c.get("kept") else c.get("reason", ""),
            "kept": bool(c.get("kept")),
            "split": c.get("split", ""),
            "meaning": MEANING[(c["kind"], c["name"])].replace("{rel}", "relation"),
        }
        for c in candidates
        if c["language"] != "en" and not c.get("problems") and "match" in c
    ]
    items.sort(key=lambda i: (KIND_ORDER[i["kind"]], i["name"], i["language"], not i["kept"], i["text"]))
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    OUT.write_text(TEMPLATE.read_text().replace("/*__ITEMS__*/", payload))
    kept = sum(i["kept"] for i in items)
    print(f"{len(items)} phrasings ({kept} kept by the checker) -> {OUT}")


if __name__ == "__main__":
    main()
