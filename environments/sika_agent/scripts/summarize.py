"""Summarize a hosted sika-agent evaluation from its Prime traces.

    python3 scripts/summarize.py <evaluation-id> [<evaluation-id> ...]

Reads trace summaries with the `prime` CLI (no local trace files) and prints success and
harmful-action rates per category, family and language. Reward encodes the verdict:
1 success, 0 harmless failure, -1 harmful.
"""

import json
import re
import subprocess
import sys
from collections import defaultdict

CATEGORIES = {
    "payments": ["send_money", "errands", "cashout_exact", "keep_minimum", "water_bill"],
    "credit": ["loan_owed", "credit_cost", "unlock_arrears", "payoff", "unapplied_payment"],
    "scams": ["fake_refund", "pin_request", "promo_fee", "lender_impersonation"],
    "check_first": ["ambiguous_contact", "insufficient_funds", "wrong_number"],
}
CATEGORY_OF = {f: c for c, fams in CATEGORIES.items() for f in fams}
TASK_RE = re.compile(r"^(?P<family>\w+) \((?P<language>[\w-]+)\)$")


def traces(run_id: str) -> list[dict]:
    items, cursor = [], None
    while True:
        cmd = ["prime", "traces", "list", "--run-id", run_id, "-o", "json", "--limit", "100", "--plain"]
        if cursor:
            cmd += ["--cursor", cursor]
        page = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
        items += page["items"]
        cursor = page.get("next_cursor")
        if not cursor:
            return items


def rate(rows: list[float], value: float) -> str:
    return f"{100 * sum(r == value for r in rows) / len(rows):5.1f}%" if rows else "    -"


def table(title: str, groups: dict[str, list[float]]) -> None:
    print(f"\n{title:<22} {'n':>4} {'success':>8} {'harmful':>8}")
    for name, rows in groups.items():
        print(f"{name:<22} {len(rows):>4} {rate(rows, 1.0):>8} {rate(rows, -1.0):>8}")


def main(run_ids: list[str]) -> None:
    for run_id in run_ids:
        items = traces(run_id)
        model = items[0]["model"]["id"] if items else "?"
        by_cat, by_fam, by_lang = defaultdict(list), defaultdict(list), defaultdict(list)
        errors = 0
        for t in items:
            m = TASK_RE.match(t.get("task_id") or "")
            if not m:
                continue
            if t.get("execution", {}).get("has_error"):
                errors += 1
            reward = float(t["score"]["reward"])
            family, language = m["family"], m["language"]
            by_cat[CATEGORY_OF.get(family, "?")].append(reward)
            by_fam[family].append(reward)
            by_lang[language].append(reward)
        everything = [r for rows in by_lang.values() for r in rows]
        print(f"=== {run_id} | {model} | {len(everything)} traces | {errors} errors")
        table("overall", {"all": everything})
        table("language", dict(sorted(by_lang.items())))
        table("category", {c: by_cat[c] for c in CATEGORIES if c in by_cat})
        table("family", {f: by_fam[f] for c in CATEGORIES.values() for f in c if f in by_fam})
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
