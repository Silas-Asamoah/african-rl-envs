"""Re-score hosted evaluations with the current grader by replaying their traces.

    uv run python scripts/rescore.py --json out.json <evaluation-id> [<evaluation-id> ...]

The wallet simulator is deterministic, so a trace's recorded tool calls, replayed from the
task's starting wallet, reproduce its final state exactly; the current `evaluate` then
scores it. Tasks are scored on the wallets they actually ran with. A task is excluded only if two of
the people it is about (recipients, numbers behind scams) hold spellings of the same Akan
day name (Afua/Efua), the clash the generator now prevents. Prints how each verdict moved, and with `--json`
writes the same counts `summarize.py --json` does, for tables and plots.
"""

import json
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sika_agent import wallet  # noqa: E402
from sika_agent.checks import Expect, evaluate  # noqa: E402
from sika_agent.scenarios import FAMILIES, same_name  # noqa: E402
from sika_agent.wallet import SikaState  # noqa: E402
from summarize import CATEGORIES, counts, traces  # noqa: E402

TOOLS = {
    "check_balance", "list_contacts", "get_fee_schedule", "list_transactions", "read_messages",
    "initiate_transfer", "confirm_transfer", "cancel_transfer", "buy_airtime", "list_saved_billers",
    "pay_bill", "get_loan_status", "make_loan_payment", "request_payment_reallocation",
    "report_fraud", "send_sms", "ask_user",
}  # fmt: skip


def text(content) -> str:
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def day_name_clash(world: dict, expect: dict) -> bool:
    """Whether two people the task is about hold spellings of the same day name."""
    numbers = {p["to"] for p in expect["transfers"] + expect["airtime"]} | set(expect["forbidden_numbers"])
    firsts = {world["registry"][n].split()[0].title() for n in numbers if n in world["registry"]}
    return any((same_name(name) - {name}) & firsts for name in firsts)


def replay(trace: dict) -> tuple[float, bool]:
    """(reward under the current grader, whether the task is free of a day-name clash)."""
    data = trace["task"]["data"]
    clean = not day_name_clash(data["world"], data["expect"])
    s = SikaState.model_validate(data["world"])
    said, last_reply, users = [], "", 0
    for node in trace["nodes"]:
        message = node.get("message", node)
        role = message.get("role")
        if role == "user":
            users += 1
            if users > 1:  # the scripted user answering a question asked in plain text
                s.events.append("ask_user")
                s.questions.append(last_reply)
        elif role == "assistant":
            last_reply = text(message.get("content"))
            said.append(last_reply)
            for call in message.get("tool_calls") or []:
                fn = call.get("function", call)
                name = fn.get("name")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    if name in TOOLS and isinstance(args, dict):
                        getattr(wallet, name)(s, **args)
                except (TypeError, ValueError, AttributeError):
                    pass  # the tool server would have rejected these arguments too
    verdict = evaluate(s, Expect.model_validate(data["expect"]), "\n".join(said))
    return (1.0 if verdict.success else -1.0 if verdict.harmful else 0.0), clean


def fetch(trace_id: str) -> dict:
    cmd = ["prime", "traces", "get", trace_id, "--raw", "--plain"]
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


def main(args: list[str]) -> None:
    out_path = None
    if args and args[0] == "--json":
        out_path, args = args[1], args[2:]
    category_of = {f: c for f, (c, _) in FAMILIES.items()}
    exported = []
    for run_id in args:
        summaries = [t for t in traces(run_id) if (t.get("score") or {}).get("reward") is not None]
        with ThreadPoolExecutor(8) as pool:
            raw = list(pool.map(lambda t: fetch(t["trace_id"]), summaries))
        moves: Counter = Counter()
        changed = 0
        by_lang, by_cat, by_fam = defaultdict(list), defaultdict(list), defaultdict(list)
        model = summaries[0]["model"]["id"] if summaries else "?"
        for summary, trace in zip(summaries, raw):
            new, clean = replay(trace)
            if not clean:
                changed += 1
                continue
            old = float(summary["score"]["reward"])
            data = trace["task"]["data"]
            if new != old:
                moves[(data["family"], old, new)] += 1
            by_lang[data["language"]].append(new)
            by_cat[category_of[data["family"]]].append(new)
            by_fam[data["family"]].append(new)
        everything = [r for rows in by_lang.values() for r in rows]
        print(f"=== {run_id} | {model} | {len(everything)} scored, {changed} excluded (day-name clash)")
        for (family, old, new), n in sorted(moves.items()):
            print(f"    {family}: {old:+.0f} -> {new:+.0f}  x{n}")
        exported.append(
            {
                "evaluation_id": run_id,
                "model": model,
                "errors": 0,
                "excluded": changed,
                "overall": counts(everything),
                "language": {k: counts(v) for k, v in sorted(by_lang.items())},
                "category": {c: counts(by_cat[c]) for c in CATEGORIES if c in by_cat},
                "family": {f: counts(v) for f, v in sorted(by_fam.items())},
            }
        )
    if out_path:
        Path(out_path).write_text(json.dumps(exported, indent=1))
        print(f"wrote {len(exported)} evaluations to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
