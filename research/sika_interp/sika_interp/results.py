"""Success and harm rates from vf-eval traces, by language and category, with Wilson 95%
intervals. Errored rollouts are left out and counted separately.

    python -m sika_interp.results outputs/*/traces.jsonl
"""

import math
import sys
from collections import defaultdict
from dataclasses import dataclass

from sika_interp.tasks import LANGUAGES
from sika_interp.traces import load


@dataclass
class Episode:
    model: str
    task_id: str
    family: str
    category: str
    language: str
    seed: int
    reward: float
    """1 success, 0 harmless miss, -1 harmful."""


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a rate of `hits` in `n`."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def episodes(paths: list[str]) -> tuple[list[Episode], int]:
    """Scored episodes from traces files, and how many rollouts errored."""
    out, errored = [], 0
    for path in paths:
        for trace in load(path):
            if not trace.get("ok", True) or trace.get("errors") or "outcome" not in trace.get("rewards", {}):
                errored += 1
                continue
            data = trace["task"]["data"]
            out.append(
                Episode(
                    model=trace["agent"]["config"]["model"],
                    task_id=data["id"],
                    family=data["family"],
                    category=data["category"],
                    language=data["language"],
                    seed=data["seed"],
                    reward=trace["rewards"]["outcome"]["score"],
                )
            )
    return out, errored


def rate(rows: list[Episode], value: float) -> str:
    hits = sum(r.reward == value for r in rows)
    if not rows:
        return "-"
    low, high = wilson(hits, len(rows))
    return f"{100 * hits / len(rows):.0f}% ({100 * low:.0f}–{100 * high:.0f})"


def table(rows: list[Episode]) -> str:
    """A markdown table per model: overall, each language, each category."""
    lines = []
    by_model: dict[str, list[Episode]] = defaultdict(list)
    for r in rows:
        by_model[r.model].append(r)
    for model, eps in by_model.items():
        lines += [f"### {model}", "", "| | n | success | harmful |", "|---|---|---|---|"]
        groups = [("all", eps)]
        groups += [(lang, [e for e in eps if e.language == lang]) for lang in LANGUAGES]
        groups += [(cat, [e for e in eps if e.category == cat]) for cat in sorted({e.category for e in eps})]
        for name, group in groups:
            if group:
                lines.append(f"| {name} | {len(group)} | {rate(group, 1.0)} | {sum(e.reward == -1.0 for e in group)} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    rows, errored = episodes(sys.argv[1:])
    print(table(rows))
    print(f"{len(rows)} episodes scored, {errored} errored and left out")
