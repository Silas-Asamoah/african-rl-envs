"""The episodes the study runs: matched twins in every language, plus extra Twi–English.

Built from the environment's own taskset with `matched` on, so prompts, system prompts and
seeds are exactly what an agent sees in a sika-agent run. Tasks that share a family and
seed are twins: same wallet, people, amounts and expected outcome, different wording.
"""

from __future__ import annotations

from sika_agent.taskset import SikaTask, SikaTaskset, SikaTasksetConfig

LANGUAGES = ("en", "tw", "en-tw")
MATCHED_SEEDS = 10
"""Seeds per family, each played in all three languages."""
EXTRA_EN_TW_SEEDS = 10
"""More seeds per family in Twi–English alone, for switch-point and cue-moving work."""
SEED_BASE = {"train": 0, "eval": 1_000_000}
"""Where the taskset starts counting seeds for each split."""


def _load(split: str, seeds: int, languages: list[str], families: list[str] | None) -> list[SikaTask]:
    cfg = SikaTasksetConfig(split=split, tasks_per_combo=seeds, languages=languages, matched=True)
    if families is not None:
        cfg.families = families
    return list(SikaTaskset(cfg).load())


def matched_tasks(
    seeds: int = MATCHED_SEEDS, families: list[str] | None = None, split: str = "eval"
) -> list[SikaTask]:
    """Every family in every language for the first `seeds` seeds of `split`."""
    return _load(split, seeds, list(LANGUAGES), families)


def extra_en_tw_tasks(
    extra: int = EXTRA_EN_TW_SEEDS,
    after: int = MATCHED_SEEDS,
    families: list[str] | None = None,
    split: str = "eval",
) -> list[SikaTask]:
    """Twi–English tasks on the `extra` seeds that follow the matched ones."""
    first = SEED_BASE[split] + after
    return [t for t in _load(split, after + extra, ["en-tw"], families) if t.data.seed >= first]


def study_tasks(families: list[str] | None = None, split: str = "eval") -> list[SikaTask]:
    """The full set: matched twins, then the extra Twi–English seeds."""
    return matched_tasks(families=families, split=split) + extra_en_tw_tasks(families=families, split=split)


def twins(tasks: list[SikaTask]) -> dict[tuple[str, int], dict[str, SikaTask]]:
    """Group tasks by (family, seed) into {language: task}."""
    groups: dict[tuple[str, int], dict[str, SikaTask]] = {}
    for task in tasks:
        groups.setdefault((task.data.family, task.data.seed), {})[task.data.language] = task
    return groups
