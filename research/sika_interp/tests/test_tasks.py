from sika_agent.scenarios import FAMILIES

from sika_interp.tasks import (
    EXTRA_EN_TW_SEEDS,
    LANGUAGES,
    MATCHED_SEEDS,
    SEED_BASE,
    extra_en_tw_tasks,
    matched_tasks,
    study_tasks,
    twins,
)


def test_matched_set_covers_every_family_language_and_seed():
    tasks = matched_tasks()
    assert len(tasks) == MATCHED_SEEDS * len(FAMILIES) * len(LANGUAGES)
    assert len({t.data.id for t in tasks}) == len(tasks)
    seeds = {t.data.seed for t in tasks}
    assert seeds == set(range(SEED_BASE["eval"], SEED_BASE["eval"] + MATCHED_SEEDS))


def test_twins_share_the_wallet_and_differ_in_wording():
    for (family, seed), group in twins(matched_tasks(seeds=2)).items():
        assert set(group) == set(LANGUAGES), (family, seed)
        en = group["en"].data
        for language in ("tw", "en-tw"):
            other = group[language].data
            assert other.system_prompt == en.system_prompt
            assert other.expect == en.expect
            assert other.prompt != en.prompt


def test_extra_seeds_are_twi_english_and_new():
    extra = extra_en_tw_tasks()
    assert len(extra) == EXTRA_EN_TW_SEEDS * len(FAMILIES)
    assert {t.data.language for t in extra} == {"en-tw"}
    matched_seeds = {t.data.seed for t in matched_tasks(seeds=MATCHED_SEEDS, families=["send_money"])}
    assert not matched_seeds & {t.data.seed for t in extra}


def test_extra_seeds_match_a_longer_matched_run():
    """An extra task is the same task a matched run with more seeds would build."""
    longer = {t.data.id: t for t in matched_tasks(seeds=MATCHED_SEEDS + 1, families=["promo_fee"])}
    for task in extra_en_tw_tasks(extra=1, families=["promo_fee"]):
        assert task.data.prompt == longer[task.data.id].data.prompt


def test_study_set_size():
    per_family = MATCHED_SEEDS * len(LANGUAGES) + EXTRA_EN_TW_SEEDS
    assert len(study_tasks()) == per_family * len(FAMILIES)
