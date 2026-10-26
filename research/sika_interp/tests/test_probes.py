import warnings

import numpy as np
import pytest
from sklearn.model_selection import KFold, cross_val_score

from sika_interp.probes import (
    chance,
    cross_language_accuracy,
    folds_by_group,
    layer_accuracy,
    load,
    probe,
    twin_placement,
)

FAMILIES = ("send_money", "promo_fee", "payoff")
LANGUAGES = ("en", "tw", "en-tw")


def episodes(seeds, families=FAMILIES, languages=LANGUAGES):
    """Labels for every family in every language on `seeds` seeds."""
    rows = [(f, 1_000_000 + s, lang) for s in range(seeds) for f in families for lang in languages]
    family, seed, language = (np.array(c) for c in zip(*rows))
    return family, seed, language


def plant(X, family, layers, first_dim=0, rows=slice(None), size=6.0):
    """Mark each family along its own direction (`first_dim` onwards) at `layers`."""
    rows = np.arange(len(family))[rows]
    dims = first_dim + np.array([FAMILIES.index(f) for f in family[rows]])
    for layer in range(X.shape[1])[layers]:
        X[rows, layer, dims] += size


def per_seed_labels(rng, seed):
    """A random label per seed and a random vector per seed shared by all its episodes:
    data a probe can only score on by having seen a twin."""
    seeds = np.unique(seed)
    labels = dict(zip(seeds, rng.permutation(np.arange(len(seeds)) % 2)))
    vectors = dict(zip(seeds, rng.normal(size=(len(seeds), 128))))
    y = np.array([labels[s] for s in seed])
    X = np.stack([vectors[s] for s in seed])[:, None, :] + 0.1 * rng.normal(size=(len(seed), 2, 128))
    return X.astype(np.float32), y


def test_chance_is_the_commonest_label_rate():
    assert chance(["a", "a", "b"]) == pytest.approx(2 / 3)
    assert chance(np.array([1, 2, 3, 4])) == 0.25


def test_folds_keep_each_seed_on_one_side():
    _, seed, _ = episodes(7)
    splits = folds_by_group(seed, folds=5)
    assert len(splits) == 5
    for train, test in splits:
        assert not set(seed[train]) & set(seed[test])
    assert len(folds_by_group(seed[seed < 1_000_003], folds=5)) == 3
    with pytest.raises(ValueError):
        folds_by_group(seed[seed == 1_000_000])


def test_probe_reads_a_label_only_from_the_layer_it_appears():
    rng = np.random.default_rng(0)
    family, seed, _ = episodes(12)
    X = rng.normal(size=(len(family), 6, 16)).astype(np.float32)
    plant(X, family, slice(3, None))
    acc = layer_accuracy(X, family, seed)
    assert acc.shape == (6,)
    assert (acc[:3] < chance(family) + 0.25).all()
    assert (acc[3:] > 0.9).all()


def test_a_layer_every_episode_shares_sits_at_chance():
    family, seed, _ = episodes(10)
    X = np.ones((len(family), 1, 8), dtype=np.float32)
    assert layer_accuracy(X, family, seed)[0] == pytest.approx(chance(family))


def test_grouping_stops_twins_leaking_the_label():
    rng = np.random.default_rng(1)
    _, seed, _ = episodes(60, families=FAMILIES[:1])
    X, y = per_seed_labels(rng, seed)
    leaky = cross_val_score(probe(), X[:, 0], y, cv=KFold(5, shuffle=True, random_state=0)).mean()
    assert leaky > 0.9
    assert (layer_accuracy(X, y, seed) < 0.7).all()


def test_cross_language_transfer_needs_a_shared_code():
    rng = np.random.default_rng(2)
    family, seed, language = episodes(30)
    X = rng.normal(size=(len(family), 4, 16)).astype(np.float32)
    plant(X, family, slice(None, 2), rows=language == "en")
    plant(X, family, slice(2, None), first_dim=3)
    acc = cross_language_accuracy(X, family, language, seed, test=("en", "tw", "en-tw"))
    assert set(acc) == {"en", "tw", "en-tw"}
    assert (acc["en"] > 0.9).all()
    for lang in ("tw", "en-tw"):
        assert (acc[lang][:2] < chance(family) + 0.25).all()
        assert (acc[lang][2:] > 0.9).all()


def test_cross_language_tests_only_on_held_out_seeds():
    rng = np.random.default_rng(3)
    _, seed, language = episodes(60, families=FAMILIES[:1])
    X, y = per_seed_labels(rng, seed)
    en = language == "en"
    seen = probe().fit(X[en, 0], y[en])
    assert seen.score(X[language == "tw", 0], y[language == "tw"]) > 0.9
    acc = cross_language_accuracy(X, y, language, seed)
    assert (acc["tw"] < 0.7).all() and (acc["en-tw"] < 0.7).all()


def placed(rng, family, seed, language, english_share):
    """En and Tw twins share a family vector and differ otherwise; each En–Tw episode is a
    mix of its twins with `english_share` of English at each layer."""
    d = 64
    fam = {f: rng.normal(size=d) for f in FAMILIES}
    index = {k: i for i, k in enumerate(zip(family, seed, language))}
    X = np.zeros((len(family), len(english_share), d), dtype=np.float32)
    for f, s in {(f, s) for f, s, _ in index}:
        en, tw = fam[f] + rng.normal(size=d), fam[f] + rng.normal(size=d)
        for layer, share in enumerate(english_share):
            noise = 0.1 * rng.normal(size=(3, d))
            X[index[(f, s, "en")], layer] = en + noise[0]
            X[index[(f, s, "tw")], layer] = tw + noise[1]
            X[index[(f, s, "en-tw")], layer] = share * en + (1 - share) * tw + noise[2]
    return X


def test_twin_placement_follows_the_mix():
    rng = np.random.default_rng(4)
    family, seed, language = episodes(10)
    X = placed(rng, family, seed, language, english_share=[0.8, 0.8, 0.2, 0.2])
    out = twin_placement(X, family, seed, language)
    assert set(out) == {"to_en", "to_tw", "closer_to_en", "mismatched"}
    assert (out["closer_to_en"][:2] == 1).all() and (out["closer_to_en"][2:] == 0).all()
    assert (out["to_en"][:2] > out["to_tw"][:2]).all() and (out["to_tw"][2:] > out["to_en"][2:]).all()
    assert (out["mismatched"] > 0.2).all()
    assert (out["to_en"][:2] > out["mismatched"][:2] + 0.2).all()


def test_twin_placement_uses_only_complete_twins():
    rng = np.random.default_rng(5)
    family, seed, language = episodes(10)
    X = placed(rng, family, seed, language, english_share=[0.8, 0.2])
    gone = (family == "payoff") & (seed == 1_000_000) & (language == "tw")
    extra_family, extra_seed, extra_language = episodes(5, languages=("en-tw",))
    family = np.concatenate([family[~gone], extra_family])
    seed = np.concatenate([seed[~gone], extra_seed + 100])
    language = np.concatenate([language[~gone], extra_language])
    X = np.concatenate([X[~gone], rng.normal(size=(len(extra_family), 2, 64)).astype(np.float32)])
    out = twin_placement(X, family, seed, language)
    assert out["closer_to_en"].tolist() == [1.0, 0.0]


def test_twin_placement_is_nan_where_every_episode_is_alike():
    rng = np.random.default_rng(6)
    family, seed, language = episodes(5)
    X = placed(rng, family, seed, language, english_share=[0.5, 0.8])
    X[:, 0] = rng.normal(size=64).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = twin_placement(X, family, seed, language)
    for values in out.values():
        assert np.isnan(values[0]) and not np.isnan(values[1])


def test_twin_placement_wants_one_row_per_episode():
    family, seed, language = episodes(2)
    X = np.zeros((len(family) + 1, 1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="more than one"):
        twin_placement(X, np.append(family, family[0]), np.append(seed, seed[0]), np.append(language, language[0]))


def test_load_stacks_one_read_point_with_labels_from_file_names(tmp_path):
    for name, value, points in [
        ("eval_promo_fee_en-tw_1000000", 1, [10, 20, 22]),
        ("eval_send_money_en_1000001", 2, [10, -1, 12]),
    ]:
        residual = np.stack([np.full((3, 4), value + k / 10, dtype=np.float16) for k in range(3)])
        np.savez(tmp_path / f"{name}.npz", residual=residual, points=np.array(points))
    X, family, language, seed = load(tmp_path)
    assert X.shape == (2, 3, 4) and X[:, 0, 0].tolist() == [1, 2]
    assert family.tolist() == ["promo_fee", "send_money"]
    assert language.tolist() == ["en-tw", "en"]
    assert seed.tolist() == [1_000_000, 1_000_001]
    reasoning = load(tmp_path, point="reasoning_end")
    assert reasoning[1].tolist() == ["promo_fee"]
    assert reasoning[0][0, 0, 0] == pytest.approx(1.1, abs=1e-2)
