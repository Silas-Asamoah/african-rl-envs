import math

import numpy as np
import pytest

from sika_interp.neurons import (
    LANGUAGES,
    candidates,
    language_preference,
    load_mlp,
    logit_lens,
    percentile_vs_controls,
    selectivity,
    wiring,
    wiring_up,
)


def activations(rng, per_language=40, layers=3, width=50):
    """Small noise activations [episodes, layers, width] with labels: the three languages in
    equal shares, every fourth episode a threat."""
    language = np.repeat(LANGUAGES, per_language)
    threat = np.arange(len(language)) % 4 == 0
    return rng.normal(scale=0.1, size=(len(language), layers, width)), language, threat


def test_selectivity_puts_a_planted_threat_neuron_on_top():
    rng = np.random.default_rng(0)
    A, _, threat = activations(rng)
    A[threat, 1, 7] += 1.0
    d = selectivity(A.astype(np.float16), threat)
    assert d.shape == (3, 50) and d.dtype == np.float32
    assert np.unravel_index(np.abs(d).argmax(), d.shape) == (1, 7)
    assert d[1, 7] > 5
    d[1, 7] = 0
    assert np.abs(d).max() < 1.5
    assert (1, 7, "threat") in candidates({"threat": selectivity(A, threat)}, per_layer=1)


def test_selectivity_is_cohens_d_with_a_pooled_sd():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 2, 4))
    A[:, 0, 0] = 2.0
    mask = np.arange(30) < 12
    a, b = A[mask, 1, 2], A[~mask, 1, 2]
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    d = selectivity(A, mask, eps=1e-12)
    assert d[1, 2] == pytest.approx((a.mean() - b.mean()) / math.sqrt(pooled), rel=1e-5)
    assert selectivity(A, mask)[0, 0] == 0
    assert selectivity(A, ~mask)[1, 2] == pytest.approx(-d[1, 2], rel=1e-5)
    with pytest.raises(ValueError):
        selectivity(A, np.ones(30, dtype=bool))


def test_language_preference_finds_twi_only_neurons():
    rng = np.random.default_rng(2)
    A, language, _ = activations(rng)
    tw, en = language == "tw", language == "en"
    A[:, 2, 3] = 0.0
    A[tw, 2, 3] = rng.normal(2.0, 0.1, tw.sum())  # silent outside Twi
    A[tw, 0, 4] += 3.0  # fires on Twi, jitters elsewhere
    A[en, 1, 9] -= 3.0  # fires negative on English
    A[:, 1, 0] = 0.0  # never fires
    out = language_preference(A.astype(np.float16), language)
    assert out["mean"].shape == out["probability"].shape == (3, 3, 50)
    assert out["entropy"].shape == out["prefers"].shape == (3, 50)
    assert out["entropy"][2, 3] == pytest.approx(0.0, abs=1e-6)
    assert out["prefers"][2, 3] == LANGUAGES.index("tw")
    assert out["mean"][1, 2, 3] == pytest.approx(2.0, abs=0.05)
    assert out["mean"][0, 2, 3] == 0
    assert np.nanargmin(out["entropy"][0]) == 4 and out["prefers"][0, 4] == LANGUAGES.index("tw")
    assert 0.8 < out["entropy"][0, 4] < 0.95  # the median's floor, see the docstring
    assert out["prefers"][1, 9] == LANGUAGES.index("en") and out["mean"][0, 1, 9] < -2.5
    assert np.isnan(out["entropy"][1, 0]) and out["prefers"][1, 0] == -1
    noise = np.delete(out["entropy"][0], 4)
    assert noise.mean() > 1.05 and noise.max() <= math.log(3) + 1e-6
    strict = language_preference(A, language, quantile=2 / 3)
    assert strict["entropy"][0, 4] < 0.1


def test_language_preference_needs_every_language():
    A = np.zeros((4, 1, 2))
    with pytest.raises(ValueError, match="en-tw"):
        language_preference(A, ["en", "tw", "en", "tw"])


def test_candidates_union_the_top_per_layer_and_keep_the_first_reason():
    nan = np.nan
    threat = np.array([[0.1, -4, 0.2, 5, 0.3], [0.1, 0.2, 3, 0.3, nan]])
    language = np.array([[0.1, 0.2, 0.3, 9, 8], [7, 0.1, 0.2, 0.3, 0.4]])
    assert candidates({"threat": threat, "language": language}, per_layer=2) == [
        (0, 1, "threat"),
        (0, 3, "threat"),
        (0, 4, "language"),
        (1, 0, "language"),
        (1, 2, "threat"),
        (1, 3, "threat"),
        (1, 4, "language"),
    ]
    assert candidates({"x": np.full((1, 3), nan)}, per_layer=2) == []
    assert len(candidates({"x": np.ones((2, 3))}, per_layer=10)) == 6


def test_logit_lens_recovers_planted_tokens_in_any_chunk_size():
    rng = np.random.default_rng(3)
    w = rng.normal(size=32)
    unembed = rng.normal(size=(500, 32))
    unembed[42], unembed[7] = 3 * w, -3 * w
    unembed = unembed.astype(np.float16)
    up, down = logit_lens(w, unembed, k=5, chunk=64)
    assert up[0][0] == 42 and down[0][0] == 7
    assert len(up) == len(down) == 5
    assert [v for _, v in up] == sorted((v for _, v in up), reverse=True)
    assert [v for _, v in down] == sorted(v for _, v in down)
    full = unembed.astype(np.float32) @ w.astype(np.float32)
    assert up[0][1] == pytest.approx(full.max(), rel=1e-5) and down[0][1] == pytest.approx(full.min(), rel=1e-5)
    whole_up, whole_down = logit_lens(w, unembed, k=5, chunk=10_000)
    assert [i for i, _ in whole_up + whole_down] == [i for i, _ in up + down]
    assert [v for _, v in whole_up + whole_down] == pytest.approx([v for _, v in up + down], rel=1e-5)


def test_logit_lens_applies_the_norm_scale():
    unembed, w = np.eye(4), np.array([1.0, 2.0, 0.0, 0.0])
    assert logit_lens(w, unembed, k=1)[0] == [(1, 2.0)]
    assert logit_lens(w, unembed, np.array([3.0, 1.0, 1.0, 1.0]), k=1)[0] == [(0, 3.0)]


def test_wiring_finds_planted_read_directions_across_chunks():
    rng = np.random.default_rng(4)
    w = rng.normal(size=64)
    rows = rng.normal(size=(1000, 64))
    rows[777] = 2 * w + 0.1 * rng.normal(size=64)
    rows[123] = -w
    rows[5] = 0.0
    rows = rows.astype(np.float16)
    top = wiring(w, rows, k=2, chunk=100)
    assert top[0][0] == 123 and top[0][1] == pytest.approx(-1, abs=1e-3)
    assert top[1][0] == 777 and top[1][1] > 0.99

    v = w / np.linalg.norm(w)
    brute = (rows.astype(np.float32) @ v) / np.linalg.norm(rows.astype(np.float32), axis=1).clip(1e-12)
    order = np.argsort(-np.abs(brute))[:25]
    for chunk in (7, 100, 5000):
        got = wiring(w, rows, k=25, chunk=chunk)
        assert [i for i, _ in got] == order.tolist()
        assert [c for _, c in got] == pytest.approx(brute[order].tolist(), abs=1e-5)

    everything = dict(wiring(w, rows, k=1000, chunk=64))
    assert len(everything) == 1000 and everything[5] == 0 and not np.isnan(list(everything.values())).any()


def test_wiring_up_reads_columns_of_down_proj_through_a_transpose():
    rng = np.random.default_rng(5)
    down = rng.normal(size=(64, 300)).astype(np.float16)  # [d_model, width], as down_proj.weight
    read = down[:, 211].astype(np.float32) + 0.05 * rng.normal(size=64)
    got = wiring_up(read, down.T, k=3, chunk=50)
    assert got[0][0] == 211 and got[0][1] > 0.99
    same = wiring(read, np.ascontiguousarray(down.T), k=3, chunk=300)
    assert [i for i, _ in got] == [i for i, _ in same]
    assert [c for _, c in got] == pytest.approx([c for _, c in same], abs=1e-6)


def test_percentile_vs_controls_at_the_extremes_and_ties():
    rng = np.random.default_rng(6)
    controls = np.arange(1, 101) * rng.choice([-1, 1], size=100)
    assert percentile_vs_controls(1000, controls) == 100.0
    assert percentile_vs_controls(-1000, controls) == 100.0
    assert percentile_vs_controls(0.0, controls) == 0.0
    assert percentile_vs_controls(50.5, controls) == 50.0
    assert percentile_vs_controls(-50, controls) == 49.5
    assert isinstance(percentile_vs_controls(3, controls), float)
    many = percentile_vs_controls(np.array([0.0, 1000.0, np.nan]), controls)
    assert many[:2].tolist() == [0.0, 100.0] and np.isnan(many[2])
    assert percentile_vs_controls(2, [1, np.nan, 3]) == 50.0
    with pytest.raises(ValueError):
        percentile_vs_controls(1, [np.nan])


def test_load_mlp_stacks_one_read_point_with_labels_from_file_names(tmp_path):
    for name, value, points in [
        ("eval_promo_fee_en-tw_1000000", 1, [10, 20, 22]),
        ("eval_send_money_en_1000001", 2, [10, -1, 12]),
    ]:
        mlp = np.stack([np.full((3, 5), value + k / 10, dtype=np.float16) for k in range(3)])
        np.savez(tmp_path / f"{name}.npz", residual=np.zeros((3, 4, 2)), mlp=mlp, points=np.array(points))
    A, family, language, seed = load_mlp(tmp_path)
    assert A.shape == (2, 3, 5) and A.dtype == np.float16 and A[:, 0, 0].tolist() == [1, 2]
    assert family.tolist() == ["promo_fee", "send_money"]
    assert language.tolist() == ["en-tw", "en"]
    assert seed.tolist() == [1_000_000, 1_000_001]
    reasoning = load_mlp(tmp_path, point="reasoning_end")
    assert reasoning[1].tolist() == ["promo_fee"]
    assert reasoning[0][0, 0, 0] == pytest.approx(1.1, abs=1e-2)
    with pytest.raises(FileNotFoundError):
        load_mlp(tmp_path / "missing")
