"""Linear probes and twin geometry on the residual stream, layer by layer.

Each episode gives one vector per layer at a read point, stacked as `X` with shape
[episodes, layers, d_model]. Three questions:

- `layer_accuracy`: from which layer can a linear probe read a label (family, category,
  language) off the residual stream?
- `cross_language_accuracy`: does a probe trained on English requests read the same label
  off Twi and Twi–English ones? Where it does, the model files the request the same way
  whatever the language.
- `twin_placement`: does a Twi–English request sit nearer its English twin or its Twi twin?

Probe folds split by seed. Twins share a seed, so a probe never meets the twin of a test
episode in training and can't pass by recognising a wallet it has seen.

Layer 0 is the embedding of the read-point token alone, before attention mixes in the
rest of the prompt: read later layers as gains over it. When every episode shares that token
(the last prompt token is the chat template's), layer 0 is the same for all of them, so
probes sit at `chance` there and `twin_placement` gives nan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def chance(y: ArrayLike) -> float:
    """Accuracy of always guessing the commonest label: the floor for reading probe scores."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return float(counts.max() / counts.sum())


def folds_by_group(groups: ArrayLike, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) index pairs with every group wholly on one side; one fold per group
    when there are fewer groups than `folds`."""
    groups = np.asarray(groups)
    n = min(folds, len(np.unique(groups)))
    if n < 2:
        raise ValueError("need at least two groups (seeds) to hold one out")
    return list(GroupKFold(n_splits=n).split(np.zeros((len(groups), 1)), groups=groups))


def probe(C: float = 0.1):
    """The linear probe: standardised features into L2-regularised logistic regression."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def layer_accuracy(X: np.ndarray, y: ArrayLike, groups: ArrayLike, folds: int = 5, C: float = 0.1) -> np.ndarray:
    """Held-out probe accuracy for `y` at each layer, shape [layers]: the mean over folds
    that split by `groups` (pass seeds, so twins and siblings stay together)."""
    y = np.asarray(y)
    splits = folds_by_group(groups, folds)
    out = np.empty(X.shape[1])
    for layer in range(X.shape[1]):
        A = X[:, layer].astype(np.float64)
        out[layer] = np.mean([probe(C).fit(A[tr], y[tr]).score(A[te], y[te]) for tr, te in splits])
    return out


def cross_language_accuracy(
    X: np.ndarray,
    y: ArrayLike,
    language: ArrayLike,
    seed: ArrayLike,
    train: str = "en",
    test: tuple[str, ...] = ("tw", "en-tw"),
    folds: int = 5,
    C: float = 0.1,
) -> dict[str, np.ndarray]:
    """Accuracy at each layer of a probe trained on `train`-language episodes and tested on
    each `test` language: {language: [layers]}, the mean over folds.

    Folds split by seed. Each probe trains on the training seeds' `train` episodes and is
    tested on the held-out seeds' episodes only, so no test episode's twin was in training.
    Put `train` in `test` for the within-language accuracy to compare against. A fold with
    no held-out episodes in a test language is skipped for that language."""
    y, language = np.asarray(y), np.asarray(language)
    plan = [
        (tr[language[tr] == train], {t: te[language[te] == t] for t in test})
        for tr, te in folds_by_group(seed, folds)
    ]
    out = {t: np.full(X.shape[1], np.nan) for t in test}
    for layer in range(X.shape[1]):
        A = X[:, layer].astype(np.float64)
        scores: dict[str, list[float]] = {t: [] for t in test}
        for tr, held_out in plan:
            fitted = probe(C).fit(A[tr], y[tr])
            for t, idx in held_out.items():
                if len(idx):
                    scores[t].append(fitted.score(A[idx], y[idx]))
        for t in test:
            if scores[t]:
                out[t][layer] = np.mean(scores[t])
    return out


def _unit(V: np.ndarray, floor: float) -> np.ndarray:
    """Rows scaled to length 1; nan for rows no longer than `floor`."""
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return np.divide(V, norms, out=np.full_like(V, np.nan), where=norms > floor)


def twin_placement(X: np.ndarray, family: ArrayLike, seed: ArrayLike, language: ArrayLike) -> dict[str, np.ndarray]:
    """Where each Twi–English episode sits between its English and Twi twins, per layer.

    Uses every (family, seed) played in all three languages, one row each. Vectors are
    centred on each layer's mean over all episodes passed, so what every prompt shares
    doesn't make everything look alike. Each value has shape [layers]:
    - "to_en": mean cosine between a Twi–English episode and its English twin;
    - "to_tw": mean cosine between a Twi–English episode and its Twi twin;
    - "closer_to_en": fraction of twins where the Twi–English episode is nearer its English twin;
    - "mismatched": mean cosine between a Twi–English episode and the English episode of the
      same family on another seed, the baseline for how close "same job, different wallet" is.
    Layers where centred vectors vanish (every episode alike) come out nan."""
    family = np.asarray(family)
    index: dict[tuple, int] = {}
    for i, key in enumerate(zip(family.tolist(), np.asarray(seed).tolist(), np.asarray(language).tolist())):
        if key in index:
            raise ValueError(f"more than one episode for {key}: pass one read point per episode")
        index[key] = i
    twins = [
        (i, index[(f, s, "en")], index[(f, s, "tw")])
        for (f, s, lang), i in index.items()
        if lang == "en-tw" and (f, s, "en") in index and (f, s, "tw") in index
    ]
    if not twins:
        raise ValueError("no (family, seed) has all three languages")
    mix, en, tw = np.array(twins).T
    others = family[mix][:, None] == family[mix][None, :]
    np.fill_diagonal(others, False)

    out = {k: np.full(X.shape[1], np.nan) for k in ("to_en", "to_tw", "closer_to_en", "mismatched")}
    for layer in range(X.shape[1]):
        A = X[:, layer].astype(np.float64)
        centre = A.mean(axis=0)
        floor = 1e-6 * np.linalg.norm(A, axis=1).max()
        m, e, t = (_unit(A[rows] - centre, floor) for rows in (mix, en, tw))
        to_en, to_tw = (m * e).sum(axis=1), (m * t).sum(axis=1)
        out["to_en"][layer] = to_en.mean()
        out["to_tw"][layer] = to_tw.mean()
        out["closer_to_en"][layer] = np.where(np.isnan(to_en - to_tw), np.nan, to_en > to_tw).mean()
        if others.any():
            out["mismatched"][layer] = (m @ e.T)[others].mean()
    return out


POINTS = ("request_end", "reasoning_end", "action_start")
"""The read points `modal_app.read_points` records, in the order it stores them (mirrors
`modal_app.POINTS`, which can't import this module's dependencies locally)."""


def load(directory: str | Path, point: str = "request_end", key: str = "residual") -> tuple[np.ndarray, ...]:
    """One model's arrays at one read point, as `modal_app.read_points` saves them
    (`<split>_<family>_<language>_<seed>.npz`, arrays [point, layer, width]): X
    [episodes, layers, width], then family, language and seed per episode, in file-name
    order. Episodes without that point (no reasoning, say) are left out."""
    at = POINTS.index(point)
    arrays, rows = [], []
    for path in sorted(Path(directory).glob("*.npz")):
        head, language, seed = path.stem.rsplit("_", 2)
        with np.load(path) as saved:
            if saved["points"][at] < 0:
                continue
            arrays.append(saved[key][at])
        rows.append((head.split("_", 1)[1], language, int(seed)))
    if not rows:
        raise FileNotFoundError(f"no {point} arrays in {directory}")
    family, language, seed = (np.array(c) for c in zip(*rows))
    return np.stack(arrays), family, language, seed


def held_out_scores(
    A: np.ndarray, y: ArrayLike, language: ArrayLike, seed: ArrayLike, train: str = "en", folds: int = 5, C: float = 0.1
) -> np.ndarray:
    """Per-episode probability of the positive class from a probe trained on `train`-language
    episodes, each scored by a probe that never saw its seed (so never its twins). `A` is one
    layer, [episodes, width]; `y` is boolean. Every language gets a score."""
    y, language = np.asarray(y, dtype=bool), np.asarray(language)
    out = np.full(len(y), np.nan)
    for tr, te in folds_by_group(seed, folds):
        trn = tr[language[tr] == train]
        fitted = probe(C).fit(A[trn].astype(np.float64), y[trn])
        out[te] = fitted.predict_proba(A[te].astype(np.float64))[:, list(fitted.classes_).index(True)]
    return out


def placement_margins(A: np.ndarray, family: ArrayLike, seed: ArrayLike, language: ArrayLike) -> dict[int, float]:
    """For each Twi–English episode with both twins, by row index: its centred cosine to its
    English twin minus that to its Twi twin, at one layer `A` [episodes, width]. Positive
    means it sits nearer the English twin."""
    family, seed, language = (np.asarray(v) for v in (family, seed, language))
    index = {k: i for i, k in enumerate(zip(family.tolist(), seed.tolist(), language.tolist()))}
    V = A.astype(np.float64) - A.astype(np.float64).mean(axis=0)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    out = {}
    for (f, s, lang), i in index.items():
        if lang == "en-tw" and (f, s, "en") in index and (f, s, "tw") in index:
            out[i] = float(V[i] @ V[index[(f, s, "en")]] - V[i] @ V[index[(f, s, "tw")]])
    return out
