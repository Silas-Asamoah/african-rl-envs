"""Single MLP neurons: which ones care about scams or language, and what each one does.

The MLPs are gated (SwiGLU): `down_proj(silu(gate_proj(x)) * up_proj(x))`. Neuron i of a
layer is the i-th input to `down_proj`. It reads the residual stream along row i of
`gate_proj.weight` and of `up_proj.weight` ([width, d_model]) and writes onto it along
column i of `down_proj.weight` ([d_model, width]). Its activation is signed: silu lets small
negatives through and `up_proj` can have either sign.

From activations stacked as `A` [episodes, layers, width], one read point per episode
(`load_mlp`):
- `selectivity`: how far each neuron moves between two groups of episodes (threat against
  the rest, Twi against English), in pooled standard deviations;
- `language_preference`: each neuron's mean per language, and how evenly it fires across
  languages (low entropy: language-specific);
- `candidates`: the neurons worth a card, the top few per layer on each score.

From the weights alone, for a neuron's card:
- `logit_lens`: the tokens its output direction pushes up and down, if nothing after it
  changed that direction;
- `wiring` and `wiring_up`: which neurons read what it writes, and which write what it reads;
- `percentile_vs_controls`: how a steering effect compares with pushes along random directions.

Weight inputs can be large (every neuron of every later layer, or the whole vocabulary), so
they are read in row chunks and cast to float32 a chunk at a time; a memory-mapped array works.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

LANGUAGES = ("en", "tw", "en-tw")
"""Order of the language axis in `language_preference` (as `tasks.LANGUAGES`)."""
POINTS = ("request_end", "reasoning_end", "action_start")
"""The read points `modal_app.read_points` records, in the order it stores them (as
`probes.POINTS`; not imported from there so this module needs only numpy)."""


def load_mlp(directory: str | Path, point: str = "request_end") -> tuple[np.ndarray, ...]:
    """MLP activations at one read point, as `modal_app.read_points` saves them
    (`<split>_<family>_<language>_<seed>.npz`, `mlp` [point, layer, width]): A [episodes,
    layers, width] as stored (float16), then family, language and seed per episode, in
    file-name order. Episodes without that point are left out, as in `probes.load`."""
    at = POINTS.index(point)
    arrays, rows = [], []
    for path in sorted(Path(directory).glob("*.npz")):
        head, language, seed = path.stem.rsplit("_", 2)
        with np.load(path) as saved:
            if saved["points"][at] < 0:
                continue
            arrays.append(saved["mlp"][at])
        rows.append((head.split("_", 1)[1], language, int(seed)))
    if not rows:
        raise FileNotFoundError(f"no {point} arrays in {directory}")
    family, language, seed = (np.array(c) for c in zip(*rows))
    return np.stack(arrays), family, language, seed


def selectivity(A: np.ndarray, mask: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    """Cohen's d of each neuron between episodes where `mask` is True and where it is False:
    the difference in means over the pooled standard deviation (plus `eps` on the variance,
    so constant neurons give 0, not nan). Shape [layers, width], float32; positive means
    higher where `mask` is True."""
    mask = np.asarray(mask, dtype=bool)
    n1, n0 = int(mask.sum()), int((~mask).sum())
    if n1 == 0 or n0 == 0 or n1 + n0 < 3:
        raise ValueError(f"need episodes on both sides and three in all (have {n1} and {n0})")
    out = np.empty(A.shape[1:], dtype=np.float32)
    for layer in range(A.shape[1]):
        a, b = A[mask, layer].astype(np.float64), A[~mask, layer].astype(np.float64)
        ma, mb = a.mean(axis=0), b.mean(axis=0)
        pooled = (((a - ma) ** 2).sum(axis=0) + ((b - mb) ** 2).sum(axis=0)) / (n1 + n0 - 2)
        out[layer] = (ma - mb) / np.sqrt(pooled + eps)
    return out


def language_preference(
    A: np.ndarray, language: ArrayLike, languages: tuple[str, ...] = LANGUAGES, quantile: float = 0.5
) -> dict[str, np.ndarray]:
    """Per neuron, how its activation differs across languages:
    - "mean": mean activation in each language, [languages, layers, width];
    - "probability": the fraction of each language's episodes where the neuron is active,
      [languages, layers, width];
    - "entropy": entropy in nats of those probabilities normalised to sum to one across
      languages, as in LAPE (Tang et al. 2024), [layers, width]. 0 means it is active in one
      language only, log(len(languages)) equally in all;
    - "prefers": index into `languages` of the highest probability, [layers, width].

    Activations are signed, so "active" can't be "above 0" as with a ReLU. A neuron counts
    as active where |activation| exceeds its own `quantile` of |activation| over every
    episode passed (the median by default). That puts each neuron on its own scale, and a
    sparse neuron, exactly 0 in most episodes, counts as active wherever it is non-zero. The
    cost: with the median, half of all episodes count as active whatever the neuron does. With
    three languages in equal shares, a neuron that fires strongly in one and only jitters in
    the others is still "active" in a quarter of the others' episodes, and its entropy sits
    near 0.87 nats (of 1.10), not 0. Read entropy as a ranking, or raise `quantile` (2/3 for
    three equal shares) to let such a neuron reach 0. A neuron active nowhere (constant)
    gets entropy nan and prefers -1."""
    language = np.asarray(language)
    groups = [language == lang for lang in languages]
    if missing := [lang for lang, g in zip(languages, groups) if not g.any()]:
        raise ValueError(f"no episodes in {missing}")
    k, (layers, width) = len(languages), A.shape[1:]
    mean = np.empty((k, layers, width), dtype=np.float32)
    probability = np.empty((k, layers, width), dtype=np.float32)
    for layer in range(layers):
        x = A[:, layer].astype(np.float32)
        size = np.abs(x)
        active = size > np.quantile(size, quantile, axis=0)
        for j, g in enumerate(groups):
            mean[j, layer] = x[g].mean(axis=0)
            probability[j, layer] = active[g].mean(axis=0)
    total = probability.sum(axis=0)
    share = np.divide(probability, total, out=np.zeros_like(probability), where=total > 0)
    terms = np.where(share > 0, share * np.log(np.where(share > 0, share, 1.0)), 0.0)
    entropy = np.where(total > 0, -terms.sum(axis=0), np.nan).astype(np.float32)
    prefers = np.where(total > 0, probability.argmax(axis=0), -1)
    return {"mean": mean, "probability": probability, "entropy": entropy, "prefers": prefers}


def candidates(scores: dict[str, np.ndarray], per_layer: int) -> list[tuple[int, int, str]]:
    """The union over `scores` ({reason: [layers, width]}) of each layer's `per_layer`
    neurons with the largest |score|, as (layer, neuron, reason) sorted by layer then
    neuron. A neuron picked by more than one score keeps the first reason in `scores`'
    order. Nan scores are never picked."""
    picked: dict[tuple[int, int], str] = {}
    for reason, score in scores.items():
        size = np.abs(np.asarray(score, dtype=np.float64))
        size[np.isnan(size)] = -np.inf
        for layer, row in enumerate(size):
            top = np.argsort(-row, kind="stable")[: min(per_layer, int((row > -np.inf).sum()))]
            for neuron in top.tolist():
                picked.setdefault((layer, neuron), reason)
    return [(layer, neuron, reason) for (layer, neuron), reason in sorted(picked.items())]


def _top(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the `k` largest `values`, largest first."""
    k = min(k, len(values))
    idx = np.argpartition(-values, k - 1)[:k] if k else np.array([], dtype=int)
    return idx[np.argsort(-values[idx], kind="stable")]


def logit_lens(
    w_out: np.ndarray, unembed: np.ndarray, norm_weight: np.ndarray | None = None, k: int = 20, chunk: int = 32768
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """A neuron's direct effect on the next-token logits: `unembed` [vocab, d_model] times its
    output direction `w_out` [d_model], scaled first by the final norm's per-dimension scale
    `norm_weight` when given. The norm's division by the residual's size is left out: it
    scales every logit alike. Pass the scale the norm actually applies, which for Qwen3.5
    (whose RMSNorm multiplies by 1 + weight) is `1 + norm.weight`.

    Returns the `k` most promoted tokens, largest first, and the `k` most suppressed, most
    negative first, each as [(token_id, value)]."""
    v = np.asarray(w_out, dtype=np.float32)
    if norm_weight is not None:
        v = v * np.asarray(norm_weight, dtype=np.float32)
    logits = np.concatenate(
        [np.asarray(unembed[s : s + chunk], dtype=np.float32) @ v for s in range(0, len(unembed), chunk)]
    )
    up, down = _top(logits, k), _top(-logits, k)
    return [(int(i), float(logits[i])) for i in up], [(int(i), float(logits[i])) for i in down]


def _cosines(v: np.ndarray, rows: np.ndarray, k: int, chunk: int) -> list[tuple[int, float]]:
    """The `k` rows with the largest |cosine| to `v`, as [(row, signed cosine)], largest
    |cosine| first. Zero rows score 0."""
    v = np.asarray(v, dtype=np.float32)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    best_idx, best_cos = np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    for start in range(0, len(rows), chunk):
        block = np.asarray(rows[start : start + chunk], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1)
        cos = np.divide(block @ v, norms, out=np.zeros(len(block), dtype=np.float32), where=norms > 1e-12)
        keep = _top(np.abs(cos), k)
        best_idx = np.concatenate([best_idx, start + keep])
        best_cos = np.concatenate([best_cos, cos[keep]])
        top = _top(np.abs(best_cos), k)
        best_idx, best_cos = best_idx[top], best_cos[top]
    return [(int(i), float(c)) for i, c in zip(best_idx, best_cos)]


def wiring(w_out: np.ndarray, later_read: np.ndarray, k: int = 20, chunk: int = 32768) -> list[tuple[int, float]]:
    """Direct-path links out of a neuron: the `k` read directions (rows of `later_read`,
    [neurons, d_model], such as later layers' `gate_proj.weight` rows stacked) whose
    |cosine| with its output direction `w_out` [d_model] is largest, as [(row, signed
    cosine)]. Positive means what it writes pushes that neuron's input up. For the path as
    the reading layer sees it, scale its rows by that layer's pre-MLP norm scale first."""
    return _cosines(w_out, later_read, k, chunk)


def wiring_up(w_read: np.ndarray, earlier_out: np.ndarray, k: int = 20, chunk: int = 32768) -> list[tuple[int, float]]:
    """Direct-path links into a neuron: the `k` output directions (rows of `earlier_out`,
    [neurons, d_model], such as earlier layers' `down_proj.weight.T` stacked) whose |cosine|
    with its read direction `w_read` [d_model] is largest, as [(row, signed cosine)]."""
    return _cosines(w_read, earlier_out, k, chunk)


def percentile_vs_controls(effect: float | ArrayLike, controls: ArrayLike) -> float | np.ndarray:
    """Where |effect| falls among |controls| (the same measurement after pushes along random
    directions), from 0 (below every control) to 100 (above every one); ties count half.
    `effect` can be an array, each compared with the same controls. Nan controls are dropped
    and a nan effect gives nan."""
    pool = np.abs(np.asarray(controls, dtype=np.float64).ravel())
    pool = np.sort(pool[~np.isnan(pool)])
    if not len(pool):
        raise ValueError("no controls to compare against")
    size = np.abs(np.asarray(effect, dtype=np.float64))
    below = np.searchsorted(pool, size, side="left") + np.searchsorted(pool, size, side="right")
    out = np.where(np.isnan(size), np.nan, 50.0 * below / len(pool))
    return float(out) if out.ndim == 0 else out
