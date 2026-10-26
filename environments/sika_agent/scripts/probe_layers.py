"""Layer-by-layer probes over a `dump_activations.py` output.

Three questions, each answered per layer:

0. **Where does the model tell the languages apart?** A probe for the language itself.
1. **Does the model understand the request in each language?** A logistic-regression probe
   predicts the task's category (payments, credit, scams, check first, injection) and,
   separately, whether the request is unsafe (a scam or planted instruction). Trained and
   tested within one language, holding out whole seeds.
2. **Is that understanding shared across languages?** The same probes trained on English
   and tested on Twi and Twi–English. High within-language but low cross-language accuracy
   means separate, language-specific representations.
3. **How close are the matched tasks?** Cosine similarity between the English and Twi
   activations of the same task (same wallet, same seed), per layer.

If `--behaviour` points at a results file, probe accuracy is set beside task success per
language, which is the comprehension-versus-policy split: a Twi probe that is as accurate as
the English one while Twi success is far lower points at the policy, not comprehension.

    uv run --with scikit-learn --with matplotlib python scripts/probe_layers.py acts/q35-4b

Writes `probes.csv`, `similarity.csv`, `fertility.csv` and `probes.png` beside the dump.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(dump: Path, position: str):
    meta = [json.loads(line) for line in (dump / "meta.jsonl").open()]
    acts = np.load(dump / f"{position}.npy").astype(np.float32)
    if len(meta) != len(acts):
        raise SystemExit(f"{len(meta)} meta rows but {len(acts)} activations")
    return meta, acts


def probe(x_train, y_train, x_test, y_test) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(set(y_train)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))


def grouped_folds(seeds: np.ndarray, k: int):
    """Split by seed, so a task's matched copies and siblings never straddle train and test."""
    unique = np.unique(seeds)
    rng = np.random.default_rng(0)
    rng.shuffle(unique)
    for fold in np.array_split(unique, min(k, len(unique))):
        test = np.isin(seeds, fold)
        yield ~test, test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--position", default="user_mean", choices=["user_mean", "last"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--every", type=int, default=1, help="Probe every Nth layer (faster on big models)")
    ap.add_argument("--behaviour", type=Path, help="JSON {language: success rate} from an eval of this model")
    args = ap.parse_args()

    meta, acts = load(args.dump, args.position)
    languages = sorted({m["language"] for m in meta}, key=["en", "tw", "en-tw"].index)
    lang = np.array([m["language"] for m in meta])
    seeds = np.array([m["seed"] for m in meta])
    labels = {
        "category": np.array([m["category"] for m in meta]),
        "unsafe": np.array([m["unsafe"] for m in meta]),
    }
    layers = list(range(0, acts.shape[1], args.every))
    rows = []

    for target, y in labels.items():
        for layer in layers:
            x = acts[:, layer]
            for language in languages:  # within one language, held-out seeds
                sel = lang == language
                scores = [
                    probe(x[sel][tr], y[sel][tr], x[sel][te], y[sel][te])
                    for tr, te in grouped_folds(seeds[sel], args.folds)
                ]
                rows.append((target, layer, language, language, float(np.nanmean(scores))))
            if "en" in languages:  # trained on English, tested on the others
                for language in languages:
                    if language == "en":
                        continue
                    scores = []
                    for tr, te in grouped_folds(seeds, args.folds):
                        train = tr & (lang == "en")
                        test = te & (lang == language)
                        scores.append(probe(x[train], y[train], x[test], y[test]))
                    rows.append((target, layer, "en", language, float(np.nanmean(scores))))
        print(f"probed {target}")

    for layer in layers:  # which language is this? (a sanity check and the "language track")
        x = acts[:, layer]
        scores = [probe(x[tr], lang[tr], x[te], lang[te]) for tr, te in grouped_folds(seeds, args.folds)]
        rows.append(("language", layer, "all", "all", float(np.nanmean(scores))))
    print("probed language")

    with (args.dump / "probes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "layer", "train_language", "test_language", "accuracy"])
        w.writerows(rows)

    sim_rows = similarity(meta, acts, layers, languages)
    with (args.dump / "similarity.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "pair", "cosine", "cosine_mismatched"])
        w.writerows(sim_rows)

    fert = defaultdict(list)
    for m in meta:
        fert[m["language"]].append(m["fertility"])
    with (args.dump / "fertility.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "mean_user_tokens_per_word", "tasks"])
        for language in languages:
            w.writerow([language, f"{np.mean(fert[language]):.2f}", len(fert[language])])

    behaviour = json.loads(args.behaviour.read_text()) if args.behaviour else None
    plot(rows, sim_rows, languages, args.dump / "probes.png", behaviour)
    summarize(rows, sim_rows, fert, languages, behaviour)


def similarity(meta, acts, layers, languages):
    """Mean cosine between matched tasks (same family and seed) in two languages, with a
    baseline of mismatched pairs (same family, different seed) to show what 'close' means."""
    index = {(m["family"], m["seed"], m["language"]): i for i, m in enumerate(meta)}
    rows = []
    for other in languages:
        if other == "en":
            continue
        pairs = [
            (index[(f, s, "en")], index[(f, s, other)])
            for (f, s, la) in index
            if la == "en" and (f, s, other) in index
        ]
        if not pairs:
            continue
        a, b = np.array(pairs).T
        rng = np.random.default_rng(0)
        shuffled = b.copy()
        for f in {meta[i]["family"] for i in a}:  # shuffle within family: same task type, other wallet
            idx = np.where([meta[i]["family"] == f for i in a])[0]
            shuffled[idx] = b[rng.permutation(idx)]
        for layer in layers:
            x = acts[:, layer]
            x = x - x.mean(0)  # centre, so the shared mean direction doesn't swamp the cosine
            rows.append((layer, f"en~{other}", cosine(x[a], x[b]), cosine(x[a], x[shuffled])))
    return rows


def cosine(u, v) -> float:
    num = (u * v).sum(1)
    den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-8
    return float((num / den).mean())


def plot(rows, sim_rows, languages, path, behaviour) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"en": "#1f6feb", "tw": "#d1495b", "en-tw": "#e0a100"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, target in zip(axes[:2], ["category", "unsafe"]):
        for language in languages:
            pts = [(r[1], r[4]) for r in rows if r[0] == target and r[2] == r[3] == language]
            ax.plot(*zip(*pts), color=colors[language], label=f"{language} (within)")
            cross = [(r[1], r[4]) for r in rows if r[0] == target and r[2] == "en" and r[3] == language != "en"]
            if cross:
                ax.plot(*zip(*cross), color=colors[language], linestyle="--", label=f"en→{language}")
        ax.set_title(f"Probe: {target}")
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out accuracy")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    ax = axes[2]
    for pair in sorted({r[1] for r in sim_rows}):
        other = pair.split("~")[1]
        pts = [(r[0], r[2]) for r in sim_rows if r[1] == pair]
        base = [(r[0], r[3]) for r in sim_rows if r[1] == pair]
        ax.plot(*zip(*pts), color=colors[other], label=f"{pair} matched")
        ax.plot(*zip(*base), color=colors[other], linestyle=":", label=f"{pair} other wallet")
    ax.set_title("Matched-task similarity")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine (centred)")
    ax.legend(fontsize=8)
    if behaviour:
        fig.suptitle("Task success: " + ", ".join(f"{k} {v:.0%}" for k, v in behaviour.items()))
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def summarize(rows, sim_rows, fert, languages, behaviour) -> None:
    print("\nBest layer per probe (held-out accuracy):")
    for target in ["category", "unsafe"]:
        for train, test in [(la, la) for la in languages] + [("en", la) for la in languages if la != "en"]:
            pts = [(r[4], r[1]) for r in rows if r[0] == target and r[2] == train and r[3] == test]
            if pts:
                acc, layer = max(pts)
                print(f"  {target:8s} {train:>5s}→{test:<5s} {acc:.0%} at layer {layer}")
    print("\nUser tokens per word: " + ", ".join(f"{la} {np.mean(fert[la]):.2f}" for la in languages))
    if behaviour:
        print("Task success:         " + ", ".join(f"{k} {v:.0%}" for k, v in behaviour.items()))


if __name__ == "__main__":
    main()
