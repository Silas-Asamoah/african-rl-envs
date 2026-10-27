"""Gather the study's results into one compact JSON for the viewer page.

    python -m sika_interp.viewer --runs <notes>/runs --out viewer-data.json [--page page.html]

Reads the traces and the summaries the Modal jobs wrote (probes, cues, tokens, anchors, maps)
from `--runs`, laid out as the study left them, and keeps only what the page draws. Missing
pieces are left out, so the page can be built before every job has finished.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sika_interp.results import episodes, wilson
from sika_interp.tasks import LANGUAGES
from sika_interp.traces import load, turns
from sika_interp.words import tag

MODELS = {
    "qwen": {"id": "Qwen/Qwen3.5-9B", "name": "Qwen3.5-9B", "run": "study"},
    "ministral": {"id": "mistralai/Ministral-3-8B-Reasoning-2512", "name": "Ministral-3-8B-Reasoning-2512", "run": "study-ministral"},
}
STUDY_RUNS = ("study-matched", "study-extra-entw", "study-more-twins")


def _read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def behaviour(runs: Path, key: str) -> dict:
    """Success and harm per language over every matched twin, and paired twin counts."""
    paths = [runs / f"{r}-{key}" / "traces.jsonl" for r in STUDY_RUNS]
    eps, errored = episodes([str(p) for p in paths if p.exists()])
    by_twin = defaultdict(dict)
    for e in eps:
        by_twin[(e.family, e.seed)][e.language] = e.reward
    twins = {k: v for k, v in by_twin.items() if set(v) == set(LANGUAGES)}
    out = {"errored": errored, "twins": len(twins), "languages": {}}
    for lang in LANGUAGES:
        rewards = [v[lang] for v in twins.values()]
        hits = sum(r == 1.0 for r in rewards)
        low, high = wilson(hits, len(rewards))
        out["languages"][lang] = {
            "n": len(rewards),
            "success": hits / len(rewards),
            "low": low,
            "high": high,
            "harmful": sum(r == -1.0 for r in rewards),
        }
    for other in ("tw", "en-tw"):
        out[f"en_vs_{other}"] = {
            "lost": sum(v["en"] == 1.0 and v[other] != 1.0 for v in twins.values()),
            "gained": sum(v["en"] != 1.0 and v[other] == 1.0 for v in twins.values()),
        }
    return out


def reasoning_language(runs: Path, key: str) -> dict:
    """Share of English, Twi and mixed words in the models' reasoning, by request language."""
    counts = {lang: defaultdict(int) for lang in LANGUAGES}
    for run in STUDY_RUNS:
        path = runs / f"{run}-{key}" / "traces.jsonl"
        if not path.exists():
            continue
        for trace in load(path):
            lang = trace["task"]["data"]["language"]
            for t in turns(trace):
                for w in tag(t.response.get("reasoning_content") or ""):
                    counts[lang][w.tag] += 1
    out = {}
    for lang, c in counts.items():
        total = c["en"] + c["tw"] + c["mixed"] or 1
        out[lang] = {"en": c["en"] / total, "tw": c["tw"] / total, "mixed": c["mixed"] / total, "words": total}
    return out


def words_alone(runs: Path) -> dict:
    """What a bag-of-words probe on the request text gets, as a baseline for the residual
    probes: within each language, and trained on English, tested on the others."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    from sika_agent.scenarios import FAMILIES
    from sika_interp.probes import folds_by_group

    rows = []
    for run in ("study-matched", "study-extra-entw"):
        path = runs / f"{run}-qwen" / "traces.jsonl"
        for trace in load(path):
            d = trace["task"]["data"]
            rows.append((d["prompt"], d["family"], FAMILIES[d["family"]][0] in ("scams", "injection"), d["language"], d["seed"]))
    text, family, threat, language, seed = (np.array(c, dtype=object) for c in zip(*rows))
    threat = threat.astype(bool)
    out = {}
    for name, y in (("family", family), ("threat", threat)):
        scores = defaultdict(list)
        for tr, te in folds_by_group(seed, 5):
            clf = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=2000))
            for lang in LANGUAGES:
                a, b = tr[language[tr] == lang], te[language[te] == lang]
                scores[f"within_{lang}"].append(clf.fit(text[a], y[a]).score(text[b], y[b]))
            a = tr[language[tr] == "en"]
            clf.fit(text[a], y[a])
            for lang in ("tw", "en-tw"):
                b = te[language[te] == lang]
                scores[f"from_en_{lang}"].append(clf.score(text[b], y[b]))
        out[name] = {k: float(np.mean(v)) for k, v in scores.items()}
    return out


def anchors(runs: Path, key: str) -> list[dict]:
    """Each replayed episode: its steps, the resampling importance of its model turns, and
    the step-to-turn links."""
    resampled = _read(runs / "anchors" / f"anchors-{key}.json")
    links = _read(runs / "anchors" / f"anchors-links-{MODELS[key]['name']}.json")
    if not resampled or not links:
        return []
    out = []
    for ep in links:
        turns_ = resampled["episodes"].get(ep["task_id"], [])
        means = [sum(t["rewards"]) / len(t["rewards"]) if t["rewards"] else None for t in turns_]
        out.append(
            {
                "task_id": ep["task_id"],
                "steps": ep["steps"],
                "turn_step": ep["turn_step"],
                "links": ep["links"],
                "outcome_from_turn": means,
                "replays": resampled["n"],
            }
        )
    return out


def neuron_map(maps: Path, key: str, background: int = 24000, seed: int = 0) -> dict | None:
    """Every candidate neuron and a random background sample, with place and scores
    quantised to int8 (scores ×20, places scaled per layer), plus cards and steering."""
    import numpy as np

    base = maps / MODELS[key]["run"] / MODELS[key]["name"]
    if not (base / "map.npz").exists():
        return None
    z = np.load(base / "map.npz")
    cards = _read(base / "cards.json") or []
    steering = _read(base / "steering.json")
    places = z["places"].astype(np.float32)  # [layers, width, 2]
    layers, width, _ = places.shape
    score_names = [k for k in z.files if k not in ("places", "prefers")]
    rng = np.random.default_rng(seed)
    flat = rng.choice(layers * width, size=min(background, layers * width), replace=False)
    chosen = sorted(set(flat.tolist()) | {c["layer"] * width + c["neuron"] for c in cards})
    L, N = np.divmod(np.array(chosen), width)
    scale = np.percentile(np.abs(places).reshape(layers, -1), 98, axis=1) + 1e-6  # outliers clip to the edge
    xy = np.clip(places[L, N] / scale[L, None] * 127, -127, 127).astype(int)
    scores = {k: np.clip(np.nan_to_num(z[k].astype(np.float32)[L, N]) * 20, -127, 127).astype(int).tolist() for k in score_names}
    return {
        "layers": int(layers),
        "width": int(width),
        "layer": L.tolist(),
        "neuron": N.tolist(),
        "x": xy[:, 0].tolist(),
        "y": xy[:, 1].tolist(),
        "prefers": z["prefers"][L, N].astype(int).tolist(),
        "scores": scores,
        "cards": cards,
        "steering": steering,
    }


def _clean(value):
    """NaN and infinities as null: JSON has no NaN, and the page reads null as missing."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def build(runs: Path) -> dict:
    data: dict = {"models": {}}
    data["words_alone"] = words_alone(runs)
    for key, meta in MODELS.items():
        read = runs / "read"
        model = {
            "name": meta["id"],
            "behaviour": behaviour(runs, key),
            "reasoning_language": reasoning_language(runs, key),
            "probes": _read(read / f"probes-{meta['run']}-{meta['name']}.json"),
            "cues": None,
            "tokens": _read(read / f"tokens-{meta['run']}-{meta['name']}.json"),
            "anchors": anchors(runs, key),
            "map": neuron_map(runs / "maps", key),
        }
        cues = _read(read / f"cues-{meta['run']}-{meta['name']}.json")
        if cues:
            layers = len(cues[0]["scores"]["as_written"])
            model["cues"] = {
                group: {
                    v: [sum(r["scores"][v][j] for r in rows) / len(rows) for j in range(layers)]
                    for v in ("as_written", "no_english", "no_twi", "random")
                }
                | {"n": len(rows)}
                for group, rows in (("threat", [r for r in cues if r["threat"]]), ("control", [r for r in cues if not r["threat"]]))
            }
        data["models"][key] = model
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--page", type=Path, help="also write the viewer page with the data inlined")
    args = ap.parse_args()
    data = build(args.runs)
    text = json.dumps(_clean(data), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    args.out.write_text(text)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    if args.page:
        template = (Path(__file__).resolve().parent.parent / "viewer" / "template.html").read_text()
        # Partial-byte tokens decode to U+FFFD; keep them as JSON escapes, not raw characters.
        inline = text.replace("</", "<\\/").replace("\ufffd", "\\ufffd")
        args.page.write_text(template.replace("/*DATA*/", inline))
        print(f"wrote {args.page}")


if __name__ == "__main__":
    main()
