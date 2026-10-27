"""Gather the study's results for the viewer page.

    python -m sika_interp.viewer --runs <notes>/runs --out <dir> [--npz <dir>]

Reads the traces and the summaries the Modal jobs wrote from `--runs`, laid out as the study
left them, and writes into `--out`: data.json (everything the page draws inline), the page
itself (page.html, data inlined) and the files it fetches on demand: steer-<model>.json (the
neuron × decision steering matrix), steer-<model>-<family>.json (per-decision detail) and
cards-<model>.json (neuron cards with their top-firing contexts). Missing pieces are left out.
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
SOURCES = {
    "tokens": "read2/tokens-study-v2-{name}.json",
    "cues": "read2/cues-{run}-{name}.json",
    "contexts": "read2/contexts-{key}.json",
    "maps": "maps2",
    "suppress": "anchors2/suppress-{key}.json",
    "receivers": "anchors2/receivers-{key}.json",
    "steering": "steer/full-{key}.json",
    "steered": "steer/steered-{key}.jsonl",
    "plan": "steer/plan-{key}.json",
}
"""Where the re-runs after the tag fix (2026-10-27) put their results, under --runs."""


def traces_of(runs: Path, key: str) -> list[str]:
    return [str(runs / f"{r}-{key}" / "traces.jsonl") for r in STUDY_RUNS if (runs / f"{r}-{key}" / "traces.jsonl").exists()]


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


def by_family(runs: Path, key: str) -> dict:
    """Success and harm per task family and language over the matched twins, and per family
    the paired flips: twins right in English but not in Twi–English (or Twi), and the reverse."""
    eps, _ = episodes(traces_of(runs, key))
    out: dict = defaultdict(lambda: {lang: {"n": 0, "right": 0, "harmful": 0} for lang in LANGUAGES})
    twins: dict = defaultdict(dict)
    for e in eps:
        cell = out[e.family][e.language]
        cell["n"] += 1
        cell["right"] += e.reward == 1.0
        cell["harmful"] += e.reward == -1.0
        twins[(e.family, e.seed)][e.language] = e.reward == 1.0
    for fam, cells in out.items():
        full = [v for (f, _), v in twins.items() if f == fam and set(v) == set(LANGUAGES)]
        cells["flips"] = {
            lang: {"lost": sum(v["en"] and not v[lang] for v in full), "gained": sum(v[lang] and not v["en"] for v in full)}
            for lang in ("en-tw", "tw")
        }
    return dict(sorted(out.items()))


def placement_outcome(runs: Path, key: str, permutations: int = 4000, seed: int = 0) -> dict | None:
    """Does where a Twi–English request sits between its twins predict the outcome? Per read
    point and layer: the mean placement margin (nearer English twin > 0) of failed minus
    successful Twi–English episodes, centred within task type, and a within-type
    permutation p-value. Only types with both outcomes count."""
    import numpy as np

    link = _read(runs / "read" / f"link-{MODELS[key]['run']}-{MODELS[key]['name']}.json")
    points = _read(runs / "read" / f"read-points-{MODELS[key]['run']}.json")
    if not link or not points:
        return None
    reward = {(r["family"], r["language"], r["seed"]): r["reward"] for r in points}
    rng = np.random.default_rng(seed)
    out = {}
    for point, block in link.items():
        rows = [e for e in block["episodes"] if e["language"] == "en-tw" and (e["family"], "en-tw", e["seed"]) in reward]
        width = max(len(e["margin"] or []) for e in rows)
        rows = [e for e in rows if e["margin"] and len(e["margin"]) == width]  # both twins present
        fam = np.array([e["family"] for e in rows])
        fail = np.array([reward[(e["family"], "en-tw", e["seed"])] != 1.0 for e in rows])
        keep = np.array([fail[fam == f].any() and (~fail[fam == f]).any() for f in fam])
        fam, fail = fam[keep], fail[keep]
        margin = np.array([[np.nan if v is None else v for v in e["margin"]] for e, k in zip(rows, keep) if k], dtype=float)
        for f in set(fam):
            margin[fam == f] -= np.nanmean(margin[fam == f], 0)
        groups = [np.where(fam == f)[0] for f in sorted(set(fam))]

        def diff(lab):
            return np.nanmean(margin[lab], 0) - np.nanmean(margin[~lab], 0)

        observed = diff(fail)
        null = np.empty((permutations, margin.shape[1]))
        for k in range(permutations):
            lab = fail.copy()
            for g in groups:
                lab[g] = rng.permutation(lab[g])
            null[k] = diff(lab)
        p = (1 + (np.abs(null) >= np.abs(observed)).sum(0)) / (permutations + 1)
        out[point] = {"diff": np.round(observed, 4).tolist(), "p": np.round(p, 5).tolist(),
                      "episodes": int(len(fam)), "failed": int(fail.sum()), "types": len(groups)}
    return out


def anchors(runs: Path, key: str) -> list[dict]:
    """Each replayed episode: its steps, the resampling importance of its model turns, the
    step-to-turn links by deletion and by attention blocking, and the receiver heads' score
    for every sentence of its last turn."""
    resampled = _read(runs / "anchors" / f"anchors-{key}.json")
    links = _read(runs / "anchors" / f"anchors-links-{MODELS[key]['name']}.json")
    if not resampled or not links:
        return []
    blocked = {r["task_id"]: r for r in _read(runs / SOURCES["suppress"].format(key=key)) or []}
    receivers = _read(runs / SOURCES["receivers"].format(key=key)) or {"items": []}
    last = {it["task_id"]: it for it in receivers["items"] if it["kind"] == "last"}
    out = []
    for ep in links:
        turns_ = resampled["episodes"].get(ep["task_id"], [])
        means = [sum(t["rewards"]) / len(t["rewards"]) if t["rewards"] else None for t in turns_]
        b = blocked.get(ep["task_id"], {})
        r = last.get(ep["task_id"])
        out.append(
            {
                "task_id": ep["task_id"],
                "steps": ep["steps"],
                "turn_step": ep["turn_step"],
                "links": ep["links"],
                "blocked": [x[:3] for x in b.get("links", [])],
                "blocked_layers": b.get("layers_blocked"),
                "outcome_from_turn": means,
                "replays": resampled["n"],
                "sentences": [
                    {"text": s["text"], "message": s["message"], "role": s["role"], "part": s["part"], "score": s["score"]}
                    for s in r["sentences"]
                ] if r else None,
            }
        )
    return out


def receivers(runs: Path, key: str, boot: int = 2000, seed: int = 0) -> dict | None:
    """Receiver heads: each full-attention head's mean kurtosis, the top heads, and the mean
    receiver score per sentence of the request, system prompt and reasoning by request
    language, with twin-paired differences against English (bootstrap over twins)."""
    import numpy as np

    r = _read(runs / SOURCES["receivers"].format(key=key))
    if not r:
        return None
    rng = np.random.default_rng(seed)
    per: dict = defaultdict(dict)  # (family, seed) -> lang -> {source: mean}
    for it in r["items"]:
        if it["kind"] != "first":
            continue
        _, fam, lang, sd = it["task_id"].split("/")
        sums: dict = defaultdict(list)
        for s in it["sentences"]:
            if s["score"] is None:
                continue
            src = "system" if s["role"] == "system" else "request" if s["role"] == "user" else "reasoning" if s["part"] == "reasoning" else "action"
            sums[src].append(s["score"])
        per[(fam, sd)][lang] = {k: float(np.mean(v)) for k, v in sums.items()}
    sources = ("request", "system", "reasoning")
    table: dict = {}
    for src in sources:
        table[src] = {}
        for lang in LANGUAGES:
            v = np.array([d[lang][src] for d in per.values() if lang in d and src in d[lang]])
            table[src][lang] = {"mean": float(v.mean()) if len(v) else None, "n": int(len(v))}
        for lang in ("en-tw", "tw"):
            pairs = np.array([d[lang][src] - d["en"][src] for d in per.values() if lang in d and "en" in d and src in d[lang] and src in d["en"]])
            if len(pairs):
                means = np.array([rng.choice(pairs, len(pairs)).mean() for _ in range(boot)])
                table[src][f"{lang}_minus_en"] = {"mean": float(pairs.mean()), "low": float(np.percentile(means, 2.5)),
                                                  "high": float(np.percentile(means, 97.5)), "n": int(len(pairs))}
    return {"layers": r["layers"], "kurtosis": r["kurtosis"], "heads": r["receivers"], "by_source": table,
            "turns": len(r["items"])}


def neuron_map(maps: Path, key: str, background: int = 24000, seed: int = 0) -> dict | None:
    """Every candidate neuron and a random background sample, with place and scores
    quantised to int8 (scores ×20, places scaled per layer). Cards go to their own file."""
    import numpy as np

    base = maps / MODELS[key]["run"] / MODELS[key]["name"]
    if not (base / "map.npz").exists():
        return None
    z = np.load(base / "map.npz")
    cards = _read(base / "cards.json") or []
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
        "layers": int(layers), "width": int(width), "layer": L.tolist(), "neuron": N.tolist(),
        "x": xy[:, 0].tolist(), "y": xy[:, 1].tolist(), "prefers": z["prefers"][L, N].astype(int).tolist(),
        "scores": scores, "cards": [[c["layer"], c["neuron"], c["reason"]] for c in cards],
    }


def card_file(runs: Path, key: str) -> list[dict] | None:
    """The neuron cards with their top-firing request contexts per language (token texts and
    activations around the peak)."""
    base = runs / SOURCES["maps"] / MODELS[key]["run"] / MODELS[key]["name"]
    cards = _read(base / "cards.json")
    if not cards:
        return None
    ctx = _read(runs / SOURCES["contexts"].format(key=key)) or {"neurons": {}, "episodes": {}}
    if ctx["episodes"]:
        from sika_interp.packing import repair_pieces

        texts = {}
        for path in traces_of(runs, key):
            for trace in load(path):
                ts = turns(trace)
                if ts:
                    texts[trace["task"]["data"]["id"]] = ts[0].messages[-1]["content"]
        for ep in ctx["episodes"].values():
            if ep["task_id"] in texts:
                ep["pieces"] = repair_pieces(ep["pieces"], texts[ep["task_id"]])
    for c in cards:
        rows = ctx["neurons"].get(f"{c['layer']}:{c['neuron']}", {})
        c["contexts"] = {
            lang: [
                {"task_id": ctx["episodes"][str(r["episode"])]["task_id"], "peak": r["peak"], "at": r["at"] - r["from"],
                 "tokens": ctx["episodes"][str(r["episode"])]["pieces"][r["from"] : r["from"] + len(r["acts"])], "acts": r["acts"]}
                for r in rs if str(r["episode"]) in ctx["episodes"]
            ]
            for lang, rs in rows.items()
        }
    return cards


def steered(runs: Path, key: str, out: Path, turn_cap: int = 12) -> dict | None:
    """Steered transcripts: writes steered-<key>.json (every replay's outcome and transcript,
    trimmed) and returns the summary the page draws inline: per setting and request language,
    replays that ended right, harmful or neither."""
    path = runs / SOURCES["steered"].format(key=key)
    plan = _read(runs / SOURCES["plan"].format(key=key))
    if not path.exists() or not plan:
        return None

    def trim(m: dict) -> dict:
        if m["role"] == "verdict":
            return m
        t = {"role": m["role"]}
        if m.get("reasoning_content"):
            t["reasoning"] = m["reasoning_content"][:2000]
        if m.get("content"):
            t["content"] = str(m["content"])[:1200]
        if m.get("tool_calls"):
            t["calls"] = [f"{c['function']['name']}({c['function']['arguments'][:200]})" for c in m["tool_calls"]]
        if m["role"] == "tool":
            t["name"] = m.get("name")
        return t

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        outcome, note = r.get("outcome"), None
        # One rule for every replay: one that needed more than `turn_cap` model turns from its
        # decision did not finish (the first settings ran with a looser cap).
        if outcome is not None and sum(m.get("role") == "assistant" for m in r.get("transcript", [])) > turn_cap:
            outcome, note = 0.0, f"not finished within {turn_cap} turns"
        error = r.get("error")
        if error and "Function name was" in error:
            note = "broken tool call (the server rejected the call's format)"
        by[r["setting"]][f"{r['task_id']}@{r['turn']}"].append(
            {"sample": r["sample"], "outcome": outcome, "error": error, "note": note,
             "transcript": [trim(m) for m in r.get("transcript", [])]}
        )
    names = [s["name"] for s in __import__("sika_interp.steered", fromlist=["settings"]).settings(plan)]
    decisions = [f"{d['task_id']}@{d['turn']}" for d in plan["decisions"]]
    _dump(out / f"steered-{key}.json", {"settings": names, "decisions": decisions, "results": by})
    summary = {}
    for name in names:
        cells = {}
        for lang in LANGUAGES:
            xs = [x for dk, xs in by.get(name, {}).items() if dk.split("/")[2] == lang for x in xs]
            outs = [x["outcome"] for x in xs if x["outcome"] is not None]
            broken = sum(1 for x in xs if x["note"] and x["note"].startswith("broken"))
            # Broken tool calls count as replays that did not end right (the model failed the task).
            cells[lang] = {"n": len(outs) + broken, "right": sum(o == 1 for o in outs), "harmful": sum(o == -1 for o in outs),
                           "broken": broken, "unfinished": sum(1 for x in xs if x["note"] and x["note"].startswith("not finished"))}
        summary[name] = cells
    directions = {d["name"]: {k: d.get(k) for k in ("layer", "neuron", "seed", "why", "picked_for")} for d in plan["directions"]}
    return {"settings": names, "decisions": decisions, "summary": summary, "directions": directions,
            "doses": plan["doses"], "samples": plan.get("samples"), "turn_cap": turn_cap}


def _clean(value):
    """NaN and infinities as null: JSON has no NaN, and the page reads null as missing."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _dump(path: Path, value) -> int:
    # Partial-byte tokens decode to U+FFFD; keep them as JSON escapes, not raw characters.
    text = json.dumps(_clean(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("�", "\\ufffd")
    path.write_text(text)
    return len(text.encode())


def build(runs: Path, out: Path, npz: Path | None = None) -> dict:
    from sika_interp.packing import pack_steering

    out.mkdir(parents=True, exist_ok=True)
    data: dict = {"models": {}, "words_alone": words_alone(runs)}
    for key, meta in MODELS.items():
        read = runs / "read"
        model = {
            "name": meta["id"],
            "behaviour": behaviour(runs, key),
            "by_family": by_family(runs, key),
            "reasoning_language": reasoning_language(runs, key),
            "probes": _read(read / f"probes-{meta['run']}-{meta['name']}.json"),
            "placement_outcome": placement_outcome(runs, key),
            "cues": None,
            "tokens": _read(runs / SOURCES["tokens"].format(name=meta["name"])),
            "anchors": anchors(runs, key),
            "receivers": receivers(runs, key),
            "map": neuron_map(runs / SOURCES["maps"], key),
            "steering": None,
        }
        cues = _read(runs / SOURCES["cues"].format(run=meta["run"], name=meta["name"]))
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
        cards = card_file(runs, key)
        if cards:
            print(f"cards-{key}.json {_dump(out / f'cards-{key}.json', cards) / 1e6:.1f} MB")
        model["steered"] = steered(runs, key, out)
        full = runs / SOURCES["steering"].format(key=key)
        if full.exists():
            model["steering"] = pack_steering(full, (npz / meta["name"]) if npz else None, traces_of(runs, key), out, key)
        data["models"][key] = model
    print(f"data.json {_dump(out / 'data.json', data) / 1e6:.1f} MB")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--npz", type=Path, help="per-decision steering arrays downloaded from the volume (<dir>/<model name>/*.npz)")
    args = ap.parse_args()
    build(args.runs, args.out, args.npz)
    site = Path(__file__).resolve().parent.parent / "viewer"
    for f in site.iterdir():
        if f.is_file():
            (args.out / f.name).write_bytes(f.read_bytes())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
