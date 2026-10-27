"""Compact encodings for the viewer's steering files, and the packer that writes them.

Numbers ship as base64 byte arrays the page decodes into typed arrays:
- shifts (nats, either sign): signed square root in int8, x = sign(q)·(q/20)²; exact to
  ~0.003 near zero, ~0.3 at ±10, saturating at ±40.3; -128 marks a missing value;
- log-probabilities (≤ 0): uint8 in quarter-nats, x = -q/4, down to -63.5; 255 marks missing;
- percentiles: uint8 0–100; 255 marks missing.
"""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SQ = 20.0
MISSING_I8 = -128
MISSING_U8 = 255


def sq8(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    q = np.clip(np.round(np.sign(x) * np.sqrt(np.abs(x)) * SQ), -127, 127)
    return np.where(np.isfinite(x), q, MISSING_I8).astype(np.int8)


def unsq8(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.where(q == MISSING_I8, np.nan, np.sign(q) * (q / SQ) ** 2)


def lp8(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    q = np.clip(np.round(-x * 4), 0, 254)
    return np.where(np.isnan(x), MISSING_U8, np.where(np.isneginf(x), 254, q)).astype(np.uint8)


def unlp8(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.where(q == MISSING_U8, np.nan, -q / 4)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def percentiles(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Share (0–100) of `controls` whose magnitude is below each value's: values [..., V],
    controls [..., C] with the same leading shape."""
    v = np.abs(values)[..., :, None]
    c = np.abs(controls)[..., None, :]
    ok = np.isfinite(c)
    below = ((c < v) & ok).sum(-1)
    n = ok.sum(-1)
    out = np.where(n > 0, 100.0 * below / np.maximum(n, 1), np.nan)
    return np.where(np.isfinite(values), out, np.nan)


def _excerpt(text: str | None, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def transcripts(paths: list[str], model: str, wanted: set[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    """For each (task_id, turn index): the request, the message just before the turn, the
    model's reasoning there and what it did."""
    from sika_interp.traces import load, turns

    out = {}
    for path in paths:
        for trace in load(path):
            if trace["agent"]["config"]["model"] != model:
                continue
            tid = trace["task"]["data"]["id"]
            idx = [i for t, i in wanted if t == tid]
            if not idx:
                continue
            ts = turns(trace)
            for i in idx:
                t = ts[i]
                calls = [
                    f"{c['function']['name']}({_excerpt(c['function'].get('arguments'), 160)})"
                    for c in t.response.get("tool_calls") or []
                ]
                out[(tid, i)] = {
                    "request": _excerpt(next((m["content"] for m in t.messages if m["role"] == "user"), ""), 600),
                    "before": _excerpt(str(t.messages[-1].get("content") or ""), 600) if i else "",
                    "before_role": t.messages[-1]["role"] if i else "",
                    "reasoning": _excerpt(t.response.get("reasoning_content"), 2400),
                    "did": calls or [_excerpt(t.response.get("content"), 600)],
                    "did_reply": not calls,
                }
    return out


def pack_steering(full: Path, npz_dir: Path | None, paths: list[str], out_dir: Path, key: str) -> dict:
    """Write the steering files for one model and return the inline summary.

    - steer-<key>.json: every decision's facts, and for every neuron × decision × sign × dose
      the shift of the right-move margin (and of the harm margin where defined) with its
      percentile among the random pushes at the neuron's layer, same decision, sign and dose;
    - steer-<key>-<family>.json: per decision, the transcript excerpt, the move distribution
      unsteered (plain pass and zero push) and under every neuron push, and the random
      pushes' margin shifts at every layer.
    The summary has, per neuron, mean shifts and percentiles by language, sign and dose."""
    data = json.loads(full.read_text())
    dirs = data["directions"]
    doses, K = data["doses"], len(data["doses"])
    neuron_idx = [j for j, d in enumerate(dirs) if d[0] == "neuron"]
    neurons = [[dirs[j][1], dirs[j][2]] for j in neuron_idx]
    layer_of = np.array([dirs[j][1] for j in neuron_idx])
    controls_by_layer: dict[int, list[int]] = defaultdict(list)
    for j, d in enumerate(dirs):
        if d[0] == "control":
            controls_by_layer[d[1]].append(j)
    layers = sorted(controls_by_layer)
    rows = [r for r in data["decisions"] if "error" not in r]
    rows.sort(key=lambda r: (r["family"], r["seed"], r["kind"], ["en", "en-tw", "tw"].index(r["language"])))
    N, D = len(neuron_idx), len(rows)

    def matrix(name: str, chosen: list[int]):
        shift = np.full((len(chosen), N, 2, K), np.nan)
        pct = np.full((len(chosen), N, 2, K), np.nan)
        for a, i in enumerate(chosen):
            s = np.asarray(rows[i][f"{name}_shift"], dtype=np.float64)  # [dirs, 2, K]
            shift[a] = s[neuron_idx]
            for layer in set(layer_of.tolist()):
                mine = np.where(layer_of == layer)[0]
                ctrl = s[controls_by_layer[layer]]  # [C, 2, K]
                vals = s[[neuron_idx[m] for m in mine]]  # [n, 2, K]
                pct[a, mine] = np.moveaxis(percentiles(np.moveaxis(vals, 0, -1), np.moveaxis(ctrl, 0, -1)), -1, 0)
        return shift, pct

    right_rows = [i for i, r in enumerate(rows) if "right_shift" in r]
    harm_rows = [i for i, r in enumerate(rows) if "harm_shift" in r]
    r_shift, r_pct = matrix("right", right_rows)
    h_shift, h_pct = matrix("harm", harm_rows)
    families = sorted({r["family"] for r in rows})
    decisions = [
        {
            "id": r["task_id"], "kind": r["kind"], "family": r["family"], "category": r["category"],
            "language": r["language"], "seed": r["seed"], "turn": r["index"], "reward": r["reward"],
            "took": r["took"], "rule": r["rule"], "right_margin": r.get("right_margin"),
            "harm_margin": r.get("harm_margin"), "flagged": bool(r.get("flagged")),
        }
        for r in rows
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"steer-{key}.json").write_text(json.dumps({
        "doses": doses, "signs": [-1, 1], "neurons": neurons, "layers": layers,
        "controls_per_layer": data["controls_per_layer"], "decisions": decisions,
        "right": {"rows": right_rows, "shift": b64(sq8(r_shift)), "pct": b64(np.where(np.isnan(r_pct), MISSING_U8, np.round(r_pct)).astype(np.uint8))},
        "harm": {"rows": harm_rows, "shift": b64(sq8(h_shift)), "pct": b64(np.where(np.isnan(h_pct), MISSING_U8, np.round(h_pct)).astype(np.uint8))},
    }, separators=(",", ":")))

    texts = transcripts(paths, data["model"], {(r["task_id"], r["index"]) for r in rows})
    for fam in families:
        detail = []
        for i, r in enumerate(rows):
            if r["family"] != fam:
                continue
            item = {"i": i, "labels": r["labels"], "base": b64(lp8(r["base"])), "text": texts.get((r["task_id"], r["index"]), {})}
            npz = npz_dir / r["saved"] if npz_dir else None
            if npz is not None and npz.exists():
                z = np.load(npz)
                eff = z["effects"].astype(np.float32)
                item["zero"] = b64(lp8(z["zero"] if "zero" in z.files else z["base"]))
                item["push"] = b64(lp8(eff[neuron_idx]))  # [N, 2, K, labels]
            for name in ("right", "harm"):
                if f"{name}_shift" in r:
                    s = np.asarray(r[f"{name}_shift"], dtype=np.float64)
                    item[f"{name}_controls"] = b64(sq8(np.stack([s[controls_by_layer[l]] for l in layers])))  # [L, C, 2, K]
            detail.append(item)
        (out_dir / f"steer-{key}-{fam}.json").write_text(json.dumps(detail, ensure_ascii=False, separators=(",", ":")))

    # Inline summary per neuron: mean shift and mean percentile by language, sign and dose.
    langs = ["en", "en-tw", "tw"]
    summary = {"neurons": neurons, "n_decisions": D, "families": families, "langs": langs, "doses": doses}
    for name, shift, pct, chosen in (("right", r_shift, r_pct, right_rows), ("harm", h_shift, h_pct, harm_rows)):
        lang_of = np.array([rows[i]["language"] for i in chosen])
        means, pcts, beats = {}, {}, {}
        for lang in langs:
            m = lang_of == lang
            with np.errstate(all="ignore"):
                means[lang] = np.round(np.nanmean(shift[m], 0), 3).tolist() if m.any() else None  # [N, 2, K]
                pcts[lang] = np.round(np.nanmean(pct[m], 0), 1).tolist() if m.any() else None
                beats[lang] = np.round(np.nanmean(pct[m] >= 97, 0), 3).tolist() if m.any() else None
        summary[name] = {"mean_shift": means, "mean_pct": pcts, "share_beats_97": beats}
    return summary


def rows_from_npz(npz_dir: Path, decisions: list[dict], names: set[str] | None = None) -> list[dict]:
    """Steering rows rebuilt from the arrays `Model.steer_moves` saves, for decisions whose
    returned rows were lost: margins and shifts from the arrays and each decision's rule, as
    `Pushes.summary` computes them (the saved pushes are float16, so shifts can differ from
    a returned row's by ~0.01 nats). `decisions` as `decisions.collect` gives them."""
    from sika_interp.decisions import margin_sides

    out = []
    for d in decisions:
        name = f"{d['task_id'].replace('/', '_')}_{d['kind']}.npz"
        path = npz_dir / name
        if (names is not None and name not in names) or not path.exists():
            continue
        z = np.load(path)
        labels = [str(x) for x in z["labels"]]
        groups = {lab: lab.split("/") for lab in labels[1:]}
        base = z["base"].astype(np.float64)
        zero = (z["zero"] if "zero" in z.files else z["base"]).astype(np.float64)
        eff = z["effects"].astype(np.float64)
        row = {k: d[k] for k in ("task_id", "family", "category", "language", "seed", "kind", "index", "reward", "took", "rule")}
        row |= {"labels": labels, "groups": groups, "base": [round(float(x), 4) for x in base], "saved": name,
                "rebuilt": True, "flagged": False}
        for margin, (a, b) in margin_sides(labels, groups, d["rule"]).items():
            m0 = np.logaddexp.reduce(base[a]) - np.logaddexp.reduce(base[b])
            start = np.logaddexp.reduce(zero[a]) - np.logaddexp.reduce(zero[b])
            pushed = np.logaddexp.reduce(eff[..., a], -1) - np.logaddexp.reduce(eff[..., b], -1)
            row[f"{margin}_margin"] = round(float(m0), 4)
            row[f"{margin}_shift"] = np.round(pushed - start, 3).tolist()
            row[f"{margin}_route_gap"] = round(float(start - m0), 4)
        out.append(row)
    return out


def assemble_from_npz(npz_dir: Path, decisions: list[dict], cards: list[dict], names: set[str], model: str,
                      run: str, doses: list[float], controls: int) -> dict:
    """A steering result in `steer_all`'s shape, rebuilt entirely from saved arrays: the
    directions in `Model.steer_moves`' order (the cards' neurons sorted, then `controls`
    random directions per layer used) and a row per decision in `names`."""
    picked = sorted({(c["layer"], c["neuron"]) for c in cards})
    used = sorted({layer for layer, _ in picked})
    directions = [["neuron", layer, i] for layer, i in picked] + [["control", layer, c] for layer in used for c in range(controls)]
    return {"model": model, "run": run, "doses": doses, "signs": [-1, 1], "controls_per_layer": controls,
            "directions": directions, "decisions": rows_from_npz(npz_dir, decisions, names)}


def repair_pieces(pieces: list[str], text: str) -> list[str]:
    """Token texts with characters split across tokens restored from the original text. A
    character whose bytes span two tokens decodes as U+FFFD in the first and nothing in the
    second; each U+FFFD stands for one character of `text` at the same place. The pieces end
    where `text` ends (they may start a little before it). Pieces are returned unchanged if
    they don't line up with `text`."""
    off = sum(len(p) for p in pieces) - len(text)
    if off < 0:
        return pieces
    out, g = [], 0
    for p in pieces:
        fixed = []
        for ch in p:
            if g >= off:
                want = text[g - off]
                if ch == "\ufffd":
                    ch = want
                elif ch != want:
                    return pieces
            fixed.append(ch)
            g += 1
        out.append("".join(fixed))
    return out
