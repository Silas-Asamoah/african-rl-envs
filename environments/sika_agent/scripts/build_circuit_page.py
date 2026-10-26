"""Build the circuit map page from a `dump_activations.py` + `probe_layers.py` output.

    python3 scripts/build_circuit_page.py acts/q35-4b \\
        --behaviour results/q35-4b-success.json --out acts/q35-4b/circuit.html

Reads `probes.csv`, `similarity.csv`, `meta.jsonl` and `config.json` from the dump folder and
embeds them in `circuit_viewer.html`. The page is self-contained: open it in a browser, or
publish it. Needs only the standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["en", "tw", "en-tw"]
# What each tool does to the wallet, for the first-action panel.
GROUPS = {
    "money": {"initiate_transfer", "confirm_transfer", "buy_airtime", "pay_bill", "make_loan_payment", "send_sms"},
    "protect": {"report_fraud", "cancel_transfer"},
}


def group_of(tool: str) -> str:
    return next((g for g, tools in GROUPS.items() if tool in tools), "check")


def first_actions(meta: list[dict]) -> dict:
    """Mean probability mass per tool group, softmaxed over the 17 tool scores of each task."""
    acc: dict = defaultdict(lambda: {"safe": defaultdict(float), "unsafe": defaultdict(float), "n_safe": 0, "n_unsafe": 0})
    for m in meta:
        scores = m.get("tool_logprobs")
        if not scores:
            continue
        top = max(scores.values())
        weights = {t: math.exp(v - top) for t, v in scores.items()}
        total = sum(weights.values())
        kind = "unsafe" if m["unsafe"] else "safe"
        a = acc[m["language"]]
        a["n_" + kind] += 1
        for tool, w in weights.items():
            a[kind][group_of(tool)] += w / total
    out = {}
    for lang, a in acc.items():
        out[lang] = {"n_safe": a["n_safe"], "n_unsafe": a["n_unsafe"]}
        for kind in ("safe", "unsafe"):
            n = max(1, a["n_" + kind])
            out[lang][kind] = {g: a[kind][g] / n for g in ("money", "check", "protect")}
    return out


def clean(v: float | None) -> float | None:
    """Rounded, with NaN (a probe that saw one class) as null for JSON."""
    return None if v is None or math.isnan(v) else round(v, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--out", type=Path, help="Default: <dump>/circuit.html")
    ap.add_argument("--behaviour", type=Path, help='JSON {"en": 0.86, "tw": 0.51, ...} of task success')
    ap.add_argument("--note", default="", help="A line shown at the top, e.g. for demo data")
    ap.add_argument("--standalone", action="store_true", help="Add <!doctype html> and head tags for opening locally")
    args = ap.parse_args()

    config = json.loads((args.dump / "config.json").read_text())
    meta = [json.loads(line) for line in (args.dump / "meta.jsonl").open()]
    languages = [la for la in ORDER if any(m["language"] == la for m in meta)]

    probes: dict = defaultdict(lambda: defaultdict(dict))
    layers: set[int] = set()
    for r in csv.DictReader((args.dump / "probes.csv").open()):
        layer = int(r["layer"])
        layers.add(layer)
        probes[r["target"]][f'{r["train_language"]}>{r["test_language"]}'][layer] = float(r["accuracy"])
    layer_index = sorted(layers)
    probes_out = {
        target: {key: [clean(by_layer.get(la)) for la in layer_index] for key, by_layer in keys.items()}
        for target, keys in probes.items()
    }

    sim: dict = defaultdict(lambda: {"matched": {}, "other": {}})
    for r in csv.DictReader((args.dump / "similarity.csv").open()):
        sim[r["pair"]]["matched"][int(r["layer"])] = float(r["cosine"])
        sim[r["pair"]]["other"][int(r["layer"])] = float(r["cosine_mismatched"])
    sim_out = {
        pair: {k: [round(v.get(la, 0.0), 4) for la in layer_index] for k, v in d.items()} for pair, d in sim.items()
    }

    fert: dict = defaultdict(list)
    for m in meta:
        fert[m["language"]].append(m["fertility"])

    data = {
        "model": Path(config["model"]).name if Path(config["model"]).exists() else config["model"],
        "layers": config["layers"],
        "tasks": len(meta),
        "position": "user_mean",
        "note": args.note,
        "languages": languages,
        "layer_index": layer_index,
        "probes": probes_out,
        "similarity": sim_out,
        "fertility": {la: sum(v) / len(v) for la, v in fert.items()},
        "actions": first_actions(meta),
        "behaviour": json.loads(args.behaviour.read_text()) if args.behaviour else None,
    }
    blob = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    page = (HERE / "circuit_viewer.html").read_text().replace("/*__DATA__*/null", blob, 1)
    if args.standalone:
        page = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + page.replace("<main>", "</head>\n<body>\n<main>", 1)
            + "\n</body>\n</html>\n"
        )
    out = args.out or args.dump / "circuit.html"
    out.write_text(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
