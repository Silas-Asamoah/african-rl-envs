"""Build the Sika Anchors page from `anchor_episode.py` and `neuron_map.py` outputs.

    python3 scripts/build_anchor_page.py maps/q35-4b/anchors.json \\
        --neurons maps/q35-4b/neurons.json --out maps/q35-4b/anchors.html

The page is self-contained apart from three.js, which it loads from a CDN for the 3D neuron
map: open it in a browser, or publish it. Needs only the standard library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("anchors", type=Path, help="anchor_episode.py output")
    ap.add_argument("--neurons", type=Path, help="neuron_map.py output")
    ap.add_argument("--out", type=Path, help="Default: anchors.html beside the anchors file")
    ap.add_argument("--note", default="", help="A line shown at the top, e.g. for demo data")
    ap.add_argument("--standalone", action="store_true", help="Add <!doctype html> and head tags for opening locally")
    args = ap.parse_args()

    data = json.loads(args.anchors.read_text())
    model = data.get("model", "")
    data["model"] = Path(model).name if model and Path(model).exists() else model
    data["note"] = args.note
    if args.neurons:
        neurons = json.loads(args.neurons.read_text())
        neurons["model"] = Path(neurons["model"]).name if Path(neurons["model"]).exists() else neurons["model"]
        data["neurons"] = neurons
    for ep in data["episodes"].values():  # JSON object keys arrive as strings
        ep["activity"] = {int(k): v for k, v in ep.get("activity", {}).items()}

    blob = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    page = (HERE / "anchors_viewer.html").read_text().replace("/*__DATA__*/null", blob, 1)
    if args.standalone:
        page = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + page.replace('<div class="wrap">', '</head>\n<body>\n<div class="wrap">', 1)
            + "\n</body>\n</html>\n"
        )
    out = args.out or args.anchors.with_name("anchors.html")
    out.write_text(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
