"""Activation patching between matched twins: is the placement of a Twi–English request at the
end of the request a cause of the failures that follow it?

A decision is patched at the chat-template tokens that follow the user's request (identical
across the twins of a seed, so they line up one to one): the residual stream there is set,
after one layer or after every layer, to what another prompt produced at the same tokens.
The model's own reasoning then runs to its action, and the margin of the right moves is
read as in the steer lens. Sources, for a failed Twi–English (or Twi) decision whose English
twin succeeded:

- `en`: the English twin (the swap);
- `half`: the decision's own state moved half way to the twin's;
- `random`: a random direction with the norm of (twin − own), the disruption control;
- `en_other`: an English request of the same task family from another seed;
- `tw`: the Twi twin (same content, other language).

The reverse arm puts the Twi–English twin's state (`en-tw`) into the succeeding English
decision, against `random`. The GPU side (`modal_app.Patches`) does the passes; this module
holds what can run without a model: the arms, the template-token rule, the condition grid and
the layer choice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import mean

from sika_interp.words import tag

SOURCES = ("en", "half", "random", "en_other", "tw")
"""Sources for the primary (Twi–English) and Twi arms, the swap first."""
REVERSE_SOURCES = ("en-tw", "random")
"""Sources for the reverse arm, the swap first."""
CONTROLS = {"primary": ("random", "en_other"), "twi": ("random", "en_other"), "reverse": ("random",)}
"""The controls the swap is measured against, per arm."""
ALL = -1
"""The layer index meaning "after every layer at once"."""


def request(messages: Sequence[dict]) -> str:
    """The user's request: the first user message."""
    return next(m.get("content") or "" for m in messages if m["role"] == "user")


def has_twi(text: str) -> bool:
    """Whether the wording has any Twi of its own (a Twi or mixed word by the tagger)."""
    return any(w.tag in ("tw", "mixed") for w in tag(text))


def shared_prefix(*sequences: Sequence) -> int:
    """How many leading items every sequence shares."""
    n = min((len(s) for s in sequences), default=0)
    for i in range(n):
        first = sequences[0][i]
        if any(s[i] != first for s in sequences[1:]):
            return i
    return n


def template_span(prompt: Sequence[int], turn0: Sequence[int], user_end: int) -> int:
    """How many template tokens follow the request at `user_end` in a decision's prompt: the
    tokens it shares there with the episode's first-turn prompt (the request followed by the
    generation prompt). At the first turn that is the whole generation prompt; at a later turn
    the shared run stops where the earlier model turn begins."""
    if user_end > len(prompt) or user_end > len(turn0) or prompt[:user_end] != turn0[:user_end]:
        return 0
    return shared_prefix(prompt[user_end:], turn0[user_end:])


def other_seed(candidates: Iterable[dict], seed: int) -> dict | None:
    """The candidate with the smallest seed above `seed`, else the smallest seed of all."""
    others = sorted((c for c in candidates if c["seed"] != seed), key=lambda c: c["seed"])
    if not others:
        return None
    return next((c for c in others if c["seed"] > seed), others[0])


def arms(decisions: Iterable[dict]) -> dict[str, list[dict]]:
    """The patching arms from first-move decisions (as `decisions.collect` gives them, all three
    languages of each seed). Each item: the decision to patch (`decision`), the first-turn
    messages of each source prompt (`sources`, with the decision's own as `own`), the swap
    source's name (`swap`) and the arm.

    - `primary`: Twi–English failed (reward below 1), its English twin succeeded, and the
      wording has Twi of its own;
    - `twi`: Twi failed, English twin succeeded;
    - `reverse`: the English twin of each primary decision, receiving the Twi–English state.

    An item needs every source: a decision whose twin never reached a first move, or whose
    family has no other seed where English succeeded, is left out (`dropped` counts why)."""
    rows = [d for d in decisions if d["kind"] == "first_move"]
    by = {(d["family"], d["seed"], d["language"]): d for d in rows}
    english_right: dict[str, list[dict]] = {}
    for (family, _, language), d in by.items():
        if language == "en" and d["reward"] == 1:
            english_right.setdefault(family, []).append(d)
    out: dict[str, list[dict]] = {"primary": [], "twi": [], "reverse": [], "dropped": []}

    def drop(d: dict, why: str) -> None:
        out["dropped"].append({"task_id": d["task_id"], "why": why})

    for (family, seed, language), d in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if language == "en" or d["reward"] == 1:
            continue
        en = by.get((family, seed, "en"))
        if en is None or en["reward"] != 1:
            continue
        if language == "en-tw" and not has_twi(request(d["messages"])):
            drop(d, "no Twi in the wording")
            continue
        tw = by.get((family, seed, "tw" if language == "en-tw" else "en-tw"))
        other = other_seed(english_right.get(family, []), seed)
        if other is None:
            drop(d, "no other seed with English right")
            continue
        sources = {"own": d["messages"][:2], "en": en["messages"][:2], "en_other": other["messages"][:2]}
        if language == "en-tw":
            if tw is None:
                drop(d, "no Twi twin with a first move")
                continue
            sources["tw"] = tw["messages"][:2]
            out["primary"].append({"arm": "primary", "decision": d, "sources": sources, "swap": "en"})
            out["reverse"].append(
                {"arm": "reverse", "decision": en, "sources": {"own": en["messages"][:2], "en-tw": d["messages"][:2]}, "swap": "en-tw"}
            )
        else:
            # The Twi arm's "tw" source is the Twi–English twin: same content, the other wording.
            if tw is None:
                drop(d, "no Twi–English twin with a first move")
                continue
            sources["tw"] = tw["messages"][:2]
            out["twi"].append({"arm": "twi", "decision": d, "sources": sources, "swap": "en"})
    return out


def conditions(n_layers: int, sources: Sequence[str]) -> list[tuple[str, int]]:
    """The condition grid: every source after each layer, then every source after all layers.
    The unpatched pass is run separately as the zero of each shift."""
    return [(s, layer) for s in sources for layer in range(n_layers)] + [(s, ALL) for s in sources]


def by_layer(rows: Sequence[tuple[str, int]], decisions: Sequence[dict], margin: str = "right") -> dict:
    """Mean shift of `<margin>_margin` per (source, layer) over the decisions that have it.
    Layer `ALL` is keyed "all"."""
    sums: dict[tuple[str, int], list[float]] = {}
    for d in decisions:
        shifts = d.get(f"{margin}_shift")
        if shifts is None:
            continue
        for (source, layer), shift in zip(rows, shifts):
            sums.setdefault((source, layer), []).append(shift)
    return {f"{source}:{'all' if layer == ALL else layer}": round(mean(v), 4) for (source, layer), v in sums.items()}


def swap_minus_controls(rows: Sequence[tuple[str, int]], decisions: Sequence[dict], swap: str, controls: Sequence[str],
                        margin: str = "right") -> dict[str, float]:
    """Per layer: the swap's mean shift minus the mean over the controls' mean shifts."""
    means = by_layer(rows, decisions, margin)
    layers = sorted({layer for _, layer in rows}, key=lambda x: (x == ALL, x))
    out = {}
    for layer in layers:
        key = "all" if layer == ALL else layer
        if f"{swap}:{key}" not in means:
            continue
        pool = [means[f"{c}:{key}"] for c in controls if f"{c}:{key}" in means]
        out[str(key)] = round(means[f"{swap}:{key}"] - (mean(pool) if pool else 0.0), 4)
    return out


def choose_layers(rows: Sequence[tuple[str, int]], decisions: Sequence[dict], swap: str, controls: Sequence[str],
                  n: int = 2, margin: str = "right") -> dict:
    """The `n` single layers with the largest swap-minus-controls shift on even seeds, and the
    same contrast at those layers on the odd seeds, held out."""
    even = [d for d in decisions if d["seed"] % 2 == 0]
    odd = [d for d in decisions if d["seed"] % 2 == 1]
    on_even = swap_minus_controls(rows, even, swap, controls, margin)
    single = {k: v for k, v in on_even.items() if k != "all"}
    chosen = sorted(single, key=lambda k: -single[k])[:n]
    on_odd = swap_minus_controls(rows, odd, swap, controls, margin)
    return {
        "chosen": [int(k) for k in chosen],
        "even": {k: on_even[k] for k in chosen},
        "odd_held_out": {k: on_odd.get(k) for k in chosen},
        "all_layers_even": on_even.get("all"),
        "all_layers_odd": on_odd.get("all"),
        "n_even": len(even),
        "n_odd": len(odd),
    }
