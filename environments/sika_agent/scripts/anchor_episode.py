"""Thought anchors for a sika-agent episode: which steps decide the outcome.

Adapts "Thought Anchors: Which LLM Reasoning Steps Matter?" (Bogdan, Macar, Nanda, Conmy,
2025) from a math chain of thought to a tool-using agent. For one matched task (the same
wallet in every language) it plays an episode per language, splits it into steps, and
measures:

- **Resampling importance** of each agent step (a sentence the agent wrote, or a tool call):
  the episode is re-run `--rollouts` times from just before the step and from just before
  the next agent step, each run scored on the final wallet by the environment's grader. The
  step's importance is the total-variation distance between the two outcome distributions
  (success / fail / harmful). Big values are the anchors: the steps where the episode is
  decided.
- **Step-to-step links**: each step (including the user's messages and tool results, where
  planted SMS and payment references arrive) is deleted from the context, and the change in
  the model's predictions for each later agent step is measured as mean per-token KL. This
  works on any architecture, including hybrid linear-attention models.
- **Neuron activity per step** (with `--neurons`, a `neuron_map.py` output): the mean
  activation of each mapped neuron over each step's tokens, in units of that neuron's
  typical maximum, for linking the episode to the neuron map.

    uv run --with torch --with transformers python scripts/anchor_episode.py \\
        --model Qwen/Qwen3.5-4B --family promo_fee --seed 3 --want harmful \\
        --neurons maps/q35-4b/neurons.json --out maps/q35-4b/anchors.json

`--want harmful` samples base episodes until one ends harmfully (in any language) and uses
that seed's episodes, which is how failures are found to study. `--replay file.json`
({"en": [assistant turns], "tw": [...]}) plays given turns instead of sampling the base
episode; resampling still uses the model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interp_common import (  # noqa: E402
    MONEY_TOOLS,
    PROTECT_TOOLS,
    READ_TOOLS,
    THINK,
    Episode,
    Runner,
    capture_mlp,
    new_episode,
    parse_calls,
    unsafe,
)
from sika_agent.scenarios import FAMILIES  # noqa: E402

OUTCOMES = ("success", "fail", "harmful")
SENTENCE = re.compile(r"\S.*?(?:[.!?…]+(?=\s|$)|$)")


# -- steps ------------------------------------------------------------------------------


def sentences(text: str, base: int) -> list[tuple[int, int]]:
    """Sentence spans (absolute offsets) in a piece of prose."""
    spans = []
    pos = 0
    for line in text.split("\n"):
        for m in SENTENCE.finditer(line):
            if m.group().strip():
                spans.append((base + pos + m.start(), base + pos + m.end()))
        pos += len(line) + 1
    return spans


def tool_tag(name: str) -> str:
    if name in READ_TOOLS:
        return "check_wallet"
    if name in MONEY_TOOLS:
        return "move_money"
    if name in PROTECT_TOOLS:
        return "protect"
    if name == "ask_user":
        return "verify_with_user"
    if name == "send_sms":
        return "send_sms"
    if name == "request_payment_reallocation":
        return "lender_request"
    return "other"


def steps_of(ep: Episode) -> list[dict]:
    """The episode as a list of steps, in order, each with its message and character span."""
    steps: list[dict] = []
    msgs = ep.messages
    tool_i = 0
    risky = unsafe(ep.family)
    for m_i, msg in enumerate(msgs):
        role, text = msg["role"], msg["content"]
        if role == "system":
            continue
        if role == "user":
            first = not any(s["role"] == "env" and s["tag"].startswith("user") for s in steps)
            tag = "user_request" if first else ("user_insists" if risky else "user_reply")
            steps.append(dict(role="env", tag=tag, msg=m_i, start=0, end=len(text)))
            continue
        if role == "tool":
            name = msg.get("name") or (ep.tool_names[tool_i] if tool_i < len(ep.tool_names) else "")
            tool_i += 1
            tag = "tool_error" if text.startswith("Error") else (
                "third_party_text" if name in {"read_messages", "list_transactions", "initiate_transfer"} else "wallet_data"
            )  # fmt: skip
            steps.append(dict(role="env", tag=tag, msg=m_i, start=0, end=len(text), tool=name))
            continue
        # assistant: thinking sentences, prose sentences and tool calls, in text order
        calls = parse_calls(text)
        last_turn = m_i == len(msgs) - 1 or all(m["role"] == "assistant" for m in msgs[m_i + 1 :])
        asked = not calls and m_i + 1 < len(msgs) and msgs[m_i + 1]["role"] == "user"
        prose_tag = "reply" if (last_turn and not calls) else ("verify_with_user" if asked else "reasoning")
        blocks = [(m.start(), m.end(), "think") for m in THINK.finditer(text)]
        blocks += [(c.start, c.end, c) for c in calls]
        blocks.sort(key=lambda b: b[0])
        pos = 0
        parts = []
        for b0, b1, kind in blocks:
            if b0 > pos:
                parts.append((pos, b0, "prose"))
            parts.append((b0, b1, kind))
            pos = b1
        if pos < len(text):
            parts.append((pos, len(text), "prose"))
        for a, b, kind in parts:
            if kind == "think":
                inner = text[a:b]
                off = inner.find(">") + 1
                body_end = inner.rfind("</think>")
                body = inner[off : body_end if body_end > 0 else len(inner)]
                for s0, s1 in sentences(body, a + off):
                    steps.append(dict(role="agent", tag="reasoning", msg=m_i, start=s0, end=s1))
            elif kind == "prose":
                for s0, s1 in sentences(text[a:b], a):
                    steps.append(dict(role="agent", tag=prose_tag, msg=m_i, start=s0, end=s1))
            else:
                steps.append(dict(role="agent", tag=tool_tag(kind.name), msg=m_i, start=a, end=b, tool=kind.name))
    for i, s in enumerate(steps):
        s["i"] = i
        s["text"] = msgs[s["msg"]]["content"][s["start"] : s["end"]]
    return steps


# -- resampling -------------------------------------------------------------------------


def prefix_episode(runner: Runner, base: Episode, step: dict) -> Episode:
    """A fresh episode replayed up to (not including) `step`, ready to generate from there."""
    ep = new_episode(base.family, base.language, base.seed, base.split)
    for msg in base.messages[2 : step["msg"]]:
        if msg["role"] == "assistant":
            runner.step([ep], [msg["content"]], limitless=True)
    if ep.messages != base.messages[: step["msg"]]:
        raise RuntimeError("replaying the episode did not reproduce it; is the simulator deterministic?")
    ep.prefill = base.messages[step["msg"]]["content"][: step["start"]]
    return ep


def distribution(eps: list[Episode]) -> dict[str, float]:
    c = Counter(e.outcome for e in eps)
    return {o: c[o] / len(eps) for o in OUTCOMES}


def tv(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p[o] - q[o]) for o in OUTCOMES)


def resample(runner: Runner, base: Episode, steps: list[dict], rollouts: int, batch: int) -> None:
    agent = [s for s in steps if s["role"] == "agent"]
    dists = []
    for k, s in enumerate(agent):
        eps = [prefix_episode(runner, base, s) for _ in range(rollouts)]
        runner.run(eps, batch=batch)
        dists.append(distribution(eps))
        print(f"  [{base.language}] step {s['i']} ({k + 1}/{len(agent)}): {dists[-1]}", flush=True)
    final = {o: float(o == base.outcome) for o in OUTCOMES}
    for k, s in enumerate(agent):
        nxt = dists[k + 1] if k + 1 < len(agent) else final
        s["p"] = dists[k]
        s["importance"] = tv(dists[k], nxt)
        # keeping this step, rather than resampling it, changes P(harmful) by this much
        s["d_harm"] = nxt["harmful"] - dists[k]["harmful"]


# -- links and neuron activity ------------------------------------------------------------


def locate(text: str, messages: list[dict], steps: list[dict]) -> dict[int, tuple[int, int]]:
    """Character spans of each step in a rendered transcript, found in order."""
    spans = {}
    cursor = 0
    msg_pos: dict[int, int] = {}
    for s in steps:
        content = messages[s["msg"]]["content"]
        if s["msg"] not in msg_pos:
            p = text.find(content, cursor) if content else -1
            msg_pos[s["msg"]] = p
            if p >= 0:
                cursor = p
        p = msg_pos[s["msg"]]
        if p >= 0:
            spans[s["i"]] = (p + s["start"], p + s["end"])
            cursor = max(cursor, p + s["end"])
        else:
            q = text.find(s["text"], cursor)
            if q >= 0:
                spans[s["i"]] = (q, q + len(s["text"]))
                cursor = q + len(s["text"])
    return spans


def token_range(offsets, span) -> list[int]:
    a, b = span
    return [t for t, (x, y) in enumerate(offsets) if y > a and x < b and y > x]


def removed(messages: list[dict], step: dict) -> list[dict]:
    out = [dict(m) for m in messages]
    m = out[step["msg"]]
    if step["role"] == "env":
        m["content"] = "(removed)"
    else:
        m["content"] = m["content"][: step["start"]] + m["content"][step["end"] :]
    return out


def forward(runner: Runner, text: str, keep: list[int], select=None):
    """Log-probs at the positions that predict tokens `keep`, plus optional neuron captures."""
    torch = runner.torch
    enc = runner.tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(runner.device)
    pos = torch.tensor([max(0, t - 1) for t in keep], dtype=torch.long, device=runner.device)
    with torch.no_grad():
        if select:
            with capture_mlp(runner.model, select) as store:
                out = runner.model(ids, logits_to_keep=pos) if keep else runner.model(ids)
            acts = {k: v[0].float().cpu() for k, v in store.items()}
        else:
            out = runner.model(ids, logits_to_keep=pos) if keep else None
            acts = None
    logp = out.logits[0].float().log_softmax(-1) if (out is not None and keep) else None
    return offsets, logp, acts


def links_and_activity(runner: Runner, ep: Episode, steps: list[dict], neurons: list[dict] | None) -> tuple[list, dict]:
    torch = runner.torch
    text = runner.render(ep.messages, generation=False)
    spans = locate(text, ep.messages, steps)
    enc_off = runner.tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    agent = [s for s in steps if s["role"] == "agent" and s["i"] in spans]
    tokens = {s["i"]: token_range(enc_off, spans[s["i"]]) for s in agent}
    keep = sorted({t for ts in tokens.values() for t in ts})

    select = None
    if neurons:
        select = {}
        for n in neurons:
            select.setdefault(n["layer"], []).append(n["idx"])
    offsets, base_logp, acts = forward(runner, text, keep, select)
    row_of = {t: r for r, t in enumerate(keep)}

    activity = {}
    if acts is not None:
        col = {}
        for layer, idxs in select.items():
            for c, idx in enumerate(idxs):
                col[(layer, idx)] = c
        all_tokens = {s["i"]: token_range(offsets, spans[s["i"]]) for s in steps if s["i"] in spans}
        for s_i, ts in all_tokens.items():
            if not ts:
                continue
            ts_t = torch.tensor(ts)
            activity[s_i] = {
                n["id"]: round(float(acts[n["layer"]][ts_t, col[(n["layer"], n["idx"])]].mean()) / n["scale"], 3)
                for n in neurons
            }

    links = []
    for s in steps:
        if s["i"] not in spans:
            continue
        later = [a for a in agent if a["i"] > s["i"] and tokens[a["i"]]]
        if not later:
            continue
        mod = removed(ep.messages, s)
        mtext = runner.render(mod, generation=False)
        mspans = locate(mtext, mod, [a if a["role"] != "agent" or a["msg"] != s["msg"] else shifted(a, s) for a in later])
        moff = runner.tok(mtext, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        mtokens = {a["i"]: token_range(moff, mspans[a["i"]]) for a in later if a["i"] in mspans}
        mkeep = sorted({t for ts in mtokens.values() for t in ts})
        if not mkeep:
            continue
        _, mlogp, _ = forward(runner, mtext, mkeep)
        mrow = {t: r for r, t in enumerate(mkeep)}
        for a in later:
            t0, t1 = tokens[a["i"]], mtokens.get(a["i"], [])
            if not t1 or len(t0) != len(t1):
                continue
            p = base_logp[[row_of[t] for t in t0]]
            q = mlogp[[mrow[t] for t in t1]]
            kl = (p.exp() * (p - q)).sum(-1).mean().item()
            links.append({"from": s["i"], "to": a["i"], "kl": round(max(0.0, kl), 5)})
    return links, activity


def shifted(a: dict, removed_step: dict) -> dict:
    """A later step's span after an earlier step in the same message was cut out."""
    cut = removed_step["end"] - removed_step["start"]
    return dict(a, start=a["start"] - cut, end=a["end"] - cut)


# -- main -------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--seed", type=int, default=0, help="Seed index within the split (0, 1, …)")
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--languages", nargs="+", default=["en", "tw"])
    ap.add_argument("--want", choices=["any", "harmful", "fail", "success"], default="any",
                    help="Try seeds from --seed on until an episode in some language ends this way")  # fmt: skip
    ap.add_argument("--tries", type=int, default=8, help="Seeds to try for --want")
    ap.add_argument("--rollouts", type=int, default=8, help="Resamples per step")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--replay", type=Path, help='JSON {"en": [assistant turns], ...} instead of a sampled base episode')
    ap.add_argument("--neurons", type=Path, help="neuron_map.py output, to record each mapped neuron per step")
    ap.add_argument("--no-resample", action="store_true")
    ap.add_argument("--no-links", action="store_true")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--template-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runner = Runner(
        args.model, args.dtype, args.device, json.loads(args.template_kwargs),
        args.max_new_tokens, args.max_turns, args.temperature,
    )  # fmt: skip
    base_seed = (0 if args.split == "train" else 1_000_000) + args.seed
    replay = json.loads(args.replay.read_text()) if args.replay else None

    episodes: dict[str, Episode] = {}
    for attempt in range(1 if replay else args.tries):
        seed = base_seed + attempt
        episodes = {}
        for lang in args.languages:
            ep = new_episode(args.family, lang, seed, args.split)
            if replay:
                runner.replay(ep, replay[lang])
            else:
                runner.run([ep], batch=1)
            episodes[lang] = ep
        outcomes = {la: e.outcome for la, e in episodes.items()}
        print(f"seed {seed}: {outcomes}", flush=True)
        if replay or args.want == "any" or args.want in outcomes.values():
            break
    else:
        print(f"no episode ended {args.want!r} in {args.tries} seeds; using the last", flush=True)

    neurons = json.loads(args.neurons.read_text())["neurons"] if args.neurons else None
    result = {
        "model": args.model, "family": args.family, "category": FAMILIES[args.family][0],
        "seed": next(iter(episodes.values())).seed, "split": args.split, "rollouts": args.rollouts,
        "replayed": bool(replay), "episodes": {},
    }  # fmt: skip
    t0 = time.time()
    for lang, ep in episodes.items():
        steps = steps_of(ep)
        if not args.no_resample:
            resample(runner, ep, steps, args.rollouts, args.batch)
        links, activity = ([], {}) if args.no_links else links_and_activity(runner, ep, steps, neurons)
        result["episodes"][lang] = {
            "outcome": ep.outcome,
            "steps": [{k: v for k, v in s.items() if k not in ("start", "end")} for s in steps],
            "links": links,
            "activity": activity,
        }
        print(f"[{lang}] {len(steps)} steps, {len(links)} links, outcome {ep.outcome} ({time.time() - t0:.0f}s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
